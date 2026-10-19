"""Approximate spatial and reference-group EDA, no identity inference."""
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from basic_shape_eda import ROOT,fmt,md_table,log
from panel_eda_common import load_panel,clock,centered,haversine,hashes,finish,verify_base,save_check
from eda_plot_utils import save_plot

OUT=ROOT/'data/eda/2026-10-19_spatial_group'
REPORT=ROOT/'reports/2026-10-19_12_공간과_참조그룹의_관계_EDA.md'
FILES=['sites.csv','train/generation.parquet'];G=[]


def main():
    sys.stdout.reconfigure(encoding='utf-8');assert sys.version_info[:2]==(3,13)
    OUT.mkdir(parents=True,exist_ok=True)
    if '--verify-only' in sys.argv:
        info,check=verify_base(OUT,REPORT,FILES,6)
        pairs=pd.read_csv(OUT/'spatial_pairs.csv',encoding='utf-8-sig')
        bins=pd.read_csv(OUT/'distance_bins.csv',encoding='utf-8-sig');groups=pd.read_csv(OUT/'groups.csv',encoding='utf-8-sig')
        assert len(pairs)==10878 and pairs.distance_km.ge(0).all() and bins.pairs.sum()==10878
        assert groups.groupby('window').plants.sum().to_dict()=={'2024 고정':148,'메타 전체':253,'전체 공통 시각':253}
        save_check(OUT,{**check,'pairs':10878,'distance_bin_accounting':True,'group_composition_accounting':True});return
    before=hashes(FILES);sites,gen,energy,cf=load_panel();ids=cf.columns;s=sites.loc[ids];t=clock(cf.index)
    day=t.hour.between(9,16);within=centered(cf).loc[day];rday=cf.loc[day].corr().to_numpy();rwithin=within.corr().to_numpy()
    dist=haversine(s.lat,s.lon);i,j=np.triu_indices(148,1)
    pairs=pd.DataFrame({'distance_km':dist[i,j],'r_day':rday[i,j],'r_within':rwithin[i,j],
                        'same_wa_station':s.wa_station.to_numpy()[i]==s.wa_station.to_numpy()[j],
                        'same_wb_station':s.wb_station.to_numpy()[i]==s.wb_station.to_numpy()[j],
                        'same_group':s.current_group.to_numpy()[i]==s.current_group.to_numpy()[j]})
    pairs['distance_bin']=pd.cut(pairs.distance_km,[0,10,30,60,100,200,np.inf],right=False,
                                 labels=['<10','10~<30','30~<60','60~<100','100~<200','>=200'])
    bins=pairs.groupby('distance_bin',observed=True).agg(pairs=('r_within','size'),median_day=('r_day','median'),
        median_within=('r_within','median'),q25=('r_within',lambda x:x.quantile(.25)),q75=('r_within',lambda x:x.quantile(.75))).reset_index()
    conditions=[]
    for condition in ['same_wa_station','same_wb_station','same_group']:
        for key,z in pairs.groupby(condition):conditions.append({'condition':condition,'same':bool(key),'pairs':len(z),
            'median_distance_km':z.distance_km.median(),'median_day':z.r_day.median(),'median_within':z.r_within.median()})
    full=gen.pivot(index='ts',columns='site_id',values='energy').sort_index()
    common=full.loc[full.notna().all(axis=1)].mask(lambda x:x<0).dropna()
    commoncf=common.div(sites.loc[common.columns,'capacity_kw'],axis=1)
    groups=[];curves=[]
    for window,x,meta in [('메타 전체',None,sites),('2024 고정',cf,s),('전체 공통 시각',commoncf,sites)]:
        for group,m in meta.groupby('current_group'):
            row={'window':window,'group':group,'plants':len(m),'capacity_kw':m.capacity_kw.sum(),
                 'median_capacity_kw':m.capacity_kw.median(),'median_lat':m.lat.median(),'median_lon':m.lon.median(),
                 'wa_distance_median_m':m.wa_dist.median(),'wb_distance_median_m':m.wb_dist.median()}
            if x is not None:
                z=x[m.index];weights=m.capacity_kw/m.capacity_kw.sum();weighted=z@weights
                tt=clock(x.index);low=tt.hour.between(9,16)
                row.update(equal_site_mean_cf=z.mean().mean(),capacity_weighted_mean_cf=weighted.mean(),
                    day_weighted_std=weighted.loc[low].std(),hours=len(z))
                for dimension in ['month','hour']:
                    for k,v in weighted.groupby(tt[dimension]).mean().items():
                        curves.append({'window':window,'group':group,'dimension':dimension,'key':int(k),'weighted_cf':v})
            groups.append(row)
    site_stats=s.copy();site_stats['mean_cf']=cf.mean();site_stats['day_std']=cf.loc[day].std();site_stats['within_day_std']=within.std()
    assoc=[]
    for x in ['lat','lon','wa_dist','wb_dist','capacity_kw']:
        for y in ['mean_cf','day_std','within_day_std']:
            assoc.append({'x':x,'y':y,'plants':148,'spearman':site_stats[x].corr(site_stats[y],method='spearman')})
    fig,axes=plt.subplots(1,2,figsize=(13,5))
    axes[0].hexbin(pairs.distance_km,pairs.r_within,gridsize=45,mincnt=1,bins='log');axes[0].set_xlabel('제공 좌표 간 근사 거리 (km)');axes[0].set_ylabel('낮 평균 편차 Pearson')
    axes[1].errorbar(range(len(bins)),bins.median_within,yerr=[bins.median_within-bins.q25,bins.q75-bins.median_within],fmt='o-')
    axes[1].set_xticks(range(len(bins)),bins.distance_bin);axes[1].set_xlabel('근사 거리 구간 (km)');axes[1].set_ylabel('중앙값과 IQR')
    save_plot(fig,OUT,G,'01_distance','발전소 사이의 거리와 동시 변동','좌표는 최대 2 km 이동/반올림된 값. 근거리의 정밀 거리와 원래 위치를 추정하지 않음. IQR은 신뢰구간이 아님.')
    fig,axes=plt.subplots(1,3,figsize=(14,5))
    for ax,key in zip(axes,['same_wa_station','same_wb_station','same_group']):
        z=pd.DataFrame(conditions);z=z.loc[z.condition==key];ax.bar(z.same.astype(str),z.median_within);ax.set_title(key);ax.set_ylim(0,1);ax.set_ylabel('낮 편차 상관 중앙값')
    save_plot(fig,OUT,G,'02_shared','같은 기상 지점과 참조 그룹의 쌍','공유 여부와 거리 및 지역 구성이 함께 다름. 기상 지점을 공유한 효과나 현재 그룹의 우수성을 추정한 결과가 아님.')
    fig,axes=plt.subplots(1,2,figsize=(12,5));groupframe=pd.DataFrame(groups)
    for ax,col in zip(axes,['plants','capacity_kw']):
        groupframe.pivot(index='window',columns='group',values=col).plot.bar(ax=ax,rot=10);ax.set_ylabel(col)
    save_plot(fig,OUT,G,'03_group_composition','참조 그룹의 발전소 수와 설비용량','current_group은 현재 참조 배정이며 과거 실제 배정 이력이나 최적 배정이 아님.')
    curve=pd.DataFrame(curves);fig,axes=plt.subplots(1,2,figsize=(13,5))
    for ax,dim in zip(axes,['hour','month']):
        for group,z in curve.loc[(curve.window=='2024 고정')&(curve.dimension==dim)].groupby('group'):ax.plot(z.key,z.weighted_cf,label=group)
        ax.set_xlabel('구간 시작 시간' if dim=='hour' else '월');ax.set_ylabel('용량 가중 발전 비율');ax.legend()
    save_plot(fig,OUT,G,'04_group_patterns','같은 2024년에서 참조 그룹별 관측 패턴','현재 메타 배정을 과거 기록에 적용한 설명용 집계. 그룹의 인과 효과나 재배정 결과가 아님.')
    fig,axes=plt.subplots(1,2,figsize=(12,5))
    for ax,col in zip(axes,['wa_dist','wb_dist']):
        ax.scatter(site_stats[col]/1000,site_stats.mean_cf,alpha=.65);ax.set_xlabel(col+' (km)');ax.set_ylabel('연평균 발전 비율')
    save_plot(fig,OUT,G,'05_station_distance','제공된 기상 지점 거리와 발전 비율','거리 자체의 인과 영향과 지점 대표성/예측 성능은 판단하지 않음. 좌표로 재계산한 거리가 아닌 제공 거리 사용.')
    fig,axes=plt.subplots(1,2,figsize=(12,5))
    for ax,col in zip(axes,['lat','lon']):
        ax.scatter(site_stats[col],site_stats.mean_cf,alpha=.65);ax.set_xlabel('제공 '+col);ax.set_ylabel('연평균 발전 비율')
    save_plot(fig,OUT,G,'06_coordinates','익명화 좌표 수준과 발전 비율','좌표와 발전량의 기술통계만 비교. 지도나 외부 위치 조회, 원위치 및 사업자 추정 없음.')
    info={'plants':148,'pairs':len(pairs),'distance_within_spearman':pairs.distance_km.corr(pairs.r_within,method='spearman')}
    finish(OUT,info,FILES,before,{'spatial_pairs':pairs,'distance_bins':bins,'shared_conditions':pd.DataFrame(conditions),
        'groups':groupframe,'group_curves':curve,'site_metadata_associations':pd.DataFrame(assoc)},G,'공간과 참조 그룹 관계 EDA')
    text=['# 공간과 참조 그룹의 관계 EDA','',f"작성: {info['finished_at']}. 실제 계산 완료.",'','## 질문과 목적','',
        '발전소 간 근사 거리와 공동 변동, 공유 기상 지점, 현재 참조 그룹의 구성과 발전 패턴을 확인한다. 좌표는 익명화된 제공값만 사용하며 원위치나 사업자를 특정하지 않았다.','',
        '## 사용 범위와 방법','','발전소 쌍은 2024년 완전 기록 148개소를 같은 시각으로 비교했다. 용량 대비 비율, 낮 시간과 발전소별 월/시간 평균을 뺀 낮 편차를 사용했다. 좌표 간 구면 거리는 기술통계용 근사치이며 배포 설명에 따르면 각 좌표는 원위치에서 최대 2 km 이동 후 반올림됐다. 따라서 두 지점 사이 거리의 왜곡은 한 지점의 이동량보다 클 수 있고 근거리 해석은 특히 제한된다.','',
        '기상 지점까지 거리는 원위치 기준의 제공 wa_dist/wb_dist를 사용하고 좌표로 재산출하지 않았다. 그룹은 sites.csv의 현재 참조 A/B를 과거 기록에 적용했다. 전체 메타, 2024 고정 집단, 253개 공통 시각을 구분했다. 검정과 인과 추정, 그룹 최적화는 수행하지 않았다.','',
        '## 근사 거리와 동시 변동','',md_table(['근사 거리 km','쌍 수','낮 상관 중앙값','낮 편차 상관 중앙값','편차 상관 IQR'],[[r.distance_bin,fmt(r.pairs),f'{r.median_day:.3f}',f'{r.median_within:.3f}',f'{r.q25:.3f} ~ {r.q75:.3f}'] for r in bins.itertuples()]),'',
        f"근사 거리와 낮 편차 상관의 Spearman은 {info['distance_within_spearman']:.3f}였다. 쌍마다 거리가 늘어날 때 상관이 반드시 단조롭게 줄어드는 뜻은 아니며, 발전소 쌍들은 지점을 공유해 서로 독립이 아니다.",'',
        md_table(['구분','공유','쌍 수','근사 거리 중앙값 km','낮 편차 상관 중앙값'],[[r['condition'],r['same'],fmt(r['pairs']),f"{r['median_distance_km']:.1f}",f"{r['median_within']:.3f}"] for r in conditions]),'',
        '## 현재 그룹의 구성과 관측 패턴','',md_table(['범위','그룹','발전소 수','총 용량 kW','용량 중앙값 kW','A/B 지점 거리 중앙값 km','용량 가중 평균 cf','낮 가중 표준편차'],[[r['window'],r['group'],r['plants'],f"{r['capacity_kw']:.1f}",f"{r['median_capacity_kw']:.1f}",f"{r['wa_distance_median_m']/1000:.1f} / {r['wb_distance_median_m']/1000:.1f}",f"{r['capacity_weighted_mean_cf']:.4f}" if 'capacity_weighted_mean_cf' in r else '해당 없음',f"{r['day_weighted_std']:.4f}" if 'day_weighted_std' in r else '해당 없음'] for r in groups]),'',
        '비교 기간, 포함 발전소와 용량 가중이 다르면 그룹 평균도 달라진다. 같은 기간의 구성 차이를 기술했으며 그룹 A/B 중 어느 배정이 우수하다고 결론내리지 않았다.','',
        '## 메타 특성과 발전소별 요약의 관계','',md_table(['메타','관측 요약','발전소 수','Spearman'],[[r['x'],r['y'],r['plants'],f"{r['spearman']:.3f}"] for r in assoc]),'',
        '연평균 cf, 낮의 산포와 달력 평균을 뺀 산포는 발전소별 요약이다. 기상 지점 거리의 상관은 예보 오차를 계산한 결과가 아니므로 거리가 멀어 예측이 나쁘다는 해석으로 확장하지 않는다.','',
        '## 한계와 다음 단계','','설비 방향, 지형, 운영 상태와 기상 차이를 동시에 분리하지 않았다. 제공 좌표와 거리로 원래 위치를 추정하지 않았다. 공간 및 그룹 관계는 관측된 범위의 기술통계로 남기고 다음으로 SMP와 발전량 관계를 살핀다. 머신러닝과 배정 최적화는 수행하지 않았다.','',
        '## 관련 코드와 결과','','- [코드](../scripts/spatial_group_eda.py)','- [전체 그림](../data/eda/2026-10-19_spatial_group/index.html)','- [거리 구간 집계](../data/eda/2026-10-19_spatial_group/distance_bins.csv)','','모든 데이터 출력은 로컬 data/에만 보관한다.','']
    REPORT.write_text('\n'.join(text),encoding='utf-8');log(str(REPORT))


if __name__=='__main__':main()
