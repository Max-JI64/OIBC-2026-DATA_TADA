"""Observed SMP-generation relationships; no price forecast or scoring."""
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from basic_shape_eda import ROOT,SOURCE,fmt,md_table,log
from panel_eda_common import load_panel,clock,hashes,finish,verify_base,save_check
from eda_plot_utils import save_plot

OUT=ROOT/'data/eda/2026-10-19_smp_generation'
REPORT=ROOT/'reports/2026-10-19_13_SMP와_발전량의_관계_EDA.md'
FILES=['sites.csv','train/generation.parquet','train/smp.csv'];G=[]


def main():
    sys.stdout.reconfigure(encoding='utf-8');assert sys.version_info[:2]==(3,13)
    OUT.mkdir(parents=True,exist_ok=True)
    if '--verify-only' in sys.argv:
        info,check=verify_base(OUT,REPORT,FILES,5)
        bins=pd.read_csv(OUT/'price_bins.csv',encoding='utf-8-sig');cond=pd.read_csv(OUT/'conditions.csv',encoding='utf-8-sig')
        assert bins.n.sum()==8784 and cond.loc[cond.condition=='전체','n'].iloc[0]==8784
        assert info['matched_price_rows']==8784 and info['price_unmatched']==0
        assert info['k_start_hours']==[10,11,12,13,14]
        save_check(OUT,{**check,'price_bin_accounting':True,'matched_hours':8784,'k_right_closed_alignment':True});return
    before=hashes(FILES);sites,gen,energy,cf=load_panel();ids=cf.columns;t=clock(cf.index)
    price=pd.read_csv(SOURCE/'train/smp.csv',encoding='utf-8-sig');price['ts']=pd.to_datetime(price.ts).dt.tz_convert('Asia/Seoul')
    assert not price.ts.duplicated().any()
    d=t.join(price.set_index('ts'),how='left',validate='one_to_one');assert d.smp.notna().all()
    d.index=pd.DatetimeIndex(d.index).tz_convert('Asia/Seoul')
    assert np.array_equal((d.index.hour-1)%24,d.hour)
    d['energy_sum']=energy.sum(axis=1);d['equal_cf']=cf.mean(axis=1);d['weighted_cf']=d.energy_sum/sites.loc[ids,'capacity_kw'].sum()
    # The supplied scoring k is defined by end timestamp hour, not interval-start hour.
    d['k']=np.where(d.index.hour.isin(range(11,16)),3,1)
    day=d.hour.between(9,16)
    within=d[['smp','weighted_cf','equal_cf']]-d.groupby(['month','hour'])[['smp','weighted_cf','equal_cf']].transform('mean')
    daily=d.groupby('day')[['smp','weighted_cf','energy_sum']].mean()
    relationships=[]
    for label,z in [('전체',d),('낮 09~17시',d.loc[day]),('같은 월/시간 편차 전체',within),('같은 월/시간 편차 낮',within.loc[day]),('일평균',daily)]:
        for col in ['weighted_cf']+(['equal_cf'] if 'equal_cf' in z else []):
            relationships.append({'view':label,'generation':col,'n':len(z),'pearson':z.smp.corr(z[col]),'spearman':z.smp.corr(z[col],method='spearman')})
    d['price_bin']=pd.qcut(d.smp,5,duplicates='drop')
    bins=d.groupby('price_bin',observed=True).agg(n=('smp','size'),price_min=('smp','min'),price_max=('smp','max'),price_mean=('smp','mean'),
        mean_cf=('weighted_cf','mean'),median_cf=('weighted_cf','median'),sum_energy=('energy_sum','sum'),day_share=('hour',lambda x:x.between(9,16).mean())).reset_index()
    high=d.smp.ge(d.smp.quantile(.9));within_high=d.smp.ge(d.groupby(['month','hour']).smp.transform(lambda x:x.quantile(.9)))
    conditions=[]
    for name,mask in [('전체',pd.Series(True,index=d.index)),('전체 SMP 상위 10% 기준 이상',high),('월/시간별 SMP 상위 10% 기준 이상',within_high),('SMP 정확히 0',d.smp.eq(0)),('k=3 시각',d.k.eq(3))]:
        z=d.loc[mask]
        conditions.append({'condition':name,'n':len(z),'share_times':len(z)/len(d),'mean_smp':z.smp.mean() if len(z) else None,
            'mean_cf':z.weighted_cf.mean() if len(z) else None,'energy_share':z.energy_sum.sum()/d.energy_sum.sum(),
            'day_share':z.hour.between(9,16).mean() if len(z) else None})
    strata=[]
    for dim in ['month','hour']:
        for key,z in d.groupby(dim):strata.append({'dimension':dim,'key':int(key),'n':len(z),'mean_smp':z.smp.mean(),
            'mean_cf':z.weighted_cf.mean(),'pearson':z.smp.corr(z.weighted_cf),'energy_sum':z.energy_sum.sum()})
    st=pd.DataFrame(strata)
    fig,axes=plt.subplots(1,2,figsize=(13,5))
    for ax,z,title in [(axes[0],d,'전체'),(axes[1],d.loc[day],'낮')]:
        ax.hexbin(z.smp,z.weighted_cf,gridsize=45,mincnt=1,bins='log');ax.set_xlabel('관측 SMP (원/kWh)');ax.set_ylabel('용량 가중 발전 비율');ax.set_title(title)
    save_plot(fig,OUT,G,'01_price_cf','같은 시각의 SMP와 발전량','2024 고정 148개소의 합산 비율, 점은 시각. 가격과 발전량의 인과 영향이나 미래 가격 정보 이용 가능성을 추정하지 않음.')
    fig,ax=plt.subplots(figsize=(8,6));ax.hexbin(within.loc[day,'smp'],within.loc[day,'weighted_cf'],gridsize=45,mincnt=1,bins='log')
    ax.set_xlabel('SMP 월/시간 평균 편차');ax.set_ylabel('발전 비율 월/시간 평균 편차')
    save_plot(fig,OUT,G,'02_within','같은 월과 시간 안의 가격-발전량 관계','그룹 평균을 뺀 기술통계이며 날짜별 수요/기상 등을 통제한 인과 분석이 아님.')
    fig,axes=plt.subplots(1,2,figsize=(13,5))
    for ax,dim in zip(axes,['hour','month']):
        z=st.loc[st.dimension==dim];ax.plot(z.key,z.mean_smp,color='#26718c',label='SMP');ax.set_ylabel('SMP 원/kWh');ax.set_xlabel(dim)
        other=ax.twinx();other.plot(z.key,z.mean_cf,color='#d7804f',label='발전 비율');other.set_ylabel('용량 가중 발전 비율')
    save_plot(fig,OUT,G,'03_clock','시간대와 월별 가격 및 발전 비율 평균','각 축의 단위가 다름. 두 곡선의 높이나 시각적 교차는 값의 일치나 인과 관계를 뜻하지 않음.')
    fig,axes=plt.subplots(1,2,figsize=(13,5));axes[0].bar(range(len(bins)),bins.mean_cf);axes[0].set_xticks(range(len(bins)),[f'{x:.1f}~{y:.1f}' for x,y in zip(bins.price_min,bins.price_max)],rotation=25)
    axes[0].set_xlabel('SMP 5분위 구간 (원/kWh)');axes[0].set_ylabel('평균 발전 비율')
    c=pd.DataFrame(conditions);axes[1].barh(c.condition,c.energy_share);axes[1].set_xlabel('전체 관측 발전량 중 비중')
    save_plot(fig,OUT,G,'04_bins_conditions','가격 구간과 특정 시각 조건의 발전량','상위 기준에 같은 가격이 있으면 정확히 10%보다 많이 선택될 수 있음. 공식 정산액/점수나 매출을 계산한 결과가 아님.')
    fig,axes=plt.subplots(1,2,figsize=(13,5))
    for ax,dim in zip(axes,['month','hour']):
        z=st.loc[st.dimension==dim];ax.bar(z.key,z.pearson);ax.set_ylim(-1,1);ax.axhline(0,color='grey',lw=.8);ax.set_xlabel(dim);ax.set_ylabel('SMP와 발전 비율 Pearson')
    save_plot(fig,OUT,G,'05_conditional_correlations','월별 및 시간별 가격-발전량 관계','각 셀에서 다른 조건은 여전히 섞여 있음. 표본은 시간적으로 독립이 아님.')
    info={'matched_price_rows':len(d),'price_unmatched':int(d.smp.isna().sum()),'k_start_hours':sorted(d.loc[d.k.eq(3),'hour'].unique().tolist())}
    finish(OUT,info,FILES,before,{'correlations':pd.DataFrame(relationships),'price_bins':bins,'conditions':c,'strata':st},G,'SMP와 발전량 관계 EDA')
    text=['# SMP와 발전량의 관계 EDA','',f"작성: {info['finished_at']}. 실제 계산 완료.",'','## 질문과 목적','',
        '발전량이 많은 시간과 가격이 높은 시간이 겹치는지, 시간/월 구성을 구분한 뒤 관계가 어떻게 달라지는지 살폈다. 제공된 관측 가격과 발전량의 기술통계이며 미래 가격 예측, 공식 점수와 수익 계산은 하지 않았다.','',
        '## 사용 범위와 방법','','2024년 완전 기록 148개 발전소, 8,784시각의 관측 발전량을 합산하고 train/smp.csv를 정확한 ts로 조인했다. 가격은 시각당 한 값이므로 발전소 행으로 반복해 독립 표본 수를 늘리지 않았다. 용량 가중 발전 비율과 발전소 동일 가중 비율을 비교했다.','',
        '낮은 시작 9~16시다. 배포 설명의 k는 종료 시각 ts의 11~15시에서 3이므로 구간 시작 10~14시에 해당한다. 이는 가격/발전 패턴의 관측 분류에만 사용했으며 공식 정산식이나 과거 예측 점수 계산은 하지 않았다. 전체/낮과 월/시간 평균 편차, 일평균 관계를 나눴다. 상위 가격은 관측된 값으로 사후 분류한 EDA이며 예측 시점에 실제 가격을 안다는 가정이 아니다.','',
        '## 가격과 발전량 관계','',md_table(['범위','발전 비율','시각/일 수','Pearson','Spearman'],[[r['view'],r['generation'],fmt(r['n']),f"{r['pearson']:.3f}",f"{r['spearman']:.3f}"] for r in relationships]),'',
        '## 가격 구간별 발전량','',md_table(['SMP 구간 원/kWh','시각 수','평균 발전 비율','중앙값 비율','낮 시각 비중'],[[f'{r.price_min:.2f} ~ {r.price_max:.2f}',fmt(r.n),f'{r.mean_cf:.4f}',f'{r.median_cf:.4f}',f'{100*r.day_share:.1f}%'] for r in bins.itertuples()]),'',
        '## 특정 조건의 발전량 비중','',md_table(['조건','시각 수','시각 비중','평균 SMP','평균 비율','전체 발전량 비중'],[[r['condition'],fmt(r['n']),f"{100*r['share_times']:.1f}%",f"{r['mean_smp']:.2f}" if r['mean_smp'] is not None else '없음',f"{r['mean_cf']:.4f}" if r['mean_cf'] is not None else '없음',f"{100*r['energy_share']:.1f}%"] for r in conditions]),'',
        '전체 상위 가격과 같은 월/시간 안의 상위 가격은 다른 질문이다. 높은 가격 시각의 발전량이 적더라도 가격이 발전량을 낮춘다는 인과 결론을 내리지 않는다. 표본 기간의 가격 수준, 계절과 시간대, 수요와 공동 기상 등이 섞여 있다.','',
        '## 한계와 다음 단계','','전체 253개소의 연간 관계를 확인한 것은 아니다. 2024년 고정 집단의 참조 그룹 구성은 치우쳐 있다. 연속한 시각은 독립 표본이 아니며 p값은 계산하지 않았다. 다음 단위에서 발전량의 자기상관, 급변과 지속 구간을 살핀다. 모델 학습과 발전량 예측은 진행하지 않았다.','',
        '## 관련 코드와 결과','','- [코드](../scripts/smp_generation_eda.py)','- [전체 그림](../data/eda/2026-10-19_smp_generation/index.html)','- [월/시간별 집계](../data/eda/2026-10-19_smp_generation/strata.csv)','','원본은 유지했고 모든 데이터 출력은 로컬 data/에만 보관한다.','']
    REPORT.write_text('\n'.join(text),encoding='utf-8');log(str(REPORT))


if __name__=='__main__':main()
