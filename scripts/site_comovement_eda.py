"""Concurrent generation movement, without optimization or prediction."""
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from basic_shape_eda import ROOT,SOURCE,fmt,md_table,log
from panel_eda_common import load_panel,clock,centered,hashes,finish,verify_base,save_check
from eda_plot_utils import save_plot

OUT=ROOT/'data/eda/2026-10-19_site_comovement'
REPORT=ROOT/'reports/2026-10-19_11_발전소_사이의_동시_변동_EDA.md'
FILES=['sites.csv','train/generation.parquet'];G=[]


def main():
    sys.stdout.reconfigure(encoding='utf-8');assert sys.version_info[:2]==(3,13)
    OUT.mkdir(parents=True,exist_ok=True)
    if '--verify-only' in sys.argv:
        info,check=verify_base(OUT,REPORT,FILES,6)
        p=pd.read_csv(OUT/'pairs.csv',encoding='utf-8-sig')
        assert len(p)==148*147//2 and not p.duplicated(['site_a','site_b']).any()
        assert p.filter(regex='^r_').abs().le(1.000001).all().all()
        assert info['raw_energy_cf_correlation_max_difference']<1e-10
        cov=pd.read_csv(OUT/'aggregation.csv',encoding='utf-8-sig')
        assert np.allclose(cov.direct_std,cov.covariance_std)
        save_check(OUT,{**check,'plant_pairs':len(p),'covariance_identity':True});return
    before=hashes(FILES);sites,gen,energy,cf=load_panel();t=clock(cf.index)
    day=t.hour.between(9,16);anomaly=centered(cf).loc[day]
    ramp=cf.diff().loc[day&t.hour.sub(1).between(9,16)]
    daily=cf.groupby(t.day).mean()
    matrices={'all':cf.corr(),'day':cf.loc[day].corr(),'within':anomaly.corr(),'ramp':ramp.corr(),'daily':daily.corr()}
    ids=cf.columns;i,j=np.triu_indices(len(ids),1)
    pairs=pd.DataFrame({'site_a':ids[i],'site_b':ids[j],**{'r_'+k:m.to_numpy()[i,j] for k,m in matrices.items()}})
    rawdiff=float(np.abs(energy.corr()-matrices['all']).to_numpy().max())
    distribution=[]
    for col in pairs.columns[2:]:
        a=pairs[col]
        distribution.append({'metric':col,'pairs':len(a),'q05':a.quantile(.05),'median':a.median(),'q95':a.quantile(.95),'negative_share':a.lt(0).mean()})
    aggregate=[]
    for name,x in [('낮 원비율',cf.loc[day]),('낮 월/시간 평균 편차',anomaly),('낮 1시간 변화',ramp)]:
        for weighting,w in [('발전소 동일',np.ones(148)/148),('용량',sites.loc[ids,'capacity_kw'].to_numpy()/sites.loc[ids,'capacity_kw'].sum())]:
            covariance=x.cov().to_numpy();direct=(x@w).std(ddof=1)
            aggregate.append({'view':name,'weights':weighting,'n_times':len(x),'mean_site_std':x.std().mean(),
                              'direct_std':direct,'covariance_std':np.sqrt(w@covariance@w),
                              'zero_covariance_reference_std':np.sqrt(np.sum(w*w*np.diag(covariance)))})
    low=[]
    for name,x in [('낮 발전 비율',cf.loc[day]),('낮 월/시간 평균 편차',anomaly)]:
        frac=x.le(x.quantile(.1)).mean(axis=1)
        for q in [.5,.9,.95,.99]:low.append({'view':name,'quantile':q,'fraction_sites_low':frac.quantile(q)})
    full=gen.pivot(index='ts',columns='site_id',values='energy').sort_index()
    common=full.loc[full.notna().all(axis=1)].mask(lambda x:x<0).dropna()
    common_cf=common.div(sites.loc[common.columns,'capacity_kw'],axis=1)
    tc=clock(common.index);cr=common_cf.loc[tc.hour.between(9,16)].corr().to_numpy()
    ci,cj=np.triu_indices(253,1);common_pairs=cr[ci,cj]
    fig,axes=plt.subplots(1,2,figsize=(13,5))
    for name in ['all','day','within','ramp']:axes[0].hist(pairs['r_'+name],bins=45,alpha=.4,label=name,density=True)
    axes[0].set_xlabel('발전소 쌍 Pearson');axes[0].legend()
    axes[1].hist(pairs.r_daily,bins=45,alpha=.6,label='2024 148개소 일평균',density=True)
    axes[1].hist(common_pairs,bins=45,alpha=.5,label='253개소 최근 공통 시각 낮',density=True);axes[1].legend(fontsize=8)
    save_plot(fig,OUT,G,'01_distributions','발전소 쌍 관계 / 시간 구성에 따른 차이','기준 표본과 최근 공통 표본은 기간과 발전소 구성이 다름. raw와 cf의 Pearson은 양의 상수 나눗셈에 불변.')
    for n,name in enumerate(['day','within'],2):
        fig,ax=plt.subplots(figsize=(9,8));im=ax.imshow(matrices[name],vmin=-1,vmax=1,cmap='RdBu_r')
        ax.set_xlabel('익명 발전소 순서');ax.set_ylabel('익명 발전소 순서');fig.colorbar(im,ax=ax)
        save_plot(fig,OUT,G,f'{n:02d}_{name}','2024년 148개 발전소 / '+name,'발전소 ID 순서이며 군집 모델을 적합하지 않음.')
    fig,ax=plt.subplots(figsize=(8,6));ax.hexbin(pairs.r_day,pairs.r_within,gridsize=45,mincnt=1,bins='log')
    ax.set_xlabel('낮 발전 비율 상관');ax.set_ylabel('같은 월/시간 평균 편차 상관')
    save_plot(fig,OUT,G,'04_pair_change','같은 발전소 쌍의 조건별 상관','점의 밀도는 발전소 쌍의 수. 쌍은 서로 독립 표본이 아님.')
    fig,ax=plt.subplots(figsize=(10,5))
    for name,x in [('낮 원비율',cf.loc[day]),('낮 평균 편차',anomaly)]:
        frac=x.le(x.quantile(.1)).mean(axis=1);ax.hist(frac,bins=35,alpha=.5,label=name)
    ax.set_xlabel('각 발전소의 자체 10백분위 이하인 발전소 비율');ax.set_ylabel('시각 수');ax.legend()
    save_plot(fig,OUT,G,'05_joint_low','낮은 값의 동시 발생','이하는 <= 기준. 임계값 집중으로 각 발전소의 발생률이 정확히 10%가 아닐 수 있음. 고장이나 예측 실패의 정의가 아님.')
    a=pd.DataFrame(aggregate);fig,ax=plt.subplots(figsize=(12,5))
    a.loc[a.weights=='발전소 동일'].set_index('view')[['mean_site_std','direct_std','zero_covariance_reference_std']].plot.bar(ax=ax,rot=10)
    save_plot(fig,OUT,G,'06_aggregation','합산 후 변동과 공분산 0 가정의 참조','관측된 전체 148개소를 묶은 기술통계이며 새 배정이나 최적화가 아님. 참조값은 공분산 항만 0으로 둔 계산.')
    info={'plants':148,'hours':8784,'day_hours':int(day.sum()),'raw_energy_cf_correlation_max_difference':rawdiff,
          'common_253_hours':len(common),'common_253_day_pair_median':float(np.median(common_pairs))}
    finish(OUT,info,FILES,before,{'pairs':pairs,'correlation_summary':pd.DataFrame(distribution),'aggregation':a,'joint_low':pd.DataFrame(low)},G,'발전소 사이의 동시 변동 EDA')
    text=['# 발전소 사이의 동시 변동 EDA','',f"작성: {info['finished_at']}. 실제 자료 계산 완료.",'','## 질문과 목적','',
          '발전소들이 같은 시각에 함께 변하는 정도와 함께 낮은 값을 보이는 빈도, 관측 집단을 합쳤을 때 변동이 어떻게 달라지는지 확인했다. 예측 오차, 발전소 배정과 최적화는 계산하지 않았다.','',
          '## 사용 범위와 방법','','2024년 완전 기록 148개 발전소, 8,784시간을 주 표본으로 사용했다. 낮은 구간 시작 9~16시이고, 낮 1시간 변화는 현재와 이전 구간 시작이 모두 이 범위인 연속 기록이다. 발전 비율은 energy/capacity_kw이며 정적 용량을 적용했다. 모든 원본은 유지했다.','',
          '전체 시간, 낮, 지점별 월/시간 평균을 뺀 낮 편차, 낮의 1시간 변화, 일평균의 발전소 쌍 Pearson을 비교했다. 야간의 공통 0과 달력 패턴이 포함된 관계와 조건을 줄인 관계를 구분했다. 그룹 평균 편차는 인과 통제가 아니다. 모든 쌍은 같은 표본이고 쌍들끼리는 독립이 아니므로 p값을 계산하지 않았다.','',
          '## 발전소 쌍의 관계','',md_table(['관계','쌍 수','5백분위','중앙값','95백분위','음의 상관 비율'],[[r['metric'],fmt(r['pairs']),f"{r['q05']:.3f}",f"{r['median']:.3f}",f"{r['q95']:.3f}",f"{100*r['negative_share']:.1f}%"] for r in distribution]),'',
          f"원시 발전량과 용량 대비 비율의 쌍별 Pearson 최대 차이는 {rawdiff:.3g}였다. 발전소마다 양의 상수 용량으로 나눠도 쌍별 선형 상관은 변하지 않는다. 비율은 서로 다른 규모의 산포와 합산 가중을 비교할 때 의미가 있다. 253개 전체의 최근 공통 시각은 음수 쌍을 보류한 후 {len(common):,}시간이며 낮 상관 중앙값은 {np.median(common_pairs):.3f}였다. 이 표본은 2024년 집단과 구성/기간이 다르다.",'',
          '## 합산의 변동','',md_table(['자료','가중','개별 표준편차 평균','관측 합산 비율 표준편차','공분산 0 참조'],[[r['view'],r['weights'],f"{r['mean_site_std']:.4f}",f"{r['direct_std']:.4f}",f"{r['zero_covariance_reference_std']:.4f}"] for r in aggregate]),'',
          '관측된 평균의 분산은 가중치와 전체 공분산 행렬로 직접 확인했다. 공분산 0 참조는 독립이라고 증명한 결과가 아니라 합산에서 공동 변동이 차지하는 영향을 보기 위한 계산이다. 가중 평균 표준편차가 개별 평균보다 작아도 실제 예측 오차가 감소했다는 뜻은 아니다.','',
          '## 낮은 값의 동시 발생','',md_table(['기준','시각 분포 분위수','낮은 발전소 비율'],[[r['view'],r['quantile'],f"{100*r['fraction_sites_low']:.1f}%"] for r in low]),'',
          '낮은 값은 발전소 자체의 10백분위 이하라는 탐색용 기준이며 정지나 고장 판정이 아니다. 평균 편차의 낮은 값은 같은 월/시간 기대 수준보다 낮다는 뜻이다.','',
          '## 한계와 다음 단계','','공동 날씨, 설비 방향과 운영 상태를 분리한 분석은 아니다. 발전소 쌍의 동시 변동을 확인했으며 다음은 제공 좌표, 기상 지점 거리와 현재 참조 그룹에 따른 차이를 확인한다. 모델 학습과 발전량 예측은 하지 않았다.','',
          '## 관련 코드와 결과','','- [코드](../scripts/site_comovement_eda.py)','- [그림](../data/eda/2026-10-19_site_comovement/index.html)','- [전체 발전소 쌍](../data/eda/2026-10-19_site_comovement/pairs.csv)','','데이터 출력은 로컬 data/에만 보관한다.','']
    REPORT.write_text('\n'.join(text),encoding='utf-8');log(str(REPORT))


if __name__=='__main__':main()
