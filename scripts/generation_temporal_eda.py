"""Descriptive lag, ramp, run and stable-cohort year patterns; no forecasts."""
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from basic_shape_eda import ROOT,fmt,md_table,log
from panel_eda_common import load_panel,clock,centered,hashes,finish,verify_base,save_check
from eda_plot_utils import save_plot

OUT=ROOT/'data/eda/2026-10-19_generation_temporal'
REPORT=ROOT/'reports/2026-10-19_14_발전량의_시차_급변_지속_EDA.md'
FILES=['sites.csv','train/generation.parquet'];G=[]
LAGS=[1,2,3,24,48,168]


def lag_records(cf):
    t=clock(cf.index);anomaly=centered(cf)
    records=[]
    for lag in LAGS:
        previous=clock(cf.index-pd.Timedelta(hours=lag))
        day=t.hour.between(9,16).to_numpy()&previous.hour.between(9,16).to_numpy()
        for view,z,mask in [('전체 비율',cf,np.ones(len(cf),dtype=bool)),('양쪽 낮 비율',cf,day),('양쪽 낮 월/시간 편차',anomaly,day)]:
            past=z.shift(lag);available=mask & past.notna().all(axis=1).to_numpy()
            current=z.loc[available];old=past.loc[available]
            corr=current.corrwith(old)
            aggregate=current.mean(axis=1).corr(old.mean(axis=1))
            for site,value in corr.items():records.append({'lag_hours':lag,'view':view,'site_id':site,'n_times':len(current),'pearson':value,'aggregate_equal_pearson':aggregate})
    return pd.DataFrame(records)


def runs(cf,threshold):
    # Explicitly break at both non-low values and each interval-start calendar day.
    t=clock(cf.index);rows=[]
    for site in cf:
        flags=(cf[site].le(threshold)&t.hour.between(9,16)).to_numpy()
        days=t.day.to_numpy();length=0;last=None
        for flag,day in zip(flags,days):
            if length and (not flag or day!=last):
                rows.append({'site_id':site,'threshold':threshold,'run_hours':length});length=0
            if flag:length+=1
            last=day
        if length:rows.append({'site_id':site,'threshold':threshold,'run_hours':length})
    return pd.DataFrame(rows)


def main():
    sys.stdout.reconfigure(encoding='utf-8');assert sys.version_info[:2]==(3,13)
    OUT.mkdir(parents=True,exist_ok=True)
    if '--verify-only' in sys.argv:
        info,check=verify_base(OUT,REPORT,FILES,6)
        lag=pd.read_csv(OUT/'lag_correlations.csv',encoding='utf-8-sig');run=pd.read_csv(OUT/'low_runs.csv',encoding='utf-8-sig')
        assert len(lag)==6*3*148 and lag.pearson.abs().le(1.000001).all()
        for h in LAGS:assert lag.loc[(lag.lag_hours==h)&(lag.view=='전체 비율'),'n_times'].eq(8784-h).all()
        assert run.run_hours.between(1,8).all()
        per=pd.read_csv(OUT/'run_summary.csv',encoding='utf-8-sig')
        for threshold,z in run.groupby('threshold'):
            row=per.loc[np.isclose(per.threshold,threshold)].iloc[0]
            assert int(z.run_hours.sum())==int(row.low_site_hours)
        assert info['stable_cohort_plants']>0 and info['stable_hours']==21168
        save_check(OUT,{**check,'lag_views':18,'plants':148,'lag_and_low_run_accounting':True,'stable_cohort_hour_grid':True});return
    before=hashes(FILES);sites,gen,energy,cf=load_panel();t=clock(cf.index)
    lag=lag_records(cf)
    summary=lag.groupby(['lag_hours','view']).agg(plants=('pearson','size'),n_times=('n_times','first'),median=('pearson','median'),
        q05=('pearson',lambda x:x.quantile(.05)),q95=('pearson',lambda x:x.quantile(.95)),aggregate=('aggregate_equal_pearson','first')).reset_index()
    daily=cf.groupby(t.day).mean();dailydev=daily-daily.groupby(daily.index.month).transform('mean')
    daylag=[]
    for shift in [1,2,7,14]:
        for label,x in [('일평균',daily),('월평균을 뺀 일평균',dailydev)]:
            old=x.shift(shift);mask=old.notna().all(axis=1);r=x.loc[mask].corrwith(old.loc[mask])
            daylag.append({'lag_days':shift,'view':label,'days':int(mask.sum()),'median':r.median(),'q05':r.quantile(.05),'q95':r.quantile(.95)})
    valid=t.hour.between(10,16)
    ramp=cf.diff().loc[valid]
    rc=clock(ramp.index);rampdev=ramp-ramp.groupby([rc.month,rc.hour]).transform('mean')
    rampstats=[]
    for label,x in [('낮 1시간 변화',ramp),('같은 월/시간 평균을 뺀 변화',rampdev)]:
        vals=x.to_numpy().ravel()
        rampstats.append({'view':label,'site_hours':len(vals),'q01':np.quantile(vals,.01),'q05':np.quantile(vals,.05),
            'median':np.median(vals),'q95':np.quantile(vals,.95),'q99':np.quantile(vals,.99),'abs_q95':np.quantile(abs(vals),.95),
            'drop_02_share':np.mean(vals<=-.2),'rise_02_share':np.mean(vals>=.2)})
    run=pd.concat([runs(cf,0),runs(cf,.05)],ignore_index=True)
    runsummary=[]
    for threshold,z in run.groupby('threshold'):
        qualifying=z.loc[z.run_hours>=3]
        runsummary.append({'threshold':threshold,'runs':len(z),'low_site_hours':int(z.run_hours.sum()),'runs_ge3':len(qualifying),
            'plants_ge3':qualifying.site_id.nunique(),'max_run':z.run_hours.max()})
    # Same plant set throughout Jan 2023 through May 2025; no changing membership.
    start=gen.ts-pd.Timedelta(hours=1)
    lo=pd.Timestamp('2023-01-01',tz='Asia/Seoul');hi=pd.Timestamp('2025-06-01',tz='Asia/Seoul')
    sub=gen.loc[(start>=lo)&(start<hi)].copy()
    expected=len(pd.date_range(lo,hi,freq='h',inclusive='left'))
    counts=sub.groupby('site_id').size();stableids=sorted(counts.loc[counts==expected].index)
    stable=sub.loc[sub.site_id.isin(stableids)].pivot(index='ts',columns='site_id',values='energy').sort_index()
    assert len(stable)==expected and stable.notna().all().all() and len(stableids)>0
    assert stable.ge(0).all().all()
    stablecf=stable.div(sites.loc[stable.columns,'capacity_kw'],axis=1);st=clock(stable.index);year=(stable.index-pd.Timedelta(hours=1)).year
    stablemean=stablecf.mean(axis=1)
    monthly=[];annual=[]
    for (y,m),z in stablemean.groupby([year,st.month]):monthly.append({'year':int(y),'month':int(m),'plants':len(stableids),'hours':len(z),'mean_cf':z.mean()})
    for y,z in stablemean.loc[st.month<=5].groupby(year[st.month<=5]):
        annual.append({'year':int(y),'period':'1~5월','plants':len(stableids),'hours':len(z),'mean_cf':z.mean()})
    fig,axes=plt.subplots(1,2,figsize=(13,5))
    for view,z in summary.groupby('view'):axes[0].plot(z.lag_hours,z['median'],marker='o',label=view)
    axes[0].set_xticks(LAGS);axes[0].set_xscale('symlog',linthresh=3);axes[0].set_ylim(-1,1);axes[0].set_xlabel('시차 시간');axes[0].set_ylabel('발전소별 Pearson 중앙값');axes[0].legend(fontsize=8)
    dl=pd.DataFrame(daylag)
    for view,z in dl.groupby('view'):axes[1].plot(z.lag_days,z['median'],marker='o',label=view)
    axes[1].set_xlabel('시차 일');axes[1].set_ylabel('발전소별 Pearson 중앙값');axes[1].set_ylim(-1,1);axes[1].legend(fontsize=8)
    save_plot(fig,OUT,G,'01_lags','발전량의 시간 시차와 일 시차 관계','기간 전체를 사용한 기술통계. 전체 시간에는 공통 야간 0과 일변화가 포함되며 과거값 예측 성능을 계산하지 않음.')
    fig,axes=plt.subplots(1,2,figsize=(13,5))
    for ax,h in zip(axes,[1,24]):
        old=cf.shift(h);mask=t.hour.between(9,16)&clock(cf.index-pd.Timedelta(hours=h)).hour.between(9,16).to_numpy()&old.notna().all(axis=1)
        ax.hexbin(old.loc[mask].mean(axis=1),cf.loc[mask].mean(axis=1),gridsize=45,mincnt=1,bins='log');ax.set_title(f'{h}시간 전과 현재');ax.set_xlabel('이전 집단 평균 비율');ax.set_ylabel('현재 집단 평균 비율')
    save_plot(fig,OUT,G,'02_lag_shapes','같은 집단의 과거와 현재 / 양쪽 낮','점은 시각이며 발전소 평균. 개별 발전소의 모든 분포를 대표하는 그림은 아님.')
    fig,axes=plt.subplots(1,2,figsize=(13,5))
    for ax,x,title in [(axes[0],ramp,'원래 낮 변화'),(axes[1],rampdev,'월/시간 평균을 뺀 변화')]:
        ax.hist(x.to_numpy().ravel(),bins=100,log=True);ax.set_title(title);ax.set_xlabel('1시간 발전 비율 변화');ax.set_ylabel('발전소-시각 수 (로그)')
    save_plot(fig,OUT,G,'03_ramps','낮의 1시간 변화 분포','양쪽 구간 시작이 9~16시인 연속 1시간만 포함. 문턱 0.2는 탐색용이며 설비 이상이나 예측 실패 판정이 아님.')
    fig,axes=plt.subplots(1,2,figsize=(12,5))
    for ax,threshold in zip(axes,[0,.05]):
        z=run.loc[run.threshold==threshold];ax.hist(z.run_hours,bins=np.arange(.5,9.5),log=True);ax.set_xticks(range(1,9));ax.set_xlabel('같은 날의 연속 시간');ax.set_ylabel('구간 수 (로그)');ax.set_title(f'cf <= {threshold:g}')
    save_plot(fig,OUT,G,'04_low_runs','낮의 낮은 발전 비율 지속','각 날짜 경계와 낮 시간 범위에서 구간을 끊음. 야간을 포함한 긴 0 구간을 낮 지속으로 오인하지 않음.')
    fig,ax=plt.subplots(figsize=(12,5))
    for y,z in pd.DataFrame(monthly).groupby('year'):ax.plot(z.month,z.mean_cf,marker='.',label=str(y))
    ax.set_xticks(range(1,13));ax.set_xlabel('달력 월');ax.set_ylabel('같은 발전소 집단의 평균 비율');ax.legend()
    save_plot(fig,OUT,G,'05_stable_years','같은 발전소 집단의 연도별 월 패턴','2023년부터 2025년 5월까지 기록이 완전한 동일 발전소만 사용. 2025년 연간 평균이나 추세를 추정하지 않음.')
    fig,ax=plt.subplots(figsize=(12,5))
    for m,z in daily.mean(axis=1).groupby(daily.index.month):ax.plot(z.index,z.to_numpy(),color='#26718c',lw=.7)
    ax.set_xlabel('2024년 날짜');ax.set_ylabel('148개소 동일 가중 일평균 비율')
    save_plot(fig,OUT,G,'06_daily_series','일별 발전 비율과 변동','관측된 일평균 시계열이며 예측 곡선이 아님. 결측 보간이나 이동평균 모형을 적용하지 않음.')
    info={'plants':148,'hours':8784,'stable_cohort_plants':len(stableids),'stable_hours':expected}
    finish(OUT,info,FILES,before,{'lag_correlations':lag,'lag_summary':summary,'daily_lags':dl,'ramp_summary':pd.DataFrame(rampstats),
        'low_runs':run,'run_summary':pd.DataFrame(runsummary),'stable_months':pd.DataFrame(monthly),'stable_jan_may':pd.DataFrame(annual)},G,'발전량 시차, 급변과 지속 EDA')
    text=['# 발전량의 시차, 급변과 지속 EDA','',f"작성: {info['finished_at']}. 실제 계산 완료.",'','## 질문과 목적','',
        '이전 시간과 전날 발전량의 관계, 낮의 급변과 낮은 값의 지속, 같은 발전소 집단의 연도별 변화를 확인했다. 과거 관측값을 이용한 예측을 만들거나 모델을 학습하지 않았다.','',
        '## 사용 범위와 방법','','시차 주 분석은 2024년 완전 기록 148개소/8,784시간이다. 시작 9~16시를 고정 낮 구간으로 두고 시차의 양쪽이 낮인 표본을 별도로 계산했다. 일평균은 24시간을 포함한다. 비율은 정적 capacity_kw로 나눴다. 같은 발전소의 월/시간 평균을 뺀 편차도 비교했으며 이는 기술통계로 미래 검증에 쓸 전처리를 적합한 것이 아니다.','',
        '시간 격자가 완전한 자료의 정확한 행 이동으로 1/2/3/24/48/168시간 시차를 만들었다. 일 시차는 1/2/7/14일을 비교했다. 시차가 다르면 유효 시각 수와 낮 시간대의 교집합도 달라지므로 수치 크기를 같은 표본의 순위로 해석하지 않는다. p값과 신뢰구간, 예측 오차는 계산하지 않았다.','',
        '## 시간 시차 관계','',md_table(['시차 시간','범위','시각 수','개별 상관 중앙값','5~95백분위','집단 평균 상관'],[[r.lag_hours,r.view,fmt(r.n_times),f'{r.median:.3f}',f'{r.q05:.3f} ~ {r.q95:.3f}',f'{r.aggregate:.3f}'] for r in summary.itertuples()]),'',
        '## 일 시차 관계','',md_table(['시차 일','범위','일 수','개별 상관 중앙값','5~95백분위'],[[r['lag_days'],r['view'],fmt(r['days']),f"{r['median']:.3f}",f"{r['q05']:.3f} ~ {r['q95']:.3f}"] for r in daylag]),'',
        '## 낮의 1시간 변화','',md_table(['범위','발전소-시간 쌍','1백분위','중앙값','99백분위','절대 변화 95백분위','-0.2 이하 비율','+0.2 이상 비율'],[[r['view'],fmt(r['site_hours']),f"{r['q01']:.3f}",f"{r['median']:.3f}",f"{r['q99']:.3f}",f"{r['abs_q95']:.3f}",f"{100*r['drop_02_share']:.2f}%",f"{100*r['rise_02_share']:.2f}%"] for r in rampstats]),'',
        '시작 10~16시의 변화만 사용해 이전 시각도 시작 9~16시 안에 있도록 했다. 0.2는 비율 차이의 탐색 기준이며 20%의 상대 증감률이 아니다. 달력 평균을 뺀 변화에도 날씨와 운영 등 여러 원인이 섞여 있다.','',
        '## 낮은 값의 지속','',md_table(['cf 문턱 이하','낮은 발전소-시간 수','전체 구간 수','3시간 이상 구간','3시간 이상 경험 발전소','최장 시간'],[[r['threshold'],fmt(r['low_site_hours']),fmt(r['runs']),fmt(r['runs_ge3']),r['plants_ge3'],r['max_run']] for r in runsummary]),'',
        '각 날짜에서 시작 9~16시 범위 안의 연속 구간만 센다. 0과 0.05 이하는 다른 기준이며 야간을 연결하지 않았다. 장시간 낮은 기록을 확인한 것이고 통신 장애나 설비 정지의 원인을 확정하지 않았다.','',
        '## 동일 집단의 연도별 비교','',f'2023년 1월부터 2025년 5월까지 {expected:,}시간이 모두 존재하는 {len(stableids)}개소를 고정했다. 연도 비교는 공통 1~5월만 같은 기간으로 집계하고 월별 전체 모양은 별도 그림에 남겼다.','',
        md_table(['연도','공통 기간','같은 발전소 수','시각 수','동일 가중 평균 비율'],[[r['year'],r['period'],r['plants'],fmt(r['hours']),f"{r['mean_cf']:.4f}"] for r in annual]),'',
        '윤년의 2월 일 수 차이는 남는다. 발전소 구성이 고정돼도 날씨와 설비 상태 등의 연도 차이는 분리하지 못하므로 노후화나 성능 추세로 결론내리지 않는다.','',
        '## 한계와 현재 상태','','이번에 제안한 발전소 간, 공간/그룹, SMP와 시차 관계 단위를 실행했다. 이것이 가능한 모든 EDA의 완료를 뜻하지 않는다. 학습/평가 구간의 계절을 맞춘 분포 비교, A/B 매핑 지점 쌍의 중복 정보 등은 아직 별도 실행하지 않았다. 모델 학습과 예측, 최적화 및 제출은 하지 않았다.','',
        '## 관련 코드와 결과','','- [코드](../scripts/generation_temporal_eda.py)','- [전체 그림](../data/eda/2026-10-19_generation_temporal/index.html)','- [시차별 집계](../data/eda/2026-10-19_generation_temporal/lag_summary.csv)','','원본을 유지하고 데이터 출력은 로컬 data/에만 보관한다.','']
    REPORT.write_text('\n'.join(text),encoding='utf-8');log(str(REPORT))


if __name__=='__main__':main()
