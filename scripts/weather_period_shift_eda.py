"""Season/station/hour-balanced EDA of supplied forecast distributions.

Evaluation forecasts are inspected as covariates only. No fitting, performance
estimation, automatic cleaning, prediction, or submission is performed.
"""
import gc
import json
import sys
from datetime import datetime
from zoneinfo import ZoneInfo
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import wasserstein_distance
from basic_shape_eda import ROOT,SOURCE,digest,fmt,md_table,log
from eda_plot_utils import save_plot,write_gallery,check_gallery

OUT=ROOT/'data/eda/2026-10-19_weather_period_shift'
REPORT=ROOT/'reports/2026-10-19_15_학습과_평가_기상분포_비교_EDA.md'
FILES=['sites.csv',*[f'{split}/forecast_{p}.parquet' for p in ['wa','wb'] for split in ['train','test']]]
PERIODS=['이전 1년','최근 1년','평가 1년']
BOUNDS=[('2023-06-01','2024-06-01'),('2024-06-01','2025-06-01'),('2025-06-01','2026-06-01')]
PROFILES=[];SHIFTS=[];MONTHS=[];AUDIT=[];EXTREMES=[];ATOMS=[];GALLERY=[];INFO={}
COLORS=['#8b969c','#26718c','#d7804f']


def quantile(x,w,q):
    # Weighted inverse empirical CDF, including masses at repeated values.
    order=np.argsort(x);x=x[order];cum=np.cumsum(w[order]);cum/=cum[-1]
    return x[np.minimum(np.searchsorted(cum,q,side='left'),len(x)-1)]


def view(x,groups,common,total_groups):
    mask=np.isfinite(x)&common[groups]
    a=x[mask];g=groups[mask]
    counts=np.bincount(g,minlength=total_groups)
    weights=1/counts[g]/common.sum()
    assert np.isclose(weights.sum(),1)
    assert np.allclose(np.bincount(g,weights=weights,minlength=total_groups)[common],1/common.sum())
    return a,g,weights


def cdf(x,w,grid):
    order=np.argsort(x);a=x[order];c=np.cumsum(w[order]);c/=c[-1]
    idx=np.searchsorted(a,grid,side='right')-1
    return np.where(idx>=0,c[np.maximum(idx,0)],0)


def profile(x,w):
    q=quantile(x,w,np.array([0,.01,.25,.5,.75,.99,1]))
    return {'n':len(x),'mean':float(np.sum(x*w)),'min':float(q[0]),'q01':float(q[1]),'q25':float(q[2]),
            'median':float(q[3]),'q75':float(q[4]),'q99':float(q[5]),'max':float(q[6]),
            'zero_share':float(np.sum(w[x==0]))}


def difference_unit(col):
    if col.endswith(('_sin','_cos')):return '무단위'
    if col.endswith(('cloud','humidity','rel_hum','prob')):return '%p'
    if 'temp' in col or col.endswith('dew_point'):return '°C'
    if col.endswith(('wind_speed','wind_gust_spd')):return 'm/s' if col.startswith('wa') else 'km/h'
    if col.endswith('press'):return 'hPa'
    if col.endswith('vis'):return 'km'
    if col.endswith('ceiling'):return 'm'
    if col=='wb_uv_idx':return '지수'
    return 'cm' if col=='wb_snow' else 'mm'


def compare(first,second,profiles,provider,r,col,comparison,groups):
    a,_,aw=first;b,_,bw=second
    pa,pb=profiles
    grid=np.unique(np.r_[quantile(a,aw,np.linspace(0,1,1001)),quantile(b,bw,np.linspace(0,1,1001))])
    distance=float(wasserstein_distance(a,b,u_weights=aw,v_weights=bw))
    scale=pa['q75']-pa['q25']
    SHIFTS.append({'provider':provider,'round':r,'variable':col,'comparison':comparison,'common_cells':groups,
                   'n_reference':len(a),'n_later':len(b),'reference_mean':pa['mean'],'later_mean':pb['mean'],
                   'mean_change':pb['mean']-pa['mean'],'median_change':pb['median']-pa['median'],
                   'wasserstein':distance,'reference_iqr':scale,'wasserstein_per_iqr':distance/scale if scale>0 else None,
                   'grid_cdf_max_gap':float(np.max(np.abs(cdf(a,aw,grid)-cdf(b,bw,grid)))),
                   'later_outside_reference_range_share':float(np.sum(bw[(b<pa['min'])|(b>pa['max'])]))})


def process(p):
    station=p+'_station';sites=pd.read_csv(SOURCE/'sites.csv',encoding='utf-8-sig')
    stations=sorted(sites[station].unique());codes={s:i for i,s in enumerate(stations)}
    tr=pd.read_parquet(SOURCE/f'train/forecast_{p}.parquet')
    te=pd.read_parquet(SOURCE/f'test/forecast_{p}.parquet')
    original=[c for c in tr if c not in [station,'round','ts']]
    assert set(original)==set(te.columns)-{station,'round','ts'}
    assert set(te[station])==set(stations) and set(tr[station])==set(stations)
    for split,d in [('학습 전체',tr),('평가 전체',te)]:
        for col in original:
            a=d[col].to_numpy(dtype=float);finite=a[np.isfinite(a)]
            EXTREMES.append({'provider':p,'split':split,'variable':col,'rows':len(a),'missing':int(np.isnan(a).sum()),
                             'min':float(finite.min()),'max':float(finite.max()),'zero_share':float(np.mean(finite==0))})
    off=tr.ts.ne(tr.ts.dt.floor('h'))
    INFO[p]={'all_train_rows':len(tr),'all_test_rows':len(te),'off_hour_train_excluded':int(off.sum()),
             'original_variables':len(original),'stations':len(stations)}
    assert te.ts.eq(te.ts.dt.floor('h')).all()
    tr=tr.loc[~off]
    periods=[]
    for split,d,lo,hi in [('train',tr,*BOUNDS[0]),('train',tr,*BOUNDS[1]),('test',te,*BOUNDS[2])]:
        start=d.ts-pd.Timedelta(hours=1)
        z=d.loc[(start>=pd.Timestamp(lo,tz='Asia/Seoul'))&(start<pd.Timestamp(hi,tz='Asia/Seoul'))].copy()
        start=z.ts-pd.Timedelta(hours=1)
        z['month']=start.dt.month.astype('int8');z['hour']=start.dt.hour.astype('int8')
        z['cell']=(z[station].map(codes)*12+(z.month-1))*24+z.hour
        assert not z.duplicated([station,'round','ts']).any()
        periods.append(z)
    direction=p+'_wind_dir'
    for z in periods:
        z[direction+'_sin']=np.sin(np.deg2rad(z[direction]));z[direction+'_cos']=np.cos(np.deg2rad(z[direction]))
    columns=[c for c in original if c!=direction]+[direction+'_sin',direction+'_cos']
    for r in [10,17]:
        data=[z.loc[z['round']==r] for z in periods]
        for label,z,(lo,hi) in zip(PERIODS,data,BOUNDS):
            hours=len(pd.date_range(pd.Timestamp(lo,tz='Asia/Seoul'),pd.Timestamp(hi,tz='Asia/Seoul'),freq='h',inclusive='left'))
            expected=len(stations)*hours
            AUDIT.append({'provider':p,'round':r,'period':label,'expected_keys':expected,'actual_keys':len(z),
                          'missing_keys':expected-len(z),'stations':z[station].nunique(),'months':z.month.nunique(),
                          'value_missing_cells':int(z[original].isna().to_numpy().sum())})
        groups=[z.cell.to_numpy(dtype=int) for z in data]
        total=len(stations)*12*24
        graph_profiles={}
        for col in columns:
            values=[z[col].to_numpy(dtype=float) for z in data]
            counts=[np.bincount(g[np.isfinite(x)],minlength=total) for x,g in zip(values,groups)]
            common=np.logical_and.reduce([n>0 for n in counts]);assert common.any()
            samples=[view(x,g,common,total) for x,g in zip(values,groups)]
            profiles=[profile(x,w) for x,_,w in samples]
            for label,stats in zip(PERIODS,profiles):PROFILES.append({'provider':p,'round':r,'variable':col,'period':label,'common_cells':int(common.sum()),**stats})
            compare(samples[0],samples[1],profiles[:2],p,r,col,'이전→최근',int(common.sum()))
            compare(samples[1],samples[2],profiles[1:],p,r,col,'최근→평가',int(common.sum()))
            # Exact month summaries with the same supported cells and weighting.
            for month in range(1,13):
                monthstats=[]
                for x,g,w in samples:
                    mask=((g//24)%12)==month-1
                    ww=w[mask]/w[mask].sum()
                    monthstats.append((float(np.sum(x[mask]*ww)),float(np.sum(ww[x[mask]==0]))))
                MONTHS.append({'provider':p,'round':r,'variable':col,'month':month,
                    'prior_mean':monthstats[0][0],'recent_mean':monthstats[1][0],'test_mean':monthstats[2][0],
                    'prior_zero_share':monthstats[0][1],'recent_zero_share':monthstats[1][1],'test_zero_share':monthstats[2][1],
                    'test_minus_recent':monthstats[2][0]-monthstats[1][0]})
            if col in ['wb_vis','wb_ceiling']:
                target=16.0934 if col=='wb_vis' else 9144.
                for label,(x,g,w) in zip(PERIODS,samples):
                    ATOMS.append({'round':r,'variable':col,'period':label,'target':target,'n':len(x),
                        'weighted_at_atom':float(w[np.isclose(x,target,rtol=0,atol=1e-9)].sum()),
                        'weighted_above_atom':float(w[x>target+1e-9].sum()),'max':float(x.max())})
            grid=np.unique(np.r_[*(quantile(x,w,np.linspace(0,1,501)) for x,_,w in samples)])
            graph_profiles[col]=(grid,[cdf(x,w,grid) for x,_,w in samples])
            del samples
        for start in range(0,len(columns),4):
            fig,axes=plt.subplots(2,2,figsize=(12.5,8))
            for ax,col in zip(axes.flat,columns[start:start+4]):
                grid,curves=graph_profiles[col]
                for label,color,y in zip(PERIODS,COLORS,curves):ax.plot(grid,y,label=label,color=color,lw=1.2)
                ax.set_title(col);ax.set_xlabel('제공 변수 값 / 풍향은 sin 또는 cos');ax.set_ylabel('균등 셀 가중 누적 비율');ax.legend(fontsize=8)
            for ax in axes.flat[len(columns[start:start+4]):]:ax.set_visible(False)
            save_plot(fig,OUT,GALLERY,f'{p}_{r}_ecdf_{start//4+1:02d}',f'{p.upper()} {r}시 / 계절과 지점을 맞춘 분포',
                      '공통 지점/월/구간 시작 시간 셀에 균등 가중. 값은 원본 유한값이며 범위 초과도 임의 제거하지 않음. CDF는 분위수 격자 표시.')
        log(f'{p.upper()} {r}: balanced comparisons complete')
    del tr,te,periods;gc.collect()


def plots():
    shifts=pd.DataFrame(SHIFTS);months=pd.DataFrame(MONTHS)
    for p in ['wa','wb']:
        a=shifts.loc[(shifts.provider==p)&(shifts.comparison=='최근→평가')]
        fig,axes=plt.subplots(1,2,figsize=(14,8))
        for ax,metric,title in [(axes[0],'grid_cdf_max_gap','CDF 차이 최대 (분위수 격자)'),(axes[1],'wasserstein_per_iqr','W1 / 최근 IQR')]:
            z=a.pivot(index='variable',columns='round',values=metric)
            z.plot.barh(ax=ax,color=COLORS[1:]);ax.set_title(title);ax.set_xlabel(metric)
        save_plot(fig,OUT,GALLERY,p+'_shift_overview',p.upper()+' / 최근 1년 대비 평가 구간 분포 차이',
                  '수치가 클수록 해당 거리 정의에서 차이가 큼. IQR=0인 변수의 표준화 거리는 계산하지 않음. 오류/성능 저하의 판정 문턱이 아님.')
        core=[p+'_temp',p+'_cloud',p+'_humidity' if p=='wa' else 'wb_rel_hum',p+'_rain']
        fig,axes=plt.subplots(2,2,figsize=(13,8))
        for ax,col in zip(axes.flat,core):
            z=months.loc[(months.provider==p)&(months['round']==17)&(months.variable==col)].sort_values('month')
            for label,color,c in zip(PERIODS,COLORS,['prior_mean','recent_mean','test_mean']):ax.plot(z.month,z[c],marker='.',label=label,color=color)
            ax.set_xticks(range(1,13));ax.set_xlabel('같은 달력 월');ax.set_ylabel('균등 셀 가중 평균');ax.set_title(col);ax.legend(fontsize=8)
        save_plot(fig,OUT,GALLERY,p+'_monthly',p.upper()+' / 같은 월별 기상 평균 비교 (17시)',
                  '각 월에 같은 지점과 시작 시간의 공통 셀을 균등 가중. 월 이름이 같아도 관측 연도/날짜와 실제 날씨가 다름.')
    fig,axes=plt.subplots(1,2,figsize=(13,5));a=pd.DataFrame(ATOMS)
    for ax,col in zip(axes,['wb_vis','wb_ceiling']):
        z=a.loc[a.variable==col].pivot(index='period',columns='round',values='weighted_at_atom').reindex(PERIODS)
        z.plot.bar(ax=ax,color=COLORS[1:],rot=10);ax.set_ylim(0,1);ax.set_title(col);ax.set_ylabel('집중값의 균등 셀 가중 비율')
    save_plot(fig,OUT,GALLERY,'wb_atoms','B 시정과 운고의 집중값 비중','집중값의 코드/상한 의미는 미확정. 해당 값에 몰린 비중이 달라졌다는 사실만 확인.')
    fig,axes=plt.subplots(1,2,figsize=(13,5));a=pd.DataFrame(AUDIT)
    for ax,p in zip(axes,['wa','wb']):
        z=a.loc[a.provider==p].pivot(index='period',columns='round',values='missing_keys').reindex(PERIODS)
        z.plot.bar(ax=ax,color=COLORS[1:],rot=10);ax.set_ylabel('없는 지점/라운드/시각 키');ax.set_title(p.upper())
    save_plot(fig,OUT,GALLERY,'key_missing','기간별 기록 부재','값 NaN과 키 자체의 부재를 구분. 정시 격자에 없는 행을 0으로 채우거나 비정시 기록을 반올림하지 않음.')


def pressure_detail():
    frames=[]
    for split,lo,hi,label in [('train',*BOUNDS[1],'최근 1년'),('test',*BOUNDS[2],'평가 1년')]:
        d=pd.read_parquet(SOURCE/f'{split}/forecast_wa.parquet',columns=['wa_station','round','ts','wa_ground_press'])
        start=d.ts-pd.Timedelta(hours=1)
        d=d.loc[(start>=pd.Timestamp(lo,tz='Asia/Seoul'))&(start<pd.Timestamp(hi,tz='Asia/Seoul'))].copy()
        start=d.ts-pd.Timedelta(hours=1)
        d['month']=start.dt.month;d['hour']=start.dt.hour
        # Equal hour cells per station-month, each hour uses its available dates.
        cell=d.groupby(['round','wa_station','month','hour'],observed=True).wa_ground_press.agg(['mean','min','max'])
        cells=cell[['mean']].rename(columns={'mean':label})
        frames.append(cells)
    joined=frames[0].join(frames[1],how='inner',validate='one_to_one')
    joined['change']=joined['평가 1년']-joined['최근 1년']
    station=joined.groupby(['round','wa_station','month'],observed=True).agg(
        recent_mean=('최근 1년','mean'),test_mean=('평가 1년','mean'),change=('change','mean'),hours=('change','size')).reset_index()
    monthly=station.groupby(['round','month'],observed=True).agg(stations=('change','size'),mean_change=('change','mean'),
        median_change=('change','median'),q05=('change',lambda x:x.quantile(.05)),q95=('change',lambda x:x.quantile(.95)),
        stations_drop5=('change',lambda x:int(x.lt(-5).sum())),stations_drop10=('change',lambda x:int(x.lt(-10).sum()))).reset_index()
    station.to_csv(OUT/'pressure_station_month.csv',index=False,encoding='utf-8-sig')
    monthly.to_csv(OUT/'pressure_month_summary.csv',index=False,encoding='utf-8-sig')
    fig,axes=plt.subplots(1,2,figsize=(14,5.5))
    for ax,r in zip(axes,[10,17]):
        a=monthly.loc[monthly['round']==r]
        ax.plot(a.month,a['mean_change'],marker='o',label='지점 평균')
        ax.plot(a.month,a.median_change,marker='.',label='지점 중앙값')
        ax.fill_between(a.month,a.q05,a.q95,alpha=.2,label='지점 5~95백분위')
        ax.axhline(0,color='grey',lw=.8);ax.set_xticks(range(1,13));ax.set_xlabel('같은 달력 월');ax.set_ylabel('평가 - 최근 지표기압 (hPa)');ax.set_title(f'{r}시 예보');ax.legend(fontsize=8)
    save_plot(fig,OUT,GALLERY,'wa_pressure_detail','A 지표기압 / 지점별 월 변화의 산포',
              '월별 각 지점의 시간 셀 평균 차이. 산포는 지점 사이 분포이며 신뢰구간이 아님. 실제 날씨와 공급 처리 변화는 분리하지 못함.')
    fig,axes=plt.subplots(1,2,figsize=(14,7))
    for ax,r in zip(axes,[10,17]):
        a=station.loc[station['round']==r].pivot(index='wa_station',columns='month',values='change')
        im=ax.imshow(a,aspect='auto',cmap='RdBu_r',vmin=-60,vmax=60)
        ax.set_xticks(range(12),range(1,13));ax.set_yticks(range(len(a)),a.index,fontsize=5)
        ax.set_xlabel('달력 월');ax.set_title(f'{r}시 / 제공 지점 ID');fig.colorbar(im,ax=ax,label='평가 - 최근 hPa')
    save_plot(fig,OUT,GALLERY,'wa_pressure_stations','A 지표기압 / 지점과 월별 평균 차이',
              '지점은 익명 제공 ID이며 위치를 추정하지 않음. 색상 범위 밖 값이 있으면 끝색으로 표시하며 수치는 집계 파일에 보존.')
    return monthly


def report():
    s=pd.DataFrame(SHIFTS);pr=pd.DataFrame(PROFILES);mo=pd.DataFrame(MONTHS)
    recent=s.loc[s.comparison=='최근→평가']
    previous=s.loc[s.comparison=='이전→최근'].set_index(['provider','round','variable'])
    main=[]
    for row in recent.itertuples():
        old=previous.loc[(row.provider,row.round,row.variable)]
        main.append([row.provider.upper(),row.round,row.variable,difference_unit(row.variable),f'{row.mean_change:+.3f}',f'{row.wasserstein:.3f}',
                     f'{row.grid_cdf_max_gap:.3f}',f'{old.grid_cdf_max_gap:.3f}',f'{100*row.later_outside_reference_range_share:.3f}%'])
    text=['# 학습과 평가 기상 분포 비교 EDA','',f"작성: {INFO['finished_at']}. 실제 자료 계산 완료.",'',
          '## 질문과 목적','','학습과 평가 구간의 제공 예보 분포가 다른지, 계절/지점/시간 구성을 맞춘 뒤에도 차이가 남는지 확인했다. 평가 구간에는 기상 실측과 발전량 정답이 없으므로 예보 정확도, 발전량 예측 성능이나 성능 저하를 판단하지 않았다.','',
          '## 사용 범위와 수행 방법','','A 61개와 B 73개 지점의 train/test 예보 파일 전체를 읽고 모든 수치 기상 변수를 비교했다. 원래 기상 열은 A 10개/B 20개이며 풍향 원각도의 선형 평균 대신 sin/cos 두 표현을 사용해 비교 변수는 A 11개/B 21개다. 라운드 10시/17시는 분리했다. 별도 실측과 발전량을 조인하지 않았다.','',
          '- 기간은 이전 1년 2023-06~2024-05, 최근 1년 2024-06~2025-05, 평가 1년 2025-06~2026-05다. 이전 기간에는 윤일이 포함되며 모두 ts - 1시간의 구간 시작 기준이다.',
          '- 각 변수와 라운드에서 세 기간 모두 유한값이 있는 동일 지점/달력 월/시작 시간 셀만 사용했다. 공통 셀 하나에 같은 질량을 주고 셀 안의 유한값은 균등 가중했다. 지점, 월, 시간의 관측 빈도 차이를 줄이지만 날짜별 날씨나 누락의 원인까지 맞추는 방법은 아니다.',
          '- 결측값과 없는 키를 0으로 채우지 않았다. 비정시 예보는 반올림하지 않고 별도 제외했다. 습도 100 초과 등 원본 유한값은 임의 제거하지 않았다. 원각도 풍향의 값 범위는 별도 원본 요약에 남겼고 sin/cos 각각의 거리는 완전한 원형분포 거리와 다르다.',
          '- 균등 셀 가중 평균/분위수/0 비율과 1차 Wasserstein 거리(W1)를 계산했다. W1은 값의 차이 단위이며 변수 간 절대 숫자를 직접 서열화하지 않는다. 최근 IQR로 나눈 값도 계산하되 IQR=0이면 계산하지 않았다.',
          '- CDF 차이는 양쪽 각각 1,001개 가중 분위수로 만든 합집합 격자의 최대 차이로, 정확한 KS 통계량이나 검정이 아니다. p값, 자동 드리프트 판정 문턱과 분류모형은 사용하지 않았다. 그림 CDF는 각 기간 501개 분위수 격자다.',
          '- 이전→최근의 분포 차이도 같은 공통 셀에서 계산해 연도 사이 변동의 참고로 두었다. 과거 한 번의 차이가 정상 변동 범위나 미래 허용 기준을 뜻하지 않는다.','',
          '## 키 가용성과 값 결측','',md_table(['자료','라운드','기간','격자 키','실제 키','없는 키','값 NaN 셀'],[[r['provider'].upper(),r['round'],r['period'],fmt(r['expected_keys']),fmt(r['actual_keys']),fmt(r['missing_keys']),fmt(r['value_missing_cells'])] for r in AUDIT]),'',
          '전체 학습 파일의 비정시 제외는 '+', '.join(f"{p.upper()} {INFO[p]['off_hour_train_excluded']}행" for p in ['wa','wb'])+'이다. 이 숫자는 세 비교 기간에서 빠진 행 수와 같지 않을 수 있다. 변수별 공통 셀과 유효값 수는 로컬 weighted_profiles.csv에 기록했다.','',
          '## 균등 셀 가중 분포 차이','',md_table(['자료','라운드','변수','평균 차이/W1 단위','평균 변화 평가-최근','W1 최근→평가','CDF 최근→평가','CDF 이전→최근','평가값의 최근 전체 범위 밖 비중'],main),'',
          'CDF 차이는 0~1의 질량 차이이며 W1과 다른 관점이다. IQR=0인 강수/눈 계열은 표준화 W1을 해석하지 않고 0 집중, 양수 값의 범위와 원단위 거리를 함께 본다. 평가값이 최근 범위를 벗어나도 새 계절의 날씨에서 자연스럽게 생길 수 있어 오류로 판정하지 않았다.','',
          '## 주요 결과의 해석','',
          '지점/월/시간 구성을 맞춘 뒤에도 A 지표기압의 차이가 두드러졌다. 17시 예보의 가중 평균은 최근 1년 1,015.895 hPa에서 평가 1년 1,009.668 hPa로 6.227 hPa 낮아졌다. CDF 격자 최대 차이는 약 0.208로 이번 64개 변수/라운드 조합 중 큰 항목에 속했다. W1도 이전→최근 0.467 hPa에서 최근→평가 6.227 hPa로 컸다.','',
          '같은 달 비교에서 차이는 평가 구간의 2026년 1~5월에 집중됐다. 지점별 월/시간 평균을 다시 계산했을 때 17시 예보의 2월은 61개 지점 모두에서 5 hPa 넘게 낮았고 51개 지점에서는 10 hPa 넘게 낮았다. 따라서 소수 지점의 극단값 하나로만 생긴 모양은 아니다. 반면 2025년 6~12월의 월평균 차이는 -0.033~+1.925 hPa였다. 실제 날씨와 기압 정의/자료 생성 방식 변화를 분리할 근거는 없어 원인을 미확정으로 남긴다.','',
          '최근 1년 17시 지표기압의 범위는 991~1,035 hPa, 평가 구간은 924~1,033 hPa였으며 평가값의 약 7.78%가 최근 전체 범위 밖이었다. 전체 학습 파일의 최소는 966 hPa로 최근 1년의 최소와 다르다. 범위 밖 값을 자동 삭제하거나 기압 차이를 보정하지 않았다.','',
          '모든 변수의 차이가 같은 정도는 아니다. 17시 기온의 최근→평가 평균 차이는 A +0.364°C, B +0.322°C였고 W1은 A 0.467°C/B 0.456°C였다. 두 값 모두 이전→최근의 W1(A 0.744°C/B 0.762°C)보다 작았다. 한 번의 과거 비교만으로 정상 범위를 확정하지는 않는다.','',
          'B 17시 강수확률의 가중 평균은 9.950%에서 7.619%로 낮아졌고, 정확한 0의 비중은 64.69%에서 68.81%로 커졌다. 강우량의 0 비중도 93.46%에서 95.10%로 커졌다. 이는 제공 예보 값의 분포이며 실제 비가 덜 왔거나 예보 성능이 나빠졌다는 결론은 아니다.','',
          '## A 지표기압의 지점별 추가 점검','',
          md_table(['라운드','달력 월','지점 수','평가-최근 평균 hPa','지점 변화 5~95백분위','5 hPa 초과 하락 지점','10 hPa 초과 하락 지점'],[[r['round'],r['month'],r['stations'],f"{r['mean_change']:.3f}",f"{r['q05']:.3f} ~ {r['q95']:.3f}",r['stations_drop5'],r['stations_drop10']] for r in pd.read_csv(OUT/'pressure_month_summary.csv',encoding='utf-8-sig').to_dict('records')]),'',
          '각 지점/월의 변화는 24개 시간 셀의 평균 차이다. 여기의 5/10 hPa 기준은 변화 범위를 기술하기 위한 문턱이며 오류 판정 기준이 아니다. 월 1~5는 평가 구간에서 2026년, 월 6~12는 2025년이다.','',
          '## B 집중값의 변화','',md_table(['변수','라운드','기간','집중값','집중값 비중','그보다 큰 비중','최대값'],[[r['variable'],r['round'],r['period'],r['target'],f"{100*r['weighted_at_atom']:.2f}%",f"{100*r['weighted_above_atom']:.3f}%",f"{r['max']:.4f}"] for r in ATOMS]),'',
          '시정 16.0934 km와 운고 9,144 m는 앞선 EDA에서 의미를 확정하지 못한 값이다. 집중값보다 큰 값의 부재도 배포 또는 수집 방식의 상한인지 실제 날씨 구성인지 이 자료만으로 구분할 수 없다. 삭제나 결측 대체는 하지 않았다.','',
          '## 한계와 다음 단계','','세 기간의 날짜와 실제 기상 조건은 서로 다르다. 비교 셀의 가중을 같게 해도 값이 없는 날짜의 선택 편향, 지점 자료의 처리 방식 변화, 정상적인 연도 간 날씨 변화를 분리하지 못한다. 가중 후 남은 차이를 공급사 오류나 기후 변화로 단정하지 않는다. 학습 파일의 전체 최소/최대 비교와 최근 1년 기준의 범위 밖 비중은 기준 기간이 다르다.','',
          '이번 작업은 제공된 평가 공변량의 기술통계다. 평가 실측이나 발전량을 추정해 대신 사용하지 않았다. 다음 미분석 질문은 발전소별 A/B 매핑 지점 쌍에서의 기상 정보 중복이며, 아직 별도 실행하지 않았다. 모델 학습과 예측, 최종 전처리, 최적화와 제출은 하지 않았다.','',
          '## 관련 코드와 로컬 결과','','- [코드](../scripts/weather_period_shift_eda.py)','- [전체 그림](../data/eda/2026-10-19_weather_period_shift/index.html)','- [균등 셀 가중 집계](../data/eda/2026-10-19_weather_period_shift/weighted_profiles.csv)','- [분포 차이](../data/eda/2026-10-19_weather_period_shift/shifts.csv)','- [월별 비교](../data/eda/2026-10-19_weather_period_shift/monthly.csv)','','원본은 유지하고 모든 데이터 출력과 그림은 로컬 data/에만 보관한다.','']
    REPORT.write_text('\n'.join(text),encoding='utf-8')


def verify():
    info=json.loads((OUT/'summary.json').read_text(encoding='utf-8'))
    pr=pd.read_csv(OUT/'weighted_profiles.csv',encoding='utf-8-sig');s=pd.read_csv(OUT/'shifts.csv',encoding='utf-8-sig')
    a=pd.read_csv(OUT/'key_audit.csv',encoding='utf-8-sig');m=pd.read_csv(OUT/'monthly.csv',encoding='utf-8-sig')
    assert len(pr)==32*2*3 and len(s)==32*2*2 and len(m)==32*2*12 and len(a)==12
    assert a.expected_keys.eq(a.actual_keys+a.missing_keys).all()
    assert a.loc[(a.provider=='wa')&(a.period=='평가 1년'),'missing_keys'].sum()==264
    assert a.loc[(a.provider=='wb')&(a.period=='평가 1년'),'missing_keys'].sum()==0
    assert s.grid_cdf_max_gap.between(-1e-9,1+1e-9).all() and s.wasserstein.ge(0).all()
    assert pr.zero_share.between(0,1+1e-9).all()
    assert (pr['min']<=pr.q01).all() and (pr.q01<=pr['median']).all() and (pr['median']<=pr.q99).all() and (pr.q99<=pr['max']).all()
    for row in pr.itertuples():
        z=m.loc[(m.provider==row.provider)&(m['round']==row.round)&(m.variable==row.variable)]
        col={'이전 1년':'prior_mean','최근 1년':'recent_mean','평가 1년':'test_mean'}[row.period]
        # Every month has equal station-hour cell counts in this source sample.
        assert row.common_cells==(61 if row.provider=='wa' else 73)*12*24
        assert np.isclose(z[col].mean(),row.mean)
    assert info['sha256']=={p:digest(SOURCE/p) for p in FILES}
    probe_x=np.array([0.,10.,20.,30.]);probe_groups=np.array([0,0,0,1])
    px,pg,pw=view(probe_x,probe_groups,np.array([True,True]),2)
    assert np.allclose(np.bincount(pg,weights=pw),[.5,.5]) and np.isclose(np.sum(px*pw),20.)
    assert np.array_equal(quantile(np.array([0.,100.]),np.array([.9,.1]),np.array([.5,.95])),[0.,100.])
    assert np.allclose(cdf(np.array([0.,0.,2.]),np.array([.2,.3,.5]),np.array([-1.,0.,1.,2.])),[0.,.5,.5,1.])
    assert '\u00b7' not in REPORT.read_text(encoding='utf-8')
    pressure=pd.read_csv(OUT/'pressure_month_summary.csv',encoding='utf-8-sig')
    stations=pd.read_csv(OUT/'pressure_station_month.csv',encoding='utf-8-sig')
    assert len(pressure)==24 and len(stations)==2*61*12 and stations.hours.eq(24).all()
    assert pressure.stations.eq(61).all() and pressure.stations_drop10.le(pressure.stations_drop5).all()
    for row in pressure.itertuples():
        expected=m.loc[(m.provider=='wa')&(m['round']==row.round)&(m.variable=='wa_ground_press')&(m.month==row.month),'test_minus_recent'].iloc[0]
        assert np.isclose(row.mean_change,expected)
    result={**check_gallery(OUT,26),'variables_after_angle_encoding':32,'round_variable_combinations':64,
            'period_profiles':len(pr),'distribution_comparisons':len(s),'month_comparisons':len(m),
            'cell_weight_and_month_accounting':True,'weighted_cdf_boundaries_checked':True,'pressure_station_accounting':True,'source_unchanged':True}
    (OUT/'verification.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result))


def main():
    sys.stdout.reconfigure(encoding='utf-8');assert sys.version_info[:2]==(3,13)
    OUT.mkdir(parents=True,exist_ok=True)
    if '--verify-only' in sys.argv:verify();return
    if '--pressure-detail' in sys.argv:
        INFO.update(json.loads((OUT/'summary.json').read_text(encoding='utf-8')))
        assert INFO['sha256']=={p:digest(SOURCE/p) for p in FILES}
        GALLERY.extend(json.loads((OUT/'gallery.json').read_text(encoding='utf-8')))
        assert not any(g['file']=='figures/wa_pressure_detail.png' for g in GALLERY)
        pressure_detail()
        assert INFO['sha256']=={p:digest(SOURCE/p) for p in FILES}
        write_gallery(OUT,GALLERY,'학습과 평가 기상 분포 비교 EDA')
        print('Pressure detail complete, 26 figures.');return
    if '--report-only' in sys.argv:
        INFO.update(json.loads((OUT/'summary.json').read_text(encoding='utf-8')))
        for target,name in [(PROFILES,'weighted_profiles'),(SHIFTS,'shifts'),(MONTHS,'monthly'),(AUDIT,'key_audit'),(EXTREMES,'raw_file_profiles'),(ATOMS,'atoms')]:
            target.extend(pd.read_csv(OUT/(name+'.csv'),encoding='utf-8-sig').to_dict('records'))
        report();print(str(REPORT));return
    before={p:digest(SOURCE/p) for p in FILES}
    for p in ['wa','wb']:process(p)
    plots();pressure_detail();assert before=={p:digest(SOURCE/p) for p in FILES}
    INFO.update(sha256=before,finished_at=datetime.now(ZoneInfo('Asia/Seoul')).isoformat(timespec='minutes'))
    for name,rows in [('weighted_profiles',PROFILES),('shifts',SHIFTS),('monthly',MONTHS),('key_audit',AUDIT),('raw_file_profiles',EXTREMES),('atoms',ATOMS)]:
        pd.DataFrame(rows).to_csv(OUT/(name+'.csv'),index=False,encoding='utf-8-sig')
    (OUT/'summary.json').write_text(json.dumps(INFO,ensure_ascii=False,indent=2),encoding='utf-8')
    write_gallery(OUT,GALLERY,'학습과 평가 기상 분포 비교 EDA');report();log(str(REPORT))


if __name__=='__main__':main()
