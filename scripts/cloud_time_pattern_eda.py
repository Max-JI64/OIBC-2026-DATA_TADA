"""Bounded audit of the supplied B cloud-cover time pattern, no model fitting."""
import json
import sys
from datetime import datetime
from zoneinfo import ZoneInfo
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from basic_shape_eda import ROOT,SOURCE,digest,fmt,md_table,log
from eda_plot_utils import save_plot,write_gallery,check_gallery

OUT=ROOT/'data/eda/2026-10-19_cloud_time_pattern'
REPORT=ROOT/'reports/2026-10-19_08_B운량_시간대_패턴_점검.md'
FILES=['sites.csv','train/generation.parquet','train/forecast_wb.parquet','train/observed_wb.parquet']
GALLERY=[];HOURLY=[];GROUPS=[];CONTRAST=[];INTERP=[];LAGS=[];INFO={}


def cohort_stations():
    gen=pd.read_parquet(SOURCE/'train/generation.parquet',columns=['site_id','ts'])
    gen=gen.loc[(gen.ts-pd.Timedelta(hours=1)).dt.year==2024]
    counts=gen.groupby('site_id').size();ids=counts.loc[counts==8784].index
    assert len(ids)==148
    sites=pd.read_csv(SOURCE/'sites.csv',encoding='utf-8-sig')
    return set(sites.loc[sites.site_id.isin(ids),'wb_station'])


def describe(a,name):
    grouped=a.groupby('hour',observed=True).wb_cloud
    z=grouped.agg(n='size',mean='mean',median='median',std='std',q25=lambda x:x.quantile(.25),q75=lambda x:x.quantile(.75),
                  zero_share=lambda x:x.eq(0).mean(),hundred_share=lambda x:x.eq(100).mean())
    for h,row in z.iterrows():HOURLY.append({'dataset':name,'hour':int(h),**row.to_dict()})


def pair_views(f,o,stations):
    for r in [10,17]:
        a=f.loc[f['round']==r].drop(columns='round')
        d=a.merge(o,on=['wb_station','ts'],how='inner',validate='one_to_one',suffixes=('_f','_o'))
        start=d.ts-pd.Timedelta(hours=1)
        d['hour']=start.dt.hour;d['month']=start.dt.month;d['year']=start.dt.year;d['day']=start.dt.floor('D')
        d=d.loc[d.wb_cloud_f.between(0,100)&d.wb_cloud_o.between(0,100)].copy()
        d['ae']=(d.wb_cloud_f-d.wb_cloud_o).abs()
        d['bias']=d.wb_cloud_f-d.wb_cloud_o
        d['equal']=d.ae.le(1e-9)
        main=d.loc[(d.year==2024)&d.wb_station.isin(stations)].copy()
        assert len(main)==47*8784
        for dimension,scope,z in [('month','2024 연결 지점',main),('wb_station','2024 연결 지점',main),('year','학습 전체 지점',d)]:
            agg=z.groupby([dimension,'hour'],observed=True).agg(n=('ae','size'),mae=('ae','mean'),bias=('bias','mean'),exact_share=('equal','mean'))
            for (group,h),row in agg.iterrows():GROUPS.append({'round':r,'dimension':dimension,'scope':scope,'group':str(group),'hour':int(h),**row.to_dict()})
        for name,col in [(f'예보 {r}시','wb_cloud_f')]+([('실측','wb_cloud_o')] if r==17 else []):
            describe(main[['hour',col]].rename(columns={col:'wb_cloud'}),name)
        # Compare the same station-date across 09, 10, 11 and 12 interval starts.
        wide=main.loc[main.hour.isin([9,10,11,12])].pivot(index=['wb_station','day'],columns='hour',values=['wb_cloud_f','wb_cloud_o','ae'])
        wide=wide.dropna()
        center=wide['ae'][[10,11]].mean(axis=1);flank=wide['ae'][[9,12]].mean(axis=1)
        contrast=pd.DataFrame({'station':[x[0] for x in wide.index],'day':[x[1] for x in wide.index],
                               'center':center.to_numpy(),'flank':flank.to_numpy()})
        contrast['delta']=contrast.center-contrast.flank
        for dimension in ['all','station','month']:
            contrast['month']=pd.DatetimeIndex(contrast.day).month
            groups=[('all',contrast)] if dimension=='all' else contrast.groupby(dimension,observed=True)
            for key,z in groups:CONTRAST.append({'round':r,'dimension':dimension,'group':str(key),'n_days':len(z),
                    'center_mae':float(z.center.mean()),'flank_mae':float(z.flank.mean()),'mean_delta':float(z.delta.mean()),
                    'median_delta':float(z.delta.median()),'lower_share':float(z.delta.lt(0).mean())})
        for source,col in [('예보','wb_cloud_f'),('실측','wb_cloud_o')]:
            z=wide[col]
            active=(z[12]-z[9]).abs()>1
            residual=pd.concat([(z[10]-(2*z[9]+z[12])/3).abs(),(z[11]-(z[9]+2*z[12])/3).abs()],axis=1)
            for scope,mask in [('모든 날짜',pd.Series(True,index=z.index)),('양끝 차이 1%p 초과',active)]:
                v=residual.loc[mask]
                INTERP.append({'round':r,'source':source,'scope':scope,'n_days':len(v),
                               'linear_both_share':float(v.max(axis=1).le(1e-6).mean()),'mean_abs_residual':float(v.to_numpy().mean())})
        if r==17:
            # Each candidate shift is evaluated on exactly the same station-time set.
            lag=main[['wb_station','ts','hour','wb_cloud_f']].copy()
            observation=o.set_index(['wb_station','ts']).wb_cloud
            for shift in range(-3,4):
                keys=pd.MultiIndex.from_arrays([lag.wb_station,lag.ts+pd.Timedelta(hours=shift)],names=['wb_station','ts'])
                lag[f'o_{shift}']=observation.reindex(keys).to_numpy()
            cols=[f'o_{s}' for s in range(-3,4)]
            good=lag[cols].apply(lambda x:x.between(0,100)).all(axis=1)
            lag=lag.loc[good]
            for shift in range(-3,4):
                ae=(lag.wb_cloud_f-lag[f'o_{shift}']).abs()
                for scope,mask in [('전체 시간',pd.Series(True,index=lag.index)),('시작 10~11시',lag.hour.isin([10,11])),('시작 9시/12시',lag.hour.isin([9,12]))]:
                    LAGS.append({'shift_hours':shift,'scope':scope,'n':int(mask.sum()),'mae':float(ae.loc[mask].mean())})
            main=main.sort_values(['wb_station','ts'])
            for source,col in [('예보 17시','wb_cloud_f'),('실측','wb_cloud_o')]:
                step=main.groupby('wb_station',observed=True)[col].diff().abs()
                contiguous=main.groupby('wb_station',observed=True).ts.diff().eq(pd.Timedelta(hours=1))
                for h,z in main.loc[contiguous].groupby('hour'):
                    vals=step.loc[z.index]
                    HOURLY.append({'dataset':source+' 1시간 변화','hour':int(h),'n':len(vals),
                                   'mean':float(vals.mean()),'median':float(vals.median()),'zero_share':float(vals.le(1e-9).mean())})
        log(f'Cloud pattern round {r} complete')


def plots():
    h=pd.DataFrame(HOURLY);g=pd.DataFrame(GROUPS);c=pd.DataFrame(CONTRAST)
    fig,axes=plt.subplots(2,2,figsize=(13,8))
    for name,z in h.loc[~h.dataset.str.contains('변화')].groupby('dataset'):
        axes[0,0].plot(z.hour,z['mean'],label=name);axes[0,1].plot(z.hour,z.hundred_share,label=name)
        axes[1,0].plot(z.hour,z.zero_share,label=name)
    for name,z in h.loc[h.dataset.str.contains('변화')].groupby('dataset'):axes[1,1].plot(z.hour,z['mean'],label=name)
    for ax,title in zip(axes.flat,['평균 운량 (%)','정확한 100 비율','정확한 0 비율','이전 시각 대비 평균 절대 변화 (%p)']):
        ax.set_title(title);ax.set_xlabel('구간 시작 시각');ax.legend(fontsize=8)
    save_plot(fig,OUT,GALLERY,'01_hourly_values','시간대별 예보와 실측의 값 분포','2024 연결 지점. 1시간 변화는 같은 지점의 연속 1시간 기록에 한정.')
    for dimension,num in [('month',2),('wb_station',3),('year',4)]:
        fig,axes=plt.subplots(1,2,figsize=(14,6))
        for ax,r in zip(axes,[10,17]):
            z=g.loc[(g.dimension==dimension)&(g['round']==r)]
            matrix=z.pivot(index='group',columns='hour',values='mae')
            if dimension!='wb_station':matrix=matrix.reindex(sorted(matrix.index,key=int))
            im=ax.imshow(matrix,aspect='auto',vmin=0,vmax=50,cmap='YlOrRd')
            ax.set_xticks(range(24),range(24),fontsize=7);ax.set_yticks(range(len(matrix)),matrix.index,fontsize=6 if dimension=='wb_station' else 9)
            ax.set_xlabel('구간 시작 시각');ax.set_title(f'{r}시 예보');fig.colorbar(im,ax=ax,label='MAE (%p)')
        save_plot(fig,OUT,GALLERY,f'{num:02d}_{dimension}','B 운량 MAE / '+dimension,'월/지점은 2024 연결 지점. 연도는 전체 학습 지점이며 연도별 기간과 가용성이 다름.')
    fig,axes=plt.subplots(1,2,figsize=(13,5))
    for ax,dimension in zip(axes,['station','month']):
        for r in [10,17]:
            z=c.loc[(c.dimension==dimension)&(c['round']==r)]
            if dimension=='month':
                z=z.assign(month=z.group.astype(int)).sort_values('month');ax.plot(z.month,z.mean_delta,marker='.',label=f'{r}시')
            else:ax.hist(z.mean_delta,bins=16,alpha=.5,label=f'{r}시')
        ax.axhline(0,color='grey',lw=.7) if dimension=='month' else ax.axvline(0,color='grey',lw=.7)
        ax.set_title(dimension);ax.set_xlabel('월' if dimension=='month' else '가운데 두 시간 MAE - 양옆 MAE (%p)');ax.legend()
    save_plot(fig,OUT,GALLERY,'05_paired_contrast','같은 지점/날짜의 시간대 차이','구간 시작 10,11시 평균과 9,12시 평균을 비교. 음수는 가운데 두 시간에서 더 작은 차이.')
    fig,ax=plt.subplots(figsize=(10,5))
    for scope,z in pd.DataFrame(LAGS).groupby('scope'):ax.plot(z.shift_hours,z.mae,marker='o',label=scope)
    ax.set_xlabel('실측 대상 시각 이동 (시간)');ax.set_ylabel('MAE (%p)');ax.legend()
    save_plot(fig,OUT,GALLERY,'06_shift_sensitivity','실측 시각 이동 민감도 / 17시 예보','모든 이동 후보가 유효한 공통 표본. 탐색용이며 시각 이동을 정당화하거나 원본을 보정한 결과가 아님.')


def report():
    c=pd.DataFrame(CONTRAST);h=pd.DataFrame(HOURLY);g=pd.DataFrame(GROUPS)
    lines=['# B 운량의 시간대 패턴 점검','',f"작성: {INFO['finished_at']}. 상태: 실행 완료.",'','## 질문과 목적','',
           '구간 시작 10시와 11시에 B 운량의 예보-실측 차이가 작아지는 모양이 월과 지점에 걸쳐 반복되는지, 어느 자료의 분포와 연결되는지 확인했다. 원인을 확정하거나 자료를 보정하는 작업은 하지 않았다.','',
           '## 사용 범위와 수행 방법','','2024년 고정 발전소 집단에 연결된 B 47개 지점의 고유 지점-시각을 주 표본으로 사용했다. 각 라운드에 412,848쌍이다. 전체 학습 기간의 모든 73개 지점은 연도별 패턴 보조 점검에 사용했다. 운량 0~100의 유효 쌍을 사용하고 비정시 예보는 반올림하지 않았다.','',
           '- 월과 시간은 우측닫힘 ts - 1시간의 시작 시각이다. 예보 10시와 17시를 분리했다.',
           '- 예보와 실측의 평균, 중앙값, IQR, 정확한 0/100 비율과 연속 1시간 변화 크기를 비교했다.',
           '- 같은 지점/날짜에서 시작 9~12시 네 기록이 모두 있는 표본으로 가운데 두 시간과 양옆의 절대 차이를 비교했다. 지점과 날짜의 구성 차이를 줄인 기술통계이며 인과 통제는 아니다.',
           '- 시작 9시와 12시 값을 잇는 직선과 10/11시 값의 차이를 수치적으로 점검했다. 양끝 차이가 1%p를 넘는 경우도 따로 계산해 상수 구간의 영향을 구분했다. 보간을 실제 적용하지 않았다.',
           '- 실측 시각을 -3~+3시간 옮기는 탐색은 모든 후보가 유효한 공통 표본으로만 계산했다. 작은 오차의 시차가 있어도 올바른 시각이라고 판단하거나 원본을 수정하지 않았다.','',
           '## 월과 지점에서의 반복','',md_table(['라운드','동일 지점/날짜 수','10/11시 MAE','9/12시 MAE','차이','가운데가 작은 날짜 비율'],[[r['round'],fmt(r['n_days']),f"{r['center_mae']:.3f}",f"{r['flank_mae']:.3f}",f"{r['mean_delta']:.3f}",f"{100*r['lower_share']:.1f}%"] for r in CONTRAST if r['dimension']=='all']),'']
    for r in [10,17]:
        a=c.loc[(c['round']==r)&(c.dimension=='station')];b=c.loc[(c['round']==r)&(c.dimension=='month')]
        lines += [f"{r}시 예보에서 가운데 두 시간의 평균 차이가 작은 지점은 {int(a.mean_delta.lt(0).sum())}/{len(a)}개, 월은 {int(b.mean_delta.lt(0).sum())}/{len(b)}개였다.",'']
    lines += ['## 시간대별 값의 모양','',md_table(['자료','시작 시각','평균 운량 %','중앙값 %','정확한 0 비율 %','정확한 100 비율 %'],[[row.dataset,row.hour,f'{row.mean:.2f}',f'{row.median:.2f}',f'{100*row.zero_share:.2f}',f'{100*row.hundred_share:.2f}'] for row in h.loc[~h.dataset.str.contains('변화')&h.hour.isin([9,10,11,12,13])].itertuples()]),'',
              '각 시간대의 표본 수가 같아 단순한 행 수 차이로 설명되지는 않는다. 평균의 움직임, 0/100 집중과 1시간 변화 그림을 함께 확인했다. 관측 방식, 갱신 주기와 자료 생성 과정은 배포 설명만으로 확정되지 않았다.','',
              '## 직선 형태와 시각 이동의 수치 점검','',md_table(['라운드','자료','범위','지점/날짜 수','두 중간값이 직선과 일치 %','직선 대비 평균 절대 차이 %p'],[[row['round'],row['source'],row['scope'],fmt(row['n_days']),f"{100*row['linear_both_share']:.2f}",f"{row['mean_abs_residual']:.3f}"] for row in INTERP]),'',
              md_table(['실측 시차 시간','범위','공통 행','MAE %p'],[[row['shift_hours'],row['scope'],fmt(row['n']),f"{row['mae']:.3f}"] for row in LAGS]),'',
              '수치적으로 직선과 일치하는 구간은 값의 반복이나 자료 생성 방식의 후보를 좁히는 단서다. 공급사가 실제로 보간했다는 증거와 같지는 않다. 시차 비교 역시 날씨의 시간적 지속성과 일변화의 영향을 받으므로 낮은 MAE를 시각 오류의 확정 근거로 쓰지 않는다.','',
              '## 한계와 다음 단계','','이 점검은 반복성, 분포와 단순 수치 패턴을 확인하는 범위에서 마쳤다. 생성 원인은 미확정으로 남긴다. 특정 시간대를 삭제하거나 원본 시각을 이동하지 않는다. 이 패턴이 있는 상태에서 전체 평균만으로 운량 자료를 평가하지 않고 월/시간별 차이를 함께 보존한다. 다음 단위는 기상 변수 사이의 중복 정보 EDA다. 모델 학습과 발전량 예측은 하지 않았다.','',
              '## 관련 코드와 결과','','- [분석 코드](../scripts/cloud_time_pattern_eda.py)','- [전체 그래프](../data/eda/2026-10-19_cloud_time_pattern/index.html)','- [월/지점/연도별 집계](../data/eda/2026-10-19_cloud_time_pattern/group_hour.csv)','- [값 분포](../data/eda/2026-10-19_cloud_time_pattern/hourly.csv)','','원본 해시를 실행 전후 확인했다. 모든 데이터 출력과 그림은 로컬 data/에만 보관한다.','']
    REPORT.write_text('\n'.join(lines),encoding='utf-8')


def verify():
    info=json.loads((OUT/'summary.json').read_text(encoding='utf-8'))
    c=pd.read_csv(OUT/'contrasts.csv',encoding='utf-8-sig');g=pd.read_csv(OUT/'group_hour.csv',encoding='utf-8-sig')
    for r in [10,17]:
        a=c.loc[(c['round']==r)&(c.dimension=='all')].iloc[0]
        assert a.n_days==47*366
        for dim in ['month','station']:
            z=c.loc[(c['round']==r)&(c.dimension==dim)]
            assert z.n_days.sum()==a.n_days and np.isclose(np.average(z.mean_delta,weights=z.n_days),a.mean_delta)
        for dim in ['month','wb_station']:assert g.loc[(g['round']==r)&(g.dimension==dim),'n'].sum()==47*8784
    assert info['sha256']=={p:digest(SOURCE/p) for p in FILES}
    check={**check_gallery(OUT,6),'paired_dates_and_strata':True,'source_unchanged':True}
    assert '\u00b7' not in REPORT.read_text(encoding='utf-8')
    (OUT/'verification.json').write_text(json.dumps(check,indent=2),encoding='utf-8');print(json.dumps(check))


def main():
    sys.stdout.reconfigure(encoding='utf-8');assert sys.version_info[:2]==(3,13)
    OUT.mkdir(parents=True,exist_ok=True)
    if '--verify-only' in sys.argv:verify();return
    before={p:digest(SOURCE/p) for p in FILES}
    stations=cohort_stations();assert len(stations)==47
    f=pd.read_parquet(SOURCE/'train/forecast_wb.parquet',columns=['wb_station','round','ts','wb_cloud'])
    o=pd.read_parquet(SOURCE/'train/observed_wb.parquet',columns=['wb_station','ts','wb_cloud'])
    assert not o.duplicated(['wb_station','ts']).any()
    f=f.loc[f.ts.eq(f.ts.dt.floor('h'))]
    pair_views(f,o,stations);plots()
    assert before=={p:digest(SOURCE/p) for p in FILES}
    INFO.update(sha256=before,finished_at=datetime.now(ZoneInfo('Asia/Seoul')).isoformat(timespec='minutes'))
    for name,rows in [('hourly',HOURLY),('group_hour',GROUPS),('contrasts',CONTRAST),('linear_pattern',INTERP),('lags',LAGS)]:
        pd.DataFrame(rows).to_csv(OUT/(name+'.csv'),index=False,encoding='utf-8-sig')
    (OUT/'summary.json').write_text(json.dumps(INFO,ensure_ascii=False,indent=2),encoding='utf-8')
    write_gallery(OUT,GALLERY,'B 운량 시간대 패턴 점검');report();log(str(REPORT))


if __name__=='__main__':main()
