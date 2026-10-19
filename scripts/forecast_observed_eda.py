"""Paired EDA of supplied day-ahead weather forecasts and observations.

No forecast model is fitted. Source files are read-only; all data outputs stay
under ignored data/. The primary sample uses unique station-times, not plant rows.
"""
from __future__ import annotations
import gc
import html
import json
import re
import sys
from datetime import datetime
from zoneinfo import ZoneInfo

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

from basic_shape_eda import ROOT, SOURCE, digest, fmt, md_table, log

OUT=ROOT/'data/eda/2026-10-19_forecast_observed'
FIG=OUT/'figures'
REPORT=ROOT/'reports/2026-10-19_07_예보와_실측의_차이_EDA.md'
MAIN='2024 연결 지점'
FULL='학습 전체 지점과 기간'
META={'wa_cloud':('운량','%'),'wa_temp':('기온','°C'),'wa_temp_max':('최대기온','°C'),
      'wa_temp_min':('최저기온','°C'),'wa_humidity':('습도','%'),'wa_ground_press':('지표기압','hPa'),
      'wa_wind_speed':('풍속','m/s'),'wa_wind_dir':('풍향','°'),'wa_rain':('강우량','mm'),
      'wa_snow':('강설량','mm'),'wb_temp':('기온','°C'),'wb_feel_temp':('체감온도','°C'),
      'wb_dew_point':('이슬점','°C'),'wb_wind_speed':('풍속','km/h'),'wb_wind_dir':('풍향','°'),
      'wb_rel_hum':('습도','%'),'wb_vis':('시정','km'),'wb_ceiling':('운고','m'),
      'wb_uv_idx':('UV','지수'),'wb_cloud':('운량','%'),'wb_rain':('강우/강수 참고','mm')}
VARS={p:[c for c in META if c.startswith(p+'_')] for p in ['wa','wb']}
FILES=['sites.csv','train/generation.parquet',*[f'train/{k}_{p}.parquet' for p in ['wa','wb'] for k in ['forecast','observed']]]
METRICS=[];AUDIT=[];ROUND=[];STRATA=[];EVENTS=[];ATOMS=[];GALLERY=[];INFO={}
COLORS=['#26718c','#d7804f']


def error_unit(col):
    return '%p' if META[col][1]=='%' else META[col][1]


def error(f,o,col):
    delta=f-o
    return (delta+180)%360-180 if col.endswith('wind_dir') else delta


def valid(s,col):
    ok=np.isfinite(s)
    if col.endswith('cloud') or col.endswith('humidity') or col.endswith('rel_hum'):
        ok &= s.between(0,100)
    elif col=='wb_vis':ok &= s.between(0,50)
    elif col=='wb_uv_idx':ok &= s.between(0,15)
    elif col.endswith('wind_speed'):ok &= s.between(0,50 if col.startswith('wa') else 200)
    elif col.endswith('wind_dir'):ok &= s.between(0,360)
    elif col.endswith('press'):ok &= s.between(900,1100)
    elif col.endswith(('rain','snow','ceiling')):ok &= s>=0
    return ok


def stats(f,o,col):
    e=error(f,o,col).to_numpy(dtype=float)
    ae=np.abs(e)
    if not len(e):return {'n':0}
    return {'n':len(e),'bias':float(e.mean()),'median_error':float(np.median(e)),
            'mae':float(ae.mean()),'rmse':float(np.sqrt(np.mean(e*e))),
            'p50_abs':float(np.quantile(ae,.5)),'p90_abs':float(np.quantile(ae,.9)),
            'p95_abs':float(np.quantile(ae,.95)),'p99_abs':float(np.quantile(ae,.99)),
            'max_abs':float(ae.max()),'correlation':None if col.endswith('wind_dir') or f.nunique()<2 or o.nunique()<2 else float(f.corr(o))}


def save(fig,name,title,note):
    fig.suptitle(title,fontsize=13,fontweight='bold')
    fig.tight_layout(rect=[0,0,1,.96])
    fig.savefig(FIG/(name+'.png'),dpi=120,bbox_inches='tight')
    plt.close(fig)
    GALLERY.append({'file':'figures/'+name+'.png','title':title,'note':note})


def pair_panel(records,p):
    cols=VARS[p]
    for start in range(0,len(cols),4):
        chunk=cols[start:start+4]
        fig,axes=plt.subplots(len(chunk),3,figsize=(14,3.5*len(chunk)),squeeze=False)
        for i,col in enumerate(chunk):
            f10,f17,o=records[col]
            name,unit=META[col]
            axes[i,0].set_title(f'{name} ({unit}) / 17시 예보와 실측')
            if len(o):
                # Evenly spaced chronological sample, used only for plotting density.
                take=np.linspace(0,len(o)-1,min(len(o),100000),dtype=int)
                xx,yy=o.iloc[take].to_numpy(),f17.iloc[take].to_numpy()
                axes[i,0].hexbin(xx,yy,gridsize=45,mincnt=1,bins='log',cmap='viridis')
                lower=min(xx.min(),yy.min());upper=max(xx.max(),yy.max())
                axes[i,0].plot([lower,upper],[lower,upper],color='#a55043',lw=1)
            axes[i,0].set_xlabel('실측');axes[i,0].set_ylabel('예보')
            errors=[error(f10,o,col).to_numpy(),error(f17,o,col).to_numpy()]
            extent=max(float(np.quantile(np.abs(x),.995)) for x in errors)
            extent=max(extent,.01)
            edges=np.linspace(-extent,extent,65)
            for e,color,r in zip(errors,COLORS,[10,17]):
                hist,_=np.histogram(e,bins=edges)
                axes[i,1].stairs(hist/len(e),edges,color=color,label=f'{r}시')
                ae=np.sort(np.abs(e))
                positions=np.linspace(0,len(ae)-1,min(len(ae),2000),dtype=int)
                axes[i,2].plot(ae[positions],(positions+1)/len(ae),color=color,label=f'{r}시')
            axes[i,1].axvline(0,color='grey',lw=.7)
            axes[i,1].set_title(f'{name} / 예보 - 실측 분포')
            axes[i,1].set_xlabel(f'차이 ({error_unit(col)}), 절댓값 99.5%까지 표시');axes[i,1].set_ylabel('전체 표본 대비 구간 비중')
            axes[i,2].set_xlim(0,extent);axes[i,2].set_ylim(0,1.01)
            axes[i,2].set_title(f'{name} / 절대 차이 누적분포')
            axes[i,2].set_xlabel(f'절대 차이 ({error_unit(col)}), 99.5%까지 표시');axes[i,2].set_ylabel('누적 비율')
            axes[i,1].legend(fontsize=8);axes[i,2].legend(fontsize=8)
        save(fig,f'{p}_pairs_{start//4+1:02d}',p.upper()+' / 같은 표본의 예보와 실측',
             '2024 연결 지점, 두 예보와 실측의 비교 범위 통과 공통 표본. 산점 밀도는 시간순 등간격 최대 10만 행이며 모든 통계는 전체 유효 표본. 풍향 차이는 최단 회전각. B 강수는 정의 차이가 있는 참고 비교.')


def strata(d,f,o,col,p,r):
    e=error(f,o,col)
    temp=d.loc[f.index,['month','hour','station']].copy()
    temp['error']=e;temp['abs_error']=e.abs();temp['sq_error']=e*e
    for dimension in ['month','hour','station']:
        result=temp.groupby(dimension,observed=True).agg(n=('error','size'),bias=('error','mean'),mae=('abs_error','mean'),mse=('sq_error','mean'))
        for idx,row in result.iterrows():
            STRATA.append({'provider':p,'round':r,'variable':col,'dimension':dimension,'group':str(idx),
                           'n':int(row.n),'bias':float(row.bias),'mae':float(row.mae),'rmse':float(np.sqrt(row.mse))})


def occurrence(f,o,p,r):
    for threshold in [.1,1.]:
        a=f>threshold;b=o>threshold
        hit=int((a&b).sum());miss=int((~a&b).sum());false=int((a&~b).sum());dry=int((~a&~b).sum())
        EVENTS.append({'provider':p,'round':r,'threshold_mm':threshold,'n':len(o),'hit':hit,'miss':miss,'false_alarm':false,'both_dry':dry,
                       'observed_event_rate':float(b.mean()),'forecast_event_rate':float(a.mean()),
                       'precision':hit/(hit+false) if hit+false else None,'recall':hit/(hit+miss) if hit+miss else None,
                       'csi':hit/(hit+miss+false) if hit+miss+false else None})


def concentration(f,o,col,r):
    fc=16.0934 if col=='wb_vis' else 9144.
    oc=16.1 if col=='wb_vis' else 9144.
    fm=pd.Series(np.isclose(f,fc,rtol=0,atol=1e-9),index=f.index)
    om=pd.Series(np.isclose(o,oc,rtol=0,atol=1e-9),index=o.index)
    for scope,mask in [('전체',pd.Series(True,index=f.index)),('예보 집중값',fm),('예보 집중값 제외',~fm),
                       ('예보/실측 모두 집중값',fm&om),('예보만 집중값',fm&~om),('실측만 집중값',~fm&om),('둘 다 집중값 아님',~fm&~om)]:
        result=stats(f.loc[mask],o.loc[mask],col)
        ATOMS.append({'variable':col,'round':r,'scope':scope,'forecast_atom':fc,'observed_atom':oc,
                      'share':int(mask.sum())/len(mask),**result})


def load_provider(p):
    station=p+'_station'
    fcols=VARS[p]
    ocols=['wb_precip_1h' if c=='wb_rain' else c for c in fcols]
    forecast=pd.read_parquet(SOURCE/f'train/forecast_{p}.parquet',columns=[station,'round','ts',*fcols])
    obs=pd.read_parquet(SOURCE/f'train/observed_{p}.parquet',columns=[station,'ts',*ocols])
    dups=obs.duplicated([station,'ts'],keep=False)
    duplicate_keys=len(obs.loc[dups,[station,'ts']].drop_duplicates())
    conflict=0
    if dups.any():
        # Collapse only duplicate-key rows; agreeing values survive per column.
        grouped=obs.loc[dups].groupby([station,'ts'],observed=True)[ocols].agg(['first','nunique'])
        counts=grouped.xs('nunique',axis=1,level=1)
        unique=grouped.xs('first',axis=1,level=1).mask(counts>1).reset_index()
        conflict=int((counts>1).to_numpy().sum())
        obs=pd.concat([obs.loc[~dups],unique],ignore_index=True)
    off=forecast.ts.ne(forecast.ts.dt.floor('h'))
    INFO[p]={'forecast_rows':len(forecast),'observed_rows_after_duplicate_view':len(obs),
             'duplicate_observed_keys':duplicate_keys,'conflicting_cells':conflict,
             'off_hour_forecast_by_round':{str(k):int(v) for k,v in forecast.loc[off,'round'].value_counts().items()}}
    forecast=forecast.loc[~off].rename(columns={station:'station',**{c:c+'_f' for c in fcols}})
    obs=obs.rename(columns={station:'station',**{c:fc+'_o' for c,fc in zip(ocols,fcols)}})
    assert not obs.duplicated(['station','ts']).any()
    return forecast,obs


def process(p,stations):
    forecast,obs=load_provider(p)
    primary={};round_data={}
    for r in [10,17]:
        f=forecast.loc[forecast['round']==r].drop(columns='round')
        assert not f.duplicated(['station','ts']).any()
        d=f.merge(obs,on=['station','ts'],how='left',validate='one_to_one',indicator=True)
        assert len(d)==len(f)
        start=d.ts-pd.Timedelta(hours=1)
        d['month']=start.dt.month.astype('int8');d['hour']=start.dt.hour.astype('int8')
        main_mask=(start.dt.year==2024)&d.station.isin(stations)
        for scope,scope_mask in [(FULL,pd.Series(True,index=d.index)),(MAIN,main_mask)]:
            current=d.loc[scope_mask]
            AUDIT.append({'provider':p,'round':r,'scope':scope,'forecast_rows':len(current),
                          'matched_rows':int(current['_merge'].eq('both').sum()),'unmatched_rows':int(current['_merge'].eq('left_only').sum()),
                          'station_count':current.station.nunique(),'unique_keys':len(current[['station','ts']].drop_duplicates())})
            for col in VARS[p]:
                fv=current[col+'_f'];ov=current[col+'_o']
                for quality,mask in [('원본 유효 쌍',np.isfinite(fv)&np.isfinite(ov)),('비교 범위 통과',valid(fv,col)&valid(ov,col))]:
                    a=fv.loc[mask];b=ov.loc[mask]
                    METRICS.append({'provider':p,'round':r,'scope':scope,'variable':col,'unit':error_unit(col),
                                    'quality':quality,'reference_only':col=='wb_rain',**stats(a,b,col)})
        # Forecast-observation matched view of primary unique station-time records.
        selected=d.loc[main_mask,['station','ts','month','hour',*[c+s for c in VARS[p] for s in ['_f','_o']]]].copy()
        round_data[r]=selected
        del d;gc.collect()
        log(f'{p.upper()} {r}: matched metrics complete')
    # The revision comparison always uses the intersection of all three valid values.
    a=round_data[10];b=round_data[17]
    both=a.merge(b[['station','ts',*[c+'_f' for c in VARS[p]]]],on=['station','ts'],how='inner',validate='one_to_one',suffixes=('_10','_17'))
    both=both.sort_values(['ts','station']).reset_index(drop=True)
    for col in VARS[p]:
        f10=both[col+'_f_10'];f17=both[col+'_f_17'];o=both[col+'_o']
        mask=valid(f10,col)&valid(f17,col)&valid(o,col)
        f10=f10.loc[mask];f17=f17.loc[mask];o=o.loc[mask]
        s10=stats(f10,o,col);s17=stats(f17,o,col)
        ae10=error(f10,o,col).abs();ae17=error(f17,o,col).abs()
        improve=(ae17<ae10-1e-9);worse=(ae17>ae10+1e-9)
        ROUND.append({'provider':p,'variable':col,'unit':error_unit(col),'reference_only':col=='wb_rain','n':len(o),
                      'mae_10':s10['mae'],'mae_17':s17['mae'],'bias_10':s10['bias'],'bias_17':s17['bias'],
                      'rmse_10':s10['rmse'],'rmse_17':s17['rmse'],'delta_mae':s17['mae']-s10['mae'],
                      'mae_change_percent':100*(s17['mae']/s10['mae']-1) if s10['mae'] else None,
                      'identical_forecast_share':float((error(f17,f10,col).abs()<=1e-9).mean()),
                      'improved_share':float(improve.mean()),'worsened_share':float(worse.mean()),
                      'tied_share':float((~(improve|worse)).mean())})
        primary[col]=(f10,f17,o)
        for r,fv in [(10,f10),(17,f17)]:
            strata(both,fv,o,col,p,r)
            if col.endswith('rain'):occurrence(fv,o,p,r)
            if col in ['wb_vis','wb_ceiling']:concentration(fv,o,col,r)
    pair_panel(primary,p)
    del forecast,obs,round_data,both,primary;gc.collect()


def panels():
    rd=pd.DataFrame(ROUND);st=pd.DataFrame(STRATA)
    fig,ax=plt.subplots(figsize=(11,9))
    x=rd.sort_values('mae_change_percent')
    ax.barh(x.variable,x.mae_change_percent,color=np.where(x.mae_change_percent<=0,COLORS[0],COLORS[1]))
    ax.axvline(0,color='grey',lw=.8);ax.set_xlabel('17시 MAE / 10시 MAE - 1 (%)')
    save(fig,'07_round_change','예보 갱신 후 차이 변화 / 같은 표본','음수는 17시의 MAE 감소. 변수마다 단위가 달라 절대 오차의 직접 서열로 해석하지 않음. B 강수는 참고 비교.')
    groups=[['wa_cloud','wa_temp','wb_cloud','wb_temp'],['wa_humidity','wa_rain','wb_rel_hum','wb_rain']]
    for page,cols in enumerate(groups,1):
        fig,axes=plt.subplots(2,2,figsize=(13,8))
        for ax,col in zip(axes.flat,cols):
            z=st.loc[(st.variable==col)&(st.dimension=='month')].copy();z['month']=z.group.astype(int)
            for r,color in zip([10,17],COLORS):
                q=z.loc[z['round']==r].sort_values('month');ax.plot(q.month,q.mae,marker='.',label=f'{r}시',color=color)
            ax.set_title(f'{col} / MAE ({error_unit(col)})');ax.set_xticks(range(1,13));ax.set_xlabel('2024년 월');ax.legend()
        save(fig,f'08_monthly_{page}','월별 예보-실측 차이','두 라운드와 실측의 공통 비교 표본, 지점-시각 행 가중. B 강수는 참고 비교.')
    fig,axes=plt.subplots(2,2,figsize=(13,8))
    for ax,col in zip(axes.flat,groups[0]):
        z=st.loc[(st.variable==col)&(st.dimension=='hour')].copy();z['hour']=z.group.astype(int)
        for r,color in zip([10,17],COLORS):
            q=z.loc[z['round']==r].sort_values('hour');ax.plot(q.hour,q.mae,label=f'{r}시',color=color)
        ax.set_title(f'{col} / MAE ({error_unit(col)})');ax.set_xlabel('대상 구간 시작 시각');ax.legend()
    save(fig,'09_hourly','시간대별 예보-실측 차이','시간대는 ts - 1시간. 대상 시각에 따라 예보 발행 후 경과 시간도 달라짐.')
    fig,axes=plt.subplots(2,4,figsize=(15,7))
    for ax,col in zip(axes.flat,groups[0]+groups[1]):
        z=st.loc[(st.variable==col)&(st.dimension=='station')].pivot(index='group',columns='round',values='mae')
        delta=z[17]-z[10]
        ax.hist(delta,bins=16,color=COLORS[0]);ax.axvline(0,color='grey',lw=.7)
        ax.set_title(col);ax.set_xlabel(f'지점별 MAE 차이 ({error_unit(col)})');ax.set_ylabel('지점 수')
    save(fig,'10_station_revision','지점별 예보 갱신 차이','같은 변수 안에서도 지점별 차이가 다를 수 있음. 음수는 17시 MAE 감소.')
    fig,axes=plt.subplots(2,2,figsize=(12,8))
    for ax,(p,t) in zip(axes.flat,[(p,t) for p in ['wa','wb'] for t in [.1,1.]]):
        z=pd.DataFrame(EVENTS);z=z.loc[(z.provider==p)&(z.threshold_mm==t)]
        bottom=np.zeros(2)
        for col,color in zip(['hit','miss','false_alarm','both_dry'],['#6a8c68','#c0746e','#d8a863','#7d9bac']):
            vals=z[col].to_numpy()/z.n.to_numpy();ax.bar(['10시','17시'],vals,bottom=bottom,label=col,color=color);bottom+=vals
        ax.set_title(f'{p.upper()} > {t:g} mm'+(' / 참고 비교' if p=='wb' else ''));ax.set_ylim(0,1);ax.legend(fontsize=8)
    save(fig,'11_rain_events','강수 기준 초과의 일치와 불일치','hit: 둘 다 초과, miss: 실측만 초과, false_alarm: 예보만 초과, both_dry: 둘 다 기준 이하. 무강수 다수로 전체 일치율이 높아질 수 있음. B는 정의 차이 있음.')
    for col,num in [('wb_vis',12),('wb_ceiling',13)]:
        a=pd.DataFrame(ATOMS);a=a.loc[(a.variable==col)&a.scope.isin(['전체','예보 집중값','예보 집중값 제외'])]
        fig,axes=plt.subplots(1,2,figsize=(13,5))
        for ax,metric,title in [(axes[0],'mae','MAE'),(axes[1],'bias','편향')]:
            z=a.pivot(index='scope',columns='round',values=metric).reindex(['전체','예보 집중값','예보 집중값 제외'])
            z.plot.bar(ax=ax,rot=15,color=COLORS);ax.set_title(title);ax.set_ylabel(META[col][1]);ax.axhline(0,color='grey',lw=.7)
        save(fig,f'{num}_atoms_{col}',col+' / 집중값 여부에 따른 예보-실측 차이','집중값은 결측으로 간주하지 않음. 값의 의미와 생성 방식이 미확정이며 제외 전후 표본 범위가 다름.')
    m=pd.DataFrame(METRICS)
    sensitivity=m.loc[m.scope==FULL].pivot(index=['variable','round'],columns='quality',values='rmse')
    ratio=(sensitivity['원본 유효 쌍']/sensitivity['비교 범위 통과']).replace([np.inf,-np.inf],np.nan)
    fig,ax=plt.subplots(figsize=(12,5))
    ratio.unstack('round').plot.bar(ax=ax,color=COLORS,rot=50)
    ax.axhline(1,color='grey',lw=.7);ax.set_yscale('log');ax.set_ylabel('원본 RMSE / 비교 범위 RMSE (로그)')
    save(fig,'14_quality_sensitivity','전체 학습 기간의 주의값 민감도','비교 범위는 확정 정제 규칙이 아님. 원본과 비교 범위 표본은 다르며 비율이 1에 가까워도 개별 주의값의 의미가 해결되는 것은 아님.')
    fig,axes=plt.subplots(1,2,figsize=(14,5))
    for ax,p in zip(axes,['wa','wb']):
        z=m.loc[(m.provider==p)&(m.scope==FULL)&(m.quality=='비교 범위 통과')].pivot(index='variable',columns='round',values='n')
        z.plot.bar(ax=ax,color=COLORS,rot=55);ax.set_title(p.upper()+' / 변수별 유효 쌍');ax.set_ylabel('행 수')
    save(fig,'15_availability','전체 학습 기간의 변수별 비교 가능 표본','실측이 없는 대상 시각, 값 결측과 열별 비교 범위 탈락을 0으로 채우지 않음.')


def outputs():
    for name,rows in [('pair_metrics',METRICS),('join_audit',AUDIT),('round_comparison',ROUND),('strata_metrics',STRATA),('rain_events',EVENTS),('concentration',ATOMS)]:
        pd.DataFrame(rows).to_csv(OUT/(name+'.csv'),encoding='utf-8-sig',index=False)
    (OUT/'summary.json').write_text(json.dumps(INFO,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    (OUT/'gallery.json').write_text(json.dumps(GALLERY,ensure_ascii=False,indent=2),encoding='utf-8')
    cards=''.join(f'<section><h2>{html.escape(g["title"])}</h2><p>{html.escape(g["note"])}</p><a href="{g["file"]}"><img src="{g["file"]}" loading="lazy"></a></section>' for g in GALLERY)
    (OUT/'index.html').write_text('''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>예보와 실측 EDA</title><style>body{font:16px/1.6 'Malgun Gothic',sans-serif;background:#f4f5f5;max-width:1200px;margin:auto;padding:24px;color:#26333c}section{background:white;margin:24px 0;padding:20px;border:1px solid #ddd}img{width:100%}h2{font-size:20px}</style><h1>제공된 기상 예보와 실측의 차이</h1><p>같은 지점과 대상 시각의 기술통계. 모델 학습과 발전량 예측 없음.</p>'''+cards+'</html>',encoding='utf-8')
    qa=OUT/'qa';qa.mkdir(exist_ok=True)
    for start in range(0,len(GALLERY),4):
        pics=[]
        for g in GALLERY[start:start+4]:
            with Image.open(OUT/g['file']) as im:
                im.load();pics.append(im.convert('RGB').resize((im.width//2,im.height//2),Image.Resampling.LANCZOS))
        w=max(im.width for im in pics);h=max(im.height for im in pics)+24
        canvas=Image.new('RGB',(w*2,h*2),'#ededed');pen=ImageDraw.Draw(canvas)
        for i,im in enumerate(pics):
            x=(i%2)*w;y=(i//2)*h;canvas.paste(im,(x,y+24));pen.text((x+5,y+4),GALLERY[start+i]['file'],fill='black')
        canvas.save(qa/f'contact_{start//4+1:02d}.png')


def report():
    rd=pd.DataFrame(ROUND);m=pd.DataFrame(METRICS);events=pd.DataFrame(EVENTS);atoms=pd.DataFrame(ATOMS)
    lines=['# 예보와 실측의 차이 EDA','',f"작성: {INFO['finished_at']}. 상태: 자료 매칭과 기술통계 실행 완료.",'',
           '## 질문과 목적','','제공된 하루전 기상 예보가 같은 제공사의 실측과 얼마나 다르고, 10시에서 17시로 갱신될 때 차이가 어떻게 바뀌는지 확인했다. 관측값의 집중과 주의값이 결과를 좌우하는지도 살폈다. 새로운 모델을 학습하거나 발전량을 예측하지 않았다.','',
           '## 사용 범위와 수행 방법','',
           f"주 분석은 앞선 관계 EDA와 같은 2024년 148개 발전소에 연결된 A {INFO['wa_primary_stations']}개, B {INFO['wb_primary_stations']}개 지점이다. 발전소마다 기상을 반복하지 않고 고유 지점-시각을 한 번씩 사용했다. 따라서 앞선 발전량 관계 EDA의 발전소-시간 가중과 다르다. 보조 점검은 전체 학습 파일의 모든 지점과 기간을 사용했다. 평가 구간에는 실측이 없으므로 평가 예보의 오차를 추정하지 않았다.",'',
           '- 배포 README에 따르면 round=10과 17은 대상일 전날의 발행 시각이다. 같은 제공사/지점/ts로 정확히 매칭했다. ts는 우측닫힘이므로 월과 시간은 ts - 1시간으로 정의했다.',
           '- 실측 중복 키는 열별로 일치값을 보존하고 충돌 셀만 NaN으로 둔 임시 뷰를 사용했다. 비정시 예보는 반올림하지 않았다. 누락값은 0으로 채우지 않았고 원본을 수정하지 않았다.',
           '- A의 공유 변수 10개, B의 공유 변수 10개를 비교했다. A 예보의 기온/최대/최저는 같은 값이므로 세 지표를 독립 예보 세 개로 해석하지 않는다. B wb_rain과 wb_precip_1h는 정의가 달라 별도의 강수 참고 비교로 추가했다. B 강수확률, 습구온도, 순간풍속 등 직접 대응 실측이 없는 예보 변수는 오차를 계산하지 않았다.',
           '- 차이는 예보 - 실측, 편향은 그 평균이다. MAE와 RMSE는 변수의 차이 단위로 표시하며 운량과 습도의 차이는 퍼센트포인트(%p)다. 풍향은 (예보 - 실측 + 180) % 360 - 180의 최단 각도 차이를 사용했고 일반 선형 상관은 계산하지 않았다. 풍향의 평균 차이는 이 부호화에 따른 기술통계로 완전한 원형통계 추정이 아니다.',
           '- 원본의 유한한 쌍과 비교 범위 통과 쌍을 모두 계산했다. 운량/습도 0~100, A 풍속 0~50 m/s, B 풍속 0~200 km/h, 시정 0~50 km, UV 0~15, 지표기압 900~1100 hPa, 풍향 0~360도, 강우/강설/운고 0 이상을 임시 비교 범위로 삼았다. 확정 오류 판정이나 최종 정제 규칙은 아니다.',
           '- 10시/17시의 주 비교는 예보 두 개와 실측이 모두 비교 범위를 통과한 동일 표본이다. 표의 개선 비율은 통계적 유의성이나 미래 성능 보장이 아니다. 월/시간/지점별 집계도 이 공통 표본을 사용했다.',
           '- 오차 히스토그램과 절대 오차 ECDF는 99.5% 지점까지만 표시하고 표의 지표는 꼬리를 포함한 전체 유효 표본으로 계산했다. 산점 밀도는 시간순 등간격 최대 10만 행으로 그렸다.','',
           '## 매칭과 가용성','',md_table(['제공사','라운드','범위','예보 정시 행','실측 키 매칭','미매칭','지점 수'],[[r['provider'].upper(),r['round'],r['scope'],fmt(r['forecast_rows']),fmt(r['matched_rows']),fmt(r['unmatched_rows']),r['station_count']] for r in AUDIT]),'',
           f"A 실측의 중복 키 {INFO['wa']['duplicate_observed_keys']}개에서 공유 열의 충돌 {INFO['wa']['conflicting_cells']}셀을 보류했다. B 비정시 예보의 라운드별 제외는 {INFO['wb']['off_hour_forecast_by_round']}행이다. 키가 매칭돼도 값 결측이나 비교 범위 탈락으로 변수별 유효 행은 다를 수 있다.",'',
           '## 동일 표본의 예보 라운드 비교','',md_table(['변수','단위','공통 행','편향 10/17','MAE 10/17','RMSE 10/17','MAE 변화 %','17시 개선/악화/절대차이 동일 %','예보값 자체 동일 %'],[[r.variable,r.unit,fmt(r.n),f'{r.bias_10:.3f} / {r.bias_17:.3f}',f'{r.mae_10:.3f} / {r.mae_17:.3f}',f'{r.rmse_10:.3f} / {r.rmse_17:.3f}',f'{r.mae_change_percent:+.2f}',f'{100*r.improved_share:.1f} / {100*r.worsened_share:.1f} / {100*r.tied_share:.1f}',f'{100*r.identical_forecast_share:.1f}'] for r in rd.itertuples()]),'']
    for p in ['wa','wb']:
        for col in [p+'_temp',p+'_cloud']:
            r=rd.loc[rd.variable==col].iloc[0]
            lines += [f"{p.upper()} {META[col][0]}의 MAE는 10시 {r.mae_10:.3f}에서 17시 {r.mae_17:.3f} {r.unit}로 {r.mae_change_percent:+.2f}% 변했다. 17시가 더 가까운 행은 {100*r.improved_share:.1f}%, 더 먼 행은 {100*r.worsened_share:.1f}%였다.",'']
    unchanged=rd.loc[rd.variable.isin(['wb_temp','wb_cloud'])]
    lines += ['B의 예보값 자체가 동일한 비율은 '+', '.join(f'{META[r.variable][0]} {100*r.identical_forecast_share:.1f}%' for r in unchanged.itertuples())+'였다. 절대 차이가 같은 경우와 예보값이 같은 경우를 별도 집계했다. 예보값 동일은 절대 허용오차 1e-9 기준이고 풍향은 같은 방향을 뜻한다.','']
    worse_rmse=rd.loc[rd.rmse_17>rd.rmse_10]
    lines += ['MAE와 RMSE가 같은 결론을 주는 것은 아니다. 17시 RMSE가 커진 변수는 '+', '.join(worse_rmse.variable)+'이다. MAE 감소만으로 큰 오차의 감소까지 확인했다고 볼 수 없다.','',
              '단위가 다른 변수의 MAE 숫자를 직접 비교하거나 A와 B의 오차를 제공사 순위로 해석하지 않는다. A와 B는 지점 위치와 표본이 다르며 실측의 측정/집계 방식도 완전히 같다고 확인하지 않았다.','',
              '## 월, 시간대와 지점별 차이','']
    st=pd.DataFrame(STRATA)
    context=[]
    for col in ['wa_cloud','wa_temp','wa_humidity','wa_rain','wb_cloud','wb_temp','wb_rel_hum','wb_rain','wb_uv_idx']:
        months=st.loc[(st.variable==col)&(st['round']==17)&(st.dimension=='month')]
        low=months.loc[months.mae.idxmin()];high=months.loc[months.mae.idxmax()]
        hours=st.loc[(st.variable==col)&(st['round']==17)&(st.dimension=='hour')]
        daytime=hours.group.astype(int).between(9,16)
        day_mae=float(np.average(hours.loc[daytime,'mae'],weights=hours.loc[daytime,'n']))
        other_mae=float(np.average(hours.loc[~daytime,'mae'],weights=hours.loc[~daytime,'n']))
        station=st.loc[(st.variable==col)&(st.dimension=='station')].pivot(index='group',columns='round',values='mae')
        context.append([col,error_unit(col),f'{int(low.group)}월 {low.mae:.3f}',f'{int(high.group)}월 {high.mae:.3f}',
                        f'{day_mae:.3f} / {other_mae:.3f}',f'{int((station[17]<station[10]-1e-9).sum())} / {len(station)}'])
    lines += [md_table(['변수','차이 단위','17시 MAE 최소 월','17시 MAE 최대 월','17시 낮/그 외 MAE','17시 MAE 감소 지점/전체'],context),'',
              '낮은 구간 시작 9~16시(09:00~17:00)이며 일출/일몰을 계산한 구분은 아니다. 월별 최소/최대는 2024년 안의 기술통계다. 월과 시간대의 날씨 구성, 예보 발행 후 경과 시간이 함께 달라지므로 원인을 하나로 확정하지 않는다. UV는 야간에 작은 값이 많아 전체 MAE와 낮 시간 MAE를 구분해 봐야 한다.','']
    cloud_hour=st.loc[(st.variable=='wb_cloud')&(st['round']==17)&(st.dimension=='hour')].set_index('group')
    lines += ['B 운량의 17시 예보 MAE는 구간 시작 '+', '.join(f'{h}시 {cloud_hour.loc[h,"mae"]:.2f}%p' for h in ['9','10','11','12','13'])+
              f'로 짧은 시간대에 크게 달랐다. 각 시간대는 모두 {fmt(int(cloud_hour.loc["10","n"]))}행으로 표본 수 차이에서 생긴 모양은 아니다. 시각 처리나 제공 자료의 생성 방식이 원인인지는 이번 집계만으로 확정하지 않았다.','',
              '## 오차의 꼬리와 원본 주의값 영향','',md_table(['변수','차이 단위','17시 원본/비교범위 행','원본/비교범위 RMSE','비교범위 절대차이 50/95/99백분위','비교범위 최대 절대차이'],[[col,error_unit(col),f'{fmt(raw.n)} / {fmt(clean.n)}',f'{raw.rmse:.3f} / {clean.rmse:.3f}',f'{clean.p50_abs:.3f} / {clean.p95_abs:.3f} / {clean.p99_abs:.3f}',f'{clean.max_abs:.3f}'] for col in META for raw in [m.loc[(m.variable==col)&(m['round']==17)&(m.scope==FULL)&(m.quality=='원본 유효 쌍')].iloc[0]] for clean in [m.loc[(m.variable==col)&(m['round']==17)&(m.scope==FULL)&(m.quality=='비교 범위 통과')].iloc[0]]]),'',
              '이 표는 전체 학습 기간이며 앞 표의 2024년 연결 지점 표본과 다르다. 원본 RMSE의 큰 증가는 드문 극단값이 제곱 오차를 지배할 수 있음을 뜻한다. 범위 제외 후에도 실측을 오류 없는 정답으로 확정하지 않았다.','',
              '## 강수 기준 초과의 참고 비교','',md_table(['자료','라운드','기준 mm','공통 행','실측/예보 초과율 %','적중','실측만 초과','예보만 초과','둘 다 기준 이하','정밀도/재현율/CSI'],[[r.provider.upper()+(' 참고' if r.provider=='wb' else ''),r.round,r.threshold_mm,fmt(r.n),f'{100*r.observed_event_rate:.2f} / {100*r.forecast_event_rate:.2f}',fmt(r.hit),fmt(r.miss),fmt(r.false_alarm),fmt(r.both_dry),f'{r.precision:.3f} / {r.recall:.3f} / {r.csi:.3f}'] for r in events.itertuples()]),'',
              '기준 초과는 엄격한 > 비교다. 적중은 예보와 실측 모두 초과, 정밀도는 예보 초과 중 적중 비율, 재현율은 실측 초과 중 적중 비율, CSI는 적중/(적중 + 실측만 초과 + 예보만 초과)이다. 무강수가 많아 전체 일치율만으로 강수 예보를 판단하기 어렵다. B는 강우량과 총 1시간 강수량의 정의 차이가 있어 정식 강수 예보 성능 지표로 단정하지 않는다.','',
              '## 시정과 운고의 집중값','',md_table(['변수','라운드','범위','행 수','비중 %','편향','MAE'],[[r.variable,r.round,r.scope,fmt(r.n),f'{100*r.share:.2f}',f'{r.bias:.3f}',f'{r.mae:.3f}'] for r in atoms.itertuples()]),'',
              '예보 시정 집중값은 16.0934 km, 실측의 대응 집중값은 16.1 km로 구분했다. 운고는 양쪽 모두 9,144 m다. 두 시정값의 차이는 0.0066 km이며 정확한 동일값 일치와 구분해 비교했다. 예보/실측의 집중 여부가 다른 행의 차이를 함께 봤으며, 특정 값이 많다는 이유로 결측값이나 측정 상한이라고 확정하지 않았다. 집중값 제외 전후의 표본과 값 범위는 다르다.','',
              '## 한계와 다음 단계','','월별, 시간별과 지점별 차이는 그래프 및 로컬 strata_metrics.csv에 기록했다. 동일 지점의 연속 시각과 여러 지점의 동시 날씨는 독립 표본이 아니다. 독립 가정의 p값과 신뢰구간을 계산하지 않았으며 지역별 공간 가중도 추가하지 않았다. 2024년 연결 지점과 전체 학습 파일은 서로 다른 표본이므로 두 결과 차이를 시간 변화 하나로 설명할 수 없다.','',
              '비교는 제공된 기상 예보와 관측의 차이다. 발전량 예측 성능이나 집합 구성 최적화의 결과가 아니다. 예보 갱신의 평균 개선이 개별 지점과 모든 시간대에서 개선을 보장하지 않는다. 다음 EDA에서는 기상 변수 사이의 중복 정보나 기상 차이가 큰 조건의 발전량 분포를 확인할 수 있다.','',
              '## 관련 코드와 로컬 결과','','- [분석 코드](../scripts/forecast_observed_eda.py)','- [그래프 갤러리](../data/eda/2026-10-19_forecast_observed/index.html)','- [전체 쌍별 집계](../data/eda/2026-10-19_forecast_observed/pair_metrics.csv)','- [동일 표본 라운드 비교](../data/eda/2026-10-19_forecast_observed/round_comparison.csv)','- [월/시간/지점별 집계](../data/eda/2026-10-19_forecast_observed/strata_metrics.csv)','','원본 해시는 실행 전후 동일했다. 그래프와 상세 데이터 출력은 로컬 data/에만 보관하고 코드, 요약 보고서와 일별 로그만 커밋한다.','']
    REPORT.write_text('\n'.join(lines),encoding='utf-8')


def verify():
    info=json.loads((OUT/'summary.json').read_text(encoding='utf-8'))
    audit=pd.read_csv(OUT/'join_audit.csv',encoding='utf-8-sig')
    rd=pd.read_csv(OUT/'round_comparison.csv',encoding='utf-8-sig')
    m=pd.read_csv(OUT/'pair_metrics.csv',encoding='utf-8-sig')
    st=pd.read_csv(OUT/'strata_metrics.csv',encoding='utf-8-sig')
    events=pd.read_csv(OUT/'rain_events.csv',encoding='utf-8-sig')
    gallery=json.loads((OUT/'gallery.json').read_text(encoding='utf-8'))
    assert len(audit)==8 and audit.forecast_rows.eq(audit.unique_keys).all()
    assert audit.forecast_rows.eq(audit.matched_rows+audit.unmatched_rows).all()
    assert audit.loc[audit.scope==MAIN,'forecast_rows'].tolist()==[40*8784,40*8784,47*8784,47*8784]
    assert len(m)==21*2*2*2 and len(rd)==21 and len(gallery)==16
    sample_sizes=m.pivot(index=['provider','round','scope','variable'],columns='quality',values='n')
    assert (sample_sizes['비교 범위 통과']<=sample_sizes['원본 유효 쌍']).all()
    assert (m.rmse+1e-9>=m.mae).all() and (m.mae+1e-9>=m.bias.abs()).all()
    assert (m.p50_abs<=m.p95_abs).all() and (m.p95_abs<=m.p99_abs).all() and (m.p99_abs<=m.max_abs).all()
    assert np.allclose(rd.improved_share+rd.worsened_share+rd.tied_share,1)
    assert rd.identical_forecast_share.between(0,1).all()
    assert (rd.identical_forecast_share<=rd.tied_share+1e-9).all()
    assert np.allclose(rd.mae_17-rd.mae_10,rd.delta_mae)
    for row in rd.itertuples():
        z=st.loc[st.variable==row.variable]
        for (r,_),group in z.groupby(['round','dimension']):
            assert group.n.sum()==row.n
            assert np.isclose(np.average(group.mae,weights=group.n),getattr(row,f'mae_{r}'))
            assert np.isclose(np.sqrt(np.average(group.rmse**2,weights=group.n)),getattr(row,f'rmse_{r}'))
    assert events[['hit','miss','false_alarm','both_dry']].sum(axis=1).eq(events.n).all()
    for g in gallery:
        with Image.open(OUT/g['file']) as im:im.verify()
    links=re.findall(r'(?:href|src)="([^"]+)"',(OUT/'index.html').read_text(encoding='utf-8'))
    assert len(links)==len(gallery)*2 and all((OUT/x).is_file() for x in links)
    assert np.allclose(error(pd.Series([359.,1.,360.]),pd.Series([1.,359.,0.]),'wa_wind_dir'),[-2,2,0])
    assert info['source_sha256']=={f:digest(SOURCE/f) for f in FILES}
    assert '\u00b7' not in REPORT.read_text(encoding='utf-8')
    result={'matched_views':len(audit),'metric_combinations':len(m),'round_comparisons':len(rd),
            'figures':len(gallery),'gallery_links':len(links),'strata_and_events_accounting':True,
            'same_sample_round_comparison':True,'circular_wind_difference':True,'source_unchanged':True}
    (OUT/'verification.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))


def main():
    sys.stdout.reconfigure(encoding='utf-8');assert sys.version_info[:2]==(3,13)
    FIG.mkdir(parents=True,exist_ok=True)
    if '--verify-only' in sys.argv:verify();return
    if '--report-only' in sys.argv:
        INFO.update(json.loads((OUT/'summary.json').read_text(encoding='utf-8')))
        for target,name in [(METRICS,'pair_metrics'),(AUDIT,'join_audit'),(ROUND,'round_comparison'),
                            (STRATA,'strata_metrics'),(EVENTS,'rain_events'),(ATOMS,'concentration')]:
            target.extend(pd.read_csv(OUT/(name+'.csv'),encoding='utf-8-sig').to_dict('records'))
        report();print(str(REPORT));return
    INFO['started_at']=datetime.now(ZoneInfo('Asia/Seoul')).isoformat(timespec='minutes')
    before={f:digest(SOURCE/f) for f in FILES}
    sites=pd.read_csv(SOURCE/'sites.csv',encoding='utf-8-sig')
    gen=pd.read_parquet(SOURCE/'train/generation.parquet',columns=['site_id','ts'])
    gen=gen.loc[(gen.ts-pd.Timedelta(hours=1)).dt.year==2024]
    counts=gen.groupby('site_id').size();ids=counts.loc[counts==8784].index
    assert len(ids)==148
    cohort=sites.loc[sites.site_id.isin(ids)]
    INFO['wa_primary_stations']=cohort.wa_station.nunique();INFO['wb_primary_stations']=cohort.wb_station.nunique()
    del gen;gc.collect()
    for p in ['wa','wb']:process(p,set(cohort[p+'_station']))
    panels()
    assert before=={f:digest(SOURCE/f) for f in FILES}
    INFO.update(source_sha256=before,source_unchanged=True,figures=len(GALLERY),finished_at=datetime.now(ZoneInfo('Asia/Seoul')).isoformat(timespec='minutes'))
    outputs();report()
    log(f'Completed: {len(METRICS)} metrics, {len(GALLERY)} figures. {REPORT}')


if __name__=='__main__':main()
