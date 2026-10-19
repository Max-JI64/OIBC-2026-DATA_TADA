"""Weather-generation descriptive EDA, no predictive fitting or submissions."""
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

OUT = ROOT/'data/eda/2026-10-19_weather_generation'
FIG = OUT/'figures'
REPORT = ROOT/'reports/2026-10-19_06_기상과_발전량의_관계_EDA.md'
VARS = {'wa': ['wa_cloud','wa_rain','wa_temp','wa_humidity'],
        'wb': ['wb_cloud','wb_rain','wb_temp','wb_rel_hum','wb_vis','wb_ceiling','wb_uv_idx','wb_precip_prob']}
FILES = ['sites.csv','train/generation.parquet',*[f'train/{kind}_{p}.parquet' for p in ['wa','wb'] for kind in ['observed','forecast']]]
CORRS, JOINS, BINS, STRATA, SENS, GALLERY = [], [], [], [], [], []
INFO = {}
COLORS = ['#26718c','#d7804f','#6a8c68']


def valid_weather(s, col):
    valid = s.notna() & np.isfinite(s)
    if col.endswith('cloud'):
        valid &= s.between(0,100)
    elif col in ['wa_humidity','wb_rel_hum']:
        valid &= s.between(0,100)
    elif col == 'wb_vis':
        valid &= s.between(0,50)
    elif col == 'wb_uv_idx':
        valid &= s.between(0,15)
    elif col.endswith('rain') or col in ['wb_ceiling','wb_precip_1h']:
        valid &= s >= 0
    return valid


def pearson(x,y):
    return float(x.corr(y)) if x.nunique()>1 and y.nunique()>1 else None


def measures(d,col,label,clean):
    mask = d[col].notna() & np.isfinite(d[col]) & (d.energy>=0)
    if clean:
        mask &= valid_weather(d[col],col)
    a = d.loc[mask, ['site_id','month','hour',col,'energy','cf']]
    count=len(a)
    day = a.loc[a.hour.between(9,16)].copy()
    noon = a.loc[a.hour==12]
    demeaned = day[[col,'cf']] - day.groupby(['site_id','month','hour'],observed=True)[[col,'cf']].transform('mean')
    result = {'dataset':label,'variable':col,'quality':'주의값 제외' if clean else '원본 기상값',
              'rows':count,'day_rows':len(day),'noon_rows':len(noon),
              'pearson_energy_all':pearson(a[col],a.energy),'pearson_cf_all':pearson(a[col],a.cf),
              'pearson_cf_day':pearson(day[col],day.cf),
              'spearman_cf_day':float(day[col].corr(day.cf,method='spearman')) if day[col].nunique()>1 else None,
              'pearson_cf_noon':pearson(noon[col],noon.cf),
              'pearson_cf_within_site_month_hour_day':pearson(demeaned[col],demeaned.cf),
              'strata':int(day.groupby(['site_id','month','hour'],observed=True).ngroups)}
    CORRS.append(result)
    return a,day,demeaned


def bins_for(s,col):
    if col.endswith('cloud'):
        labels=['0','(0,20]','(20,40]','(40,60]','(60,80]','(80,100)','100']
        middle=pd.cut(s.where(s.gt(0)&s.lt(100)),[0,20,40,60,80,100],labels=labels[1:-1])
        result=middle.astype('string')
        result.loc[s.eq(0)]='0'
        result.loc[s.eq(100)]='100'
        return pd.Series(pd.Categorical(result,categories=labels,ordered=True),index=s.index)
    return pd.cut(s,[-.001,0,.1,1,5,10,np.inf],labels=['0','(0,0.1]','(0.1,1]','(1,5]','(5,10]','>10'])


def binned(d,col,label,scope):
    a=d.copy()
    a['bin']=bins_for(a[col],col)
    grouped=a.groupby('bin',observed=True).agg(n=('cf','size'),mean_cf=('cf','mean'),median_cf=('cf','median'),
            q25=('cf',lambda x:x.quantile(.25)),q75=('cf',lambda x:x.quantile(.75)),mean_energy=('energy','mean'))
    grouped['dataset']=label
    grouped['variable']=col
    grouped['scope']=scope
    BINS.extend(grouped.reset_index().astype({'bin':str}).to_dict('records'))
    return grouped


def save(fig,name,title,note):
    fig.suptitle(title,fontsize=13,fontweight='bold')
    fig.tight_layout(rect=[0,0,1,.94])
    fig.savefig(FIG/f'{name}.png',dpi=125,bbox_inches='tight')
    plt.close(fig)
    GALLERY.append({'file':f'figures/{name}.png','title':title,'note':note})


def cloud_rain_plot(d,label,key,p):
    fig,axes=plt.subplots(2,3,figsize=(16,8.3))
    rain_col='wb_precip_1h' if 'wb_precip_1h' in d else p+'_rain'
    for row,col in enumerate([p+'_cloud',rain_col]):
        a,day,_=measures(d,col,label,True)
        allbin=binned(a,col,label,'전체 시각')
        daybin=binned(day,col,label,'09~17시 구간')
        for ax,g,title in [(axes[row,0],allbin,'전체 시각'),(axes[row,1],daybin,'구간 시작 09~16시')]:
            x=np.arange(len(g))
            ax.plot(x,g.mean_cf,color=COLORS[0],marker='o',label='평균 cf')
            ax.plot(x,g.median_cf,color=COLORS[1],marker='.',label='중앙값 cf')
            ax.fill_between(x,g.q25,g.q75,color=COLORS[0],alpha=.17,label='IQR')
            ax.set_xticks(x,g.index.astype(str),rotation=25,ha='right')
            ax.set_ylabel('용량 대비 발전 비율')
            ax.set_title(f'{col} / {title}')
            ax.legend(fontsize=7)
        ax=axes[row,2]
        pairs=[]
        for (month,hour),z in day.groupby(['month','hour'],observed=True):
            c=pearson(z[col],z.cf)
            pairs.append({'dataset':label,'variable':col,'month':int(month),'hour':int(hour),'n':len(z),'pearson':c})
        STRATA.extend(pairs)
        panel=pd.DataFrame(pairs).pivot(index='month',columns='hour',values='pearson').reindex(index=range(1,13),columns=range(9,17))
        im=ax.imshow(panel,aspect='auto',vmin=-1,vmax=1,cmap='RdBu_r')
        ax.set_xticks(range(8),range(9,17)); ax.set_yticks(range(12),range(1,13))
        ax.set_xlabel('구간 시작 시각'); ax.set_ylabel('2024년 월')
        ax.set_title(f'{col} / 같은 월과 시간 안의 Pearson')
        fig.colorbar(im,ax=ax,shrink=.8)
    save(fig,key+'_cloud_rain',label+' / 운량과 강수의 관계',
         '음수 발전량과 해당 기상 열 주의값 제외. IQR은 관측 분포이며 신뢰구간이 아님. 표본 수는 로컬 bin_summary.csv에 기록.')
    # Count panels separate from curves so sparse extreme bins are visible.
    fig,axes=plt.subplots(1,2,figsize=(12,4.4))
    for ax,col in zip(axes,[p+'_cloud',rain_col]):
        records=[r for r in BINS if r['dataset']==label and r['variable']==col and r['scope']=='09~17시 구간']
        ax.bar(range(len(records)),[r['n'] for r in records],color=COLORS[0])
        ax.set_xticks(range(len(records)),[r['bin'] for r in records],rotation=25,ha='right')
        ax.set_yscale('log');ax.set_ylabel('행 수 (로그)');ax.set_title(col)
    save(fig,key+'_counts',label+' / 낮 시간 기상 구간별 표본 수','같은 기상 지점을 공유하는 발전소의 반복 기록이 포함돼 독립 기상 표본 수와 다름.')


def process(base,p,kind):
    station=p+'_station'
    rel=f'train/{kind}_{p}.parquet'
    cols=VARS[p] if kind=='forecast' else [c for c in VARS[p] if c!='wb_precip_prob']
    if p=='wb' and kind=='observed':
        cols=['wb_precip_1h' if c=='wb_rain' else c for c in cols]
    w=pd.read_parquet(SOURCE/rel,columns=[station,'ts',*(['round'] if kind=='forecast' else []),*cols])
    duplicate_mask=w.duplicated([station,'ts'],keep=False) if kind=='observed' else pd.Series(False,index=w.index)
    conflicts=0
    if duplicate_mask.any():
        # Keep values that agree within a duplicate key; conflicting cells become NaN.
        agg=w.groupby([station,'ts'],observed=True)[cols].agg(['first','nunique'])
        collapsed=agg.xs('first',level=1,axis=1).copy()
        counts=agg.xs('nunique',level=1,axis=1)
        conflicts=int((counts>1).to_numpy().sum())
        w=collapsed.mask(counts>1).reset_index()
    off=w.ts.ne(w.ts.dt.floor('h'))
    off_total=int(off.sum())
    w=w.loc[~off]
    # Equality joining, no timestamp rounding and no forecast-round averaging.
    start=w.ts-pd.Timedelta(hours=1)
    w=w.loc[start.dt.year==2024]
    rounds=[10,17] if kind=='forecast' else [None]
    for r in rounds:
        weather=w.loc[w['round']==r].drop(columns='round') if r else w
        assert not weather.duplicated([station,'ts']).any()
        d=base.merge(weather,on=[station,'ts'],how='left',validate='many_to_one',indicator=True)
        assert len(d)==len(base)
        label=f'{p.upper()} '+('실측' if r is None else f'예보 {r}시')
        key=p+'_'+('observed' if r is None else str(r))
        JOINS.append({'dataset':label,'source':rel,'left_rows':len(base),'joined_rows':len(d),
                      'unmatched_rows':int(d['_merge'].eq('left_only').sum()),'duplicate_conflicting_cells_all_dates':conflicts,
                      'off_hour_source_rows_excluded_all_dates':off_total,
                      'matched_station_times':len(d.loc[d['_merge'].eq('both'),[station,'ts']].drop_duplicates())})
        for col in cols:
            measures(d,col,label,False)
            # Cloud and rain cleaned metrics are calculated by plotting below.
            if col not in [p+'_cloud',p+'_rain','wb_precip_1h']:
                measures(d,col,label,True)
        cloud_rain_plot(d,label,key,p)
        for col,target in [('wb_vis',16.0934),('wb_ceiling',9144.)] if p=='wb' else []:
            mask=valid_weather(d[col],col)&(d.energy>=0)&d.hour.between(9,16)
            a=d.loc[mask,['site_id','month','hour',col,'cf']]
            for scope,z in [('전체',a),('집중값 제외',a.loc[~np.isclose(a[col],target,rtol=0,atol=1e-9)])]:
                within=z[[col,'cf']]-z.groupby(['site_id','month','hour'],observed=True)[[col,'cf']].transform('mean')
                SENS.append({'dataset':label,'variable':col,'scope':scope,'rows':len(z),'target':target,
                             'pearson':pearson(z[col],z.cf),'within_pearson':pearson(within[col],within.cf)})
        del d
        gc.collect()
        log(f'Completed: {label}')


def overview_plots():
    c=pd.DataFrame(CORRS)
    clean=c.loc[c.quality=='주의값 제외']
    matrix=clean.pivot(index='dataset',columns='variable',values='pearson_cf_within_site_month_hour_day')
    fig,ax=plt.subplots(figsize=(14,5))
    im=ax.imshow(matrix,aspect='auto',vmin=-1,vmax=1,cmap='RdBu_r')
    ax.set_xticks(range(len(matrix.columns)),matrix.columns,rotation=40,ha='right')
    ax.set_yticks(range(len(matrix.index)),matrix.index)
    for i in range(len(matrix)):
        for j in range(len(matrix.columns)):
            val=matrix.iloc[i,j]
            if pd.notna(val):ax.text(j,i,f'{val:.2f}',ha='center',va='center',fontsize=8)
    fig.colorbar(im,ax=ax,label='Pearson (같은 발전소/월/시간 그룹 평균을 뺀 편차)')
    save(fig,'13_within_correlations','시간과 발전소 그룹을 맞춘 기상-발전량 관계','관찰된 그룹 평균 편차를 비교한 기술통계. 인과 통제나 예측모형의 잔차가 아님. 빈 셀은 해당 제공사에 변수 없음.')
    for col,name in [('wb_vis','14_visibility_sensitivity'),('wb_ceiling','15_ceiling_sensitivity')]:
        a=pd.DataFrame(SENS);a=a.loc[a.variable==col]
        fig,axes=plt.subplots(1,2,figsize=(12,4.5))
        for ax,metric,title in [(axes[0],'pearson','낮 시간 단순 관계'),(axes[1],'within_pearson','발전소/월/시간 평균 편차 관계')]:
            z=a.pivot(index='dataset',columns='scope',values=metric)
            z.plot.bar(ax=ax,color=COLORS[:2],rot=0)
            ax.axhline(0,color='grey',lw=.7);ax.set_ylim(-1,1);ax.set_ylabel('Pearson');ax.set_title(title)
        save(fig,name,col+' 집중값 포함/제외 민감도','값의 의미가 미확정이므로 제거를 권장하는 결과가 아님. 제외하면 표본과 지지 범위가 바뀜.')


def artifacts():
    for name,rows in [('correlations',CORRS),('join_audit',JOINS),('bin_summary',BINS),('month_hour_correlations',STRATA),('concentration_sensitivity',SENS)]:
        pd.DataFrame(rows).to_csv(OUT/f'{name}.csv',index=False,encoding='utf-8-sig')
    (OUT/'summary.json').write_text(json.dumps(INFO,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    (OUT/'gallery.json').write_text(json.dumps(GALLERY,ensure_ascii=False,indent=2),encoding='utf-8')
    cards=''.join(f'<section><h2>{html.escape(g["title"])}</h2><p>{html.escape(g["note"])}</p><a href="{g["file"]}"><img src="{g["file"]}" loading="lazy"></a></section>' for g in GALLERY)
    (OUT/'index.html').write_text('''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>기상과 발전량 EDA</title><style>body{font:16px/1.6 'Malgun Gothic',sans-serif;background:#f4f5f5;color:#26333c;max-width:1200px;margin:auto;padding:24px}section{background:white;padding:22px;margin:24px 0;border:1px solid #ddd}img{width:100%}h1{font-size:26px}h2{font-size:19px}</style><h1>2024 고정 집단의 기상과 발전량 관계</h1><p>실측과 예보 라운드를 분리한 기술통계. 모델 학습과 예측 없음.</p>'''+cards+'</html>',encoding='utf-8')
    qa=OUT/'qa';qa.mkdir(exist_ok=True)
    for start in range(0,len(GALLERY),4):
        pics=[]
        for g in GALLERY[start:start+4]:
            with Image.open(OUT/g['file']) as im:
                im.load();pics.append(im.convert('RGB').resize((im.width//2,im.height//2),Image.Resampling.LANCZOS))
        w,h=max(im.width for im in pics),max(im.height for im in pics)+24
        sheet=Image.new('RGB',(w*2,h*2),'#ededed');pen=ImageDraw.Draw(sheet)
        for i,im in enumerate(pics):
            x,y=(i%2)*w,(i//2)*h;sheet.paste(im,(x,y+24));pen.text((x+5,y+4),GALLERY[start+i]['file'],fill='black')
        sheet.save(qa/f'contact_{start//4+1:02d}.png')


def report():
    c=pd.DataFrame(CORRS);main=c.loc[(c.quality=='주의값 제외')&c.variable.str.endswith(('cloud','rain','precip_1h'))]
    quality=c.loc[c.quality=='원본 기상값'].merge(c.loc[c.quality=='주의값 제외'],on=['dataset','variable'],suffixes=('_raw','_clean'))
    changed=quality.loc[quality.rows_raw!=quality.rows_clean]
    p=['# 기상과 발전량의 관계 EDA','',f"작성: {INFO['finished_at']}. 상태: 실제 자료 조인과 기술통계 실행 완료.",'',
       '## 질문과 목적','','운량과 강수에 따라 발전량과 용량 대비 발전 비율이 어떻게 달라지는지, 전체 시각과 같은 월/시간대 안의 관계가 다른지 확인한다. 기온, 습도, 시정, 운고와 UV 등은 보조 변수로 같은 조건에서 확인했다. 실측, 10시 예보와 17시 예보를 분리했다. 모델 학습과 예측은 하지 않았다.','',
       '## 사용 범위와 수행 방법','',
       f"2024년 전 시간 기록이 있는 발전소 {INFO['sites']}개, 발전량 {fmt(INFO['generation_rows'])}행을 사용했다. 전 보고서와 같은 집단이다. 발전량을 정적 capacity_kw와 1시간으로 나눈 cf를 비교 대상으로 삼았다. 원본 발전량 음수는 이 집단에 없었다. 기상 지점은 sites.csv의 매핑을 그대로 사용했다.",'',
       '- 시각은 우측닫힘으로 맞추고 ts에서 1시간을 뺀 시작 시각으로 월과 시간을 정의했다. 낮 시간 비교는 모든 계절에 같은 09:00~17:00 구간(시작 시각 9~16시)을 사용한다. 천문학적 주간 전체나 일출 시각을 계산한 것은 아니다.',
       '- A 실측 중복은 같은 키/같은 열에서 값이 일치하면 보존하고 값이 충돌하면 그 셀만 NaN으로 처리한 임시 분석 뷰를 만들었다. 원본은 수정하지 않았다. 조인은 many_to_one 검증을 통과했고 왼쪽 발전량 행을 유지했다.',
       '- 비정시 예보는 반올림하지 않고 별도 제외했다. 예보 round를 평균하지 않고 각각 조인했다. 없는 예보 행과 값 결측은 0으로 채우지 않았다.',
       '- B 실측의 강수 열은 wb_precip_1h(1시간 강수량), B 예보의 강우 열은 wb_rain이다. 열 이름을 같은 것으로 가정하지 않고 각각 원래 이름으로 분석했다. 강수 정의 차이가 있어 숫자만으로 실측과 예보의 일치도를 판단하지 않는다.',
       '- 원본 기상값과 열별 주의값 제외를 함께 집계했다. 운량/습도는 0~100, 시정은 0~50 km, UV는 0~15, 강우량/운고는 0 이상을 비교용 범위로 두었다. 습도 100 초과가 배포에 포함된다는 사실과 구분하며 이 범위를 최종 정제 규칙으로 확정하지 않았다.',
       '- Pearson을 원시 발전량과 cf, 전체 시간과 낮 시간, 정오 한 시간에 각각 계산했다. 낮 시간 Spearman도 계산했다. 발전소/월/구간 시작 시간별 기상과 cf의 각 관측 평균을 뺀 편차의 Pearson을 추가했다. 이는 그룹 내 기술통계이며 회귀모형을 적합하거나 완전한 교란 통제를 한 결과가 아니다.',
       '- 월별/시간별 셀 안의 상관과 고정 기상 구간별 평균, 중앙값, IQR, 표본 수를 확인했다. 독립 표본 가정의 p값과 신뢰구간은 계산하지 않았다. 같은 기상 지점의 자료가 여러 발전소에 반복 사용되며 상관은 발전소-시간 행 가중이다.','',
       '## 조인과 가용성','',md_table(['자료','왼쪽/조인 후 행','기상 키 미매칭','매칭된 고유 지점-시각','전 기간 원본의 비정시 제외'],[[r['dataset'],f"{fmt(r['left_rows'])} / {fmt(r['joined_rows'])}",fmt(r['unmatched_rows']),fmt(r['matched_station_times']),fmt(r['off_hour_source_rows_excluded_all_dates'])] for r in JOINS]),'',
       '미매칭은 기상 키 기준이다. 매칭돼도 특정 열에 NaN이나 충돌 셀이 있을 수 있어 변수별 유효 표본 수는 다르다. 전 기간 비정시 제외 수를 2024년 고정 집단의 손실 행 수와 혼동하지 않는다.','',
       '## 운량과 강수의 주요 결과','',md_table(['자료','변수','전체 cf Pearson','낮 cf Pearson','낮 cf Spearman','정오 cf Pearson','발전소/월/시간 편차 Pearson'],[[r.dataset,r.variable,*[f'{v:.4f}' if pd.notna(v) else '계산 불가' for v in [r.pearson_cf_all,r.pearson_cf_day,r.spearman_cf_day,r.pearson_cf_noon,r.pearson_cf_within_site_month_hour_day]]] for r in main.itertuples()]),'',
       '운량은 두 제공사의 실측과 두 예보 라운드 모두에서 발전 비율과 음의 관계를 보였다. 전체 시각의 Pearson은 -0.0757~-0.1481이지만 낮 시간에는 -0.3183~-0.4940, 같은 발전소/월/시간 평균을 뺀 편차에서는 -0.4334~-0.6598이었다. 낮 시간대와 고정 그룹 안에서도 방향이 이어져, 전체 시간의 작은 상관만으로 운량과의 관계가 약하다고 판단하기 어렵다.','',
       '강수도 모든 자료에서 음의 관계였다. 낮 시간 Spearman의 절댓값이 Pearson보다 컸으며, 강수 0의 집중과 양수 구간의 비선형 모양을 함께 볼 필요가 있다. A 17시 예보에서 낮 시간의 강수 0 구간 평균 cf는 0.4382, 1~5 mm는 0.1500, 10 mm 초과는 0.0527이었다. 마지막 구간은 450행뿐이고 같은 지점 자료가 발전소마다 반복되므로 독립 기상 표본이 충분하다는 의미는 아니다.','',
       '운량 또는 강수와의 음의 관계가 나타나더라도 이것만으로 인과 효과를 추정한 것은 아니다. 전체 시각에는 야간과 낮의 차이가 섞여 있어 같은 월/시간대와 그룹 평균 편차 결과를 함께 봐야 한다. 실측 관계가 강해도 예측시점에 실측값을 사용할 수 있다는 의미는 아니다.','',
       '## 기상 구간별 중심과 산포','',
       '아래는 예보 17시의 낮 시간 표다. 운량은 정확한 0과 100의 집중을 중간 구간과 분리했고 강수는 정확한 0과 양수를 분리했다. 강수 단위는 두 제공사 모두 mm다. 표의 IQR은 기록의 산포로 예측 오차나 신뢰구간이 아니다.','']
    for label in ['WA 예보 17시','WB 예보 17시']:
        rows=[r for r in BINS if r['dataset']==label and r['scope']=='09~17시 구간']
        p += [f'### {label}','',md_table(['변수','기상 구간','행 수','평균 cf','중앙값 cf','25~75백분위'],[[r['variable'],r['bin'],fmt(r['n']),f"{r['mean_cf']:.4f}",f"{r['median_cf']:.4f}",f"{r['q25']:.4f} ~ {r['q75']:.4f}"] for r in rows]),'']
    p+=['## 주의값과 집중값 민감도','',md_table(['자료','변수','원본/비교범위 행','낮 Pearson 원본/제외','편차 Pearson 원본/제외'],[[r.dataset,r.variable,f'{fmt(r.rows_raw)} / {fmt(r.rows_clean)}',f'{r.pearson_cf_day_raw:.4f} / {r.pearson_cf_day_clean:.4f}',f'{r.pearson_cf_within_site_month_hour_day_raw:.4f} / {r.pearson_cf_within_site_month_hour_day_clean:.4f}'] for r in changed.itertuples()]),'',
         md_table(['자료','변수','비교','낮 표본 수','낮 Pearson','편차 Pearson'],[[r['dataset'],r['variable'],r['scope'],fmt(r['rows']),f"{r['pearson']:.4f}",f"{r['within_pearson']:.4f}"] for r in SENS]),'',
         '이번 2024년 고정 집단에서 비교 범위 제외로 표본이 바뀐 열은 A 예보 습도뿐이었다. 10시 예보 27행과 17시 예보 4행이 빠졌으며 상관 변화는 작았다. 앞선 전체 기간 점검에서 발견한 극단값이 모두 안전하다는 결론은 아니다. 표본의 연도와 발전소 범위가 달라 이번 비교에 포함되지 않은 문제가 있다.','',
         '예보 시정은 집중값 포함 여부에 특히 민감했다. B 17시 예보의 낮 시간 Pearson은 0.2068에서 -0.0860으로, 그룹 평균 편차 Pearson은 0.2462에서 -0.1556으로 바뀌었다. B 10시 예보에서도 같은 방향 전환이 나타났다. B 17시 예보 운고의 편차 Pearson은 0.4279에서 0.2403으로 낮아졌지만 양의 방향은 유지됐다. 시정과 운고를 단순한 연속값 하나로 해석하기 전에 집중값의 생성 방식을 확인해야 한다.','',
         '시정 16.0934 km와 운고 9,144 m의 의미는 미확정이다. 포함/제외 결과가 달라도 결측 코드라고 확정할 수 없다. 집중값을 제외하면 표본과 변수 범위도 바뀌고 그룹 평균도 다시 계산되므로 같은 모집단에서의 성능 비교가 아니다. 실측 시정 최빈값 16.1 km는 이 정확한 집중값 비교에서 제외되지 않는다.','',
         '## 보조 변수의 관계','',md_table(['자료','변수','낮 유효 행','낮 cf Pearson','낮 cf Spearman','발전소/월/시간 편차 Pearson'],[[r.dataset,r.variable,fmt(r.day_rows),f'{r.pearson_cf_day:.4f}',f'{r.spearman_cf_day:.4f}',f'{r.pearson_cf_within_site_month_hour_day:.4f}'] for r in c.loc[(c.quality=='주의값 제외')&~c.variable.str.endswith(('cloud','rain','precip_1h'))].itertuples()]),'',
         '습도는 모든 자료에서 낮 시간 cf와 음의 관계였고 그룹 평균 편차의 Pearson은 약 -0.64~-0.70이었다. UV는 양의 관계였으며 B 실측의 낮 시간 Pearson은 0.6662였다. 기온도 양의 관계지만 계절과 일사 등의 동반 변화를 완전히 제거한 결과가 아니므로 기온 상승의 효과로 해석하지 않는다. 이 표는 기상 변수 각각과 발전량의 관계이며 변수 사이의 중복 정보나 독립 기여를 평가한 결과는 아니다.','',
         '## 한계와 다음 단계','','고정 집단은 148개소, 한 해의 관측으로 전체 253개소나 다른 연도에 대한 일반화는 확인하지 않았다. 동일 기상 지점을 공유하고 시간적으로 연속한 행은 독립 표본이 아니다. 발전소/월/시간 평균 편차도 날짜별 공통 날씨, 일사, 설비 방향과 운영 상태를 모두 통제하지 않는다. rain의 많은 0과 극단 구간의 작은 표본 수는 상관 크기와 곡선 해석에 영향을 준다.','',
         '실측과 예보 결과는 각 자료에서 유효한 행으로 집계해 완전히 같은 표본이 아니다. 예보와 실측의 숫자 차이를 제공사 정확도 우열이나 예측 성능으로 해석하지 않는다. 다음 EDA에서 예보와 실측을 동일 지점/대상 시각으로 맞춘 오차를 별도로 확인할 수 있다. 모델 학습, 발전량 예측, 집합 구성 최적화와 제출은 하지 않았다.','',
         '## 관련 코드와 결과','','- [기상 관계 EDA 코드](../scripts/weather_generation_eda.py)','- [그래프 갤러리](../data/eda/2026-10-19_weather_generation/index.html)','- [전체 상관 집계](../data/eda/2026-10-19_weather_generation/correlations.csv)','- [구간별 집계와 표본 수](../data/eda/2026-10-19_weather_generation/bin_summary.csv)','- [조인 검증](../data/eda/2026-10-19_weather_generation/join_audit.csv)','','그래프와 상세 데이터 출력은 Git에서 제외된 로컬 data/에만 보관한다. 원본 해시는 실행 전후 동일했다. 코드와 요약 보고서, 일별 로그만 로컬 커밋한다.','']
    REPORT.write_text('\n'.join(p),encoding='utf-8')


def main():
    sys.stdout.reconfigure(encoding='utf-8');assert sys.version_info[:2]==(3,13)
    FIG.mkdir(parents=True,exist_ok=True)
    if '--verify-only' in sys.argv:
        info=json.loads((OUT/'summary.json').read_text(encoding='utf-8'))
        joins=pd.read_csv(OUT/'join_audit.csv',encoding='utf-8-sig')
        corrs=pd.read_csv(OUT/'correlations.csv',encoding='utf-8-sig')
        gallery=json.loads((OUT/'gallery.json').read_text(encoding='utf-8'))
        assert joins.left_rows.eq(joins.joined_rows).all() and len(joins)==6
        assert info['sites']==148 and info['generation_rows']==148*8784
        assert joins.left_rows.eq(info['generation_rows']).all() and joins.unmatched_rows.eq(0).all()
        assert corrs.groupby(['dataset','variable']).size().eq(2).all()
        assert len(corrs)==70 and len(gallery)==15
        bins=pd.read_csv(OUT/'bin_summary.csv',encoding='utf-8-sig')
        strata=pd.read_csv(OUT/'month_hour_correlations.csv',encoding='utf-8-sig')
        for (dataset,variable),z in bins.groupby(['dataset','variable']):
            expected=corrs.loc[(corrs.dataset==dataset)&(corrs.variable==variable)&(corrs.quality=='주의값 제외')].iloc[0]
            assert z.loc[z.scope=='전체 시각','n'].sum()==expected.rows
            assert z.loc[z.scope=='09~17시 구간','n'].sum()==expected.day_rows
            cells=strata.loc[(strata.dataset==dataset)&(strata.variable==variable)]
            assert len(cells)==12*8 and cells.n.sum()==expected.day_rows
        probe=pd.Series([0.,.1,20.,20.0001,80.,99.9995,100.,np.nan])
        assert bins_for(probe,'wa_cloud').astype('string').fillna('NA').tolist()==['0','(0,20]','(0,20]','(20,40]','(60,80]','(80,100)','100','NA']
        assert ((corrs.filter(regex='pearson|spearman').abs()<=1.000001)|corrs.filter(regex='pearson|spearman').isna()).all().all()
        for g in gallery:
            with Image.open(OUT/g['file']) as im:im.verify()
        links=re.findall(r'(?:href|src)="([^"]+)"',(OUT/'index.html').read_text(encoding='utf-8'))
        assert len(links)==30 and all((OUT/p).is_file() for p in links)
        assert info['source_sha256']=={r:digest(SOURCE/r) for r in FILES}
        assert '\u00b7' not in REPORT.read_text(encoding='utf-8')
        check={'joined_datasets':6,'variable_quality_combinations':len(corrs),'figures_checked':15,'links_checked':30,'source_unchanged':True,'no_row_expansion':True}
        (OUT/'verification.json').write_text(json.dumps(check,indent=2),encoding='utf-8');print(json.dumps(check,indent=2));return
    before={r:digest(SOURCE/r) for r in FILES}
    INFO['started_at']=datetime.now(ZoneInfo('Asia/Seoul')).isoformat(timespec='minutes')
    sites=pd.read_csv(SOURCE/'sites.csv',encoding='utf-8-sig')
    gen=pd.read_parquet(SOURCE/'train/generation.parquet')
    start=gen.ts-pd.Timedelta(hours=1)
    gen=gen.loc[start.dt.year==2024].copy()
    counts=gen.groupby('site_id').size();ids=counts[counts==8784].index
    gen=gen.loc[gen.site_id.isin(ids)]
    base=gen.merge(sites[['site_id','capacity_kw','wa_station','wb_station']],on='site_id',validate='many_to_one')
    assert len(ids)==148 and len(base)==148*8784 and (base.energy>=0).all()
    base['cf']=base.energy/base.capacity_kw
    start=base.ts-pd.Timedelta(hours=1);base['month']=start.dt.month.astype('int8');base['hour']=start.dt.hour.astype('int8')
    INFO.update(sites=len(ids),generation_rows=len(base))
    del gen;gc.collect()
    for p in ['wa','wb']:
        for kind in ['observed','forecast']:process(base,p,kind)
    overview_plots()
    assert before=={r:digest(SOURCE/r) for r in FILES}
    INFO.update(source_sha256=before,source_unchanged=True,figures=len(GALLERY),finished_at=datetime.now(ZoneInfo('Asia/Seoul')).isoformat(timespec='minutes'))
    artifacts();report()
    log(f'Completed: {len(CORRS)} variable/quality metrics, {len(GALLERY)} figures. {REPORT}')


if __name__=='__main__':main()
