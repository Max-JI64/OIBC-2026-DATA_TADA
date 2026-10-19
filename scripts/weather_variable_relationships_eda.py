"""Relationships among supplied weather variables; descriptive EDA only."""
import gc
import json
import sys
from datetime import datetime
from itertools import combinations
from zoneinfo import ZoneInfo
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from basic_shape_eda import ROOT,SOURCE,digest,fmt,md_table,log
from eda_plot_utils import save_plot,write_gallery,check_gallery

OUT=ROOT/'data/eda/2026-10-19_weather_variables'
REPORT=ROOT/'reports/2026-10-19_09_기상_변수_사이의_관계_EDA.md'
FILES=['sites.csv','train/generation.parquet',*[f'train/{k}_{p}.parquet' for p in ['wa','wb'] for k in ['observed','forecast']]]
GALLERY=[];PAIRS=[];AUDIT=[];SENS=[];INFO={}


def valid(s,col):
    a=np.isfinite(s)
    if col.endswith(('cloud','humidity','rel_hum','prob')):a &= s.between(0,100)
    elif col.endswith('wind_speed'):a &= s.between(0,50 if col.startswith('wa') else 200)
    elif col.endswith('wind_gust_spd'):a &= s.between(0,200)
    elif col.endswith('wind_dir'):a &= s.between(0,360)
    elif col.endswith(('press','pressure')):a &= s.between(900,1100)
    elif col=='wb_vis':a &= s.between(0,50)
    elif col=='wb_uv_idx':a &= s.between(0,15)
    elif col.endswith(('rain','snow','ice','ceiling','total_liq','precip_1h')):a &= s>=0
    return a


def load(p,kind,stations):
    d=pd.read_parquet(SOURCE/f'train/{kind}_{p}.parquet')
    station=p+'_station'
    d=d.loc[((d.ts-pd.Timedelta(hours=1)).dt.year==2024)&d[station].isin(stations)].copy()
    d=d.loc[d.ts.eq(d.ts.dt.floor('h'))]
    cols=[c for c in d if c not in [station,'ts','round']]
    if kind=='observed':
        dup=d.duplicated([station,'ts'],keep=False)
        if dup.any():
            z=d.loc[dup].groupby([station,'ts'],observed=True)[cols].agg(['first','nunique'])
            fixed=z.xs('first',axis=1,level=1).mask(z.xs('nunique',axis=1,level=1)>1).reset_index()
            d=pd.concat([d.loc[~dup],fixed],ignore_index=True)
    return d,cols


def heat(matrix,label,name,note):
    size=max(9,len(matrix)*.54)
    fig,ax=plt.subplots(figsize=(size+2,size))
    im=ax.imshow(matrix,vmin=-1,vmax=1,cmap='RdBu_r')
    ax.set_xticks(range(len(matrix)),matrix.columns,rotation=65,ha='right',fontsize=8)
    ax.set_yticks(range(len(matrix)),matrix.index,fontsize=8)
    if len(matrix)<=12:
        for i in range(len(matrix)):
            for j in range(len(matrix)):
                v=matrix.iloc[i,j]
                if pd.notna(v):ax.text(j,i,f'{v:.2f}',ha='center',va='center',fontsize=7)
    fig.colorbar(im,ax=ax,shrink=.7,label='상관계수')
    save_plot(fig,OUT,GALLERY,name,label,note+' 빈 셀은 상수 열 등으로 상관 계산 불가.')


def concentration(d,cols,label,station):
    pairs=[('wb_vis','wb_cloud'),('wb_vis','wb_rel_hum'),('wb_ceiling','wb_cloud'),('wb_vis','wb_ceiling')]
    day=d.loc[d.hour.between(9,16)].copy()
    for a,b in pairs:
        for scope in ['집중값 포함','해당 쌍 집중값 제외']:
            z=day
            if scope.endswith('제외'):
                mask=pd.Series(True,index=day.index)
                for c in [a,b]:
                    if c=='wb_vis':
                        target=16.1 if '실측' in label else 16.0934
                        mask &= ~np.isclose(day[c],target,rtol=0,atol=1e-9)
                    if c=='wb_ceiling':mask &= ~np.isclose(day[c],9144,rtol=0,atol=1e-9)
                z=day.loc[mask]
            centered=z[[a,b]]-z.groupby([station,'month','hour'],observed=True)[[a,b]].transform('mean')
            SENS.append({'dataset':label,'a':a,'b':b,'scope':scope,'n':len(z),'pearson_day':z[a].corr(z[b]),
                         'pearson_within_day':centered[a].corr(centered[b])})


def process(d,cols,label,key,p):
    station=p+'_station'
    before=len(d)
    # Use one complete comparison sample per dataset so matrices share support.
    good=pd.Series(True,index=d.index)
    for col in cols:good &= valid(d[col],col)
    d=d.loc[good].copy()
    assert not d.duplicated([station,'ts']).any()
    angle=p+'_wind_dir'
    d[angle+'_sin']=np.sin(np.deg2rad(d[angle]));d[angle+'_cos']=np.cos(np.deg2rad(d[angle]))
    cols=[c for c in cols if c!=angle]+[angle+'_sin',angle+'_cos']
    start=d.ts-pd.Timedelta(hours=1)
    d['month']=start.dt.month;d['hour']=start.dt.hour
    day=d.loc[d.hour.between(9,16)].copy()
    centered=day[cols]-day.groupby([station,'month','hour'],observed=True)[cols].transform('mean')
    matrices={'pearson_all':d[cols].corr(),'spearman_all':d[cols].corr(method='spearman'),
              'pearson_day':day[cols].corr(),'pearson_within_day':centered.corr()}
    constants=[c for c in cols if d[c].nunique()<2]
    AUDIT.append({'dataset':label,'before_rows':before,'complete_rows':len(d),'excluded_rows':before-len(d),
                  'day_rows':len(day),'columns':len(cols),'constant_columns':','.join(constants)})
    for name,matrix in matrices.items():matrix.to_csv(OUT/f'{key}_{name}.csv',encoding='utf-8-sig')
    for a,b in combinations(cols,2):
        values=d[[a,b]]
        zero=(values[a].eq(0)&values[b].eq(0))
        mask=(values[a]>0)&(values[b]>0)
        positive_corr=None
        if zero.mean()>.5 and int(mask.sum())>=10 and not any('wind_dir' in x for x in [a,b]):
            positive_corr=values.loc[mask,a].corr(values.loc[mask,b],method='spearman')
        PAIRS.append({'dataset':label,'a':a,'b':b,'n_all':len(d),'n_day':len(day),
                      **{name:float(m.loc[a,b]) for name,m in matrices.items()},
                      'exact_equal_share':float(np.isclose(values[a],values[b],rtol=0,atol=1e-9).mean()),
                      'both_zero_share':float(zero.mean()),'both_positive_n':int(mask.sum()),'spearman_both_positive':positive_corr,
                      'max_abs_difference':float((values[a]-values[b]).abs().max())})
    heat(matrices['spearman_all'],label+' / 전체 시간 Spearman',key+'_spearman',
         '2024 연결 지점의 자료별 완전 비교 표본. 단순 관계이며 월과 시간 패턴이 함께 포함됨.')
    heat(matrices['pearson_within_day'],label+' / 같은 지점, 월, 시간 안의 Pearson',key+'_within',
         '시작 9~16시, 지점/월/시간별 평균을 뺀 편차. 모형 적합이나 완전한 인과 통제가 아님.')
    if p=='wb':concentration(d,cols,label,station)
    if key in ['wa_17','wb_17','wb_observed']:
        pairs=[(p+'_temp',p+'_humidity' if p=='wa' else 'wb_rel_hum'),(p+'_cloud',p+'_humidity' if p=='wa' else 'wb_rel_hum')]
        pairs+=[('wa_cloud','wa_rain')] if p=='wa' else [('wb_cloud','wb_uv_idx')]
        fig,axes=plt.subplots(1,3,figsize=(15,4.8))
        take=np.linspace(0,len(day)-1,min(len(day),80000),dtype=int)
        for ax,(a,b) in zip(axes,pairs):
            ax.hexbin(day[a].iloc[take],day[b].iloc[take],gridsize=45,mincnt=1,bins='log',cmap='viridis')
            ax.set_xlabel(a);ax.set_ylabel(b)
        save_plot(fig,OUT,GALLERY,key+'_pairs',label+' / 낮 시간 변수 쌍 모양',
                  '밀도 그림만 행 순서 등간격 최대 8만 표본. 모든 상관과 집계는 전체 비교 표본으로 계산.')
    log(label+' variable relationships complete')


def get_pair(df,label,a,b):
    z=df.loc[(df.dataset==label)&(((df.a==a)&(df.b==b))|((df.a==b)&(df.b==a)))]
    return z.iloc[0] if len(z) else None


def report():
    pairs=pd.DataFrame(PAIRS)
    core=[]
    for label in [x['dataset'] for x in AUDIT]:
        p='wa' if label.startswith('WA') else 'wb';humidity=p+'_humidity' if p=='wa' else 'wb_rel_hum'
        requested=[(p+'_temp',humidity),(p+'_cloud',humidity),(p+'_cloud',p+'_rain' if p=='wa' or '예보' in label else 'wb_precip_1h')]
        if p=='wb':requested += [('wb_cloud','wb_uv_idx'),('wb_temp','wb_uv_idx')]
        for a,b in requested:
            row=get_pair(pairs,label,a,b)
            if row is not None:core.append([label,a,b,*[f'{row[m]:.3f}' for m in ['pearson_all','spearman_all','pearson_day','pearson_within_day']]])
    exact=pairs.loc[(pairs.exact_equal_share>=1-1e-12)&(pairs.max_abs_difference<=1e-9)]
    high=pairs.loc[pairs.pearson_within_day.abs()>=.9].copy()
    high=high.loc[~high.apply(lambda r:'wind_dir' in r.a or 'wind_dir' in r.b,axis=1)]
    high=high.loc[high.exact_equal_share<1-1e-12].sort_values('pearson_within_day',key=lambda x:x.abs(),ascending=False).head(20)
    text=['# 기상 변수 사이의 관계 EDA','',f"작성: {INFO['finished_at']}. 상태: 실행 완료.",'',
          '## 질문과 목적','','발전량과 관계가 있던 기상 변수들이 서로 얼마나 같은 움직임을 보이는지, 그 관계가 시간대와 계절을 구분한 뒤에도 이어지는지 확인했다. 단순히 0이 많은 변수와 값이 완전히 같은 변수를 구분했다. 모델 학습, 변수 선택과 발전량 예측은 하지 않았다.','',
          '## 사용 범위와 수행 방법','','2024년 고정 집단 148개 발전소에 연결된 A 40개와 B 47개 지점이다. 발전소별 반복 없이 고유 지점-시각을 사용했으며 실측, 10시 예보와 17시 예보를 분리했다. 자료마다 제공되는 모든 수치 기상 열을 포함했다. 제공사 사이의 직접 상관은 지점 위치가 달라 이번 범위에 넣지 않았다.','',
          '- 운량/습도/확률은 0~100, 풍향 0~360, A 풍속 0~50 m/s, B 풍속/돌풍 0~200 km/h, 기압 900~1100 hPa, 시정 0~50 km, UV 0~15, 강수/강설/착빙/운고는 0 이상을 비교 범위로 사용했다. 확정 정제 규칙은 아니다.',
          '- 중복 충돌 셀은 임시 뷰에서 NaN으로 두고 자료별 모든 열이 유효한 완전 비교 표본으로 상관 행렬을 만들었다. 이 때문에 앞선 변수별 유효 쌍 집계와 표본 수가 다르다. 결측을 0으로 채우지 않았다.',
          '- 풍향은 0/360도의 경계를 고려해 sin과 cos로 나타냈다. 두 값은 동일 각도의 표현이며 독립 측정 두 개가 아니다. 풍속이 약할 때 풍향의 의미가 약해지는 문제는 별도 검증하지 않았다.',
          '- 전체 시간 Pearson과 Spearman, 구간 시작 9~16시의 Pearson, 같은 지점/월/시간 평균을 뺀 낮 시간 편차의 Pearson을 계산했다. 평균 편차는 회귀모형이나 완전한 교란 통제가 아니다.',
          '- 값의 동일성은 절대 허용오차 1e-9와 최대 절대 차이를 함께 확인했다. 단위가 다르거나 상수인 변수는 값 일치만으로 같은 정보를 뜻하지 않는다. 함께 0인 비율이 높은 쌍은 양쪽 모두 양수인 경우의 상관도 추가했다.',
          '- B 시정/운고 집중값은 의미 미확정 상태로 유지하고 해당 변수 쌍에서만 제외하는 민감도를 계산했다. 제외 후에는 표본과 그룹 평균이 바뀌므로 처리 방법의 성능 우열을 뜻하지 않는다.','',
          '## 비교 표본','',md_table(['자료','범위 점검 전','완전 비교 표본','제외 행','낮 시간','각도 표현 후 열 수','상수 열'],[[r['dataset'],fmt(r['before_rows']),fmt(r['complete_rows']),fmt(r['excluded_rows']),fmt(r['day_rows']),r['columns'],r['constant_columns'] or '없음'] for r in AUDIT]),'',
          '## 주요 기상 변수 쌍','',md_table(['자료','변수 1','변수 2','전체 Pearson','전체 Spearman','낮 Pearson','같은 지점/월/시간 Pearson'],core),'',
          '전체와 낮/그룹 내 관계의 차이는 공통 일변화와 계절 패턴이 관계의 일부에 포함됐음을 보여줄 수 있다. 상관이 남아도 어느 변수가 독립적으로 발전량을 설명하는지, 어느 변수를 삭제해야 하는지 결정한 결과는 아니다.','',
          '## 값이 같은 쌍과 강한 관계','',md_table(['자료','변수 1','변수 2','비교 행','같은 값 비율 %','둘 다 0 비율 %'],[[r.dataset,r.a,r.b,fmt(r.n_all),f'{100*r.exact_equal_share:.2f}',f'{100*r.both_zero_share:.2f}'] for r in exact.itertuples()]),'',
          '위 표의 완전 일치도 상수 열이나 단위가 다른 열의 우연한 값 일치를 포함할 수 있다. 비상수 여부와 배포 정의를 함께 해석해야 한다. 이번 표본에서의 일치가 다른 기간까지의 완전한 중복을 보장하지 않는다.','',
          md_table(['자료','변수 1','변수 2','편차 Pearson','값 일치 %','둘 다 0 %','둘 다 양수 행','양수 구간 Spearman'],[[r.dataset,r.a,r.b,f'{r.pearson_within_day:.3f}',f'{100*r.exact_equal_share:.2f}',f'{100*r.both_zero_share:.2f}',fmt(r.both_positive_n),f'{r.spearman_both_positive:.3f}' if pd.notna(r.spearman_both_positive) else '별도 계산 대상 아님'] for r in high.itertuples()]),'',
          '강한 쌍 표는 절댓값 0.9 이상 중 상위 20개다. 전체 쌍은 로컬 pair_relationships.csv에 있으며 표에 없다는 이유로 관계가 없다고 판단하지 않는다. 희소한 눈/착빙과 확률 변수는 0의 집중과 양수 표본 수를 함께 본다.','',
          '## 집중값 포함과 제외','',md_table(['자료','변수 1','변수 2','비교','낮 행','낮 Pearson','그룹 내 Pearson'],[[r['dataset'],r['a'],r['b'],r['scope'],fmt(r['n']),f"{r['pearson_day']:.3f}",f"{r['pearson_within_day']:.3f}"] for r in SENS]),'',
          '## 한계와 다음 단계','','시간적으로 연속한 행과 여러 지점의 같은 날씨는 독립 표본이 아니다. p값과 신뢰구간은 계산하지 않았다. 2024년 연결 지점의 관계를 전체 기간이나 평가 구간으로 일반화하지 않았다. B 운량의 시간대별 구조는 원인이 미확정이고 이 관계 분석에서도 그대로 유지했다.','',
          '이번 결과는 중복 정보 후보와 조건별 관계를 확인한 EDA다. 변수의 채택/제거, 최종 전처리, 모델 학습과 예측은 수행하지 않았다. 다음 단위는 이번 세션의 EDA 결과와 미확정 사항을 종합하는 문서다.','',
          '## 관련 코드와 결과','','- [분석 코드](../scripts/weather_variable_relationships_eda.py)','- [전체 그래프](../data/eda/2026-10-19_weather_variables/index.html)','- [전체 변수 쌍 집계](../data/eda/2026-10-19_weather_variables/pair_relationships.csv)','- [집중값 민감도](../data/eda/2026-10-19_weather_variables/concentration.csv)','','원본 해시는 실행 전후 동일했다. 데이터 출력과 그림은 로컬 data/에만 보관한다.','']
    REPORT.write_text('\n'.join(text),encoding='utf-8')


def verify():
    info=json.loads((OUT/'summary.json').read_text(encoding='utf-8'))
    a=pd.read_csv(OUT/'sample_audit.csv',encoding='utf-8-sig');pairs=pd.read_csv(OUT/'pair_relationships.csv',encoding='utf-8-sig')
    assert len(a)==6 and len(pairs)==sum(int(r.columns)*(int(r.columns)-1)//2 for r in a.itertuples())
    for row in a.itertuples():
        z=pairs.loc[pairs.dataset==row.dataset]
        assert z.n_all.eq(row.complete_rows).all() and z.n_day.eq(row.day_rows).all()
        assert row.complete_rows<=row.before_rows and row.day_rows<=row.complete_rows
    for col in ['pearson_all','spearman_all','pearson_day','pearson_within_day']:
        assert (pairs[col].abs().le(1.000001)|pairs[col].isna()).all()
    assert pairs.exact_equal_share.between(0,1).all() and pairs.both_zero_share.between(0,1).all()
    assert info['sha256']=={f:digest(SOURCE/f) for f in FILES}
    result={**check_gallery(OUT,15),'datasets':6,'variable_pairs':len(pairs),'common_sample_accounting':True,'source_unchanged':True}
    assert '\u00b7' not in REPORT.read_text(encoding='utf-8')
    (OUT/'verification.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result))


def main():
    sys.stdout.reconfigure(encoding='utf-8');assert sys.version_info[:2]==(3,13)
    OUT.mkdir(parents=True,exist_ok=True)
    if '--verify-only' in sys.argv:verify();return
    before={p:digest(SOURCE/p) for p in FILES}
    sites=pd.read_csv(SOURCE/'sites.csv',encoding='utf-8-sig')
    gen=pd.read_parquet(SOURCE/'train/generation.parquet',columns=['site_id','ts'])
    gen=gen.loc[(gen.ts-pd.Timedelta(hours=1)).dt.year==2024]
    counts=gen.groupby('site_id').size();ids=counts.loc[counts==8784].index;assert len(ids)==148
    cohort=sites.loc[sites.site_id.isin(ids)];del gen;gc.collect()
    for p in ['wa','wb']:
        stations=set(cohort[p+'_station'])
        for kind in ['observed','forecast']:
            d,cols=load(p,kind,stations)
            if kind=='observed':process(d,cols,p.upper()+' 실측',p+'_observed',p)
            else:
                for r in [10,17]:process(d.loc[d['round']==r].copy(),cols,p.upper()+f' 예보 {r}시',p+'_'+str(r),p)
            del d;gc.collect()
    assert before=={p:digest(SOURCE/p) for p in FILES}
    INFO.update(sha256=before,finished_at=datetime.now(ZoneInfo('Asia/Seoul')).isoformat(timespec='minutes'))
    for name,rows in [('sample_audit',AUDIT),('pair_relationships',PAIRS),('concentration',SENS)]:
        pd.DataFrame(rows).to_csv(OUT/(name+'.csv'),index=False,encoding='utf-8-sig')
    (OUT/'summary.json').write_text(json.dumps(INFO,ensure_ascii=False,indent=2),encoding='utf-8')
    write_gallery(OUT,GALLERY,'기상 변수 사이의 관계 EDA');report();log(str(REPORT))


if __name__=='__main__':main()
