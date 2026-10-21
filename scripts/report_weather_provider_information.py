"""Render saved DA-005 results without refitting statistical structures."""
import argparse
import json
import sys

import numpy as np

from weather_provider_information import OUT,REPORT,dump

NAMES = ['기온','습도','운량','풍속','강우 여부','log 강우량','UV']
COLORS = ['#286c86','#bb6b3c']


def table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']
                     + ['| '+' | '.join(map(str,row))+' |' for row in rows])


def figures(s):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from eda_plot_utils import save_plot,write_gallery,check_gallery
    plt.rcParams.update({'font.family':'Malgun Gothic','axes.unicode_minus':False,'font.size':9,
                         'axes.spines.top':False,'axes.spines.right':False})
    gallery = []
    rank = np.arange(1,7)
    fig,axes = plt.subplots(1,2,figsize=(12,5.5))
    for ax,rd in zip(axes,[10,17]):
        a = s['cases'][f'{rd}_2024']['views']
        b = s['cases'][f'{rd}_2023']['views']['within']
        ax.plot(rank,a['level']['roots'],'o-',color='#94a8b1',label='2024 전체 낮 수준')
        ax.plot(rank,a['within']['roots'],'o-',color=COLORS[0],label='2024 달력 편차')
        ax.plot(rank,b['roots'],'s--',color=COLORS[1],label='2023 편차 재적합')
        ax.plot(rank,a['within']['frozen_2023_correlations'],'x:',color='#618455',label='2024 가중치 고정, 2023 적용')
        ax.set(title=f'{rd}시 예보',xlabel='정준축 순서',ylabel='상관',ylim=(-.02,1.04),xticks=rank)
        ax.legend(fontsize=8)
    save_plot(fig,OUT,gallery,'01_correlations','전체 수준과 달력 편차, 연도 재현','전체 수준도 09~16시 범위다. 고정 가중치 적용과 2023년 자체 재적합은 다르다.')

    fig,axes = plt.subplots(1,2,figsize=(12,6.5))
    for ax,provider,n in zip(axes,['a','b'],[6,7]):
        for offset,rd,c in [(-.16,10,COLORS[0]),(.16,17,COLORS[1])]:
            fit = s['cases'][f'{rd}_2024']['views']['within']
            values = 100*np.asarray(fit['remaining_'+provider])
            ci = 100*np.asarray(fit['interval_remaining_'+provider])
            ax.barh(np.arange(n)+offset,values,height=.3,color=c,alpha=.8,label=f'{rd}시')
            ax.errorbar(values,np.arange(n)+offset,xerr=np.abs(ci-values[:,None]).T,fmt='none',ecolor='#333333',linewidth=.8,capsize=2)
        ax.set(yticks=range(n),yticklabels=NAMES[:n],xlim=(0,102),xlabel='반대쪽 전체 블록으로 선형 설명되지 않는 분산 (%)',title=provider.upper()+' 변수군')
        ax.invert_yaxis()
        ax.legend()
    save_plot(fig,OUT,gallery,'02_remaining','2024년 달력 편차의 변수별 잔여 분산','오차 막대는 7일 공동 블록 200회의 기술적 백분위 구간이다. 잔여 변동은 고유 유용 신호와 같지 않다.')

    fig,axes = plt.subplots(1,2,figsize=(10,8))
    for ax,rd in zip(axes,[10,17]):
        fit = s['cases'][f'{rd}_2024']['views']['within']
        loading = np.vstack([fit['a_loadings'],fit['b_loadings']])[:,:2]
        im = ax.imshow(loading,cmap='RdBu_r',vmin=-1,vmax=1,aspect='auto')
        for i in range(13):
            for j in range(2):
                ax.text(j,i,f'{loading[i,j]:+.2f}',ha='center',va='center',fontsize=8,color='white' if abs(loading[i,j])>.65 else '#222222')
        ax.set(xticks=[0,1],xticklabels=['C1','C2'],yticks=range(13),
               yticklabels=['A '+v for v in NAMES[:6]]+['B '+v for v in NAMES],title=f'{rd}시 예보')
    save_plot(fig,OUT,gallery,'03_loadings','강한 첫 두 공유축의 변수 구성 (상관, 색 범위 -1~1)','첫 두 축은 주로 기온과 습도의 혼합이며 물리 원인 요인명을 확정하지 않는다.')

    fig,ax = plt.subplots(figsize=(11,5.5))
    labels,first,all_shared = [],[],[]
    for rd in [10,17]:
        fit = s['cases'][f'{rd}_2024']['views']['within']
        for provider in ['a','b']:
            labels.append(f'{rd}시 {provider.upper()}')
            first.append(100*fit['redundancy_'+provider+'_first2'])
            all_shared.append(100*(1-np.mean(fit['remaining_'+provider])))
    ax.bar(np.arange(4)-.18,first,width=.36,color=COLORS[0],label='반대쪽 첫 두 정준점수로 설명')
    ax.bar(np.arange(4)+.18,all_shared,width=.36,color=COLORS[1],label='반대쪽 변수군 전체로 설명')
    for i,(x,y) in enumerate(zip(first,all_shared)):
        ax.text(i-.18,x+.5,f'{x:.1f}',ha='center',fontsize=8)
        ax.text(i+.18,y+.5,f'{y:.1f}',ha='center',fontsize=8)
    ax.set(xticks=range(4),xticklabels=labels,ylabel='표준화 열의 동일 가중 평균 설명 분산 (%)',ylim=(0,65))
    ax.legend()
    save_plot(fig,OUT,gallery,'04_redundancy','정준상관과 원래 변수군의 공유 분산은 다르다','정준상관 약 0.9가 원래 변수 전체 분산 90% 공유를 뜻하지 않는다. 이 평균은 열 선택과 변환에 의존한다.')

    fig,axes = plt.subplots(1,2,figsize=(12,5.5))
    for ax,rd in zip(axes,[10,17]):
        fit = s['cases'][f'{rd}_2024']['views']['within']
        r = np.load(OUT/f'results_{rd}_2024.npz')
        values = r['null_roots'][:,0]
        hist = ax.hist(values,bins=20,color='#9cafb9',alpha=.85)
        height = max(hist[0])
        ax.axvline(fit['null_first_root_q95'],color='#444444',linestyle='--',label='공동 참조 첫 상관 95백분위')
        for j,value in enumerate(fit['roots']):
            ax.axvline(value,color=COLORS[0],alpha=.55,linewidth=1)
            ax.text(value,height*(1.02+.09*(j%2)),f'C{j+1}',ha='center',fontsize=8)
        ax.set(title=f'{rd}시 예보, 참조 199회',xlabel='첫 정준상관의 참조 분포와 관측 상관',ylabel='참조 표본 수',xlim=(0,1.02),ylim=(0,height*1.28))
        ax.legend(fontsize=7)
    save_plot(fig,OUT,gallery,'05_reference','동시 날짜 대응을 끊은 경험적 참조','각 달에서 B 전체를 함께 날짜 순환이동했다. 독립 행의 정식 p값이나 공유 차원 확정치가 아니다.')

    fig,axes = plt.subplots(1,2,figsize=(12,5.5))
    for ax,rd in zip(axes,[10,17]):
        fit = s['cases'][f'{rd}_2024']['views']['within']
        roots = np.asarray(fit['roots'])
        ci = np.asarray(fit['interval_roots'])
        ax.errorbar(rank,roots,yerr=np.abs(ci-roots[:,None]).T,fmt='o',color=COLORS[0],capsize=4,label='2024 편차와 블록 구간')
        ax.plot(rank,fit['frozen_2023_correlations'],'s--',color=COLORS[1],label='고정 가중치 2023 상관')
        ax.axhline(.5,color='#777777',linestyle='--',linewidth=.8,label='강한 공유 후보의 0.5 기준')
        strong = np.asarray(fit['strong_shared_axes'])
        ax.scatter(rank[strong],roots[strong]+.04,marker='*',s=100,color='#77743b',label='세 기준 모두 통과')
        ax.set(title=f'{rd}시 예보',xlabel='정준축 순서',ylabel='상관',ylim=(0,1.06),xticks=rank)
        ax.legend(fontsize=7)
    save_plot(fig,OUT,gallery,'06_stability','공유 관계의 강도와 기간/블록 재현','세 번째 축은 참조와 2023 적용을 통과해도 2024년 재표집 하한이 0.5보다 낮아 강한 후보에서 제외됐다.')

    fig,axes = plt.subplots(1,2,figsize=(13,5.5))
    for ax,rd in zip(axes,[10,17]):
        fit = s['cases'][f'{rd}_2024']['views']['within']
        remain = 100*np.r_[fit['remaining_a'],fit['remaining_b']]
        matched = 100*np.r_[fit['matched']['remaining_a'],fit['matched']['remaining_b']]
        ax.plot(range(13),remain,'o-',color=COLORS[0],label='52개 고유 쌍')
        ax.plot(range(13),matched,'s--',color=COLORS[1],label='38개 중복 없는 지점 매칭')
        ax.set(xticks=range(13),xticklabels=['A '+v for v in NAMES[:6]]+['B '+v for v in NAMES],
               ylabel='선형 잔여 분산 (%)',title=f'{rd}시 예보',ylim=(0,102))
        ax.tick_params(axis='x',rotation=65)
        ax.legend(fontsize=8)
    save_plot(fig,OUT,gallery,'07_matching','지점을 한 번씩만 쓰는 부분집합의 민감도','중복 제거와 지점 구성 변화가 함께 일어난다. 관계가 비슷해도 새로운 지점 모집단으로 일반화한 검증은 아니다.')

    fig,axes = plt.subplots(1,2,figsize=(12,5.5))
    for ax,provider,n in zip(axes,['a','b'],[6,7]):
        for rd,c in zip([10,17],COLORS):
            for year,style,marker in [(2024,'-','o'),(2023,'--','x')]:
                fit = s['cases'][f'{rd}_{year}']['views']['within']
                ax.plot(range(n),100*np.asarray(fit['remaining_'+provider]),linestyle=style,marker=marker,color=c,label=f'{rd}시 {year}')
        ax.set(xticks=range(n),xticklabels=NAMES[:n],ylabel='반대쪽 전체 블록 대비 선형 잔여 분산 (%)',title=provider.upper()+' 변수군',ylim=(0,102))
        ax.tick_params(axis='x',rotation=20)
        ax.legend(fontsize=7,ncol=2)
    save_plot(fig,OUT,gallery,'08_year_remaining','두 해의 변수별 잔여 변동','연도별 선형 투영은 각각 재적합했다. 잔여 비율의 반복은 유용한 신호나 예보 정확도의 검증이 아니다.')
    write_gallery(OUT,gallery,'DA-005 A/B 공통 구조와 잔여 변동')
    dump(OUT/'figure_verification.json',check_gallery(OUT,8))


def report(s):
    prefix = REPORT.read_text(encoding='utf-8').split('## 결과\n\n')[0]
    sample,roots,residuals,red,match,load = [],[],[],[],[],[]
    for rd in [10,17]:
        for year in [2024,2023]:
            audit = s['cases'][f'{rd}_{year}']['audit']
            sample.append([rd,year,audit['dates'],f"{audit['rows']:,}",audit['excluded_dates']])
        fit = s['cases'][f'{rd}_2024']['views']['within']
        raw = s['cases'][f'{rd}_2024']['views']['level']
        rep = s['cases'][f'{rd}_2023']['views']['within']
        for j in range(6):
            ci = fit['interval_roots'][j]
            roots.append([rd,j+1,f"{raw['roots'][j]:.3f}",f"{fit['roots'][j]:.3f}",f'[{ci[0]:.3f}, {ci[1]:.3f}]',
                          f"{rep['roots'][j]:.3f}",f"{fit['frozen_2023_correlations'][j]:.3f}",'통과' if fit['strong_shared_axes'][j] else '미통과'])
        for provider,n in [('a',6),('b',7)]:
            for j in range(n):
                ci = np.asarray(fit['interval_remaining_'+provider][j])*100
                residuals.append([rd,provider.upper(),NAMES[j],f"{100*fit['remaining_'+provider][j]:.1f}",
                                  f'[{ci[0]:.1f}, {ci[1]:.1f}]',f"{100*rep['remaining_'+provider][j]:.1f}"])
            red.append([rd,provider.upper(),f"{100*fit['redundancy_'+provider+'_first2']:.1f}",f"{100*(1-np.mean(fit['remaining_'+provider])):.1f}"])
        root_diff = np.max(np.abs(np.asarray(fit['roots'])-fit['matched']['roots']))
        rem_diff = 100*max(np.max(np.abs(np.asarray(fit['remaining_a'])-fit['matched']['remaining_a'])),np.max(np.abs(np.asarray(fit['remaining_b'])-fit['matched']['remaining_b'])))
        match.append([rd,f'{root_diff:.4f}',f'{rem_diff:.2f}',f"{fit['null_first_root_q95']:.3f}",fit['bootstrap_success']])
        for provider,n in [('a',6),('b',7)]:
            for j in [0,1,2]+([6] if n==7 else []):
                load.append([rd,provider.upper(),NAMES[j],f"{fit[provider+'_loadings'][j][0]:+.3f}",f"{fit[provider+'_loadings'][j][1]:+.3f}"])
    text = '''## 결과

**A/B는 기온과 습도 중심의 강한 선형 공유 구조를 갖지만, 이를 전체 예보의 중복으로 해석할 수는 없다.** 달력 편차에서도 첫 두 정준상관은 약 0.90~0.98이고 2023년 적용에서도 유지됐다. 반면 운량, 풍속, 강수와 B UV에는 반대쪽 전체 변수군으로 선형 설명되지 않는 분산이 많이 남았다. 그 변동의 실제 유용성은 미검증이다.

### 실제 사용 표본과 실행

공통 표본 매핑의 고유 52쌍에서 A 40개/B 47개 지점을 사용했다. 최대 이분 매칭은 38쌍으로, 각 A/B 지점을 한 번만 쓴 민감도 표본이다. 모든 월이 14일 기준을 통과했다.

'''
    text += table(['라운드','연도','잔존 날짜','쌍/시간 행','제외 날짜'],sample)
    text += '''

선택 범위의 비정시/중복 예보 키는 없었다. 17시 2024년 A 습도 범위 밖 1셀과 키 가용성 때문에 일부 균형 날짜가 제외됐고 대치는 하지 않았다. 사전 가용성 확인 뒤 같은 계약으로 본 계산을 2026-10-21 13:53 KST에 정규 CPython 3.13.1에서 완료했다. 전체/편차 8개 적합, 매칭 2개 적합과 참조 398회, 블록 재표집 400회를 수행했다. 발전량 예측 학습/추론/평가/제출은 0회다.

### 첫 두 공유축은 달력 평균 제거 후에도 강하다

2024년 정준축 가중치와 척도를 고정해 2023년 편차에 적용한 상관을 함께 읽는다. 2023년 자체 재적합과 같은 검증이 아니다. 구간은 2024년 날짜 공동 블록 재표집의 기술적 구간이며 근접한 축의 개별 방향 안정성을 보장하지 않는다.

'''
    text += table(['라운드','축','2024 수준','2024 편차','편차 블록 구간','2023 재적합','고정 가중치 2023','강한 후보 기준'],roots)
    text += '''

두 라운드 모두 첫 두 축만 사전 강한 후보 기준을 모두 통과했다. 세 번째 축의 2024년 편차 상관은 0.504/0.523이고 2023년 고정 가중치 적용 상관은 0.585/0.566이지만, 블록 하한이 0.428/0.453으로 0.5보다 낮았다. 이 축을 관계 없음으로 처리하지 않으며 강한 재현 후보와 구분한다. 다섯 번째/여섯 번째 축은 약하고 첫 축의 높은 상관만으로 모든 변수를 압축할 수 없다.

첫 두 축의 변수-점수 상관은 다음과 같다. 축 이름을 기온/습도 하나로 고정하기보다 두 변수의 혼합 방향으로 읽는다. 부호는 분석 표시 규칙이다.

'''
    text += table(['라운드','변수군','변수','C1 상관','C2 상관'],load)
    text += '''

### 높은 정준상관은 변수군 전체 분산의 공유율이 아니다

다음은 각 표준화 열을 동일 가중한 평균 설명 분산이다. 기온의 섭씨 분산과 강우량의 mm 분산을 그대로 합친 물리량이 아니며, 강우를 두 열로 표현한 이번 목록에 의존한다.

'''
    text += table(['라운드','대상 변수군','반대쪽 첫 두 점수로 설명 (%)','반대쪽 전체 블록으로 설명 (%)'],red)
    text += '''

첫 두 정준상관이 0.9를 넘더라도 그 두 점수가 설명하는 평균 분산은 A 약 41%, B 약 32%였다. 반대쪽 변수군 전체를 사용해도 A 약 47~48%, B 약 38~39%다. 따라서 “A/B가 90% 이상 중복” 또는 “두 축이면 전체 예보 대부분을 대체”라는 결론은 성립하지 않는다.

### 변수별로 남는 변동의 규모가 다르다

표의 수치는 반대쪽 전체 블록으로 설명되지 않는 분산 비율이다. 값이 클수록 선형 중복이 적다는 뜻이며, 예보 오차가 크다거나 발전량 예측에 더 유용하다는 뜻은 아니다.

'''
    text += table(['라운드','대상','변수','2024 잔여 분산 (%)','블록 구간 (%)','2023 잔여 분산 (%)'],residuals)
    text += '''

기온에는 약 10~11%, 습도에는 약 17~18%의 잔여 분산이 남았다. 운량은 A 약 60~61%/B 약 68~71%, 풍속은 양쪽 약 67~69%가 남아 기온과 다른 수준의 중복을 보였다. B의 강우 여부와 log 강우량에는 약 91~93%, UV에는 약 80~82%가 남았다. A 강우 두 열도 약 77~82%의 잔여 분산을 보였다. 2023년 수치와 구간은 표에서 구분한다.

B UV는 A에 직접 대응 열이 없고 기온/습도/운량/풍속/강우의 선형 조합으로 대부분 설명되지 않았다. 이는 B에 남는 변동을 보여 주지만, 비선형으로 공유되는 부분, 위치/정의 차이와 잡음은 이번 투영에서 분리하지 않았다. 앞선 DA-001에서 B 내부에 공통축이 있었다는 사실과도 모순되지 않는다. B 내부의 변수 관계가 A에서 모두 재구성된다는 뜻은 아니기 때문이다.

### 날짜 공동 참조와 지점 반복 민감도

'''
    text += table(['라운드','매칭 시 정준상관 최대 차이','잔여 분산 최대 차이 (%p)','참조 첫 상관 95백분위','성공 블록 수 /200'],match)
    text += '''

경험적 참조의 첫 정준상관 95백분위보다 첫 네 관측 상관이 두 라운드에서 모두 컸다. 이는 같은 달 안의 날짜 대응을 끊은 참조보다 관계가 강하다는 비교다. 강한 공유 후보 기준을 통과한 축은 둘이며 참조 통과 개수 넷과 구분한다. 독립 행을 가정한 유의확률이나 공유 차원 확정치로 해석하지 않는다.

각 지점 한 번만 남긴 38쌍에서의 정준상관/잔여 비율은 52쌍 결과와 비슷했다. 선택한 부분집합에서는 공유 지점 반복만으로 주 결론이 만들어진 모습은 약했지만, 이 비교는 중복 제거와 지점 구성 변화가 함께 일어나는 민감도다. 날짜 공동 재표집도 지점망 자체를 새로 추출하지 않아 새로운 지역으로 일반화하는 구간은 아니다.

### 검증

합성 테스트 8개가 통과했다. 척도 불변성, 순위 부족, 동일 변수군과 독립 잔여 열, 가중 달력 제거, 고정 가중치 점수 상관과 FFT의 모든 합성 이동을 확인했다.

원본 키 재정렬로 변환 특징 7,890,272개를 별도 재구성해 저장 패널과 일치시켰다. QR 공간의 교차 SVD로 8개 정준상관 해를 검산하고 직접 최소제곱으로 양방향 잔여 분산을 확인했다. 적재 상관과 첫 두 점수의 설명 분산, 연도 고정 적용, 별도 증가 경로 알고리즘의 최대 매칭 크기와 두 매칭 적합을 대조했다. 각 라운드의 최초 참조 이동과 최초 블록 표본은 직접 행을 재구성해 확인했고, 전체 저장 재표집의 구간과 강한 후보 판정을 검산했다. 모든 재표집을 독립 재실행했다는 뜻은 아니다.

원본 네 파일과 계산 코드의 해시를 확인했다. 로컬 그림 8개와 갤러리 링크 16개를 검사하고, 절반 크기 검토 시트 두 장에서 상관/구간과 적재 상관, 잔여 분산, 공동 참조 및 연도/매칭 민감도를 시각 확인했다. 적재 상관 그림의 색 범위 표기를 보완하고 다시 확인했다. 모든 패널, 가중치와 재표집, 그림은 `data/analysis/2026-10-21_provider_information/`에 로컬로만 보존한다.

## 한계 및 다음 단계

선형 CCA와 선형 투영은 비선형 공통성을 고유 변동처럼 남길 수 있다. 두 강우 열의 0 집중과 변환, 서로 다른 위치/정의, B에만 포함한 UV, 동일 가중 표준화 열 선택도 결과에 영향을 준다. 지점 쌍은 실제 같은 장소의 독립 관측 쌍이 아니며 익명 좌표로 차이의 원인을 확정하지 않는다. 연도별 달력 평균 제거와 과거 재현은 미래 정답을 남긴 예측 평가가 아니다.

분석상 활용 가설은 A/B를 통째로 중복 취급하지 않고 강한 공통 부분과 남는 변동을 구분해 보는 것이다. 어느 쪽 변수를 추가/제거하거나 발전량 예측에 더 좋은지는 이번 결과로 확정하지 않는다.

정한 범위의 공유 구조, 잔여 변동과 연도/블록/매칭 민감도를 확인해 DA-005를 종료한다. 초기 DA-001~005 다섯 질문은 각 사전 범위에서 완료했다. DA-006/007/009는 선택 후보이며 다음 심화 검토는 변화일 해석에 직접 영향을 주는 DA-009를 우선 후보로 둔다. 기존 보류를 필수 작업으로 바꾸거나 예측 모델링을 자동 재개하지 않는다.

## 관련 코드

- [계산](../scripts/weather_provider_information.py)
- [독립 검산](../scripts/verify_weather_provider_information.py)
- [그림과 보고서](../scripts/report_weather_provider_information.py)
- [합성 검증](../tests/test_weather_provider_information.py)
'''
    REPORT.write_text(prefix+text,encoding='utf-8')


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=['figures','report','all'])
    args = parser.parse_args()
    s = json.loads((OUT/'summary.json').read_text(encoding='utf-8'))
    if args.mode in ['figures','all']:
        figures(s)
    if args.mode in ['report','all']:
        report(s)
    print('DA-005 saved-result rendering complete.')
