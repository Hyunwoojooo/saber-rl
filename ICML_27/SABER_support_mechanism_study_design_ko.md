# SABER support의 의미와 재가중 효과 검증 설계

대상: ICML 재투고 준비. 이 문서는 이론 분석 두 개와 대조 실험 두 개의 설계안이다. 기존 원고나 완료 결과는 변경하지 않는다.

## 1. 리뷰가 제기한 문제

사람 리뷰어 두 명의 공통 지적은 **핵심 아이디어보다 정의와 구현 세부를 먼저 이해해야 한다**는 것이었다. 4점 리뷰어는 손실을 재가중한다는 사실보다, **왜 이 binary support 규칙으로 배정해야 하는지**를 문제 삼았다. 6점 리뷰어는 개입의 기술적 타당성을 인정하면서도 수작업 context, 이산화·최소 개수, Tail 후보 규칙에 대한 설명과 검증이 부족하다고 봤다. AI 리뷰에서 실험으로 가장 직접 대응할 질문은 **과거 행동에서 학습한 내용이 현재 정책 행동의 Q 예측까지 전달되는가**였다. 상세 근거는 [AAAI 리뷰 인계 보고서](postAAAI_Review.md#2-어떤-리뷰를-받았는가)에 있다.

가독성 문제와 근거 부족은 별개다. Method를 다시 쓰면 정의는 잘 전달할 수 있지만, 잘 전달된 규칙이 더 좋은 규칙이라는 뜻은 아니다. 반대로 결과에 표를 더해도 두 행동의 역할을 구별하지 않으면 왜 재가중하는지 이해하기 어렵다.

| 검증 | 질문 | 이 검증이 필요한 이유 |
|---|---|---|
| 이론 1. 주변 경험과 가치의 식별 가능성 | 정책 행동 주변에 기록이 없으면 무엇을 알 수 없는가? | 기존 replay loss가 작다는 것과 현재 행동의 Q가 잘 정해진다는 것은 다르다. |
| 실험 1. 주변 데이터 제거·복구 | 기록 수를 맞춰도 정책 행동 근처의 데이터가 실제로 도움이 되는가? SABER 판정은 그 기록을 얼마나 잘 포착하는가? | 주변 경험과 성능 사이의 상관관계 대신 데이터 접근성을 조작한다. 이산 support를 참값으로 사용하지 않는다. |
| 이론 2. 재가중과 정책 행동 오차 | 같은 가중치 개수로 배정할 때, 어떤 조건에서 support가 Random·TD-error보다 유리한가? | 판정하는 행동과 학습하는 행동이 다르다. 그 사이의 전달을 식에 드러내야 한다. |
| 실험 2. 같은 체크포인트에서 critic 분기 | 실제 optimizer에서 배정만 바꿔도 별도 정책 행동의 Q 예측이 나아지는가? | 기존 online 성능 비교는 배정 외에 이후의 정책·데이터까지 달라진다. 기존 고정 target 기준의 진단도 실제 return과의 오차는 측정하지 않는다. |

이 네 작업은 **coverage의 중요성, support 지표의 적합성, 배정의 효과**를 분리한다. PAL과의 성능 비교, 환경 간 전이, wall-clock, 하이퍼파라미터 최적성은 별도 질문이다. 일반적인 loss reweighting이 새롭다고 주장하지 않는다. PAL 자체가 균일 replay를 유지하면서 TD-error에 대응하는 손실을 쓰는 비교 사례다. [Fujimoto et al., NeurIPS 2020, §4](https://proceedings.neurips.cc/paper_files/paper/2020/file/a3bf6e4db673b6449c2f7d13ee6ec9c0-Paper.pdf)

기존 context-eligible count-matched 대조는 세 과제 모두 AUC 차이의 95% CI가 0을 포함했다. 이후 DrawerOpen–DDPG의 Random·TD Top-k 비교에서도 평균 방향만으로 추가 가치를 확정할 수 없었다. 이 대조들은 **최종 task 성능**을 측정했고, 고정된 critic에서 정책 행동의 오차가 어느 방향으로 바뀌는지는 보여주지 않는다. 실험 2의 평가 변수를 바꾸는 이유다. [기존 대조 결과](postAAAI_Review.md#61-support-vs-random-vs-td-error-top-k-배정-비교)

## 2. Method에서 먼저 설명할 경험의 의미

예를 들어 두 가지 행동을 구별한다. **이전에 실행한 행동**은 저장된 transition의 일부이며, critic은 이 행동의 Bellman 오차를 줄인다. **현재 정책이 고르는 행동**은 actor가 값을 조회하는 위치다. 두 행동은 같을 필요가 없다. DDPG의 actor는 이 위치의 Q값 자체만이 아니라 행동 방향의 Q 기울기를 사용한다. [Silver et al., ICML 2014, §4](https://proceedings.mlr.press/v32/silver14.pdf)

SABER는 다음 두 질문에 모두 예일 때 해당 minibatch 위치를 support-positive로 표시한다.

1. 제한된 조회 후보 중 관측·HER로 설정된 목표·남은 시간에서 만든 동일한 세부 context에 속하는 유효 replay 기록이 최소 4개 있는가?
2. 그 기록 중 하나 이상의 **실제로 실행된 행동**이 현재 정책의 조회 행동과 같은 다차원 action bin에 속하는가? 기본값은 행동 차원당 5개 구간이다.

실행 행동과 전이는 경험에서 왔지만, 목표는 hindsight로 바뀔 수 있다. 동일 context도 동일한 물리 상태는 아니다. 그러므로 이 값은 **유사하게 분류된 상황에서 행동 근거가 조회된 여부**이지, 동일 목표에서 동일 행동을 실행한 증거, 성공 여부, 발생 빈도, 분포 support 전체, 불확실성 추정치가 아니다. 최소 4개는 context 기록 수이며, 정책 bin 안의 기록 수는 1개여도 된다. Negative는 현재 조회·구간 규칙에서 찾지 못했다는 뜻이다. 조회하는 행동과 가중치를 받는 샘플의 실행 행동은 다를 수 있다.

원래 뽑힌 모든 샘플을 사용한다. Positive에 2, 나머지에 1을 준 뒤 배치 평균으로 나누고, **replay 행동에서 계산한 critic loss**에만 곱한다. 평균 가중치가 1이라는 것은 평균 loss나 gradient norm, Adam step 크기가 같다는 뜻은 아니다. Method 그림은 `정책 행동 → support 조회 → 가중치`와 `replay 행동 → Bellman 오차`가 **손실 식에서 만나는 구조**를 보여주고, Tail-512·slot 검증·context 표는 그다음에 설명한다.

## 3. 이론 1: 주변 관측이 정책 행동의 Q를 제한하는 이유

### 3.1 질문과 전제

고정된 정책에서 관측과 목표를 묶은 입력을 $x$, 현재 정책 행동을 $a_\pi$, critic의 실제 입력을 $z_\pi=(x,a_\pi)$라 둔다. 분석에서 남은 시간을 조건으로 고정할 수 있지만, 실제 critic에 새로운 입력을 추가하지 않는다. 우선 **동일한 물리 상태·목표에서 행동만 바뀌는 경우**를 다룬다. 다른 상태까지 비교하려면 별도로 가치에 적절한 거리와 그 거리에서의 연속성을 가정해야 한다. 같은 이산 context만으로 그 가정을 만족한다고 보지 않는다.

BCQ와 BEAR에서의 extrapolation 문제는 왜 policy-action coverage를 볼 만한지 설명한다. 다만 두 논문의 offline RL 행동 제약이나 Bellman backup에 대한 결론을 온라인 HER의 loss 재가중 보장으로 가져오지 않는다. 여기서는 **고정된 입력과 데이터로 무엇을 구별할 수 있는지**만 분석한다. [BCQ, ICML 2019, §4](https://proceedings.mlr.press/v97/fujimoto19a/fujimoto19a.pdf), [BEAR, NeurIPS 2019](https://papers.neurips.cc/paper_files/paper/2019/hash/c2073ffa77b5357a498057413bb09d3a-Abstract.html)

### 3.2 데이터 공백의 하한

관측된 행동 중 가장 가까운 행동까지의 거리를 $r>0$이라 하자. 데이터가 공통값 $q_0$와 일치하는 사례에서, 다음 두 함수는 관측점에서 완전히 같고 둘 다 $L$-Lipschitz다.

$$
f_\pm(a)=q_0\pm L\,[r-d(a,a_\pi)]_+.
$$

모든 관측점에서는 괄호가 0이다. 반면 정책 행동에서는 두 값이 $2Lr$만큼 다르다. 따라서 이 데이터로 같은 예측을 내는 어떤 추정기도 **두 경우 중 하나에서는 최소 $Lr$만큼 오차**가 난다. 이는 최악의 경우를 보이는 구성이지, 모든 정책 행동에서 critic의 실제 오차가 그만큼 크다는 뜻은 아니다. 원하면 같은 값을 즉시 보상으로 하는 1-step 연속 행동 문제로 두 경우를 구현할 수 있다. 고정된 로봇 환경에 대한 반례라고 해석하지 않는다.

미분 가능한 함수군에서는 데이터가 없는 열린 근방 안에만 비영인 매끄러운 함수를 더해, 관측점 예측을 바꾸지 않으면서 정책 행동의 값 또는 행동 미분을 바꿀 수 있다. 반대로 정책 행동에서 값 하나를 알아도 기울기가 자동으로 정해지지는 않는다. 기울기 분석에는 주변 데이터의 방향별 배치와 추가적인 매끄러움이 필요하다.

### 3.3 가까운 관측이 줄 수 있는 상한

$r$ 이내의 데이터 지점 $z_j$에서 실제 $Q^\pi$ 대비 critic 오차가 $\varepsilon$ 이하이고, 그 근방에서 critic과 기준 가치가 각각 $L_\theta,L_\pi$-Lipschitz라면 삼각부등식으로 다음이 성립한다.

$$
|Q_\theta(z_\pi)-Q^\pi(z_\pi)|
\leq \varepsilon+(L_\theta+L_\pi)r.
$$

오차를 신뢰할 수 있는 가까운 점이 있으면 외삽에 대한 최악의 여지가 줄어든다. **작은 training TD residual을 $\varepsilon$으로 대신 넣을 수는 없다.** Bootstrap target 자체가 틀릴 수 있기 때문이다. 희소 보상과 접촉 경계에서 가치가 급변할 수도 있다. 또한 인접한 두 행동이 bin 경계의 양편에 있으면 같은 bin이 아니며, 같은 bin이라도 가치가 현저히 다를 수 있다.

**원고에서 방어할 주장:** 주변 관측의 유무는 정책 행동 value 추정을 얼마나 데이터가 제약하는지와 관련된다. 이것은 SABER가 uncertainty를 측정한다거나, 항상 positive에 더 큰 비중을 줘야 한다는 정리가 아니다. 실험 1은 실제 환경에서 이 동기가 맞는지, 그리고 우리 indicator가 그 동기에 맞는 경험을 포착하는지를 별도로 검사한다.

## 4. 실험 1: 정책 행동 주변 데이터 제거와 복구

### 4.1 가설과 대조군

고정한 물리 상태·목표·정책에서 **같은 양의 먼 데이터 대신 주변 행동 기록을 주면**, 보지 않은 정책 행동의 Q 예측 오차 또는 모델 간 예측 불일치가 줄어드는지 검사한다. 같은 상태에서 진행하므로 방문하기 쉬운 상태에만 데이터가 많다는 선택 효과를 피한다. 실제 replay의 분포를 그대로 복제하는 실험이 아니라, action coverage만 통제해 보는 국소 회귀 실험이다.

| 조건 | 공통 데이터 | 바꾸는 데이터 | 총수 |
|---|---:|---|---:|
| Keep | 먼 행동 24개 | 주변 행동 8개 | 32 |
| Remove | 동일한 먼 행동 24개 | 추가 먼 행동 8개 | 32 |
| Restore | 동일한 먼 행동 24개 | Keep과 겹치지 않는 새 주변 행동 8개 | 32 |

Remove를 단순히 데이터 8개가 적은 조건으로 만들지 않는다. Restore는 새 모델을 같은 초기값에서 다시 적합한다. Remove 모델을 추가로 학습하면 계산 예산과 경로까지 바뀌므로 복구 효과가 분리되지 않는다. 정책 행동 자체는 어느 학습 조건에도 넣지 않는다.

### 4.2 데이터와 기준값

1. DrawerOpen–DDPG의 HER/SABER에서 seeds 6–10의 최종 정책 10개를 고정한다. 이 머신에 checkpoint 파일 10개가 있다. 본 실행 전 config·step·hash를 확인한다. 두 종류는 **정책의 출처**이며, 우열의 비교군이 아니다. 각 정책을 고정된 reset seed로 실행하고 서로 다른 8개 episode에서 예정된 시점의 전체 simulator 상태를 저장한다. Pilot과는 평가 snapshot을 분리하고, 성공·실패·접촉 여부를 보고 상태를 교체하지 않는다. 이 cohort는 최종 정책의 진단이며 초기 학습에 자동 일반화하지 않는다.
2. 각 상태에서 원 정책 행동을 기준으로 행동 box 안에서 중복 없이 뽑는다. 정규화된 행동 좌표에서 최대 절대차가 0.02–0.10이면 주변, 0.50 이상이면 먼 행동이다. 경계에서 clipping으로 중복을 만들지 않으며, 해당 수를 뽑지 못하면 기록한다. 이 설계값은 기존 제거 실험 계획을 계승한 것이고 bin 수 5의 근거가 아니다. 주 대조 집합을 reward나 Q값으로 고르지 않는다.
3. 같은 **완전한** simulator snapshot을 복원해 첫 step에 해당 행동을 실행하고, 이후는 고정 정책으로 진행한다. 같은 목표·보상·할인과 일관된 종료 규칙으로 return을 계산한다. 평가용 $G_H$는 별도 복원에서 첫 행동을 정책 행동으로 실행해 얻는다. 학습에는 다른 행동의 return만 사용한다. 정해진 행동을 실제로 실행한 label이므로 기존 replay 행동의 reward·next state를 새 행동에 붙이지 않는다.
4. 기본 할인은 원 DrawerOpen 설정인 0.98이다. 원 target이 timeout을 bootstrap한다면 외부 시간 제한을 제거한 환경에서 제한 전 transition parity를 감사하고, 실제 terminal에서는 멈춘다. 보상 절댓값이 1 이하라는 검사까지 통과하면 길이 1,024의 생략 tail 상한은 약 $5.18\times10^{-8}$이다. 호환되는 continuing task를 만들 수 없으면 유한 horizon 기준으로 이름과 결론 범위를 명시하고 따로 보고한다. 정해진 seed의 결정적 snapshot은 한 rollout으로 충분하지만, 확률적 환경에서는 반복과 기준값 추정 오차를 별도 처리한다.

### 4.3 Fitting과 평가

원 critic과 같은 256–256 ReLU MLP를 새로 적합한다. 5개 초기화를 만들고 **같은 초기화를 세 조건에 각각 복제**한다. 동일한 입력 스케일, optimizer, full-batch 학습 예산을 사용한다. 기존 critic이나 target을 초기값으로 불러오지 않는다. 그 모델은 제거할 데이터를 이미 학습했을 수 있다. 기존 제거·복구 계획의 시작 설정은 Adam(학습률 0.0003), gradient norm clip 10, 모델당 2,000 updates다. 행동 생성·초기화·평가의 난수 stream을 구분하고, 아래 fitting 기준을 다시 통과한 공통 설정을 동결한다.

한 상태에서 모델 예측 평균이 $\bar Q$이면, 주 정확도 지표는 $(\bar Q-G_H)^2$, 불일치 지표는 5개 모델의 예측 표본분산이다. 두 지표를 각각 계산한다. 모든 모델이 같은 방향으로 틀릴 수 있으므로 분산은 calibrated uncertainty가 아니다. 이 측정 방식의 참조는 [Deep Ensembles, NeurIPS 2017](https://papers.neurips.cc/paper_files/paper/2017/hash/9ef2ed4b7fd2c810847ffa5fa85bce38-Abstract.html)이며, 그 논문의 분포 모델링 전체를 재현하는 것은 아니다.

**기술적 선결 조건:** 2026-11-10 로컬 CPU pilot에서 복원은 통과했지만 36 fits 중 최대 training RMSE가 1.786875로 기존 0.05 기준을 넘었다. 4개 중 2개 상태의 기준 return 범위도 거의 0이었다. 파일 생성은 끝났지만 본 측정 기준을 통과한 것은 아니다. 먼저 별도의 기술용 snapshot에서 입력·label 스케일, 최적화 곡선, 동일 데이터에서 단순한 회귀 모델도 fit하는지 확인한다. **학습 데이터에서의 fitting만** 보고 공통 예산을 수정한다. 조건마다 다른 예산을 주거나, 평가에서 유리한 쪽으로 반경을 고치지 않는다. Fitting 문제를 없애지 못하면 최적화와 coverage가 분리되지 않은 결과로 기록한다. Return이 거의 상수인 상태는 제외하지 않고 분포와 함께 표시한다. Pilot 원시 데이터는 이 설계서에 포함하지 않는다.

**판정:** 사전의 두 제거 대조는 `Remove−Keep`의 오차와 불일치, 두 복구 대조는 `Remove−Restore`의 동일 지표다. Episode → 정책 출처 → 원 학습 seed 순으로 평균하고, 대응된 5개 seed의 원값과 구간을 보고한다. 네 대조를 같은 seed 블록으로 재표본화해 95% 구간과 명목상 가족오류 보정 구간을 제시한다. 5개 seed는 탐색적 근거이며 80개 상태나 1,200개 fits가 독립 학습 반복은 아니다. 오차만 커지면 정확도 근거, 불일치만 커지면 불일치 근거라고 쓴다. 차이가 불확실하면 가설을 입증한 것도, 두 조건이 같음을 입증한 것도 아니다.

**규모:** 80개 상태와 상태당 49개 행동/reference로 최대 3,920개 rollout, 1,024 horizon에서 약 401만 simulator steps다. 정책을 다시 학습하지 않는다. 초기 계획은 1,200 fits × 2,000 updates였으며, fitting gate를 다시 통과한 뒤 동일 예산을 고정한다. rollout label은 비교군과 모델 초기화가 공유한다.

### 4.4 실제 SABER 판정과 경험의 대응 검사

위에서 context를 고정했기 때문에 그 결과만으로 SABER의 이산화가 좋다고 결론 내릴 수 없다. **같은 실험의 보조 분석**으로, 보존된 실제 replay에서 결과를 보지 않고 뽑은 post-HER query에 아래 세 기준을 대조한다. 실제 학습 데이터를 제거하거나 weight를 바꾸지 않는다.

| 기준 | 무엇을 비교하는가 |
|---|---|
| 현재 Tail-512 판정 | 논문과 동일한 context·bin·최소 개수로 얻은 기본 지표 |
| 동일 coarse bucket의 유효 후보 전체 조회 | context·bin·최소 개수는 그대로 두고 512개 제한 때문에 놓친 기록 분리 |
| 독립 연속 거리 검사 | 행동 좌표 거리와 연속적인 상태·목표·남은 시간 차이를 따로 계산. Drawer의 손-손잡이 상대좌표·drawer 위치 등 물리량과 scale은 결과 전에 고정 |

최근접 행동 거리, 근접한 실행 행동의 개수와 출처 episode 수, bin 경계 근처의 놓침, 같은 context 내 정책 bin 분산을 보고한다. **Positive 중 자기 replay 행동도 정책 bin에 속하는 비율**과 자기 transition을 빼면 판정이 사라지는 비율은 다르므로 따로 계산한다. 자체 slot 제거와 동일 episode 제거를 감도 분석에서 분리하고, 최소 개수를 잃은 경우와 유일한 행동 근거를 잃은 경우도 나눈다. 전체 검색은 현재 이산 규칙에 대한 기준이지 연속 거리나 유용성의 정답은 아니다. 연속 거리의 근접값 역시 정의에 조건부이므로 하나의 cutoff에서 정확도를 과장하지 않는다.

구성의 정당화 수준도 구분한다. 최소 개수는 드문 context와 충분히 채워진 context를 구별하려는 조건이며, bin은 행동 근접성을 빠르게 이산화한 것이다. 고정 scan은 관측할 수 있는 근거의 범위와 비용을 바꾼다. 이 의도를 설명하는 것과 4·5·512가 좋은 값임을 보이는 것은 다르다. **고정 replay 보조 민감도**로 각 요소를 하나씩 바꿔 `최소 2/4/8`, `K=3/5/7`, `128/512/전체 후보`에서 거리 대응, 누락, 포화, 판정 비용을 비교한다. 기본값은 변경하지 않는다. 이 분석이 online AUC의 민감도나 2:1 가중 강도의 최적성을 입증하지는 않는다.

**원고에서 방어할 주장:** 주변 행동을 제거했다가 별도 주변 기록으로 되돌리는 대조가 지지되면, 통제된 상태에서 관측 위치가 정책 행동 Q 예측에 영향을 준다는 실증 근거다. 지표 점검까지 맞으면 SABER의 support가 실제 replay의 **경험적 근접성**을 어느 정도 포착한다고 말할 수 있다. 둘 중 하나만 맞으면 해당 결론만 남긴다.

## 5. 이론 2: 언제 support에 따라 critic loss를 재가중할 근거가 있는가

### 5.1 동일한 가중치 예산에서의 정확한 1차 비교

actor, critic 초기값, target, HER goal과 minibatch를 고정한다. $E$는 context 조건을 만족하는 배치 위치, $M\subseteq E$는 support-positive 위치, $k=|M|$, $B$는 배치 크기, $p=k/B$다. 대조군 $C\subseteq E$도 정확히 $k$개를 고른다. 세 배정 모두 2:1을 평균 1로 정규화하므로 분모는 $1+p$로 같다. 배치와 분리된 정책 행동 평가 오차를 $R_\pi$, replay 행동의 샘플 loss를 $\ell_i$로 두고, 다음을 정의한다.

$$
u_i=\langle\nabla_\theta R_\pi,\nabla_\theta\ell_i\rangle.
$$

$u_i$가 양수면 이 샘플의 충분히 작은 SGD step이 **평가 오차를 내리는 방향**이다. $g_W=B^{-1}\sum_i w_i\nabla\ell_i$를 가중 gradient라고 하면, 공통인 기본 gradient가 상쇄된다. Taylor 전개 결과는 다음과 같다.

$$
\begin{aligned}
R_\pi(\theta-\eta g_M)-R_\pi(\theta-\eta g_C)
&=-\eta\,\frac{p}{1+p}(\bar u_M-\bar u_C)+O(\eta^2),\\
\mathbb E_{\mathrm{Rand}}[\bar u_C]&=\bar u_E.
\end{aligned}
$$

Random은 $E$에서 균일 비복원으로 뽑기 때문에, **action-bin 조건이 context eligibility 너머의 정보를 주는지**는 $\bar u_M-\bar u_E$로 나타난다. TD-error Top-k는 $\bar u_M-\bar u_{\mathrm{TD}}$를 본다. Uniform에 대한 계수는 $p(1-p)/(1+p)$이며 비교 평균은 positive 대 나머지다. $k=0$이면 모두 uniform, $k=|E|$이면 세 배정이 같지만 $E$가 배치 전체가 아니면 Uniform과는 다를 수 있다. Random의 **1차항** 기대를 평균 gradient로 계산해도, 실제 유한 step의 risk 기대가 평균 gradient의 risk와 같다는 뜻은 아니다.

이 분석은 validation gradient와 학습 gradient의 정렬을 이용하는 [Ren et al., ICML 2018, §3.2](https://proceedings.mlr.press/v80/ren18a/ren18a.pdf)와 관련된다. 우리는 evaluation return을 사용하는 가중치 학습 알고리즘을 제안하는 것이 아니라 **이미 정해진 SABER mask가 그 방향과 맞는지** 검사한다. 이 식 자체는 일반적인 binary mask에도 성립하여, SABER 고유의 정당화는 아래 구조 분석과 실제 측정에 있다.

### 5.2 두 행동 사이에서 실제로 무엇이 전달되는가

설명을 위해 $\ell_i=\delta_i^2/2$와 $R_\pi=\mathbb E_q[e(q)^2]/2$로 쓰고 $\delta_i=Q_\theta(z_i^r)-y_i$로 정의한다. 실제 trainer에 1/2가 없으면 검산에는 그 gradient 배율을 반영하고, native Adam은 코드 그대로 사용한다. 별도의 정책 query $q$의 기준값 대비 오차를 $e(q)$, $J(z)=\nabla_\theta Q_\theta(z)$라 하면:

$$
u_i=\delta_i\,\mathbb E_q\left[e(q)\,\langle J(q),J(z_i^r)\rangle\right].
$$

큰 TD error, 큰 Jacobian 결합, 올바른 **오차 방향**은 다른 요소다. 결합이 크더라도 replay target이 편향됐거나 query가 이미 정확하면 더 큰 가중치가 도움되지 않을 수 있다. Jacobian 내적은 [Jacot et al., NeurIPS 2018](https://papers.neurips.cc/paper_files/paper/2018/hash/5a4be1fa34e62bb8a6ec6b91d2462f5a-Abstract.html)의 kernel 관점과 유사하지만, 여기서는 저장된 한 모델에서 측정한 일차 결합이고 무한 폭이나 고정 kernel 수렴 가정을 하지 않는다.

**작동하는 모델부터 증명한다.** Context와 action bin으로 만든 각 cell에 독립 Q parameter가 있고, 각 context의 정책 bin이 일정하며, 충분한 context 기록과 누락 없는 조회, 정확한 cell별 target을 가정한다. 그러면 어떤 배치 위치의 **자기 replay 행동**이 policy-query cell에 있으면 그 위치는 SABER-positive다. Query cell $c$의 배치 샘플 수를 $N_c$, 비교 mask가 그중 골라낸 수를 $t_c$라 하면 해당 cell의 오차 감소 계수는 SABER가 $2N_c/[B(1+p)]$, 비교군은 $(N_c+t_c)/[B(1+p)]$다. 충분히 작은 SGD step에서는 각 cell의 초기 오차에 각각 $1-\eta\,(\text{계수})$를 곱하므로 SABER가 적어도 같은 정도로 줄인다. **오차가 남은 평가 cell에서 비교군이 샘플을 놓친 경우에만** 엄격한 이득을 보인다. Positive가 모두 유용하다고 가정한 증명이 아니다. 다른 bin의 positive 샘플은 이 모델에서 그 query에 영향을 주지 않는다.

**실제 critic은 이 모델과 다르다.** 실제 mask에서 각 query cell의 학습 샘플을 SABER와 대조군이 각각 몇 개 선택하는지 센다. 이를 $s_c,t_c$, 그 cell의 평가 비중을 $\nu_c$, 초기 cell 오차를 $e_c$라 두면, 실제 mask를 이상적 cell에서 평가한 1차 이득은 $\sum_c \nu_c e_c^2(s_c-t_c)/[B(1+p)]$다. 이 식은 실제 mask가 제한된 scan이나 context 조건 때문에 유용한 기록을 놓치면 $s_c=N_c$라는 이상적 성질부터 깨진다는 점을 드러낸다.

다음으로 $\Gamma=\langle\nabla R_\pi,g_M-g_C\rangle$라 놓고, 실제 샘플 유용성과 cell 모델 유용성의 차이를 $|u_i-u_i^{\mathrm{cell}}|\leq\epsilon_i$로 제한할 수 있는 조건을 본다. 두 mask가 다른 위치에서만 가중치 차이가 나므로 삼각부등식으로:

$$
\Gamma\ \geq\ \Gamma_{\mathrm{cell}}-\frac{1}{B(1+p)}\sum_{i\in M\triangle C}\epsilon_i.
$$

교차 Jacobian, cell 내 오차 편차, TD target과 rollout 기준 사이의 차이가 $\epsilon_i$에 들어간다. 5.2절 첫 식의 세 항을 하나씩 cell 모델로 치환하고 절댓값을 합하면 어느 차이에서 bound가 커지는지 분해할 수 있다. 충분히 작은 값이 실제로 관측된다고 가정하지 않는다. $R_\pi$의 gradient가 두 step 구간에서 $L_R$-Lipschitz라면 Taylor 나머지의 절댓값은 $\tfrac12L_R\eta^2(\|g_M\|^2+\|g_C\|^2)$ 이하다. 예측된 개선량 $\eta\Gamma$가 이 상한을 넘는 것은 유한 SGD step에서 개선될 **충분조건**이다.

**실패 반례도 남긴다.** 같은 context 안에서 critic이 행동에 전혀 의존하지 않는 모델을 생각하자. Context A는 policy bin이 조회됐지만 이미 정확하다. Context B는 그 bin의 관측이 없으나 다른 행동에서의 오차 수정이 query에 그대로 전달된다. 두 샘플의 초기 오차가 각각 0과 1이라면 같은 배치에서 SABER의 weight $(4/3,2/3)$는 B의 유용한 gradient를 줄인다. Half-squared loss와 학습률 1의 예에서 B의 오차는 Uniform 후 $1/2$, SABER 후 $2/3$다. 이 경우 eligible Random의 기대 평가 오차도 SABER보다 작다. 두 context의 population은 모두 충분할 수 있다. 따라서 같은 bin의 기록이 있다는 사실만으로 현재 규칙의 우위를 보장할 수 없다.

**원고에서 방어할 주장:** 제한된 모델에서는 배정의 이득 조건과 실패 조건을 유도할 수 있고, 실제 모델에서는 이를 검증할 수 있다. “행동 간 일반화가 약하면 무조건 SABER가 우세하다”는 단조 법칙은 아니다. 가정이 성립하지 않으면 1차 항등식은 유지되어도 **SABER의 부호**는 미정이다. 실제 Adam, 여러 단계, 장기 성능에 대한 정리가 아니다.

## 6. 실험 2: 같은 체크포인트에서 가중치 배정만 바꾸기

### 6.1 이미 아는 것과 새로 확인할 것

기존 [A1 초반부 진단][early]은 HER/SABER에서 48개 저장 시점을 분석했다. 일부 SABER 출처에서 gradient 정렬과 action 간 결합이 관찰됐지만 실제 1·100회 업데이트의 정책 query 오차 개선은 일관적이지 않았다. 기존 [후반부 진단][late]에서도 비슷했다. 예컨대 2M 성공 실행 20개 중 100회 후 SABER가 Uniform보다 고정 target 오차가 낮은 경우가 원 목표 query에서는 8개, HER 목표 query에서는 10개였다. 실패 실행과 포화도 따로 기록돼 있다. 이 숫자는 **frozen target에 대한 오차**이고 실제 Q 정확도도, 새 실험에서 양의 효과가 나올 것이라는 근거도 아니다.

별도로 실제 rollout을 쓴 소규모 2M pilot도 있었다. 한 SABER 체크포인트에서 학습 residual은 1.2103에서 1.1056으로 줄었는데 별도 정책 행동의 Q MSE는 183.2572에서 186.8298로 커졌다. 적은 query에 오차가 집중된 사례이며 왜 그런 변화가 났는지나 평균 효과가 어떤지는 식별하지 않는다. [정책 행동 Q 진단 미팅 자료][meeting]

따라서 새 이름으로 같은 진단을 반복하지 않는다. 기존 [A2 protocol][a2]의 **확장**으로 수행한다. 핵심은 **실제 정책 행동의 rollout 기준을 여러 시점과 matched 대조군에 동일하게 적용**하는 것이다. 1차 분석의 부호와 native optimizer의 결과를 직접 대조한다. DDPG에서 두 행동을 먼저 분석하며 SAC의 squashed-mean query로 자동 일반화하지 않는다.

### 6.2 저장본과 공통 배치

원 A1 inventory는 4개 출처(HER·SABER·Random·TD) × seeds 101–103 × 13개 시점이다. 0-step은 replay가 없어 배정 비교가 불가능하다. 모든 `.pt`가 이 머신에 있다는 뜻은 아니다. 원 파일·source·환경 hash를 받기 전에는 A1의 결과 CSV만으로 optimizer 분기를 재현할 수 없다. 2M final은 포화·상수 reference로 구분력이 떨어질 수 있으므로 A1 대신 조용히 끼워 넣지 않는다.

**신규 rollout 측정의 권장 고정 subset:** 첫 학습 업데이트 전 4,800과 20k·100k·300k·500k를 네 출처와 세 seed에서 모두 사용한다(최대 60개). 이는 기존 10k–500k fixed-target 결과와 별개로, 새 reference 비용을 제한하면서 학습 전후를 포함하는 안이다. 4,800 step의 출처 간 동일 원 상태는 중복 측정·독립 표본으로 세지 않는다. 파일 확보가 안 되면 로컬 2M 저장본의 결과는 **별도 endpoint 기술 검증**으로 명시한다. 과거 성능을 보고 cohort를 고르지 않는다.

각 저장본에서 critic parameter·Adam moment와 step counter·actor·target·replay index·normalizer·config를 검증한다. 동일한 replay transition 순서, HER goal과 relabeled reward, timeout mask, 고정된 Bellman targets로 batch 계획을 만든다. 샘플 절차는 원 구현과 일치시키며, 같은 batch에서 E와 M은 원 scanner와 대조한다. Early-exit 로그의 숫자를 정확한 local count로 간주하지 않는다.

| 분기 | 높은 가중치를 줄 위치 | 답하는 질문 |
|---|---|---|
| Uniform | 없음. 전부 weight 1 | 재가중 자체의 효과가 있는가? |
| SABER | 실제 scanner의 $M$ | 현재 규칙 |
| Eligible Random | $E$에서 균일 비복원 $k$개 | context와 개수는 같을 때 action-bin 정보가 추가로 유용한가? |
| Eligible TD Top-k | $E$에서 TD error 절댓값 상위 $k$개 | 같은 예산에서 단순 TD 크기로 대체 가능한가? |

한 step에서는 완전히 동일한 시작 critic에서 TD 점수를 계산한다. 누적 비교의 주 프로토콜은 각 분기의 현재 critic으로 TD 점수를 다시 계산하는 **배정 규칙 간 비교**다. Actor·target·replay를 동결하므로 매 batch의 context 조건과 $k$는 그대로 공유된다. TD mask까지 초기 critic 점수로 고정한 비교는 필요할 때만 gradient 효과를 분리하는 보조 분석으로 둔다. TD 동점은 배치 위치를 기준으로 결정하고, Random은 고정된 독립 stream을 여러 번 사용한다. Random 복제는 선택 변동을 평균내기 위한 것이지 독립 training seed가 아니다.

### 6.3 기준값과 업데이트

1. **주 평가 bank:** 원 replay와 분리된 reset episode에서 checkpoint 정책과 탐색 잡음을 이용해 행동 trajectory를 수집한다. 미리 지정한 episode·시점에서 상태를 저장하고, 기존 HER 규칙으로 미래 achieved goal을 설정한다. 그 물리 상태를 복원하고 **새 목표에서 정책이 실제로 고른 행동**부터 실행한다. 이후 같은 목표의 고정 정책을 따르며, 해당 목표 기준으로 reward와 종료를 일관되게 계산한다. Relabeling된 목표만 바꾸고 원 환경의 물리 상태·controller를 바꾸지 않는지 검증한다. 원 replay에 simulator 상태가 없으면 관측으로 상태를 역추정하지 않는다. 새 bank는 당시 replay 전체의 정확한 분포가 아니라, HER 이후 actor가 평가하는 위치에 가까운 독립 bank다.
2. **보조 bank:** 원 desired goal로 탐색 잡음 없이 실행한 상태를 별도로 유지한다. 기존 R1/R2와 비교하려면 같은 bank에서 해당 실제 action으로 만든 **고정 Bellman target 오차**도 별도로 기록한다. 이 오차를 실제 return 오차와 합치지 않는다. 실험 1과 같은 복원·목표·종료 gate를 적용한다. 원 정책이 고정이므로 예측 기준 return은 모든 critic 분기가 공유한다.
3. **이론 검산과 실제 업데이트를 분리:** float64 작은 SGD step으로 $u_i$의 자동미분 내적과 finite difference가 수렴하는지 확인한다. 실제 비교는 저장된 Adam을 분기마다 정확히 복제하고 native clipping을 포함한 1·10·100 update를 적용한다. Adam 분모가 달라질 수 있으므로 저장 moment만으로 SGD의 부호가 보존된다고 가정하지 않는다. 실제 parameter 이동을 기록하고 $\nabla R_\pi$와의 내적으로 1차 예측을 점검한다. Clip 전후 norm과 실제 이동을 따로 저장한다.
4. **판정 가능한 배치:** 전체 배치의 결과와 $0<k<|E|$인 배치에서의 1-step 결과를 모두 보고한다. $k=0$이나 $k=|E|$는 같은 배정으로 시작하는 조건의 코드 검증에 유용하다. 누적 학습에서 이전 step에 critic이 달라졌다면 현재 mask가 같아도 결과가 같을 필요는 없다. Positive의 자기 행동-bin 일치, context 내 policy-bin 변동, TD 크기와 행동 거리로 층화한 signed Jacobian 결합, 표본별 $u_i$, phase·실행 행동 나이도 기록한다. 높은 결합이 올바른 방향으로 오차를 줄인다는 뜻은 아니다.

원 A2의 `20261008` master/bootstrap seed와 대응 집계를 계승한다. 기술 pilot에서 복원, mask와 loss parity, 원본 checkpoint 불변성을 확인한 뒤 두 batch 일정과 8개 Random stream을 고정한다. 권장 예산은 checkpoint당 HER-goal 16개와 원 목표 8개의 독립 reset episode query다. 각 episode에서 0·25·50·75·100·125·150·175 중 미리 배정한 한 시점만 취한다. 종료돼 예정 시점이 없으면 대체하지 않고 누락을 적는다. 최대 60개 저장본 기준 11분기 × 2일정 × 100으로 약 13.2만 critic updates, 결정적 reference라면 최대 1,440개 rollout이다. 같은 분기의 1·10회 측정치는 100회 경로에서 얻는다. 모델 학습은 offline 분기뿐이며 새 2M online 재학습을 전제하지 않는다.

### 6.4 무엇을 주 결과로 판정할지

주 endpoint는 **100회 native update 뒤, HER-goal query에서 고정 정책 rollout에 대한 평가 MSE**다. 두 주 대조는 `Random MSE−SABER MSE`, `TD Top-k MSE−SABER MSE`다. 양수는 SABER의 상대적 우위이며, 시작 critic보다 실제로 좋아졌는지는 별도로 표시한다. 초기 MSE가 거의 0인 경우 퍼센트 변화를 주 지표로 사용하지 않는다. Uniform 대비, 1·10회의 변화, 고정 target 오차와 원 목표 bank는 보조다. 100회 측정은 장기 성능의 대체물이 아니라 효과가 수 회 업데이트를 넘어 지속되는지 보는 고정 시점이다.

Episode → batch 일정 → checkpoint 시점·출처 → 원 training seed의 대응 구조를 유지한다. 세 seed에서는 효과 크기·부호·시점별 그림과 전체 포함 결과가 중심이며 통계적 우월성을 주장하지 않는다. 추론을 강화할 추가 독립 seed나 별도 task는 효과를 본 뒤 골라 더하지 않는다. 사전 확정한 두 비교를 하나의 계열로 다루고, 충분한 독립 seed를 확보했다면 seed 블록으로 구간과 다중비교 보정을 적용한다. 성공한 정책 또는 좋은 시점만 골라 주 결론을 내리지 않는다.

**Actor와의 연결은 보조 endpoint:** 고정 subset에 대해 critic 분기 뒤, 동일한 시작 actor·optimizer·actor batch에서 DDPG actor를 한 번만 갱신한다. 고정된 별도 시작 상태에서 새 정책의 실제 return과 성공률을 대응 비교한다. 미갱신 critic에서 actor만 한 번 갱신한 기준과 원 정책도 함께 남긴다. 이 결과가 Q 평가를 사후 대체하지 않는다. 값 오차가 줄어도 행동 기울기와 정책이 좋아진다는 보장은 없으며, 좁은 행동 범위나 접촉 경계에서 직접적인 유한차분 기울기 해석도 조건부다.

## 7. 결과 해석과 본문 반영

| 관측 결과 | 쓸 수 있는 결론 | 피할 결론 |
|---|---|---|
| 제거 후 오차 증가, 독립 주변 기록으로 감소. 거리 점검도 지지 | 시험한 상태·정책에서 주변 행동 기록이 예측을 제약하며, 현재 지표가 그 기록을 일정 범위에서 포착 | support가 모든 context나 task에서 uncertainty를 정확히 측정한다 |
| 제거·복구 효과는 있지만 실제 mask의 거리 대응이 약함 | 주변 경험의 동기는 지지되지만 bin·context·조회 근사를 개선하거나 주장 범위를 좁혀야 함 | 근접성이 중요하므로 지금의 K=5·최소 4가 옳다 |
| SABER가 Uniform보다 낫지만 Random 또는 TD와 구별되지 않음 | 해당 조건에서 재가중 이점은 관찰됐지만 정확한 support 배정의 추가 가치는 미확인 | Action-bin 일치가 향상의 원인이다 |
| 실제 rollout 오차에서 두 대조군보다 낫고 여러 seed·시점에 유지 | 정해진 고정 critic 분기에서 같은 예산을 support로 배정한 효과를 지지 | 장기 학습 성능 향상의 원인 전체를 설명했다 |
| Jacobian 결합이나 SGD 분석만 양수, native return 오차는 개선되지 않음 | 전이 경로는 관찰되나 방향·target·optimizer·포화로 순효과가 없거나 섞여 있음. 무엇이 원인인지는 측정 안 한 항까지 단정하지 않음 | 수학적 조건부 명제만으로 실제 SABER를 정당화했다 |
| Gate 통과 후에도 일관된 이득이 없음 | 효과 크기와 불확실성을 보고하고, 온라인 성능을 조건별 실측 결과로만 서술 | 부정 결과를 감추거나 작은 국소 효과까지 모두 배제한다 |

**원고 배치:** Introduction은 HER가 목표를 바꾸되 실행 행동은 바꾸지 않는다는 점에서 시작한다. [HER 원 논문, NeurIPS 2017](https://papers.neurips.cc/paper_files/paper/2017/hash/453fadbd8a1a3af50a9df4df899537b5-Abstract.html) Method는 두 행동을 보여주는 그림, 두 질문으로 된 판정 규칙, 한 줄 가중 손실, 검증할 가설 순이다. 본문 이론은 공백의 직관과 동일 개수 비교의 조건을 중심으로 두고, 하한 증명과 cell 반례·유한 step 조건은 appendix에 둔다. Experiments에는 실험 1의 제거·복구 대응 그림과 실험 2의 **실제 return 오차** 대응 그림을 넣는다. 기존 AUC는 성능 결과로 유지하되, 별도의 고정 진단이 그 AUC 차이를 매개했다는 결론을 내리지 않는다.

**순서:** 원고의 두 행동과 support 정의부터 수정안으로 정리한다. 측정에서는 저비용으로 context 내 bin 분산, 자기-bin 일치, Tail/전체·거리 대응과 저장본을 확인한다. 실험 1 fitting·복원 gate를 재점검하고, 실험 2가 사용할 실제 rollout bank의 목표·종료 정합성을 먼저 확인한다. 이후 짧은 native 분기를 검증하고 고정된 전체 분석을 실행한다. 기술 검증의 통과 여부로만 실행을 결정하며, SABER에 유리한 pilot인가로 결정하지 않는다.

[a2]: https://github.com/Hyunwoojooo/saber-rl/blob/af0dfed917ca60fa4afa394c0f1be365667117fd/ICML2027/experment_design/01_policy_query_value_diagnostic.md
[early]: https://github.com/Hyunwoojooo/saber-rl/blob/af0dfed917ca60fa4afa394c0f1be365667117fd/ICML2027/paper_artifacts/first_order_usefulness_20261109/early_phase_a1/REPORT_ko.md
[late]: https://github.com/Hyunwoojooo/saber-rl/blob/af0dfed917ca60fa4afa394c0f1be365667117fd/ICML2027/paper_artifacts/first_order_usefulness_20261109/late_ext2m/REPORT_ko.md
[meeting]: https://github.com/Hyunwoojooo/saber-rl/blob/af0dfed917ca60fa4afa394c0f1be365667117fd/ICML2027/meetings/2026-11-05_post_review_first_meeting_ko.md#L233-L271
