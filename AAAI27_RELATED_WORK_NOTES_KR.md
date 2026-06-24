# SABER AAAI-27 Related Work Notes

작성일: 2026-06-24

목적: SABER의 closest prior를 정리하고, HER relabeling / HER prioritization / PER / offline support constraint와의 차이를 논문에서 방어 가능한 축으로 고정한다.

## 1. 결론

SABER의 related work framing은 다음 한 문장으로 잡는 것이 가장 안전하다.

> Prior HER variants mainly decide which goals, episodes, or transitions to replay; SABER keeps HER replay sampling and actor updates unchanged, and only changes how much the critic should trust each Bellman error using local replay-action support.

한국어로는 다음과 같다.

> 기존 HER 계열은 주로 어떤 goal, episode, transition을 replay할지 바꾼다. SABER는 HER sampling과 actor update를 유지하고, local replay-action support를 기준으로 critic Bellman error의 fitting weight만 조정한다.

이 구분이 논문의 novelty 방어선이다. SABER를 “새 HER 방법”이라고 쓰면 HGG, HGR, EBP, CHER, MHER 계열과 정면 충돌한다. 대신 “HER 이후 critic-side Bellman error reweighting objective”로 써야 한다.

## 2. Related Work 분류

### 2.1 HER와 goal relabeling

HER는 sparse binary reward setting에서 실패 trajectory를 achieved goal 기준으로 relabel하여 학습 가능한 goal-conditioned transition을 늘린다. HER의 핵심은 target quantity 증가다.

SABER와의 차이:

- HER: 어떤 training goal을 만들 것인가.
- SABER: 만들어진 Bellman error를 critic loss에서 얼마나 강하게 fitting할 것인가.
- SABER는 HER ratio, future-goal strategy, replay sampling distribution을 바꾸지 않는다.

논문 문장 후보:

> HER increases the number of goal-conditioned Bellman targets; SABER addresses a different question: whether each relabeled Bellman update should contribute equally to critic fitting.

### 2.2 Hindsight goal generation / curriculum 계열

HGG는 HER가 heuristic하게 hindsight goals를 구성한다는 점에서 출발해, 단기적으로 달성 가능하면서 장기 target으로 가는 데 도움이 되는 hindsight goals를 생성한다. CHER는 MHER 논문에서 Curriculum-guided HER로 비교되며, hindsight goal selection에서 exploration-exploitation trade-off를 조정하는 계열로 설명된다.

SABER와의 차이:

- HGG/CHER: relabeled goal distribution 또는 curriculum을 바꾼다.
- SABER: relabeled goal은 그대로 두고, critic loss weighting만 바꾼다.
- HGG/CHER는 “어떤 goal을 학습할 것인가”에 가깝고, SABER는 “같은 goal-conditioned update를 critic이 얼마나 신뢰할 것인가”에 가깝다.

주의:

- CHER 원문은 별도 arXiv abstract page로는 검색되지 않았지만, MHER arXiv source의 `main.bbl`에서 reference metadata를 확인했다.
- 확인된 항목: Meng Fang, Tianyi Zhou, Yali Du, Lei Han, Zhengyou Zhang, "Curriculum-guided hindsight experience replay", NeurIPS 2019, pp. 12623--12634.
- 원고 bibliography에는 `fang2019cher`로 추가했다.

### 2.3 Hindsight prioritization 계열

EBP는 trajectory energy를 사용해 어떤 hindsight episode를 더 replay할지 우선순위를 둔다. HGR은 episode 평균 TD-error와 future hindsight goal의 TD-error를 사용해 replay buffer에서 episode/goal을 더 높은 확률로 sampling한다.

SABER와의 차이:

- EBP/HGR: replay selection 또는 hindsight goal sampling probability를 바꾼다.
- HGR: TD-error를 proxy로 사용한다.
- SABER: uniform replay sampling을 유지하고, TD-error magnitude가 아니라 local replay-action support를 사용한다.
- SABER의 main variant는 actor objective도 바꾸지 않는다.

리뷰어 공격:

> SABER도 결국 prioritized HER 아닌가?

답변:

> No. SABER does not prioritize episodes, transitions, or hindsight goals for replay. It samples the same HER minibatches as the baseline and reweights only the critic Bellman residual.

### 2.4 MHER / MHER(lambda) / MMHER

MHER는 single-step relabeling 대신 n-step relabeled returns를 사용해 더 많은 reward signal을 만들려 한다. 그러나 off-policy n-step bias가 생길 수 있어 MHER(lambda)와 model-based MHER가 bias-reduced variant로 제안된다.

SABER와의 차이:

- MHER: Bellman target 자체를 multi-step relabeled return으로 바꾼다.
- SABER: DDPG/TD3/SAC의 Bellman target form은 유지하고 loss weight만 바꾼다.
- MHER는 return estimation / target construction 문제이고, SABER는 critic regression reliability weighting 문제다.

논문 문장 후보:

> MHER modifies the hindsight target by using multi-step relabeled returns, whereas SABER leaves the one-step HER Bellman target unchanged and reweights the critic residual.

### 2.5 PER와 TD-error weighting

PER는 TD-error가 큰 transition을 더 자주 replay한다. HGR도 TD-error를 hindsight goal ranking의 주요 proxy로 사용한다. PER-style loss weighting은 sampling을 바꾸지 않더라도 TD-error magnitude를 직접 weight로 사용한다.

SABER와의 차이:

- PER: surprise / learning progress proxy인 TD-error magnitude를 사용한다.
- SABER: local replay-action support를 사용한다.
- PER는 replay sampling distribution을 바꾸는 것이 핵심이다.
- SABER는 sampling은 그대로 두고 critic loss만 reweight한다.

필요한 실험:

- `corr(psi_i, |delta_i|)`
- support-positive sample과 top-k TD-error sample overlap
- support-positive vs support-negative residual reduction

이 diagnostic이 있어야 “PER 변형” 공격을 강하게 막을 수 있다.

### 2.6 High replay ratio / REDQ+HER 계열

REDQ는 high update-to-data ratio, Q ensemble, random subset target minimization을 조합해 sample efficiency를 높인다. Hiraoka의 sparse goal-conditioned REDQ work는 REDQ를 HER와 결합하고 target Q bounding을 사용해 sparse goal-conditioned tasks에서 강한 sample efficiency를 보인다.

SABER와의 관계:

- REDQ+HER는 강한 sample-efficiency baseline 후보다.
- 하지만 SABER의 main claim은 “SOTA sparse-goal RL”이 아니라 “critic-side support-aware reweighting mechanism”이다.
- REDQ+HER는 backbone/compute regime이 다르므로 main table에 넣기 어렵다면 limitation 또는 future baseline으로 정직하게 쓰는 것이 안전하다.

리뷰어가 물을 수 있는 점:

> 왜 최신 high replay-ratio HER baseline과 비교하지 않았는가?

답변 방향:

> We focus on isolating the critic-side reweighting mechanism under matched HER actor-critic backbones. High replay-ratio ensemble methods are complementary and substantially change the compute/backbone regime; we discuss them as strong future baselines rather than claiming state-of-the-art sparse-goal RL.

### 2.7 Offline RL support constraint

BCQ, BEAR, CQL 등 offline RL은 dataset support 밖 action에 대한 value overestimation 또는 bootstrapping error를 줄이기 위해 policy constraint, backup constraint, conservative value learning을 사용한다.

SABER와의 차이:

- Offline RL: fixed dataset에서 policy를 dataset support 안으로 제한하거나 OOD action value를 낮춘다.
- SABER: online off-policy HER setting이며, actor action을 제한하지 않는다.
- SABER: conservative lower bound나 policy constraint를 만들지 않는다.
- SABER: replay support가 있는 local context의 critic Bellman residual에 더 큰 weight를 준다.

따라서 SABER는 offline support mismatch literature와 개념적으로 연결되지만, 알고리즘적으로는 훨씬 좁다.

## 3. Comparison Table

| Method family | 대표 논문 | Intervention point | Signal | Replay sampling 변경 | Goal relabeling 변경 | Actor objective 변경 | SABER와의 핵심 차이 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| HER | Andrychowicz et al. 2017 | goal relabeling | achieved future goals | 보통 uniform | Yes | No | target quantity를 늘림. SABER는 target reliability weight를 조정 |
| HGG | Ren et al. 2019 | hindsight goal generation | short-term achievable + long-term useful goals | goal selection 변경 | Yes | No | goal curriculum/generation. SABER는 goal distribution 불변 |
| CHER | Fang et al. 2019 | curriculum-guided hindsight goal selection | exploration-exploitation curriculum | goal selection 변경 | Yes | No | curriculum relabeling 계열. SABER는 goal selection 불변 |
| EBP | Zhao and Tresp 2018 | hindsight episode prioritization | trajectory energy | Yes | 간접적 | No | energy-based replay priority. SABER는 sampling 불변 |
| HGR | Luu and Yoo 2021 | episode/goal ranking | TD-error | Yes | Yes | No | TD-error 기반 goal/replay priority. SABER는 support 기반 critic loss weight |
| MHER / MHER(lambda) | Yang et al. 2021 | Bellman target construction | n-step relabeled returns, lambda return | No 또는 기본 HER sampling | Yes, multi-step | No | target 자체 변경. SABER는 target form 유지 |
| PER | Schaul et al. 2015 | replay sampling | TD-error magnitude | Yes | No | No | surprise prioritization. SABER는 support-positive critic residual weighting |
| PER-style loss weighting | control baseline | critic loss weighting | TD-error magnitude | No | No | No | intervention point는 유사하지만 signal이 TD-error |
| REDQ+HER / high RR | Chen et al. 2021; Hiraoka 2023 | update regime/backbone | high UTD, Q ensemble, target bounding | No 또는 HER | HER 사용 | backbone 변경 | strong baseline 후보지만 compute/backbone regime이 다름 |
| BCQ/BEAR/CQL | Fujimoto et al. 2019; Kumar et al. 2019/2020 | offline support/conservatism | dataset action support or conservative Q | offline dataset | No | policy/value constraint | policy를 support 안으로 묶거나 OOD value를 낮춤. SABER는 online critic weighting |

## 4. 논문 Related Work 구조 제안

현재 draft의 Related Work는 다음 구조로 바꾸는 것이 좋다.

1. Goal-Conditioned RL and HER
   - HER와 benchmark context.
   - SABER는 HER replacement가 아니라고 명시.

2. Goal Relabeling and Hindsight Goal Selection
   - HGG, CHER, EBP, HGR, MHER.
   - `goal/episode/transition selection` vs `critic residual weighting` 구분.

3. Prioritized Replay and TD-error Weighting
   - PER, HGR의 TD-error proxy, PER-style loss weighting.
   - SABER의 signal은 TD-error가 아니라 support.
   - sample-level diagnostic 필요.

4. Actor-Critic Error and High Replay-ratio Methods
   - DDPG/TD3/SAC.
   - REDQ+HER는 strong but different compute/backbone baseline.

5. Dataset Support and Offline RL
   - BCQ/BEAR/CQL.
   - SABER는 offline RL이 아니고 policy constraint가 아니라 critic-side weighting.

6. MetaCog-RL Lineage
   - full MetaCog가 아니라 PRACTICE/incompetence idea의 narrow instantiation.

## 5. 안전한 Related Work 문장 초안

### HER variants와의 차이

> A large body of HER variants improves sparse goal-conditioned learning by changing which goals, episodes, or transitions are replayed. HGG generates more useful hindsight goals, EBP prioritizes episodes using trajectory energy, HGR ranks hindsight goals using TD-error, and MHER modifies the relabeled Bellman target with multi-step returns. SABER is orthogonal to these approaches: it keeps the HER relabeling and replay sampling protocol fixed, and instead reweights the critic Bellman residual according to local replay-action support.

### PER와의 차이

> Unlike PER or TD-error-based HER prioritization, SABER does not use Bellman error magnitude as its priority signal. Its weight is computed before the critic update from a local state-goal-horizon replay context and the current policy action's replay-action support. This design separates supported Bellman supervision from surprise-based replay.

### Offline RL과의 차이

> SABER is motivated by support mismatch but is not an offline RL algorithm. It does not constrain the actor to remain close to the replay action distribution and does not learn a conservative lower bound. Instead, in an online HER actor-critic setting, it uses local replay support as a critic-side reliability proxy for weighting Bellman residuals.

### REDQ+HER와의 관계

> High replay-ratio ensemble methods such as REDQ+HER provide strong sample-efficiency baselines for sparse goal-conditioned tasks. We treat them as complementary to SABER because they alter the update-to-data ratio and critic ensemble architecture, whereas SABER isolates a single critic-loss reweighting mechanism under matched DDPG/TD3/SAC HER backbones.

## 6. Missing Baseline 리스크

리뷰어가 요구할 수 있는 strongest missing baselines:

| Baseline | 리스크 | 대응 |
| --- | --- | --- |
| HGR | TD-error 기반 hindsight prioritization이라 SABER/PER 차이와 직접 연결됨 | Related Work에서 반드시 비교. 실험은 가능하면 appendix/future |
| EBP | HER prioritization classic baseline | 로봇 manipulation HER prioritization으로 언급 |
| HGG/CHER | goal selection/curriculum 계열 | SABER가 goal selection을 바꾸지 않는다는 축으로 방어 |
| MHER(lambda) | multi-step HER 계열 | target construction과 loss weighting 차이를 명시 |
| REDQ+HER | 최신 sample-efficient sparse-goal baseline | SOTA claim을 피하고 complementary baseline으로 limitation 처리 |

현재 결과만으로는 “SABER is SOTA sparse goal-conditioned RL”을 주장하면 위험하다. 대신 다음 claim이 안전하다.

> SABER improves HER baselines under matched off-policy actor-critic backbones and isolates a support-aware critic-side reweighting mechanism.

## 7. BibTeX 후보

최종 `.bib`에 추가할 후보:

```bibtex
@article{plappert2018multigoal,
  title = {Multi-Goal Reinforcement Learning: Challenging Robotics Environments and Request for Research},
  author = {Plappert, Matthias and Andrychowicz, Marcin and Ray, Alex and McGrew, Bob and Baker, Bowen and Powell, Glenn and Schneider, Jonas and Tobin, Josh and Chociej, Maciek and Welinder, Peter and Kumar, Vikash and Zaremba, Wojciech},
  journal = {arXiv preprint arXiv:1802.09464},
  year = {2018}
}

@inproceedings{zhao2018ebp,
  title = {Energy-Based Hindsight Experience Prioritization},
  author = {Zhao, Rui and Tresp, Volker},
  booktitle = {Conference on Robot Learning},
  year = {2018}
}

@inproceedings{ren2019hgg,
  title = {Exploration via Hindsight Goal Generation},
  author = {Ren, Zhizhou and Dong, Kefan and Zhou, Yuan and Liu, Qiang and Peng, Jian},
  booktitle = {Advances in Neural Information Processing Systems},
  year = {2019}
}

@article{luu2021hgr,
  title = {Hindsight Goal Ranking on Replay Buffer for Sparse Reward Environment},
  author = {Luu, Tung M. and Yoo, Chang D.},
  journal = {IEEE Access},
  year = {2021}
}

@article{yang2021mher,
  title = {Bias-reduced Multi-step Hindsight Experience Replay for Efficient Multi-goal Reinforcement Learning},
  author = {Yang, Rui and Lyu, Jiafei and Yang, Yu and Yan, Jiangpeng and Luo, Feng and Luo, Dijun and Li, Lanqing and Li, Xiu},
  journal = {arXiv preprint arXiv:2102.12962},
  year = {2021}
}

@inproceedings{chen2021redq,
  title = {Randomized Ensembled Double Q-Learning: Learning Fast Without a Model},
  author = {Chen, Xinyue and Wang, Che and Zhou, Zijian and Ross, Keith},
  booktitle = {International Conference on Learning Representations},
  year = {2021}
}

@article{hiraoka2023sparsegoalredq,
  title = {Efficient Sparse-Reward Goal-Conditioned Reinforcement Learning with a High Replay Ratio and Regularization},
  author = {Hiraoka, Takuya},
  journal = {arXiv preprint arXiv:2312.05787},
  year = {2023}
}

@article{kumar2019bear,
  title = {Stabilizing Off-Policy Q-Learning via Bootstrapping Error Reduction},
  author = {Kumar, Aviral and Fu, Justin and Tucker, George and Levine, Sergey},
  journal = {Advances in Neural Information Processing Systems},
  year = {2019}
}
```

CHER metadata confirmed from the MHER arXiv source `main.bbl` and added to `paper/saber_refs.bib`.

## 8. Source Notes

Primary/near-primary sources checked:

- HER: https://arxiv.org/abs/1707.01495
- Robotics benchmark: https://arxiv.org/abs/1802.09464
- EBP: https://arxiv.org/abs/1810.01363
- HGG: https://arxiv.org/abs/1906.04279
- HGR: https://arxiv.org/abs/2110.15043
- MHER: https://arxiv.org/abs/2102.12962
- REDQ: https://arxiv.org/abs/2101.05982
- Sparse-goal REDQ+HER: https://arxiv.org/abs/2312.05787
- PER: https://arxiv.org/abs/1511.05952
- BCQ: https://arxiv.org/abs/1812.02900
- BEAR: https://arxiv.org/abs/1906.00949
- CQL: https://arxiv.org/abs/2006.04779
