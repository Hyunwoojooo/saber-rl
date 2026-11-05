# Seed 1–10 / 11–20 source 재감사

검사시각: 2026-10-30T07:02:37.623500+00:00. GitHub main commit `1c6f2872acb1bc9bccd81908a8ebcceab6392efc`.

**전체 source 폴더가 동일한 것은 아니다. 핵심 학습 코드 `code/` 26개와 `upstream/` 25개는 바이트까지 동일하다.**

## 대조 방식

1. 최신 GitHub Git blob → 로컬 보존 original → 실제 확장 study의 파일별 SHA256과 원문 비교.
2. 과거 30개 status의 manifest hash → 원본 manifest → 파일별 hash 대조. 확장 30개도 같은 방식으로 대조.
3. 별도 Python 프로세스 두 개에서 각 source를 import하고 실제 module 경로·hash 확인. Config, actor/critic, trainer, replay/HER, weighting, selector, task, env, scanner 등 10개 module이 각 전용 code/src에서 로드되는지 확인.
4. 같은 seed 1–10, 세 arm, CPU/CUDA 각각의 config 60개 비교. 원본 함수를 현재 환경에서 호출한 재구성 검증이며 과거 실행 환경의 재현이 아니다.
5. 원본 함수의 seed 허용목록만 audit process 메모리에서 11–20으로 넓혀, 원본 함수 본문으로 만든 30개 config와 실제 저장된 requested_config/config 비교. 디스크의 원본을 수정하지 않고 trainer를 호출하지 않았다.

## 결과

| 항목 | 결과 |
|---|---|
| GitHub ↔ 보존 original | 62개 모두 동일 |
| GitHub ↔ 확장 study의 code/ | 26개 모두 동일 |
| GitHub ↔ 확장 study의 upstream/ | 25개 모두 동일 |
| 실제 imported 학습 module | 10개 모두 전용 code/src를 사용하며 구현 hash 동일 |
| 같은 seed·device로 생성한 config | 60/60 동일 |
| 확장 30회의 실제 저장 학습 config | 원본 함수에서 재구성한 11–20 설정과 30/30 일치 |
| 실행 source hash | 과거 30개는 원본 manifest, 새 30개는 확장 manifest와 일치 |

다른 폴더의 metacog_lite_v0가 import되는 문제, support/TD/Random label 교체, actor/critic LR·loss·target·HER·replay·support 식 변경은 이 비교에서 발견하지 못했다.

## 달라진 파일과 영향

| 파일 | 변경 | 학습 수치에 대한 판단 |
|---|---|---|
| protocol.py | 허용 seed 범위, bootstrap seed, 분석 version | 학습 설정 생성 본문과 RNG seed 공식 동일. Bootstrap은 사후 통계 전용 |
| train.py | CLI seed 범위, PID 기록, 학습 종료 후 checkpoint 감사 | run_training 호출 전후 학습 경로·함수는 동일. 추가 감사는 학습 종료 이후 |
| launch.py | 최대3 worker, 자원 검사, 실패시 대기열 중단, polling, NUMBA_NUM_THREADS=1, pyc 쓰기 방지 | 실행 조건 변경은 존재. 새 조건이 원래 실행 조건과 같다고 주장할 수 없음 |
| validate.py | smoke 검증 seed를 1→11로 이동 | 검증 대상만 변경 |
| tests/test_assignment.py | 통합 분석 fixture를 새 seed 범위로 이동 | 테스트만 변경 |
| freeze.py / source_manifest.json | 새 snapshot ID·origin 및 파일 hash | 추적/무결성 metadata 변경 |
| README.md / PROTOCOL.md | 확장 범위 설명 추가 | 문서 변경 |
| completion.py (추가) | final checkpoint/replay/optimizer/RNG 감사 | 학습 종료 후 검사 |

원본 대비 9개 파일이 변경되고 1개가 추가됐다. 별도 최상위 queue·분석·검증 wrapper도 추가됐다. 전체 diff는 [source_diff.patch](source_diff.patch), 파일 hash는 [file_hashes.csv](file_hashes.csv)에 있다.

## 실행 조건에서 중요한 미확인점

**추가 30회는 실제 CPU 실행임이 config와 run.log 양쪽에서 확인된다. 원본 launcher/config의 기본값은 CUDA지만, 과거 30회가 실제 어떤 --device 값으로 실행됐는지는 공개 run별 config/log가 없어 확인하지 못한다.**
원본도 --device cpu를 지원하므로 기본값만 보고 GPU→CPU 변경이 실제로 일어났다고 단정할 수 없다. 반대로 두 cohort가 같은 device·package 환경이었다고 단정해서도 안 된다.
실행별 시작·종료 timestamp로 계산한 lifecycle 중첩은 과거 최대 5개, 새 cohort 최대 3개다. 과거에는 Support/Random/TD를 섞어 실행했고 새 실험은 사용자 요청대로 SABER→Random→TD 순서였다. 이는 보존된 attempt 기록의 동시 활성 수이며 개별 CPU/GPU kernel의 실제 동시 실행량을 측정한 것은 아니다.
NUMBA_NUM_THREADS=1을 새 launcher가 명시했다. 보존된 support kernel은 parallel=True/prange를 쓰지 않는 직렬 njit 구현이어서 알고리즘의 선택식 변경으로 보이지 않지만 실행 환경 차이로 공개한다. OMP/MKL/OPENBLAS=1과 torch.set_num_threads(1)은 원본에도 있었다.

## 정확한 결론

**“핵심 학습 소스와 설정 생성식은 동일하다”는 확인된다. “전체 실행 코드가 동일하고 seed만 바뀌었다”는 표현은 정확하지 않다.** 실행·검증 wrapper는 수정됐고, 과거의 실제 device/package 동일성은 아직 검증되지 않았다.
이 검사는 고정된 snapshot과 보존된 run 기록을 대조한 것이다. 과거 서버에 접속해 원 실행 환경이나 누락된 원시 config를 감사한 것은 아니다. 기존 학습 code와 결과 파일은 변경하지 않았다.
