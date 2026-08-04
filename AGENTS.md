# alis-dwq — 작업 규약 (에이전트 필독)

이 리포는 "영수증 있는 저비트 양자화"의 정본이다: 방법론(`alis_dwq/`), 교훈(`docs/`),
그리고 **모델별 케이스 스터디+원자료**(`examples/<model>/`)가 한 몸이다.

## 게시 완결 규칙 (2026-08-05 신설 — 위반 전례: Kimi-K3가 docs만 있고 examples 누락)

alis-dwq로 빌드·감사한 모델은 아래 **4종이 전부 갖춰져야 캠페인 종결**로 간주한다:

1. **docs/ 교훈** — 성공뿐 아니라 부정 결과·함정 포함(기존 파일 확장 가능)
2. **examples/<model>/README.md 케이스 스터디** — 품질·속도 수치표 + 채택/기각 판정
   + **원자료 영수증**(측정 로그, npz/원데이터, 재현 스크립트). 문서 링크만으로 대체 불가
3. **메인 README 모델 색인표에 행 추가** + docs↔examples 상호 링크
4. **검증된 명령만 게시** — 실행해보지 않은 명령·플래그 금지(§2b `-m` 오류 전례)

체크 방법: 캠페인을 닫기 전 `ls examples/ | grep <model>` 과 `grep <model> README.md`가
모두 비어 있지 않아야 한다. 하나라도 비면 미완결.

## 양방향 동기화 (상시)

- 세션 시작: 이 리포·상류(PR·이슈)에서 새 정보 흡수
- 세션 중: 검증된 교훈(부정 결과 포함)은 미루지 말고 즉시 커밋
- 수치는 "measured, not estimated" — 추정치는 추정임을 명시

<claude-mem-context>
# Memory Context

# [alis-dwq] recent context, 2026-07-15 12:33pm GMT+9

Legend: 🎯session 🔴bugfix 🟣feature 🔄refactor ✅change 🔵discovery ⚖️decision
Format: ID TIME TYPE TITLE
Fetch details: get_observations([IDs]) | Search: mem-search skill

Stats: 3 obs (1,254t read) | 21,965t work | 94% savings

### Jul 15, 2026
4109 10:32a ⚖️ alis-dwq PR #10 — 3-Lens Adversarial Review Protocol Initiated
4116 10:37a ⚖️ alis-dwq PR #10 — 3-Lens Adversarial Review Launched (Lens 1: Refutation)
4124 10:44a 🟣 alis-dwq PR #10 — Adversarial Lens 1 Review Launched (Claims C1–C8)

Access 22k tokens of past work via get_observations([IDs]) or mem-search skill.
</claude-mem-context>