# Qwen3.8-27B MLX 캠페인 — 총정리 인계 노트

작성 2026-08-17. **사전 지식 0을 전제**로 쓴다. 이 문서만 읽고 캠페인을 이어받을 수
있게 하는 것이 목적이다. AIF 규약([[aif-note-format]])을 따르되, 이 문서는 **지도**이고
**정본은 원장** `~/qwen38/DSPARK_FINDINGS.md`(1,140행, [I1]–[I121] · [RA1]–[RA28] ·
[CA1]–[CA15] · [PA1]–[PA44] · [J7])이다. 숫자가 어긋나면 원장이 이긴다.

---

## 0. 3분 요약 — 이 캠페인이 무엇인가

Alibaba의 **Qwen3.8-27B**(27.78B 밀집 하이브리드 멀티모달, 48 GatedDeltaNet + 16 풀-어텐션
층, vocab 248,320, 262K 컨텍스트, 비전 타워 333텐서 + 벤더 MTP 헤드 31텐서)를 Apple
Silicon(M3 Ultra 512GB ×2, TB5 연결)에서 **양자화·가속·게시**한 작업이다.

성과를 한 줄로: **디코드 37.6 → 74.2 tok/s(2박스, 2.07×) / 프리필 430 → 733 tok/s(1.72×)**,
그리고 그 과정에서 얻은 **측정 규율과 6건의 정직한 기각**.

산출물 3종:
1. **HF 게시 빌드 3개**(8/6/4bit) — 전부 비전 타워 + MTP 헤드 보존
2. **GitHub 전용 리포** github.com/avlp12/qwen38_alis_mlx — 원장·문서 8편·하네스·원자료
3. **포크 2개** — mlx-lm(avlp12/mlx-lm), mlx 코어(avlp12/mlx, 브랜치 `alis`)

---

## 1. 먼저 읽어야 할 것 (우선순위 순)

- **[I1]** 정본 원장 `~/qwen38/DSPARK_FINDINGS.md` — 모든 수치의 출처. 게시본 사본은
  리포의 `docs/LEDGER.md`.
- **[I2]** 리포 문서 8편(`~/qwen38_alis_mlx/docs/`):
  `speed-journey.md`(전 시도 표 — **여기부터 읽으면 빠르다**) · `methodology.md`(측정 규율,
  각 규칙마다 그것을 산 사고가 붙어 있다) · `speculative.md`(투기 디코딩 전사, §6~§8) ·
  `two-box.md`(2박스 프리필 + TP2 서빙) · `kernels.md`(커널 여정) ·
  `conversion-integrity.md`(비전/MTP 보존 기구) · `kl-tiers.md`(품질 티어 차트) ·
  `external-dossiers.md`(외부 스택 대조 — 남의 수치를 우리 수치와 나란히 놓기 전 필독)
- **[I3]** 상시 규칙(사용자 메모리): AIF 노트 규약 · 측정물 /tmp 금지 · NAS 야간 창
  01:00-08:00 · 외부 커뮤니케이션 1인칭 단수(I) · 서빙 자동 복구 금지 · 양 머신 상시 풀가동 ·
  omlx(:8002, GuruNote 소유) 절대 접촉 금지 · TERM-불응 KILL 금지.

---

## 2. 하드웨어와 배선

- **[I4]** gesicht = Mac Studio M3 Ultra 512GB(주 작업기). epsilon = 동일 사양(10.0.0.2).
  둘은 **en4 ↔ en4 TB5 직결, 10.0.0.1/10.0.0.2**. LaunchDaemon `com.alis.k3tb`가 부팅 시
  IP를 설정한다. jaccl RDMA(`rdma_en4`)로 all_sum 21.87µs — TCP ring은 459µs로 탈락.
- **[I5]** 같은 gesicht에 **URAN**(Windows, Core Ultra 9 285K/Z890)이 TB4로 붙어 있고
  bridge0(192.168.7.x + IPv6 링크로컬)을 쓴다. **10.0.0.x와 물리 버스·대역 모두 분리**되어
  간섭하지 않는다. 썬더볼트 버스 6개 중 2개만 사용(Bus 0=URAN 40Gb/s, Bus 2=epsilon 80Gb/s).
- **[CA1']** 재부팅 함정: macOS "Thunderbolt Bridge"(bridge0) 서비스가 과거 수동 설정
  10.0.0.1을 갖고 있으면 en4와 충돌해 라우팅이 bridge0으로 빠진다([I117]). 증상은
  `route -n get 10.0.0.2`가 bridge0을 가리키고 ping 불통. 현재는 bridge0이 192.168.7.2로
  옮겨져 해소됨. 재부팅 후 링크가 죽으면 `netstat -rn -f inet | grep '^10.0.0'`로
  경로 존재를 먼저 확인하라(경로만 사라지는 경우가 있다 — 부록 B).

---

## 3. 게시물 — 어디에 무엇이 있나

- **[I6]** **HF 빌드 3종**(전부 비전 333텐서 원본 바이트 + MTP 헤드 보존):
  - `avlp12/Qwen3.8-27B-Alis-MLX-8bit` 27.9GB · 21.8 tok/s · bf16과 코퍼스 PPL 구분불가
  - `avlp12/Qwen3.8-27B-Alis-MLX-6bit` 21.5GB · 27.3 tok/s · 균형 기본값
  - `avlp12/Qwen3.8-27B-Alis-MLX-4bit` 15.2GB · **AWQ + 재정렬 MTP 헤드** · main `739a5587`
    (벤더 원본 헤드는 `pre-align` 브랜치 `a71171b8`에 보존)
  카드 최신 커밋: 4bit `f9a9e88e` / 6bit `c1275405` / 8bit `512b4be3`.
- **[I7]** **전용 리포** github.com/avlp12/qwen38_alis_mlx (최신 `0562c50`).
- **[I8]** **mlx-lm 포크** avlp12/mlx-lm main — 이 캠페인의 코드가 사는 곳. 주요 커밋:
  `deca373`(2박스 프리필) · `58ae6ec`(절단 기각-샘플링) · `63fd914`(융합 투영 옵트인) ·
  `6ff2474`(fast_qmm 비-affine 폴백) · `fcd986a`(capture-and-rerun 옵트인) ·
  `1e22e21`(fast_qmm TP 샤딩 클래스 확장) · `b8a8e7c`(서버 게이트 MTP) ·
  `d3eb157`(read_last 옵트인).
- **[I9]** **mlx 코어 포크** avlp12/mlx 브랜치 `alis` = v0.32.0 + SDPA head_dim-256 융합
  (`b01cc5c8d`), 로드맵 `docs/ALIS_KERNELS.md`.
- **[I10]** 상류 기여: mlx-lm PR #1735(이중 RMSNorm 시프트 무성 손상 수정) · mlx #4265
  (소량-M qmm — **CONTRIBUTOR가 커널 랜딩 지지**, 코어 PR 초대 상태) · #4253(gather_mm
  무성 오답, **수정 확인·종결**) · #4246(gather_qmm MoE 소그룹).

---

## 4. 헤드라인 실측 (정본 프레임)

**[I11]** 정본 프레임 정의: 4bit 빌드 · 4프롬프트(chat/code/math/**한국어**) · **EOS-컷** ·
의존-사슬 타이밍 · 서멀 교대. 이 프레임을 벗어난 수치는 비교 불가다.

| 디코드 | tok/s | 배수 |
|---|---:|---:|
| 평문 | 37.6 | 1.00× |
| MTP k=2 (무게이트) | 46.8 | 1.24× |
| **게이트 MTP k=4 + min_draft_p 0.6 — 권장 운용점** | **52.8** | **1.40×** |
| DSpark 드래프터(block 8) | 48.3 | 1.28× |
| DSpark + capture-and-rerun(옵트인) | 51.7 | 1.38× |
| 실사용 샘플링(t1·top-p.95·top-k20) 게이트 MTP | 48.1@240 / 45.1@1024 | 1.29× / 1.22× |
| 서버 HTTP 스트리밍, 게이트 MTP | 53.1 greedy / 47.1 t1 | 서버 세금 ≈0 |
| **2박스 TP2 × 게이트 MTP (in-process)** | **74.2** | **2.07×** |
| **2박스 TP2, 서버** | **62.9 greedy / 57.7 t1** | 1박스 서버 대비 +18.5% / +22.4% |

| 프리필(8K) | tok/s |
|---|---:|
| 1박스 | 약 430 (엔진 상한의 96–99%) |
| **2박스 층-파이프(bitwise 동일)** | **733** (1.72×@8K, 1.89×@32K) |
| TP2 스택(서빙과 같은 스택) | 약 650 (TTFT 12.8s) |

**[I12]** 품질: 코퍼스 PPL로 8bit=bf16 구분불가. 전-어휘 정확 KL(10빌드 스윕):
q8awq3 0.00172 < q8v 0.00184 ≪ q6awq3 0.00591 < q6v 0.00664 ≪ **q4awq3m 0.06536** <
q4v 0.07626 < nvfp4 0.09621 < mxfp4 0.14374.

---

## 5. 반드시 알아야 할 추론 (여기가 캠페인의 지식이다)

- **[RA1]** **투기 이득은 하드웨어 조건부다.** 대역폭이 낮은 기계일수록 배율이 크다
  (같은 모델이 M4 Max에서 1.6×, DGX Spark에서 7.4×, 우리 M3 Ultra에서 1.40×). 절대
  속도는 우리가 위인데 배율은 남이 크다 — **분모가 다르기 때문**이다. 교차-스택 배율
  비교는 무의미하다.
- **[RA2]** **디스패치는 이미 은닉돼 있다.** 평문 디코드 루프에서 `async_eval` 이중버퍼가
  런치 갭을 가린다(실효 단가 1.2µs). 그래서 커널 융합은 평문에 +1.5%가 상한이고,
  **host-sync가 노출된 투기 루프에만 값을 지불한다**. Motif(+21%)·h3.c의 융합 성과가
  우리에게 이월되지 않은 이유.
- **[RA3]** **투기 루프의 고정비는 가중치 읽기가 아니라 체인 스케줄링/동기다.** TP2
  2단계(MTP+lm_head 샤딩)가 +4.7%에 그친 이유이자, 남은 레버가 ①넓은-M 커널로 깊은 k를
  여는 것 ②드래프트-그래프 융합인 이유.
- **[RA4]** **분산이 빨라질수록 서빙 계층이 병목으로 올라온다.** 1박스 서버 세금 ~0% vs
  TP2 15% — TP2가 스텝을 13ms로 줄이자 rank0의 SSE·detokenizer 고정비가 드러났다.
- **[RA5]** **수락률은 함정이다.** 스루풋을 예측하는 건 패스당 평균 토큰이지 수락률이
  아니다. 우리 게이트(min_draft_p)와 k-경제학이 그 위에 서 있다.
- **[RA6]** **pending-carry의 실세금은 재공급이 아니라 수락-비례 드래프트-슬롯 손실**이다
  (capture-and-rerun이 고수락 콘텐츠에서만 이기는 이유).
- **[RA7]** **AWQ 이득은 비트폭 단조**다. 4bit에서 전 slice 유의(KL −14.3%), 6bit에서
  −11%지만 잡음 경계, 8bit에서 한국어만 유의. 그래서 4bit만 AWQ로 게시했다.
- **[RA8]** **fp4는 하드웨어 조건부로 지배당한다.** Apple Silicon엔 FP4 매트멀 유닛이 없어
  nvfp4/mxfp4가 동급 크기에서 affine int4+AWQ에 진다. DGX Spark 같은 네이티브 FP4
  하드웨어에서는 역전될 것 — 미검증, 검증 가치 높음.

---

## 6. 측정 규율 — 각 규칙은 사고가 사서 얻은 것이다

- **[I13]** **EOS-컷 필수.** EOS 미적용 하네스가 수학 프롬프트의 사후-EOS 자기복사로
  수락 4.53을 만들어 헤드라인을 오염시켰다([J7]/[I78]). 구 수치(MTP 50.4/DSpark 62.2·71.9)는
  **소급 철회**됐고 카드·리포 전면 정정됐다. **이 사건 이후의 수치만 유효하다.**
- **[I14]** **의존-사슬 벤치.** 독립 matmul 루프는 GPU가 겹쳐 실행해 ×1.26 인플레를 만든다.
  디코드는 직렬이므로 반드시 사슬로 잰다. (mlx#4265에서 외부 CONTRIBUTOR가 독립 재현했다.)
- **[I15]** **대응표본 + 블록 SE.** 62토큰 프로브가 영어에서 부호를 뒤집은 사건 이후,
  코퍼스 PPL/KL(≈100K 토큰, 512-블록 SE)로 전면 교체했다.
- **[I16]** **배선 grep 오라클.** fast_qmm 커널이 프로덕션 경로에서 호출 0건인 채로
  "성과"를 낸 사고 이후, 모든 최적화는 실제 호출 여부를 grep으로 확인한 뒤 측정한다.
- **[I17]** **서멀 교대 + 냉각**(드룹 −8~9%), **길이 assert**(짧은 문장 반복이 2048 프리필을
  못 채워 4.5배 부풀린 사고), **조용한 박스**, **측정물 영구 경로**(/tmp 금지).
- **[RA9]** **오라클의 두 팔은 가중치가 같아야 배선을 잰다**([I103]) — bf16 donor와 배포
  4bit 헤드를 비교하던 오라클이 "배선 불일치"를 오발동시켰다.
- **[RA10]** **웨지 규칙**([RA27], 2026-08-17 신설, 가장 비싸게 산 규칙):
  **분산 collective 데드락을 한 번 겪은 박스는 다음 실험 전 재부팅을 전제로 한다.**
  프로세스 TERM 정리로는 회수되지 않으며, **그 상태에서 낸 실패는 코드에 관한 증거로
  채택하지 않는다.** 이 규칙이 없어서 존재하지 않는 결함(RNG 발산)을 3회 쫓고 수정 2종을
  헛되이 시도했다. 결정적 대조는 "동작하던 코드로 복원 → 여전히 실패 → 재부팅 → 같은
  코드가 첫 시도 통과"였다.

---

## 7. 기각된 것들 (다시 하지 말 것 — 근거와 함께)

- **[CA2]** 글루-융합 2·3단계(메가커널): 실효 런치 단가 재보정(3.0–4.9µs → 1.2µs)으로
  기대치가 +2.4–3.1%로 붕괴. 1단계(연접+접기)만 옵트인 채택(+1.5%, 비트 동일).
- **[CA3]** 게이트 k=8: −8.3%. 진범은 경제학이 아니라 **검증 폭 9가 split-K 커널의
  M∈[6,8] 창을 이탈**하는 것(드래프팅 폭 vs 검증 폭 회계 오류였음).
- **[CA4]** b8 > b7 우위: 공정 7v7 비교에서 동급 — 과거 우세는 **7v6 비교 인공물**.
- **[CA5]** capture-and-rerun의 한국어 구원: 기각. 한국어 병목은 롤백 오버헤드가 아니라
  수락 수준 자체(한국어는 게이트 MTP 유지).
- **[CA6]** 6bit AWQ 게시 교체: 사전 등록 게이트 미달(PPL 유의 1/3, KL 0/3)로 기각.
  자산은 `results/kl_out/`에 보존, 재론 조건은 코퍼스 확장.
- **[CA7]** TP2 평문(투기 없이): 1.37×로 1.4× 게이트 미달 — 경계 산술이 정확히 예측.
- **[CA8]** TP2 2단계 샤딩: +4.7% < +8% 게이트. 서류 산술 약 87 tok/s는 반증.
- **[CA9]** `prefill_step_size` 8192: 무개선(2048 플래토).
- **[CA10]** jtdavies 제보(4bit×고 effort 사고 폭주): **기각**. 16k캡 대응표본에서
  q4v 25.0% vs q8v 20.8%(비 1.20 < 기각선 1.25) — 비종결은 effort 속성, 양자화 무관.
- **[CA11]** RNG 발산 가설: **원인이 아니었음**([I120]). 데드락은 전부 웨지의 산물.
  시드동기 코드는 기본 OFF 옵트인으로 남아 있으나 문제 해결과 무관.

---

## 8. 채택된 코드 — 무엇이 어디에 있고 어떻게 켜나

**[I18]** 포크 `~/glm5.2/mlx-lm`(gesicht) / `/Users/m3ms/mlx-lm-fork`(epsilon, editable 설치).
**하네스는 반드시 포크를 고정하고 스톡이면 하드 실패**시킬 것(스톡 임포트가 norm 이중
시프트로 nll 17.46을 만든 전례).

| 기능 | 파일 | 켜는 법 |
|---|---|---|
| split-K MMA 커널(소량-M) | `mlx_lm/fast_qmm.py` | `load()`에서 자동. 끄기 `MLXLM_NO_FAST_QMM=1` |
| 게이트 MTP 투기 | `mlx_lm/generate.py:mtp_speculative_generate_step` | `num_draft_tokens=4, min_draft_p=0.6` |
| 절단 기각-샘플링 | 같은 함수 | `spec_temp/spec_top_k/spec_top_p/spec_draft_temp` |
| DSpark 드래프터 | `mlx_lm/dspark_generate.py` | `rollback="rerun"`(옵트인), `read_last`(옵트인) |
| 융합 투영 | `mlx_lm/models/qwen3_5.py` | `QWEN35_FUSED_PROJ=1`(기본 OFF) |
| 2박스 층-파이프 프리필 | `mlx_lm/prefill_2box/` | 서버 `--prefill-2box <host:port>` |
| 서버 게이트 MTP | `mlx_lm/server.py` | `--mtp --mtp-num-draft-tokens 4 --mtp-min-draft-p 0.6` |
| TP2 온디맨드 서빙 | `~/qwen38/serving_full2box/` | `launch_full2box.sh` / `stop_full2box.sh` |

**[I19]** 보존 기구(비전+MTP): `mlx_lm/utils.py:save_passthrough_weights` +
`qwen3_5.py:passthrough_patterns = ("model.visual.", "vision_tower.", "mtp.")`.
변환기 관문에서 **바이트 복사 후 재검증**한다. AWQ 경로도 이 관문을 지난다.

---

## 9. 누적된 데이터 — 무엇을 다시 만들 필요가 없나

**[I20]** `~/qwen38/` 아래(측정물은 전부 영구 경로, /tmp 금지 규칙):

| 경로 | 내용 | 재생성 비용 |
|---|---|---|
| `DSPARK_FINDINGS.md` | **정본 원장 1,140행** | 불가 |
| `eval_corpus/` | en(wikitext-2)·ko(한국어 위키)·code(CPython) 평가 코퍼스 | 낮음 |
| `ppl_out/` | 빌드별 대응표본 NLL 배열(.npy) | 중간(빌드당 약 10분) |
| `kl_out/` | 10빌드 전-어휘 KL JSON + 티어 차트 + 생성기 | 높음(빌드당 약 4분 × 10) |
| `exp3/` `exp4/` | MTP 스윕 · temp1 실사용 실측 | 중간 |
| `exp5_fusion/` | 글루-융합 검증 자산(비트 동일 오라클 포함) | 중간 |
| `exp6_rollback/` | capture-and-rerun 검증·벤치 | 중간 |
| `exp7_termination/` | 종결 평가 192런(1차 144 + 2차 48) | **높음(약 9시간)** |
| `exp8_server/` `exp8_ab/` | 서버 게이트 MTP · 소형 A/B | 중간 |
| `mtp_align/` `q4awq3m_align/` | **헤드 재정렬 파이프라인 + 훈련된 헤드** | 높음(약 1시간) |
| `tp2_spike/` | TP2 산술·하네스·발사 스크립트 + out/out2 실측 | 중간 |
| `serving_full2box/` | TP2 온디맨드 서빙 스크립트 + 실측 | 중간 |
| `q8awq3/ q6awq3/ q4v/ q4awq3m/` | 로컬 빌드(28G/22G/15G/15G) | 높음 |

**[I21]** X10 외장에 아카이브된 중간 산출물(sha256 전수 검증 후 원본 삭제):
`/Volumes/Crucial X10/qwen38-archive-2026-08-16/` — q8m·q4m·q4·q6awq·q6awq2·q4awq·
q4awq2·q4awq3·tap_dumps·src(bf16 원본). **재다운 전 반드시 여기부터 확인**.

---

## 10. 열린 작업 — 다음 사람이 할 것 (우선순위 순)

- **[PA1]** **넓은-M 커널(M≤16) + 깊은 k(8–12) 재스윕.** [CA3]의 진범이 커널 창이었으므로
  창을 넓히면 깊은 드래프트 경제학이 열린다. mlx #4265의 "커버 범위" 질문에 대한 답이기도
  하다. **가장 기대값이 높은 다음 수.**
- **[PA2]** **mlx 코어 PR** — #4265에서 CONTRIBUTOR가 랜딩을 지지했고 설계 질문 2개
  (기존 steel qmm 디스패치 내 split-K 변형 vs 별도 커널 / 커버 범위)를 던졌다. 응답 대기 중.
- **[PA3]** **서빙 계층 최적화**([RA4]) — TP2의 15% 세금. 청크 배칭·detokenizer 오프로드.
- **[PA4]** **MTP 서빙 프리픽스-캐시 재사용**([I98]) — 현재 MTP 모드는 요청마다 전량
  프리필한다(미커밋 꼬리가 공유 캐시를 오염시킬 위험 때문). 외부 정량화로는 공유
  프리픽스에서 14–22× 이득. 커밋-경계 스냅샷 설계가 필요.
- **[PA5]** **24GB Mac 타깃 검증** — 4bit 카드는 "24-32GB Mac용"이라 쓰여 있는데 512GB
  기계에서만 쟀다. M4 Mac mini 24GB로 실측해 카드에 실을 것(대역폭 기준 예측 평문 ~8,
  게이트 MTP 10-12). 신뢰도 공백을 메우는 저비용 고효용 작업.
- **[PA6]** **DGX Spark 교차 측정** — 우리 정본 프로토콜을 그들 하드웨어에 그대로 걸어
  최초의 유효한 Apple-vs-Spark 비교를 만든다. 특히 [RA8]의 fp4 역전 검증.
- **[PA7]** 미완의 소소한 것들: 6bit AWQ 재론(코퍼스 확장 시) · 8bit AWQ 한국어 우세
  게시 여부 · TP2 프리필 vs 층-파이프 자동 선택 로직.

---

## 부록 A — 운영 명령 (plain block, AIF 그래프 밖)

```bash
# TP2 온디맨드 서빙(프리필 약 650 + 디코드 62.9)
~/qwen38/serving_full2box/launch_full2box.sh
~/qwen38/serving_full2box/stop_full2box.sh

# 2박스 층-파이프 프리필(733 tok/s) — epsilon 러너 먼저
ssh 10.0.0.2 'cd ~/qwen38 && ./launch_runner.sh'      # :39919
# 그 뒤 gesicht 서버에 --prefill-2box 10.0.0.2:39919

# 단일박스 권장 운용점(in-process)
#   mtp_speculative_generate_step(..., num_draft_tokens=4, min_draft_p=0.6,
#                                 spec_temp=1.0, spec_top_k=20, spec_top_p=0.95)

# TB5 링크 점검(재부팅 후 필수)
ping -c2 10.0.0.2
netstat -rn -f inet | grep '^10.0.0'     # 경로 없으면 부록 B
```

## 부록 B — 알려진 함정과 대처 (plain block)

| 증상 | 원인 | 대처 |
|---|---|---|
| epsilon ping 불통, en4엔 IP 있음 | 10.0.0.0/24 링크 경로 소실 | `sudo route -n add -net 10.0.0.0/24 -interface en4` |
| 라우팅이 bridge0으로 빠짐 | Thunderbolt Bridge가 10.0.0.1 선점 | bridge0을 다른 대역으로(현재 192.168.7.2로 해소됨) |
| 분산 런치가 반복 실패 | **collective 데드락 잔재(웨지)** | **해당 박스 재부팅**([RA10]). 코드 탓하지 말 것 |
| 화면 멈춤 | GPU 스톨(완료 안 되는 커맨드 버퍼 / 발사-실패 churn) | TERM으로 정리, **KILL 금지**(wired 누수 회수 불가) |
| 벤치 수치가 이상하게 높음 | EOS 미적용 / 독립 루프 / 프리필 길이 미달 | [I13][I14][I17] 점검 |
| 하네스가 스톡 mlx-lm을 잡음 | sys.path 선삽입 누락 | 포크 고정 + 스톡이면 하드 실패 |
| mlx.launch가 스크립트를 못 찾음 | 절대경로는 상대 박스에서 깨짐 | **홈-상대경로**로 발사 + `--python` |
| ExFAT 외장 sha 전수 불일치 | AppleDouble(`._*`) 메타 파일 | 양쪽에서 `._*`/`.DS_Store` 제외 후 대조 |

## 부록 C — 접촉 금지 목록 (plain block)

- epsilon `:8002` omlx — GuruNote 소유. 절대 건드리지 말 것.
- epsilon `:39919` 프리필 러너 — 지시 없이 재시작/종료 금지(현재 재부팅으로 내려가 있음).
- NAS 쓰기는 **01:00-08:00 창**에서만(소음 민원). 러너 `~/local_claude_code/nas/`.
- HF 게시 교체 시 **반드시 pre-* 브랜치로 현 main을 보존한 뒤** 진행.
