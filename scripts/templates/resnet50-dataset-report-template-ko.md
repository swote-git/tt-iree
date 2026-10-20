# ResNet50 ImageNet validation BF16 비교 결과

동일한 torchvision `ResNet50_Weights.IMAGENET1K_V1` weight와 실제 ImageNet-1K
validation 이미지 {sample_count}장을 PyTorch, tt-iree, TTNN에서 평가했다.
tt-iree의 matrix engine 설정은 `{matrix_engine}`이다. 전체 {sample_count}장,
이미지당 1,000개 logits를 비교했으며, 완료된 실행과 입력/weight metadata를
검증한 결과만 이 문서에 포함했다. 중간 activation checkpoint는 수집하지 않았다.

## 데이터와 평가 범위

- Dataset: [{dataset}](https://huggingface.co/datasets/{dataset}), revision `{revision}`.
- Split: 정답 라벨이 있는 `validation`. 공식 unlabeled test split은 사용하지 않았다.
- 평가 이미지: 서로 다른 {sample_count}장, {class_count}개 클래스, 기존 demo 이미지 제외.
- 선택 방법: `{sampling}`.
- 모델 head: 원래 ImageNet의 1,000개 클래스. Subset 클래스만 추려 분류하지 않았다.
- 입력: V1 표준 RGB / resize 256 / center crop 224 / normalization, 동일한 저장 입력 사용.
- PyTorch/tt-iree batch 1, TTNN batch {ttnn_batch}. 이미지 복제나 마지막 batch padding 없음.

이 표본은 validation 첫 Parquet shard 내에서 선정했다. 전체 50,000장에 대한
균등 무작위 표본이나 class-balanced benchmark가 아니다. 아래 Top-1/Top-5를
공식 전체 ImageNet accuracy로 일반화하지 않는다. 이 평가에서는 한 장이 약
{one_image_percentage_points:.3f} percentage point에 해당한다.

`IMAGENET1K_V1`은 weight 버전 이름이다. FP32와 BF16 기준을 각각 남겼으며,
주 수치 비교 기준은 `model.to(torch.bfloat16)`인 PyTorch BF16 module이다.
V1 전처리 정의는 [torchvision 공식 문서](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet50.html)를 사용했다.
Dataset card는 원본 이미지가 resample되지 않았다고 설명한다. 저장 이미지는
다운로드한 Parquet bytes를 그대로 추출했고 크기/SHA256 및 ClassLabel/synset
순서와 원본 filename synset을 대조했다. ImageNet 연구용 이용 조건은 그대로 적용된다.

## 정답 라벨 기준 분류 결과

{accuracy_table}

Top-1은 가장 높은 logit의 클래스가 라벨과 같은지, Top-5는 상위 다섯 클래스에
라벨이 있는지 검사한다. 정답 개수가 같아도 같은 이미지에 맞았다는 뜻은 아니다.
이미지별 prediction과 correctness는 `*_predictions.json`에 있다.

## 1000개 Logits 수치 비교

{metric_table}

PCC는 모든 이미지의 logits를 이어 붙인 pooled correlation이며 정답률이 아니다.
Relative-L2는 해당 비교 reference의 전체 L2 norm 대비 오차 norm이다.
표의 `%`는 이 비율을 100배 표시한 값이다. Scale은 reference 방향으로의
최소제곱 배율로, 1에 가깝다고 모든 원소가 같은 것은 아니다.
Top-1 agreement는 reference 예측과의 일치이지 정답 라벨 기준 정확도가 아니다.

PyTorch BF16 hardware-batch reference와 batch-one reference의 비교는 CPU
batch 크기에 따른 numerical variation을 별도로 보여준다. TTNN과 tt-iree를
IEEE FP32 dot이나 bitwise 동일한 arithmetic 구현으로 간주하지 않는다.

## 이미지별 오차와 예측 변화

{pcc_table}

PCC가 가장 낮은 5장씩:

{worst_table}

PyTorch BF16과 비교해 tt-iree 또는 TTNN의 Top-1이 달라진 이미지는 총
{disagreements}장이다. `top1_disagreements.json`에는 세 구현의 예측, label,
correctness 및 PyTorch Top-2 logit margin을 기록했다. 작은 margin에서는
작은 BF16 차이로 순위가 바뀔 수 있지만, 이를 확인하지 않고 모든 불일치의
원인을 rounding으로 단정하지 않는다.

{change_table}

Top-5 label correctness가 달라진 이미지와 정답 클래스의 순위:

{top5_table}

`top5_disagreements.json`에 정답 logit, 5/6위 점수, 동점 여부를 저장했다.
분류는 모든 구현에서 같은 stable sort를 사용하며, 동점에서는 낮은 class
index를 먼저 선택한다. 5위 경계에 동점인 label을 추가로 정답 처리하는
tie-inclusive 규칙으로 accuracy를 바꾸지 않았다. BF16의 낮은 해상도 때문에
경계 동점이 나타날 수 있지만, logit 차이가 어느 operation에서 생겼는지는
별도 checkpoint 분석이 필요하다.

## 실행 경로와 시간

tt-iree는 이미지마다 58개, 총 {dispatch_count}개 dispatch의 native 여부,
BF16 input/output storage, 요청한 matrix engine routing을 검사했다. Tensix
경로에서는 54개 matrix dispatch가 builtin 12, 나머지 4개가 builtin 11이다.
여기서 native 검사는 numerical dispatch에 대한 계약이며, model I/O나
HAL buffer transfer를 포함한 모든 host 활동이 제거됐다는 뜻은 아니다.

{timing_table}

시간은 host wall 기준이다. IREE invocation에는 dispatch 외 HAL/VM 작업이
포함되고, process wall에는 session 생성과 종료가 포함된다. Dispatch host
duration도 profiler의 device kernel-only latency는 아니다. TTNN batch 시간을
{ttnn_batch}으로 나눈 값은 throughput 환산값이지 batch-one latency가 아니다.
같은 batch·trace·transfer 범위의 성능 비교가 아니므로 공정한 latency 비율로
직접 해석하지 않는다. Dispatch 밖 시간의 세부 원인은 이번 정확도 평가에서
추가 계측하지 않았다.

TTNN activation/weight는 BF16, 요청한 fidelity는 HiFi4, stem activation block은
512다. AvgPool audit의 실제 config는 다음과 같다. 따라서 모델의 모든
operation이 HiFi4/FP32 Dst라고 주장하지 않는다.

```json
{ttnn_avgpool}
```

TTNN의 BN folding과 FC/AvgPool 누적 설정, tt-iree의 explicit BF16 epilogue는
같은 arithmetic 경로가 아니다. 구현 간 오차를 이 한 번의 final-logits 평가로
특정 operation이나 하드웨어 결함에 귀속할 수 없다. 필요하면 최저 PCC나
Top-1 불일치 이미지를 기존 checkpoint/replay 도구로 별도 조사해야 한다.

## 재현 자료와 검증

- Golden: `{golden}`.
- tt-iree run: `{iree_run}`.
- TTNN run: `{ttnn_run}`.
- Torch / torchvision: `{torch_version}` / `{torchvision_version}`.
- Pretrained checkpoint SHA256: `{checkpoint_sha}`.
- Saved state_dict SHA256: `{weights_sha}`.
- Input NPY SHA256: `{input_sha}`.
- 분석 결과: `summary.json`, `classification.csv`, `logits_metrics.csv`, `per_image_metrics.csv`.

Golden build는 eval mode에서 각 reference를 계산했고 각 image/weight/input의
hash를 남겼다. TTNN도 동일한 saved state_dict를 로드했다. tt-iree는 동일한
BF16 export를 Tensix 옵션으로 컴파일했고 final output의 실제 BF16 bits를 읽었다.
반복 session 도구는 실제 hardware에서 A/B/A 입력을 기존 단일 실행 결과와
bitwise 비교했다. 별도의 CPU transport fixture는 입력 교체, 기존 파일
덮어쓰기 거부, short input과 잘못된 F32 output 거부를 검사했다.

이번 평가는 기존 numerical acceptance tolerance를 바꾸는 gate가 아니라
추가 accuracy diagnostic이다. 중간 operation별 오차는 이번 {sample_count}장
평가에서 측정하지 않았으므로 기존 소규모 checkpoint report와 구분한다.
TTNN 종료 시 nanobind reference-leak warning이 있었지만 실행 exit code는 0이며,
완료 상태와 전체 logits 수를 검증했다. 해당 warning의 원인은 이번 작업에서
해결하거나 검증하지 않았다. `third_party/` source/submodule은 수정하지 않았다.
