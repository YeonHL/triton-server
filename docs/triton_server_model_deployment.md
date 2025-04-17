# Triton Server

- [모델 저장소](#모델-저장소)
  - [지원 모델 형태](#지원-모델-형태)
    - [TensorRT Models](#tensorrt-models)
    - [ONNX Models](#onnx-models)
    - [TorchScript Models](#torchscript-models)
    - [TensorFlow Models](#tensorflow-models)
    - [OpenVINO Models](#openvino-models)
    - [DALI Models](#dali-models)
    - [Python Models](#python-models)
- [모델 구성](#모델-구성)
  - [최소 모델 구성](#최소-모델-구성)
    - [이름, 플랫폼 및 백엔드](#이름-플랫폼-및-백엔드)
    - [모델 트랜잭션 정책](#모델-트랜잭션-정책)
    - [최대 배치 크기](#최대-배치-크기)
    - [입력과 출력](#입력과-출력)
      - [PyTorch 백엔드를 위한 특별 규칙](#pytorch-백엔드를-위한-특별-규칙)
  - [자동 생성 모델 설정](#자동-생성-모델-설정)
  - [사용자 정의 모델 설정](#사용자-정의-모델-설정)
    - [기본 최대 배치 크기와 동적 배처](#기본-최대-배치-크기와-동적-배처)
  - [데이터 타입](#데이터-타입)
  - [리쉐이프](#리쉐이프)
  - [형상 텐서](#형상-텐서)
  - [비선형 I/O 형식](#비선형-io-형식)
  - [버전 정책](#버전-정책)
  - [인스턴스 그룹](#인스턴스-그룹)
    - [다중 모델 인스턴스](#다중-모델-인스턴스)
    - [CPU 모델 인스턴스](#cpu-모델-인스턴스)
    - [호스트 정책](#호스트-정책)
    - [속도 제한기 설정](#속도-제한기-설정)
      - [리소스](#리소스)
      - [우선순위](#우선순위)
    - [앙상블 모델 인스턴스 그룹](#앙상블-모델-인스턴스-그룹)
  - [CUDA 연산 능력](#cuda-연산-능력)
  - [최적화 정책](#최적화-정책)
  - [모델 웜업](#모델-웜업)
  - [응답 캐시](#응답-캐시)
  - [모델 매개변수](#모델-매개변수)
    - [앙상블 모델](#앙상블-모델)
    - [전처리 모델](#전처리-모델)
    - [후처리 모델](#후처리-모델)
    - [tensorrt_llm 모델](#tensorrt_llm-모델)
      - [디코딩 모드](#디코딩-모드)
      - [최적화](#최적화)
      - [스케줄링](#스케줄링)
      - [메두사](#메두사)
      - [이글](#이글)
    - [tensorrt_llm_bls 모델](#tensorrt_llm_bls-모델)
    - [모델 입력 및 출력](#모델-입력-및-출력)
      - [공통 입력](#공통-입력)
      - [공통 출력](#공통-출력)
      - [tensorrt_llm 모델의 고유 입력](#tensorrt_llm-모델의-고유-입력)
      - [tensorrt_llm 모델의 고유 출력](#tensorrt_llm-모델의-고유-출력)
      - [tensorrt_llm_bls 모델의 고유 입력](#tensorrt_llm_bls-모델의-고유-입력)
      - [tensorrt_llm_bls 모델의 고유 출력](#tensorrt_llm_bls-모델의-고유-출력)
- [Python 백엔드](#python-백엔드)
  - [auto_complete_config](#auto_complete_config)
  - [initialize](#initialize)
  - [execute](#execute)
    - [기본 모드](#기본-모드)
    - [오류 처리](#오류-처리)
    - [요청 취소 처리](#요청-취소-처리)
    - [분리 모드](#분리-모드)
      - [사용 사례](#사용-사례)
      - [비동기 실행](#비동기-실행)
    - [요청 재스케줄링](#요청-재스케줄링)
  - [`finalize`](#finalize)

## 모델 저장소

저장소 레이아웃은 다음과 같아야 합니다:

```plaintext
<model-repository-path>/
    <model-name>/
      [config.pbtxt]
      [<output-labels-file> …]
      [configs]/
        [<custom-config-file> …]
      <version>/
        <model-definition-file>
      <version>/
        <model-definition-file>
      …
    <model-name>/
      [config.pbtxt]
      [<output-labels-file> …]
      [configs]/
        [<custom-config-file> …]
      <version>/
        <model-definition-file>
      <version>/
        <model-definition-file>
      …
    …
```

최상위 모델 저장소 디렉토리 내에는 0개 이상의 하위 디렉토리가 있어야 합니다. 각 하위 디렉토리에는 해당 모델의 저장소 정보가 포함됩니다. `config.pbtxt` 파일은 해당 모델의 모델 구성을 설명합니다. 일부 모델의 경우 `config.pbtxt`가 필수이지만 다른 모델의 경우에는 선택 사항입니다. 자세한 내용은 자동 생성된 모델 구성을 참조하세요.

각 디렉토리에는 선택적으로 `configs` 하위 디렉토리가 포함될 수 있습니다. `configs` 디렉토리 내에는 `.pbtxt` 파일 확장자를 가진 0개 이상의 파일이 있어야 합니다. Triton에서 사용자 정의 모델 구성을 처리하는 방법에 대한 자세한 내용은 사용자 정의 모델 구성을 참조하세요.

각 디렉토리에는 모델의 버전을 나타내는 최소 하나의 숫자 하위 디렉토리가 있어야 합니다. Triton에서 모델 버전을 처리하는 방법에 대한 자세한 내용은 모델 버전을 참조하세요. 각 모델은 특정 백엔드에 의해 실행됩니다. 각 버전 하위 디렉토리 내에는 해당 백엔드에 필요한 파일이 있어야 합니다. 예를 들어, TensorRT, PyTorch, ONNX, OpenVINO 및 TensorFlow와 같은 프레임워크 백엔드를 사용하는 모델은 프레임워크별 모델 파일을 제공해야 합니다.

**작명 규칙**

- 알파벳(a-z, A-Z), 숫자(0-9), 언더스코어(`_`), 대시(`-`) 사용 가능
- 특수문자나 공백은 사용할 수 없음
- 대소문자 구분됨
- 시작 문자로 숫자는 권장되지 않음

### 지원 모델 형태

Triton Server은 아래 형태의 모델을 지원합니다.
지원하지 않는 형태의 모델을 로드하고 싶다면 모델을 변환하거나, Python Model로 구현해야 합니다.

#### TensorRT Models

TensorRT 모델 정의는 Plan이라고 합니다. TensorRT Plan은 기본적으로 `model.plan`이라는 이름을 가진 단일 파일이어야 합니다. 이 기본 이름은 모델 구성에서 `default_model_filename` 속성을 사용하여 재정의할 수 있습니다.

TensorRT Plan은 GPU의 [CUDA Compute Capability](https://developer.nvidia.com/cuda-gpus)에 따라 다릅니다. 따라서 TensorRT 모델은 모델 구성에서 cc_model_filenames 속성을 설정하여 각 Plan 파일을 해당하는 Compute Capability와 연결해야 합니다.

TensorRT 모델의 최소 모델 저장소는 다음과 같습니다:

```
<model-repository-path>/
    <model-name>/
      config.pbtxt
      1/
        model.plan
```

#### ONNX Models

ONNX 모델은 단일 파일이거나 여러 파일이 포함된 디렉토리입니다. 기본적으로 파일 또는 디렉토리의 이름은 `model.onnx`여야 합니다. 이 기본 이름은 모델 구성에서 `default_model_filename` 속성을 사용하여 재정의할 수 있습니다.

Triton은 Triton에서 사용하는 [ONNX Runtime](https://github.com/Microsoft/onnxruntime) 버전이 지원하는 모든 ONNX 모델을 지원합니다. [오래된 ONNX opset 버전](https://github.com/Microsoft/onnxruntime/blob/master/docs/Versioning.md#version-matrix)을 사용하거나 [지원되지 않는 타입의 연산자가 포함된](https://github.com/microsoft/onnxruntime/issues/1122) 모델은 지원되지 않습니다.

단일 파일로 구성된 ONNX 모델의 최소 모델 저장소는 다음과 같습니다:

```
<model-repository-path>/
    <model-name>/
      config.pbtxt
      1/
        model.onnx
```

여러 파일로 구성된 ONNX 모델은 디렉토리에 포함되어야 합니다. 기본적으로 이 디렉토리의 이름은 `model.onnx`여야 하지만 모델 구성에서 `default_model_filename` 속성을 사용하여 재정의할 수 있습니다. 이 디렉토리 내의 주 모델 파일은 `model.onnx`라는 이름을 가져야 합니다. 디렉토리에 포함된 ONNX 모델의 최소 모델 저장소는 다음과 같습니다:

```
<model-repository-path>/
    <model-name>/
      config.pbtxt
      1/
        model.onnx/
           model.onnx
           <other model files>
```

#### TorchScript Models

TorchScript 모델은 기본적으로 `model.pt`라는 이름을 가진 단일 파일이어야 합니다. 이 기본 이름은 [모델 구성](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/model_configuration.html)에서 `default_model_filename` 속성을 사용하여 재정의할 수 있습니다. PyTorch의 다른 버전으로 추적된 일부 모델은 기본 `opset`의 변경으로 인해 Triton에서 지원되지 않을 수 있습니다.

TorchScript 모델의 최소 모델 저장소는 다음과 같습니다:

```
<model-repository-path>/
    <model-name>/
      config.pbtxt
      1/
        model.pt
```

#### TensorFlow Models

TensorFlow는 GraphDef 또는 SavedModel의 두 가지 형식으로 모델을 저장합니다. Triton은 두 형식을 모두 지원합니다.

TensorFlow GraphDef는 기본적으로 `model.graphdef`라는 이름을 가진 단일 파일이어야 합니다. TensorFlow SavedModel은 여러 파일이 포함된 디렉토리입니다. 기본적으로 디렉토리의 이름은 `model.savedmodel`이어야 합니다. 이러한 기본 이름은 [모델 구성](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/model_configuration.html)에서 `default_model_filename` 속성을 사용하여 재정의할 수 있습니다.

TensorFlow GraphDef 모델의 최소 모델 저장소는 다음과 같습니다:

```
<model-repository-path>/
    <model-name>/
      config.pbtxt
      1/
        model.graphdef
```

TensorFlow SavedModel 모델의 최소 모델 저장소는 다음과 같습니다:

```
<model-repository-path>/
    <model-name>/
      config.pbtxt
      1/
        model.savedmodel/
           <saved-model files>
```

#### OpenVINO Models

OpenVINO 모델은 `*.xml`과 `*.bin` 두 파일로 표현됩니다. 기본적으로 `*.xml` 파일의 이름은 `model.xml`이어야 합니다. 이 기본 이름은 [모델 구성](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/model_configuration.html)에서 `default_model_filename` 속성을 사용하여 재정의할 수 있습니다.

OpenVINO 모델의 최소 모델 저장소는 다음과 같습니다:

```
<model-repository-path>/
    <model-name>/
      config.pbtxt
      1/
        model.xml
        model.bin
```

#### DALI Models

[DALI 백엔드](https://github.com/triton-inference-server/dali_backend)를 사용하면 Triton 내에서 [DALI 파이프라인](https://github.com/NVIDIA/DALI)을 모델로 실행할 수 있습니다. 이 백엔드를 사용하려면 기본적으로 `model.dali`라는 이름의 파일을 생성하여 모델 저장소에 포함해야 합니다. `model.dali`를 생성하는 방법에 대한 설명은 [DALI 백엔드 문서](https://github.com/triton-inference-server/dali_backend#how-to-use)를 참조하십시오. 기본 모델 파일 이름은 [모델 구성](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/model_configuration.html)에서 `default_model_filename` 속성을 사용하여 재정의할 수 있습니다.

DALI 모델의 최소 모델 저장소는 다음과 같습니다:

```
<model-repository-path>/
    <model-name>/
      config.pbtxt
      1/
        model.dali
```

#### Python Models

[Python 백엔드](https://github.com/triton-inference-server/python_backend)를 사용하면 Triton 내에서 Python 코드를 모델로 실행할 수 있습니다. 기본적으로 Python 스크립트의 이름은 `model.py`여야 하지만 이 기본 이름은 모델 구성에서 `default_model_filename` 속성을 사용하여 재정의할 수 있습니다.

Python 모델의 최소 모델 저장소는 다음과 같습니다:

```
<model-repository-path>/
    <model-name>/
      config.pbtxt
      1/
        model.py
```

## 모델 구성

> 원본: [Model Configuration — NVIDIA Triton Inference Server](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/model_configuration.html)

> 참조: [가이드](https://github.com/triton-inference-server/tutorials/tree/main/Conceptual_Guide/Part_1-model_deployment#model-configuration), [예제](https://github.com/triton-inference-server/tutorials/tree/main/HuggingFace#examples)

모델 저장소의 각 모델은 모델에 대한 필수 및 선택적 정보를 제공하는 모델 구성을 포함해야 합니다. 일반적으로 이 구성은 [ModelConfig protobuf](https://github.com/triton-inference-server/common/blob/main/protobuf/model_config.proto)로 지정된 `config.pbtxt` 파일에서 제공됩니다. 자동 생성된 모델 구성에서 설명된 일부 경우에는 모델 구성이 Triton에 의해 자동으로 생성될 수 있으므로 명시적으로 제공할 필요가 없습니다.

이 섹션에서는 가장 중요한 모델 구성 속성을 설명하지만 [ModelConfig protobuf](https://github.com/triton-inference-server/common/blob/main/protobuf/model_config.proto)의 문서도 참조해야 합니다.

### 최소 모델 구성

최소 모델 구성은 [platform 및/또는 backend 속성](https://github.com/triton-inference-server/backend/blob/main/README.md#backends), `max_batch_size` 속성, 그리고 모델의 입력과 출력 텐서를 지정해야 합니다.

예를 들어 두 개의 입력 input0와 input1, 그리고 하나의 출력 output0을 가진 TensorRT 모델을 살펴보겠습니다. 모든 텐서는 16개의 float32 항목으로 구성되어 있습니다. 최소 구성은 다음과 같습니다:

```yaml
platform: "tensorrt_plan"
max_batch_size: 8
input [
  {
    name: "input0"
    data_type: TYPE_FP32
    dims: [ 16 ]
  },
  {
    name: "input1"
    data_type: TYPE_FP32
    dims: [ 16 ]
  }
]
output [
  {
    name: "output0"
    data_type: TYPE_FP32
    dims: [ 16 ]
  }
]
```

#### 이름, 플랫폼 및 백엔드

모델 구성의 `name` 속성은 선택사항입니다. 모델의 이름이 구성에 지정되지 않은 경우, 모델이 포함된 모델 저장소 디렉토리와 동일한 것으로 간주됩니다. `name`이 지정된 경우 모델이 포함된 모델 저장소 디렉토리의 이름과 일치해야 합니다. `platform`과 `backend`에 필요한 값은 백엔드 문서에 설명되어 있습니다.

#### 모델 트랜잭션 정책

`model_transaction_policy` 속성은 ==모델에서 예상되는 트랜잭션의 특성을 설명==합니다.

`Decoupled`: 이 boolean 설정은 모델에서 생성된 응답이 요청과 [분리](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/decoupled_models.html)되어 있는지 여부를 나타냅니다. 분리형을 사용하면 모델이 생성하는 응답의 수가 발행된 요청의 수와 다를 수 있으며, 응답이 요청 순서와 다르게 순서가 바뀔 수 있습니다. 기본값은 `false`이며, 이는 모델이 각 요청에 대해 정확히 하나의 응답을 생성한다는 의미입니다.

#### 최대 배치 크기

`max_batch_size` 속성은 Triton이 활용할 수 있는 [배치 처리 유형](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/architecture.html#models-and-schedulers)에 대해 모델이 지원하는 최대 배치 크기를 나타냅니다. 모델의 배치 차원이 첫 번째 차원이고 모든 입력과 출력이 이 배치 차원을 가지고 있다면, Triton은 [동적 배치 처리기](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/model_configuration.html#dynamic-batcher) 또는 [시퀀스 배치 처리기](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/model_configuration.html#sequence-batcher)를 사용하여 모델과 함께 자동으로 배치 처리를 사용할 수 있습니다. 이 경우 `max_batch_size`는 Triton이 모델과 함께 사용해야 하는 최대 배치 크기를 나타내는 1 이상의 값으로 설정해야 합니다.

배치 처리를 지원하지 않거나 위에서 설명한 특정 방식으로 배치 처리를 지원하지 않는 모델의 경우 `max_batch_size`를 0으로 설정해야 합니다.

#### 입력과 출력

각 모델의 입력과 출력은 이름, 데이터 타입, 그리고 형태를 지정해야 합니다. 입력 또는 출력 텐서에 지정된 이름은 모델이 예상하는 이름과 일치해야 합니다.

##### PyTorch 백엔드를 위한 특별 규칙

**명명 규칙:**

TorchScript 모델 파일에서 입력/출력에 대한 충분한 메타데이터가 없기 때문에, 구성의 입력/출력의 "name" 속성은 특정 명명 규칙을 따라야 합니다. 자세한 내용은 다음과 같습니다.

1. [입력에만 해당] 입력이 텐서의 딕셔너리가 아닌 경우, 구성 파일의 입력 이름은 모델 정의에서 forward 함수의 입력 인수 이름을 반영해야 합니다.

예를 들어, Torchscript 모델의 forward 함수가 forward(self, input0, input1)로 정의된 경우, 첫 번째와 두 번째 입력은 각각 "input0"과 "input1"로 명명되어야 합니다.

2. `<name>__<index>`: 여기서 `<name>`은 임의의 문자열이고 `<index>`는 해당 입력/출력의 위치를 나타내는 정수 인덱스입니다.

이는 두 개의 입력과 두 개의 출력이 있는 경우, 첫 번째와 두 번째 입력은 "INPUT**0"과 "INPUT**1"로, 첫 번째와 두 번째 출력은 "OUTPUT**0"과 "OUTPUT**1"로 각각 명명될 수 있다는 의미입니다.

3. 모든 입력(또는 출력)이 동일한 명명 규칙을 따르지 않는 경우, 모델 구성에서 엄격한 순서를 적용합니다. 즉, 구성의 입력(또는 출력) 순서가 이러한 입력의 실제 순서라고 가정합니다.

**텐서의 딕셔너리를 입력으로 사용:**

PyTorch 백엔드는 텐서의 딕셔너리 형태로 모델에 입력을 전달하는 것을 지원합니다. 이는 문자열에서 텐서로의 매핑을 포함하는 딕셔너리 유형의 단일 입력이 있는 경우에만 지원됩니다. 예를 들어, 다음과 같은 형태의 입력을 기대하는 모델이 있다고 가정해보겠습니다:

```python
{'A': tensor1, 'B': tensor2}
```

이 경우 구성의 입력 이름은 위의 명명 규칙 `<name>__<index>`를 따르지 않아야 합니다. 대신, 이 경우의 입력 이름은 해당 특정 텐서의 문자열 값 'key'에 매핑되어야 합니다. 이 경우 입력은 "A"와 "B"가 되며, 여기서 입력 "A"는 tensor1에 해당하는 값을, "B"는 tensor2에 해당하는 값을 나타냅니다.

입력 및 출력 텐서에 허용되는 데이터 타입은 모델의 유형에 따라 다릅니다. [데이터타입](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/model_configuration.html#datatypes) 섹션에서는 허용되는 데이터 타입과 각 모델 유형의 데이터 타입에 매핑되는 방법을 설명합니다.

입력 형태는 모델이 예상하고 Triton이 추론 요청에서 예상하는 입력 텐서의 형태를 나타냅니다. 출력 형태는 모델이 생성하고 Triton이 추론 요청에 대한 응답으로 반환하는 출력 텐서의 형태를 나타냅니다. 입력과 출력 형태 모두 1 이상의 순위를 가져야 합니다. 즉, 빈 형태 `[ ]`는 허용되지 않습니다.

입력과 출력 형태는 `max_batch_size`와 입력 또는 출력 `dims` 속성으로 지정된 차원의 조합으로 지정됩니다. `max_batch_size`가 0보다 큰 모델의 경우, 전체 형태는 `[ -1 ] + dims`로 형성됩니다. `max_batch_size`가 0인 모델의 경우, 전체 형태는 `dims`로 형성됩니다. 예를 들어, 다음 구성에서 "input0"의 형태는 `[ -1, 16 ]`이고 "output0"의 형태는 `[ -1, 4 ]`입니다.

먼저 첫 번째 설정 예시를 보겠습니다:

```protobuf
platform: "tensorrt_plan"
max_batch_size: 8
input [
  {
    name: "input0"
    data_type: TYPE_FP32
    dims: [ 16 ]
  }
]
output [
  {
    name: "output0"
    data_type: TYPE_FP32
    dims: [ 4 ]
  }
]
```

max_batch_size가 0인 동일한 설정의 경우, "input0"의 shape는 `[16]`이고 "output0"의 shape는 `[4]`입니다:

```protobuf
platform: "tensorrt_plan"
max_batch_size: 0
input [
  {
    name: "input0"
    data_type: TYPE_FP32
    dims: [ 16 ]
  }
]
output [
  {
    name: "output0"
    data_type: TYPE_FP32
    dims: [ 4 ]
  }
]
```

가변 크기 차원을 가진 입력 및 출력 텐서를 지원하는 모델의 경우, 해당 차원을 입력 및 출력 설정에서 -1로 지정할 수 있습니다. 예를 들어, 모델이 첫 번째 차원은 크기가 4여야 하고 두 번째 차원은 임의의 크기일 수 있는 2차원 입력 텐서가 필요한 경우, 해당 입력의 모델 설정에는 `dims: [4, -1]`이 포함됩니다. 그러면 Triton은 해당 입력 텐서의 두 번째 차원이 0보다 크거나 같은 임의의 값인 추론 요청을 수락합니다. 모델 설정은 기본 모델이 허용하는 것보다 더 제한적일 수 있습니다. 예를 들어, 프레임워크 모델 자체는 두 번째 차원이 임의의 크기일 수 있도록 허용하더라도 모델 설정은 `dims: [4, 4]`로 지정될 수 있습니다. 이 경우 Triton은 입력 텐서의 shape가 정확히 `[4, 4]`인 추론 요청만 수락합니다.

Triton이 추론 요청에서 받는 입력 `shape`와 모델이 예상하는 입력 `shape` 사이에 불일치가 있는 경우 `reshape` 속성을 사용해야 합니다. 마찬가지로 모델이 생성하는 출력 `shape`와 Triton이 추론 요청에 대한 응답으로 반환하는 `shape` 사이에 불일치가 있는 경우에도 `reshape` 속성을 사용해야 합니다.

모델 입력은 입력이 ragged input임을 나타내기 위해 `allow_ragged_batch`를 지정할 수 있습니다. 이 필드는 dynamic batcher와 함께 사용되어 모든 요청에서 동일한 shape를 강제하지 않고도 배치 처리를 허용합니다.

### 자동 생성 모델 설정

Triton에 배포될 각 모델에는 필수 설정이 포함된 모델 설정 파일이 있어야 합니다. 경우에 따라 Triton이 모델 설정의 필수 부분을 자동으로 생성할 수 있습니다. ==필수 모델 설정은 최소 모델 설정에 표시된 설정==입니다. 기본적으로 Triton은 이러한 섹션을 완성하려고 시도합니다. 그러나 `--disable-auto-complete-config` 옵션으로 Triton을 시작하면 백엔드 측에서 모델 설정을 자동 완성하지 않도록 설정할 수 있습니다. 단, 이 옵션을 사용하더라도 Triton은 누락된 `instance_group` 설정을 기본값으로 채웁니다.

Triton은 대부분의 TensorRT, TensorFlow saved-model, ONNX 모델 및 OpenVINO 모델에 대해 모든 필수 설정을 자동으로 도출할 수 있습니다. Python 모델의 경우, `max_batch_size`, `input` 및 `output` 속성을 `set_max_batch_size`, `add_input` 및 `add_output` 함수를 사용하여 제공하기 위해 Python 백엔드에서 `auto_complete_config` 함수를 구현할 수 있습니다. 이러한 속성을 통해 Triton은 설정 파일이 없는 상태에서 최소 모델 설정으로 Python 모델을 로드할 수 있습니다. 다른 모든 모델 유형은 반드시 모델 설정 파일을 제공해야 합니다.

사용자 정의 백엔드를 개발할 때는 설정에 필수 설정을 채우고 `TRITONBACKEND_ModelSetConfig` API를 호출하여 완성된 설정을 Triton 코어로 업데이트할 수 있습니다. 이를 달성하는 방법의 예시로 TensorFlow와 Onnxruntime 백엔드를 참조할 수 있습니다. 현재는 `inputs`, `outputs`, `max_batch_size` 및 dynamic batching 설정만 백엔드에서 채울 수 있습니다. 사용자 정의 백엔드의 경우 `config.pbtxt` 파일에 backend 필드가 포함되어 있거나 모델 이름이 `<model_name>.<backend_name>` 형식이어야 합니다.

또한 모델 설정 엔드포인트를 사용하여 Triton이 모델에 대해 생성한 모델 설정을 볼 수 있습니다. 이를 수행하는 가장 쉬운 방법은 curl과 같은 유틸리티를 사용하는 것입니다:

```bash
$ curl localhost:8000/v2/models/<model name>/config
```

이는 생성된 모델 설정의 JSON 표현을 반환합니다. 여기에서 JSON의 `max_batch_size`, `inputs` 및 `outputs` 섹션을 가져와 `config.pbtxt` 파일로 변환할 수 있습니다. Triton은 최소 모델 설정 부분만 생성합니다. `config.pbtxt` 파일을 편집하여 모델 설정의 선택적 부분을 직접 제공해야 합니다.

### 사용자 정의 모델 설정

때로는 여러 장치에서 실행되는 Triton 인스턴스가 하나의 모델 저장소를 공유할 때, 최상의 성능을 달성하기 위해 각 플랫폼에서 모델을 다르게 설정해야 할 필요가 있습니다. Triton은 사용자가 `--model-config-name` 옵션을 설정하여 사용자 정의 모델 설정 이름을 선택할 수 있도록 합니다.

예를 들어, `./tritonserver --model-repository=</path/to/model/repository> --model-config-name=h100`을 실행할 때, 서버는 로드되는 각 모델에 대해 `/path/to/model/repository/<model-name>/configs` 디렉토리에서 사용자 정의 설정 파일 `h100.pbtxt`를 검색합니다. `h100.pbtxt`가 존재하면 이 모델의 설정으로 사용됩니다. 그렇지 않으면 설정에 따라 기본 설정 `/path/to/model/repository/<model-name>/config.pbtxt` 또는 자동 생성된 모델 설정이 선택됩니다.

사용자 정의 모델 설정은 Explicit 및 Poll 모델 제어 모드에서도 작동합니다. 사용자는 새로운 사용자 정의 설정을 삭제하거나 추가할 수 있으며, 서버는 로드된 각 모델에 대한 설정 파일을 동적으로 선택합니다.

> 사용자 정의 모델 설정 이름에는 공백 문자가 포함되어서는 안 됩니다.

예제 1: `–model-config-name=h100`

```
.
└── model_repository/
    ├── model_a/
    │   ├── configs/
    │   │   ├── v100.pbtxt
    │   │   └── **h100.pbtxt**
    │   └── config.pbtxt
    ├── model_b/
    │   ├── configs/
    │   │   └── v100.pbtxt
    │   └── **config.pbtxt**
    └── model_c/
        ├── configs/
        │   └── config.pbtxt
        └── **config.pbtxt**
```

예제 2: `–model-config-name=config`

```
.
└── model_repository/
    ├── model_a/
    │   ├── configs/
    │   │   ├── v100.pbtxt
    │   │   └── h100.pbtxt
    │   └── **config.pbtxt**
    ├── model_b/
    │   ├── configs/
    │   │   └── v100.pbtxt
    │   └── **config.pbtxt**
    └── model_c/
        ├── configs/
        │   └── **config.pbtxt**
        └── config.pbtxt
```

예제 3: `–model-config-name`이 설정되지 않은 경우

```
.
└── model_repository/
    ├── model_a/
    │   ├── configs/
    │   │   ├── v100.pbtxt
    │   │   └── h100.pbtxt
    │   └── **config.pbtxt**
    ├── model_b/
    │   ├── configs/
    │   │   └── v100.pbtxt
    │   └── **config.pbtxt**
    └── model_c/
        ├── configs/
        │   └── config.pbtxt
        └── **config.pbtxt**
```

#### 기본 최대 배치 크기와 동적 배처

모델이 자동 완성 기능을 사용할 때, `--backend-config=default-max-batch-size=<int>` 명령줄 인수를 사용하여 기본 최대 배치 크기를 설정할 수 있습니다. 이를 통해 배칭이 가능하고 자동 생성된 모델 구성을 사용하는 모든 모델에 대해 기본 최대 배치 크기를 설정할 수 있습니다. 이 값은 기본적으로 4로 설정됩니다. 백엔드 개발자는 `TRITONBACKEND_BackendConfig` API에서 이 기본 최대 배치 크기를 가져와 사용할 수 있습니다. 현재 이러한 기본 배치 값을 활용하고 생성된 모델 구성에서 동적 배칭을 활성화하는 백엔드는 다음과 같습니다:

1. TensorFlow 백엔드 (https://github.com/triton-inference-server/tensorflow_backend)
2. Onnxruntime 백엔드 (https://github.com/triton-inference-server/onnxruntime_backend)
3. TensorRT 백엔드 (https://github.com/triton-inference-server/tensorrt_backend)
   - TensorRT 모델은 최대 배치 크기를 명시적으로 저장하며 기본 최대 배치 크기 매개변수를 사용하지 않습니다. 그러나 max_batch_size > 1이고 스케줄러가 제공되지 않은 경우 동적 배치 스케줄러가 활성화됩니다.

모델에 대해 1보다 큰 최대 배치 크기 값이 설정되고 구성 파일에 스케줄러가 제공되지 않은 경우 `dynamic_batching` 구성이 설정됩니다.

### 데이터 타입

아래 표는 Triton에서 지원하는 텐서 데이터 타입을 보여줍니다. 첫 번째 열은 모델 구성 파일에 나타나는 데이터 타입의 이름을 보여줍니다. 다음 네 개의 열은 지원되는 모델 프레임워크에 대한 해당 데이터 타입을 보여줍니다. 모델 프레임워크에 특정 데이터 타입에 대한 항목이 없는 경우, Triton은 해당 모델에 대해 해당 데이터 타입을 지원하지 않습니다. "API"라고 표시된 여섯 번째 열은 TRITONSERVER C API, TRITONBACKEND C API, HTTP/REST 프로토콜 및 GRPC 프로토콜에 대한 해당 데이터 타입을 보여줍니다. 마지막 열은 Python numpy 라이브러리에 대한 해당 데이터 타입을 보여줍니다.

| Model Config | TensorRT | TensorFlow | ONNX Runtime | PyTorch | API    | NumPy         |
| ------------ | -------- | ---------- | ------------ | ------- | ------ | ------------- |
| TYPE_BOOL    | kBOOL    | DT_BOOL    | BOOL         | kBool   | BOOL   | bool          |
| TYPE_UINT8   | kUINT8   | DT_UINT8   | UINT8        | kByte   | UINT8  | uint8         |
| TYPE_UINT16  |          | DT_UINT16  | UINT16       |         | UINT16 | uint16        |
| TYPE_UINT32  |          | DT_UINT32  | UINT32       |         | UINT32 | uint32        |
| TYPE_UINT64  |          | DT_UINT64  | UINT64       |         | UINT64 | uint64        |
| TYPE_INT8    | kINT8    | DT_INT8    | INT8         | kChar   | INT8   | int8          |
| TYPE_INT16   |          | DT_INT16   | INT16        | kShort  | INT16  | int16         |
| TYPE_INT32   | kINT32   | DT_INT32   | INT32        | kInt    | INT32  | int32         |
| TYPE_INT64   | kINT64   | DT_INT64   | INT64        | kLong   | INT64  | int64         |
| TYPE_FP16    | kHALF    | DT_HALF    | FLOAT16      |         | FP16   | float16       |
| TYPE_FP32    | kFLOAT   | DT_FLOAT   | FLOAT        | kFloat  | FP32   | float32       |
| TYPE_FP64    |          | DT_DOUBLE  | DOUBLE       | kDouble | FP64   | float64       |
| TYPE_STRING  |          | DT_STRING  | STRING       |         | BYTES  | dtype(object) |
| TYPE_BF16    | kBF16    |            |              |         | BF16   |               |

TensorRT의 경우 각 값은 `nvinfer1::DataType` 네임스페이스에 있습니다. 예를 들어, `nvinfer1::DataType::kFLOAT`는 32비트 부동 소수점 데이터 타입입니다.

TensorFlow의 경우 각 값은 tensorflow 네임스페이스에 있습니다. 예를 들어, `tensorflow::DT_FLOAT`는 32비트 부동 소수점 값입니다.

ONNX Runtime의 경우 각 값 앞에 `ONNX_TENSOR_ELEMENT_DATA_TYPE_`가 붙습니다. 예를 들어, `ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT`는 32비트 부동 소수점 데이터 타입입니다.

PyTorch의 경우 각 값은 torch 네임스페이스에 있습니다. 예를 들어, `torch::kFloat`는 32비트 부동 소수점 데이터 타입입니다.

Numpy의 경우 각 값은 numpy 모듈에 있습니다. 예를 들어, `numpy.float32`는 32비트 부동 소수점 데이터 타입입니다.

### 리쉐이프

모델 구성의 입력 또는 출력에 대한 `ModelTensorReshape` 속성은 추론 API가 받아들이는 입력 또는 출력 형태가 기본 프레임워크 모델이나 커스텀 백엔드가 예상하거나 생성하는 입력 또는 출력 형태와 다르다는 것을 나타냅니다.

입력의 경우, `reshape`를 사용하여 입력 텐서를 프레임워크나 백엔드가 예상하는 다른 형태로 변형할 수 있습니다. 일반적인 사용 사례는 배칭을 지원하는 모델이 배치된 입력이 `[batch-size]` 형태를 가질 것으로 예상하는 경우입니다. 이는 배치 차원이 형태를 완전히 설명한다는 것을 의미합니다. 추론 API의 경우 각 입력이 비어있지 않은 `dims`를 지정해야 하므로 동등한 형태 `[batch-size, 1]`을 지정해야 합니다. 이 경우 입력은 다음과 같이 지정해야 합니다:

```
input [
    {
      name: "in"
      dims: [ 1 ]
      reshape: { shape: [ ] }
    }
```

출력의 경우, `reshape`를 사용하여 프레임워크나 백엔드가 생성한 출력 텐서를 추론 API가 반환하는 다른 형태로 변형할 수 있습니다. 일반적인 사용 사례는 배칭을 지원하는 모델이 배치된 출력이 `[batch-size]` 형태를 가질 것으로 예상하는 경우입니다. 추론 API의 경우 각 출력이 비어있지 않은 `dims`를 지정해야 하므로 동등한 형태 `[batch-size, 1]`을 지정해야 합니다. 이 경우 출력은 다음과 같이 지정해야 합니다:

```
output [
    {
      name: "in"
      dims: [ 1 ]
      reshape: { shape: [ ] }
    }
```

### 형상 텐서

형상 텐서를 지원하는 모델의 경우, 형상 텐서 역할을 하는 입력과 출력에 대해 `is_shape_tensor` 속성을 적절히 설정해야 합니다. 다음은 형상 텐서를 지정하는 예시 구성입니다.

```yaml
name: "myshapetensormodel"
platform: "tensorrt_plan"
max_batch_size: 8
input [
  {
    name: "input0"
    data_type: TYPE_FP32
    dims: [ 1 , 3]
  },
  {
    name: "input1"
    data_type: TYPE_INT32
    dims: [ 2 ]
    is_shape_tensor: true
  }
]
output [
  {
    name: "output0"
    data_type: TYPE_FP32
    dims: [ 1 , 3]
  }
]
```

위에서 설명한 것처럼, Triton은 입력 또는 출력 텐서 `dims`에 나열되지 않은 첫 번째 차원을 따라 배치가 발생한다고 가정합니다. 하지만 형상 텐서의 경우, 배치는 첫 번째 형상 값에서 발생합니다. 위 예시에서 추론 요청은 다음과 같은 형태의 입력을 제공해야 합니다.

```
"input0": [ x, 1, 3]
"input1": [ 3 ]
"output0": [ x, 1, 3]
```

여기서 x는 요청의 배치 크기입니다. Triton은 배치를 사용할 때 모델에서 형상 텐서를 형상 텐서로 표시하도록 요구합니다. "input1"의 형상이 모델 구성에 설명된 `[2]`가 아닌 `[3]`임에 주목하세요. myshapetensormodel이 배치 모델이므로 배치 크기를 추가 값으로 제공해야 합니다. Triton은 모델에 요청을 보내기 전에 배치 차원에서 "input1"에 대한 모든 형상 값을 누적합니다.

예를 들어, 클라이언트가 다음과 같은 세 가지 요청을 Triton에 보낸다고 가정해보겠습니다:

```
Request1:
input0: [[[1,2,3]]] <== 이 텐서의 형상 [1,1,3]
input1: [1,4,6] <== 이 텐서의 형상 [3]

Request2:
input0: [[[4,5,6]], [[7,8,9]]] <== 이 텐서의 형상 [2,1,3]
input1: [2,4,6] <== 이 텐서의 형상 [3]

Request3:
input0: [[[10,11,12]]] <== 이 텐서의 형상 [1,1,3]
input1: [1,4,6] <== 이 텐서의 형상 [3]
```

이러한 요청들이 함께 배치되면 다음과 같이 모델에 전달됩니다:

모델에 대한 배치된 요청:

```
input0: [[[1,2,3]], [[4,5,6]], [[7,8,9]], [[10,11,12]]] <== 이 텐서의 형상 [4,1,3]
input1: [4, 4, 6] <== 이 텐서의 형상 [3]
```

현재는 TensorRT만 형상 텐서를 지원합니다. 형상 텐서에 대해 더 자세히 알아보려면 [Shape Tensor I/O](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#shape_tensor_io)를 참조하세요.

### 비선형 I/O 형식

비선형 형식으로 입력 또는 출력 데이터를 처리하는 모델의 경우, `is_non_linear_format_io` 속성을 설정해야 합니다. 다음 예시 모델 구성은 INPUT0과 INPUT1이 비선형 I/O 데이터 형식을 사용하도록 지정하는 방법을 보여줍니다.

```yaml
name: "mytensorrtmodel"
platform: "tensorrt_plan"
max_batch_size: 8
input [
  {
    name: "INPUT0"
    data_type: TYPE_FP16
    dims: [ 3,224,224 ]
    is_non_linear_format_io: true
  },
  {
    name: "INPUT1"
    data_type: TYPE_FP16
    dims: [ 3,224,224 ]
    is_non_linear_format_io: true
  }
]
output [
  {
    name: "OUTPUT0"
    data_type: TYPE_FP16
    dims: [ 1,3 ]
   }
]
```

현재는 TensorRT만 이 속성을 지원합니다. I/O 형식에 대해 자세히 알아보려면 [I/O Formats documentation](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#reformat-free-network-tensors)을 참조하세요.

### 버전 정책

각 모델은 하나 이상의 버전을 가질 수 있습니다. 각 버전은 자체적으로 숫자로 된 이름의 하위 디렉토리에 저장되며, 하위 디렉토리의 이름은 모델의 버전 번호에 해당합니다. 숫자로 된 이름이 아니거나 0으로 시작하는 이름을 가진 하위 디렉토리는 무시됩니다. 각 모델 구성은 주어진 시간에 Triton이 모델 저장소에서 사용할 수 있는 버전을 제어하는 버전 정책을 지정합니다.

모델 구성의 `ModelVersionPolicy` 속성은 다음 정책 중 하나를 설정하는 데 사용됩니다.

- `All`: 모델 저장소에서 사용 가능한 모든 버전의 모델을 추론에 사용할 수 있습니다. `version_policy: { all: {}}`

- `Latest`: 저장소에 있는 최신 'n'개 버전의 모델만 추론에 사용할 수 있습니다. 모델의 최신 버전은 숫자적으로 가장 큰 버전 번호입니다. `version_policy: { latest: { num_versions: 2}}`

- `Specific`: 명시적으로 나열된 버전의 모델만 추론에 사용할 수 있습니다. `version_policy: { specific: { versions: [1,3]}}`

버전 정책이 지정되지 않은 경우, Latest(n=1)가 기본값으로 사용되어 모델의 가장 최신 버전만 Triton에서 사용할 수 있게 됩니다. 모든 경우에서 모델 저장소에서 버전 하위 디렉토리를 추가하거나 제거하면 후속 추론 요청에 사용되는 모델 버전이 변경될 수 있습니다.

다음 구성은 모델의 모든 버전을 서버에서 사용할 수 있도록 지정합니다.

```yaml
platform: "tensorrt_plan"
max_batch_size: 8
input [
  {
    name: "input0"
    data_type: TYPE_FP32
    dims: [ 16 ]
  },
  {
    name: "input1"
    data_type: TYPE_FP32
    dims: [ 16 ]
  }
]
output [
  {
    name: "output0"
    data_type: TYPE_FP32
    dims: [ 16 ]
  }
]
version_policy: { all { }}
```

### 인스턴스 그룹

Triton은 해당 모델에 대한 여러 추론 요청을 동시에 처리할 수 있도록 모델의 여러 인스턴스를 제공할 수 있습니다. 모델 구성 `ModelInstanceGroup` 속성은 사용 가능하게 만들어야 하는 실행 인스턴스의 수와 이러한 인스턴스에 사용해야 하는 컴퓨팅 리소스를 지정하는 데 사용됩니다.

#### 다중 모델 인스턴스

기본적으로 시스템에서 사용 가능한 각 GPU에 대해 모델의 단일 실행 인스턴스가 생성됩니다. 인스턴스 그룹 설정을 사용하여 모든 GPU 또는 특정 GPU에만 모델의 여러 실행 인스턴스를 배치할 수 있습니다. 예를 들어, 다음 구성은 각 시스템 GPU에서 모델의 두 실행 인스턴스를 사용할 수 있도록 합니다.

```yaml
instance_group [
  {
    count: 2
    kind: KIND_GPU
  }
]
```

그리고 다음 구성은 GPU 0에 하나의 실행 인스턴스를 배치하고 GPU 1과 2에 두 개의 실행 인스턴스를 배치합니다.

```yaml
instance_group [
  {
    count: 1
    kind: KIND_GPU
    gpus: [ 0 ]
  },
  {
    count: 2
    kind: KIND_GPU
    gpus: [ 1, 2 ]
  }
]
```

인스턴스 그룹 사용에 대한 더 자세한 예시는 [이 가이드](https://github.com/triton-inference-server/tutorials/tree/main/Conceptual_Guide/Part_2-improving_resource_utilization#concurrent-model-execution)를 참조하세요.

#### CPU 모델 인스턴스

인스턴스 그룹 설정은 CPU에서 모델 실행을 활성화하는 데도 사용됩니다. 시스템에 GPU가 있더라도 모델을 CPU에서 실행할 수 있습니다. 다음은 CPU에 두 개의 실행 인스턴스를 배치합니다.

```yaml
instance_group [
  {
    count: 2
    kind: KIND_CPU
  }
]
```

KIND_CPU 인스턴스 그룹에 대해 count가 지정되지 않은 경우, 선택된 백엔드(Tensorflow 및 Onnxruntime)의 기본 인스턴스 수는 2가 됩니다. 다른 모든 백엔드는 기본값이 1이 됩니다.

#### 호스트 정책

인스턴스 그룹 설정은 호스트 정책과 연관되어 있습니다. 다음 설정은 인스턴스 그룹 설정으로 생성된 모든 인스턴스를 "`policy_0`" 호스트 정책과 연결합니다. 기본적으로 호스트 정책은 인스턴스의 디바이스 종류에 따라 설정되며, 예를 들어 `KIND_CPU`는 "`cpu`", `KIND_MODEL`은 "`model`", `KIND_GPU`는 "`gpu_<gpu_id>`"로 설정됩니다.

```
instance_group [
  {
    count: 2
    kind: KIND_CPU
    host_policy: "policy_0"
  }
]
```

#### 속도 제한기 설정

인스턴스 그룹은 선택적으로 속도 제한기 설정을 지정할 수 있으며, 이는 그룹 내 인스턴스들에 대한 속도 제한기의 작동 방식을 제어합니다. 속도 제한이 꺼져 있으면 속도 제한기 설정은 무시됩니다. 속도 제한이 켜져 있고 `instance_group`이 이 설정을 제공하지 않으면, 이 그룹에 속한 모델 인스턴스의 실행은 속도 제한기에 의해 어떤 방식으로도 제한되지 않습니다. 설정에는 다음과 같은 사항들이 포함됩니다:

##### 리소스

모델 인스턴스를 실행하는 데 필요한 리소스 집합입니다. "`name`" 필드는 리소스를 식별하고 "`count`" 필드는 그룹 내 모델 인스턴스가 실행하는 데 필요한 리소스의 복사본 수를 나타냅니다. "`global`" 필드는 리소스가 디바이스별인지 또는 시스템 전체에서 공유되는지를 지정합니다. 로드된 모델은 동일한 이름의 리소스를 전역과 비전역 모두로 지정할 수 없습니다. 리소스가 제공되지 않으면 triton은 모델 인스턴스 실행에 리소스가 필요하지 않다고 가정하고 모델 인스턴스가 사용 가능해지는 즉시 실행을 시작합니다.

##### 우선순위

우선순위는 모든 모델의 모든 인스턴스들 간의 우선순위를 지정하는 데 사용되는 가중치 값입니다. 우선순위가 2인 인스턴스는 우선순위가 1인 인스턴스에 비해 1/2의 스케줄링 기회를 받게 됩니다.

다음 예시는 그룹 내 인스턴스들이 실행을 위해 4개의 "R1"과 2개의 "R2" 리소스를 필요로 함을 지정합니다. "R2" 리소스는 전역 리소스입니다. 추가로, `instance_group`의 속도 제한기 우선순위는 2입니다.

```
instance_group [
  {
    count: 1
    kind: KIND_GPU
    gpus: [ 0, 1, 2 ]
    rate_limiter {
      resources [
        {
          name: "R1"
          count: 4
        },
        {
          name: "R2"
          global: True
          count: 2
        }
      ]
      priority: 2
    }
  }
]
```

위의 설정은 각 디바이스(0, 1, 2)에 하나씩, 총 3개의 모델 인스턴스를 생성합니다. 세 인스턴스는 "R1"이 자신의 디바이스에 대해 로컬이기 때문에 서로 간에 "R1"을 두고 경쟁하지 않습니다. 하지만 "R2"는 시스템 전체에서 공유되는 전역 리소스로 지정되었기 때문에 "R2"를 두고는 경쟁합니다. 이 인스턴스들은 서로 간에 "R1"을 두고 경쟁하지는 않지만, 리소스 요구사항에 "R1"을 포함하고 동일한 디바이스에서 실행되는 다른 모델 인스턴스들과는 "R1"을 두고 경쟁하게 됩니다.

#### 앙상블 모델 인스턴스 그룹

[앙상블 모델](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/architecture.html#ensemble-models)은 Triton이 사용자 정의 모델 파이프라인을 실행하는 데 사용하는 추상화입니다. 앙상블 모델과 연관된 물리적 인스턴스가 없기 때문에, `instance_group` 필드를 지정할 수 없습니다.

그러나 앙상블을 구성하는 각 모델은 자체 설정 파일에서 `instance_group`을 지정할 수 있으며, 앙상블이 여러 요청을 받을 때 위에서 설명한 대로 개별적으로 병렬 실행을 지원할 수 있습니다.

### CUDA 연산 능력

`default_model_filename` 필드와 유사하게, 선택적으로 `cc_model_filenames` 필드를 지정하여 모델 로드 시 GPU의 [CUDA 연산 능력](https://developer.nvidia.com/cuda-gpus)을 해당하는 모델 파일 이름에 매핑할 수 있습니다. 이는 특히 TensorRT 모델에 유용한데, 일반적으로 특정 연산 능력에 연결되어 있기 때문입니다.

```
cc_model_filenames [
  {
    key: "7.5"
    value: "resnet50_T4.plan"
  },
  {
    key: "8.0"
    value: "resnet50_A100.plan"
  }
]
```

### 최적화 정책

모델 구성의 `ModelOptimizationPolicy` 속성은 모델의 최적화 및 우선순위 설정을 지정하는 데 사용됩니다. 이러한 설정은 백엔드가 모델을 최적화하는 방법과 Triton이 모델을 스케줄링하고 실행하는 방법을 제어합니다. 현재 사용 가능한 설정에 대해서는 [ModelConfig protobuf](https://github.com/triton-inference-server/common/blob/main/protobuf/model_config.proto)와 [optimization](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/optimization.html) 문서를 참조하세요.

### 모델 웜업

Triton이 모델을 로드할 때 해당 모델에 대한 [백엔드](https://github.com/triton-inference-server/backend/blob/main/README.md)가 초기화됩니다. 일부 백엔드의 경우, 이러한 초기화의 일부 또는 전체가 모델이 첫 번째 추론 요청(또는 처음 몇 개의 추론 요청)을 받을 때까지 지연됩니다. 결과적으로, 지연된 초기화로 인해 처음(몇 개)의 추론 요청이 상당히 느릴 수 있습니다.

이러한 초기의 느린 추론 요청을 피하기 위해, Triton은 첫 번째 추론 요청을 받기 전에 모델이 완전히 초기화되도록 "웜업"할 수 있는 구성 옵션을 제공합니다. 모델 구성에서 `ModelWarmup` 속성이 정의되면, Triton은 모델 웜업이 완료될 때까지 모델이 추론할 준비가 되었다고 표시하지 않습니다.

모델 구성의 `ModelWarmup`은 모델의 웜업 설정을 지정하는 데 사용됩니다. 이 설정은 Triton이 각 모델 인스턴스를 웜업하기 위해 생성할 일련의 추론 요청을 정의합니다. 모델 인스턴스는 요청을 성공적으로 완료한 경우에만 서비스됩니다. 모델 웜업의 효과는 프레임워크 백엔드에 따라 다르며, Triton이 모델 업데이트에 덜 반응하게 만들 수 있으므로, 사용자는 ==자신의 필요에 맞는 구성을 실험하고 선택==해야 합니다. 현재 사용 가능한 설정에 대해서는 [ModelWarmup protobuf](https://github.com/triton-inference-server/common/blob/main/protobuf/model_config.proto) 문서를 참조하고, 다양한 웜업 샘플 지정에 대한 예시는 [L0_warmup](https://github.com/triton-inference-server/server/blob/main/qa/L0_warmup/test.sh)을 참조하세요.

### 응답 캐시

모델 구성의 `response_cache` 섹션에는 이 모델에 대한 응답 캐시를 활성화하는 데 사용되는 `enable` 불리언이 있습니다.

```json
response_cache {
  enable: true
}
```

모델 구성에서 캐시를 활성화하는 것 외에도, 서버 측에서 캐싱을 활성화하려면 서버를 시작할 때 `--cache-config`를 지정해야 합니다. 서버 측 캐싱 활성화에 대한 자세한 내용은 [Response Cache](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/response_cache.html) 문서를 참조하세요.

### 모델 매개변수

다음 표는 [all_models/inflight_batcher_llm](https://github.com/triton-inference-server/tensorrtllm_backend/blob/main/all_models/inflight_batcher_llm)의 모델들의 `config.pbtxt`에서 배포 전에 수정할 수 있는 매개변수들을 보여줍니다. 최적의 성능이나 사용자 지정 매개변수를 위해서는 [perf_best_practices](https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/performance/perf-best-practices.md)를 참조하세요.

아래 나열된 매개변수의 이름은 [`fill_template.py`](https://github.com/triton-inference-server/tensorrtllm_backend/blob/main/tools/fill_template.py) 스크립트를 사용하여 수정할 수 있는 `config.pbtxt`의 값입니다.

> 쉼표가 값으로 포함된 필드(예: `gpu_device_ids`, `participant_ids`)의 경우, 백슬래시로 쉼표를 이스케이프해야 합니다. 예를 들어, `gpu_device_ids`를 `0,1`로 설정하려면 `python3 fill_template.py -i config.pbtxt "gpu_device_ids:0\,1"`을 실행해야 합니다.

필수 매개변수는 모델 실행을 위해 반드시 설정되어야 합니다. 선택적 매개변수는 필수는 아니지만 모델을 사용자 지정하기 위해 설정할 수 있습니다.

> **모델 구성을 위한 몇 가지 팁 (Some tips for model configuration)**
>
> 다음은 최적의 성능을 위한 모델 구성 팁입니다. 이러한 권장 사항은 실험을 바탕으로 하며 모든 사용 사례에 적용되지 않을 수 있습니다. 다른 매개변수에 대한 지침은 [perf_best_practices](https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/performance/perf-best-practices.md)를 참조하세요.
>
> - **인플라이트 배칭을 더 잘 활용하기 위한 모델의 `instance_count` 설정**
>
>   config.pbtxt 파일의 `instance_count` 매개변수는 실행할 모델 인스턴스의 수를 지정합니다. 이상적으로는 TRT 엔진이 지원하는 최대 배치 크기와 일치하도록 설정해야 합니다. 이는 동시 요청 실행을 허용하고 성능 병목 현상을 줄이기 때문입니다. 하지만 더 많은 CPU 메모리 리소스를 소비하게 됩니다. 최적값은 미리 결정할 수 없지만, 일반적으로 1과 같은 매우 작은 값으로 설정해서는 안 됩니다. 대부분의 사용 사례에서 `instance_count`를 5로 설정하는 것이 다양한 워크로드에서 잘 작동한다는 것을 실험을 통해 확인했습니다.
>
> - **인플라이트 배칭을 최적화하기 위한 `max_batch_size`와 `max_num_tokens` 조정** > `max_batch_size`와 `max_num_tokens`는 인플라이트 배칭을 최적화하는 데 중요한 매개변수입니다. 모델 구성 파일에서 `max_batch_size`를 수정할 수 있으며, `max_num_tokens`는 `trtllm-build` 명령을 사용하여 TRT-LLM 엔진으로 변환하는 동안 설정됩니다. 다양한 시나리오에 대해 이러한 매개변수를 조정하는 것이 필요하며, 현재로서는 실험이 최적값을 찾는 가장 좋은 방법입니다. 일반적으로 총 요청 수는 `max_batch_size`보다 작아야 하고, 총 토큰 수는 `max_num_tokens`보다 작아야 합니다.

#### 앙상블 모델

앙상블 모델에 대해 더 자세히 알아보려면 [여기](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/architecture.html#ensemble-models)를 참조하세요.

**필수 매개변수**

| 이름                    | 설명                                                                                                                                                                                                                                                                                                                                              |
| ----------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `triton_max_batch_size` | Triton 모델 인스턴스가 실행될 최대 배치 크기입니다. `tensorrt_llm` 모델의 경우 실제 런타임 배치 크기는 `triton_max_batch_size`보다 클 수 있습니다. 런타임 배치 크기는 큐에서 사용 가능한 요청 수와 엔진 빌드 `trtllm-build` 매개변수(`max_num_tokens` 및 `max_batch_size` 등)와 같은 여러 매개변수를 기반으로 TRT-LLM 스케줄러에 의해 결정됩니다. |
| `logits_datatype`       | 컨텍스트 및 생성 로짓의 데이터 타입입니다.                                                                                                                                                                                                                                                                                                        |

#### 전처리 모델

**필수 매개변수**

| 이름                           | 설명                                                     |
| ------------------------------ | -------------------------------------------------------- |
| `triton_max_batch_size`        | Triton이 모델과 함께 사용해야 하는 최대 배치 크기입니다. |
| `tokenizer_dir`                | 모델의 토크나이저 경로입니다.                            |
| `preprocessing_instance_count` | 실행할 모델 인스턴스의 수입니다.                         |

**선택적 매개변수**

| 이름                 | 설명                                                                                                                                                                                         |
| -------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `add_special_tokens` | [HF 토크나이저](https://huggingface.co/transformers/v2.11.0/main_classes/tokenizer.html#transformers.PreTrainedTokenizer.add_special_tokens)에서 사용되는 `add_special_tokens` 플래그입니다. |
| `visual_model_path`  | 멀티모달 워크플로우에서 사용되는 비전 엔진 경로입니다.                                                                                                                                       |
| `engine_dir`         | 모델의 엔진 경로입니다. 이 매개변수는 멀티모달 처리에서 `fake_prompt_id` 매핑을 위해 engine_dir의 config.json에서 `vocab_size`를 추출하는 데에만 필요합니다.                                 |

#### 후처리 모델

**필수 매개변수**

| 이름                            | 설명                                                     |
| ------------------------------- | -------------------------------------------------------- |
| `triton_max_batch_size`         | Triton이 모델과 함께 사용해야 하는 최대 배치 크기입니다. |
| `tokenizer_dir`                 | 모델의 토크나이저 경로입니다.                            |
| `postprocessing_instance_count` | 실행할 모델 인스턴스의 수입니다.                         |

**선택적 매개변수**

| 이름                  | 설명                                                                                                                                                                                |
| --------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `skip_special_tokens` | [HF 디토크나이저](https://huggingface.co/transformers/v2.11.0/main_classes/tokenizer.html#transformers.PreTrainedTokenizer.decode)에서 사용되는 `skip_special_tokens` 플래그입니다. |

#### tensorrt_llm 모델

대부분의 `tensorrt_llm` 모델 파라미터와 입출력 텐서는 [`executor.h`](https://github.com/NVIDIA/TensorRT-LLM/blob/main/cpp/include/tensorrt_llm/executor/executor.h)에 정의된 TRT-LLM C++ 런타임 API의 파라미터에 매핑될 수 있습니다. 아래 파라미터에 대한 자세한 설명은 `executor.h`의 Doxygen 주석을 참조하세요.

**필수 파라미터 (Mandatory parameters)**

| 이름                               | 설명                                                                                                                                                                                                                                                                                                                                           |
| ---------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `triton_backend`                   | 모델에 사용할 백엔드입니다. C++ TRT-LLM 백엔드 구현을 사용하려면 `tensorrtllm`으로 설정하세요. TRT-LLM Python 런타임을 사용하려면 `python`으로 설정하세요.                                                                                                                                                                                     |
| `triton_max_batch_size`            | Triton 모델 인스턴스가 실행될 최대 배치 크기입니다. `tensorrt_llm` 모델의 경우 실제 런타임 배치 크기가 `triton_max_batch_size`보다 클 수 있습니다. 런타임 배치 크기는 큐의 사용 가능한 요청 수와 엔진 빌드 `trtllm-build` 파라미터(`max_num_tokens`와 `max_batch_size` 등)와 같은 여러 파라미터를 기반으로 TRT-LLM 스케줄러에 의해 결정됩니다. |
| `decoupled_mode`                   | 분리 모드 사용 여부입니다. `stream` 텐서를 `true`로 설정하는 요청의 경우 `true`로 설정해야 합니다.                                                                                                                                                                                                                                             |
| `max_queue_delay_microseconds`     | 마이크로초 단위의 최대 큐 지연 시간입니다. 이 파라미터를 0보다 큰 값으로 설정하면 `max_queue_delay_microseconds` 내에 도착한 두 요청이 동일한 TRT-LLM 반복에서 스케줄링될 가능성이 높아집니다.                                                                                                                                                 |
| `max_queue_size`                   | 새 요청을 거부하기 전에 TRT-LLM 큐에서 허용되는 최대 요청 수입니다.                                                                                                                                                                                                                                                                            |
| `engine_dir`                       | 모델의 엔진 경로입니다.                                                                                                                                                                                                                                                                                                                        |
| `batching_strategy`                | 사용할 배칭 전략입니다. 인플라이트 배칭 지원을 활성화할 때는 `inflight_fused_batching`으로 설정하세요. 인플라이트 배칭을 비활성화하려면 `V1`으로 설정하세요.                                                                                                                                                                                   |
| `encoder_input_features_data_type` | 입력 텐서 `encoder_input_features`의 dtype입니다. mllama 모델의 경우 `TYPE_BF16`이어야 합니다. whisper와 같은 다른 모델의 경우 `TYPE_FP16`입니다.                                                                                                                                                                                              |
| `logits_datatype`                  | 컨텍스트 및 생성 로짓의 데이터 타입입니다.                                                                                                                                                                                                                                                                                                     |

**선택적 파라미터 (Optional parameters)**

- **일반 (General)**

| 이름                           | 설명                                                                                                                                                                                                             |
| ------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `encoder_engine_dir`           | 인코더-디코더 모델을 실행할 때, 인코더 모델의 구성과 엔진이 포함된 폴더의 경로입니다.                                                                                                                            |
| `max_attention_window_size`    | 슬라이딩 윈도우 어텐션과 같은 기술을 사용할 때, 하나의 토큰을 생성하기 위해 주목하는 최대 토큰 수입니다. 기본값은 시퀀스의 모든 토큰에 주목합니다. (기본값=max_sequence_length)                                  |
| `sink_token_length`            | 어텐션 윈도우에 항상 유지할 싱크 토큰의 수입니다.                                                                                                                                                                |
| `exclude_input_in_output`      | 응답에서 완성 토큰만 반환하려면 `true`로 설정하세요. 프롬프트 토큰을 생성된 토큰과 연결하여 반환하려면 `false`로 설정하세요. (기본값=`false`)                                                                    |
| `cancellation_check_period_ms` | 취소 확인 스레드가 다음 확인을 수행하기 전에 대기하는 시간입니다. triton을 통해 현재 활성 요청 중 취소된 요청이 있는지 확인하고 추가 실행을 방지합니다. (기본값=100)                                             |
| `stats_check_period_ms`        | 통계 보고 스레드가 다음 확인을 수행하기 전에 대기하는 시간입니다. (기본값=100)                                                                                                                                   |
| `recv_poll_period_ms`          | 오케스트레이터 모드에서 수신 스레드가 다음 확인을 수행하기 전에 대기하는 시간입니다. (기본값=0)                                                                                                                  |
| `iter_stats_max_iterations`    | 통계를 유지할 최대 반복 횟수입니다. (기본값=ExecutorConfig::kDefaultIterStatsMaxIterations)                                                                                                                      |
| `request_stats_max_iterations` | 요청별 통계를 유지할 최대 반복 횟수입니다. (기본값=executor::kDefaultRequestStatsMaxIterations)                                                                                                                  |
| `normalize_log_probs`          | 로그 확률의 정규화 여부를 제어합니다. `output_log_probs`의 정규화를 건너뛰려면 `false`로 설정하세요. (기본값=`true`)                                                                                             |
| `gpu_device_ids`               | 이 모델에 사용할 GPU ID의 쉼표로 구분된 목록입니다. 여러 모델 인스턴스를 구분하려면 세미콜론을 사용하세요. 제공되지 않으면 모델은 모든 가시적 GPU를 사용합니다. (기본값=지정되지 않음)                           |
| `participant_ids`              | -disable-spawn-process가 있는 오케스트레이터 모드를 사용할 때 필수인 MPI 랭크의 쉼표로 구분된 목록입니다. (기본값=지정되지 않음)                                                                                 |
| `gpu_weights_percent`          | 0.0에서 1.0 사이의 숫자로 설정하여 CPU 대신 GPU에 상주하고 런타임 중에 스트리밍 로드되는 가중치의 비율을 지정합니다. 1.0 미만의 값은 `weight_streaming`이 켜진 상태로 빌드된 엔진에서만 지원됩니다. (기본값=1.0) |

- **KV cache**

config.pbtxt에서 `enable_trt_overlap` 파라미터가 제거되었음을 참고하세요. 이 옵션은 CPU 오버헤드를 숨기기 위해 두 마이크로 배치의 실행을 오버랩할 수 있게 했습니다. CPU 오버헤드를 줄이기 위한 최적화 작업이 수행되었고 마이크로 배치의 오버랩이 추가적인 이점을 제공하지 않는다는 것이 확인되었습니다.

| 이름                             | 설명                                                                                                                                                                                                                                     |
| -------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `max_tokens_in_paged_kv_cache`   | 토큰 수로 표시된 KV 캐시의 최대 크기입니다. 지정되지 않은 경우 값은 '무한'으로 해석됩니다. KV 캐시 할당은 max_tokens_in_paged_kv_cache와 아래의 kv_cache_free_gpu_mem_fraction에서 파생된 값 중 더 작은 값입니다. (기본값=지정되지 않음) |
| `kv_cache_free_gpu_mem_fraction` | 0과 1 사이의 숫자로 설정하여 KV 캐시에 사용할 수 있는 GPU 메모리(모델 로딩 후)의 최대 비율을 나타냅니다. (기본값=0.9)                                                                                                                    |
| `cross_kv_cache_fraction`        | 0과 1 사이의 숫자로 설정하여 크로스 어텐션에 사용할 수 있는 KV 캐시의 최대 비율을 나타내며, 나머지는 셀프 어텐션에 사용됩니다. 선택적 매개변수이며 인코더-디코더 모델에만 설정해야 합니다. (기본값=0.5)                                  |
| `kv_cache_host_memory_bytes`     | 주어진 바이트 크기의 호스트 메모리로 오프로딩을 활성화합니다.                                                                                                                                                                            |
| `enable_kv_cache_reuse`          | 이전에 계산된 KV 캐시 값을 재사용하려면 `true`로 설정하세요(예: 시스템 프롬프트의 경우)                                                                                                                                                  |

- **LoRA cache**

| 이름                              | 설명                                                                                                                                                                     |
| --------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `lora_cache_optimal_adapter_size` | 캐시 페이지의 크기를 조정하는 데 사용되는 최적의 어댑터 크기입니다. 일반적으로 최적으로 크기가 조정된 어댑터는 정확히 1개의 캐시 페이지에 맞습니다. (기본값=8)           |
| `lora_cache_max_adapter_size`     | 캐시 페이지의 최소 크기를 설정하는 데 사용됩니다. 페이지는 단일 모듈, 단일 레이어 adapter_size `maxAdapterSize` 가중치 행을 수용할 수 있을 만큼 커야 합니다. (기본값=64) |
| `lora_cache_gpu_memory_fraction`  | LoRA 캐시에 사용되는 GPU 메모리의 비율입니다. 엔진 로드 후와 KV 캐시가 로드된 후 남은 메모리의 비율로 계산됩니다. (기본값=0.05)                                          |
| `lora_cache_host_memory_bytes`    | 바이트 단위의 호스트 LoRA 캐시 크기입니다. (기본값=1G)                                                                                                                   |

##### 디코딩 모드

| 이름             | 설명                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         |
| ---------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `max_beam_width` | 실행기에 전송될 요청의 빔 너비 값 (기본값=1)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 |
| `decoding_mode`  | 다음 중 하나로 설정: `{top_k, top_p, top_k_top_p, beam_search, medusa, redrafter, lookahead, eagle}` 디코딩 모드를 선택합니다. `top_k` 모드는 샘플링에 Top-K 알고리즘만 사용하고, `top_p` 모드는 샘플링에 Top-P 알고리즘만 사용합니다. top_k_top_p 모드는 요청의 런타임 샘플링 매개변수에 따라 Top-K와 Top-P 알고리즘을 모두 사용합니다. top_k_top_p 옵션은 `top_k` 또는 `top_p`를 개별적으로 사용하는 것보다 더 많은 메모리를 사용하고 런타임이 더 길다는 점에 유의하세요. 따라서 필요한 경우에만 사용해야 합니다. `beam_search`는 빔 서치 알고리즘을 사용합니다. 지정하지 않으면 `max_beam_width == 1`일 때 기본값으로 `top_k_top_p`를 사용하고, 그렇지 않으면 `beam_search`가 사용됩니다. Medusa 모델을 사용하는 경우 `medusa` 디코딩 모드를 설정해야 합니다. 하지만 TensorRT-LLM은 로드된 Medusa 모델을 감지하고 경고와 함께 디코딩 모드를 `medusa`로 덮어씁니다. ReDrafter, Lookahead 및 Eagle에도 동일하게 적용됩니다. |

##### 최적화

| 이름                           | 설명                                                                     |
| ------------------------------ | ------------------------------------------------------------------------ |
| `enable_chunked_context`       | 컨텍스트 청킹을 활성화하려면 `true`로 설정합니다. (기본값=`false`)       |
| `multi_block_mode`             | 멀티 블록 모드를 비활성화하려면 `false`로 설정합니다. (기본값=`true`)    |
| `enable_context_fmha_fp32_acc` | FMHA 러너 FP32 누적을 활성화하려면 `true`로 설정합니다. (기본값=`false`) |
| `cuda_graph_mode`              | cuda 그래프를 활성화하려면 `true`로 설정합니다. (기본값=`false`)         |
| `cuda_graph_cache_size`        | CUDA 그래프 캐시의 크기를 CUDA 그래프 수로 설정합니다. (기본값=0)        |

##### 스케줄링

| 이름                     | 설명                                                                                                                                                                                                                                                                                                                                                  |
| ------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `batch_scheduler_policy` | 현재 진행 중인 배치 반복에서 가능한 한 많은 요청을 탐욕적으로 패킹하려면 `max_utilization`으로 설정합니다. 이는 처리량을 최대화하지만 실행 중 KV 캐시 제한에 도달하면 요청 일시 중지/재개로 인한 오버헤드가 발생할 수 있습니다. 시작된 요청이 절대 일시 중지되지 않도록 보장하려면 `guaranteed_no_evict`로 설정합니다. (기본값=`guaranteed_no_evict`) |

##### 메두사

| 이름             | 설명                                                                                                               |
| ---------------- | ------------------------------------------------------------------------------------------------------------------ |
| `medusa_choices` | 예를 들어 "{0, 0, 0}, {0, 1}" 형식으로 Medusa 선택 트리를 지정합니다. 기본적으로 `mc_sim_7b_63` 선택이 사용됩니다. |

##### 이글

| 이름            | 설명                                                                                                                          |
| --------------- | ----------------------------------------------------------------------------------------------------------------------------- |
| `eagle_choices` | 예를 들어 "{0, 0, 0}, {0, 1}" 형식으로 서버별 기본 Eagle 선택 트리를 지정합니다. 기본적으로 `mc_sim_7b_63` 선택이 사용됩니다. |

#### tensorrt_llm_bls 모델

BLS 모델에 대해 자세히 알아보려면 [여기](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/python_backend/README.html#business-logic-scripting)를 참조하세요.

**필수 매개변수**

| 이름                    | 설명                                                                                                                                                                        |
| ----------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `triton_max_batch_size` | 모델이 처리할 수 있는 최대 배치 크기입니다.                                                                                                                                 |
| `decoupled_mode`        | 분리 모드를 사용할지 여부입니다.                                                                                                                                            |
| `bls_instance_count`    | 실행할 모델 인스턴스의 수입니다. 앙상블 대신 BLS 모델을 사용할 때는 동시 요청 실행을 허용하기 위해 모델 인스턴스 수를 TRT 엔진이 지원하는 최대 배치 크기로 설정해야 합니다. |
| `logits_datatype`       | 컨텍스트 및 생성 로짓의 데이터 타입입니다.                                                                                                                                  |

**선택적 매개변수**

**일반 (General)**

| 이름                | 설명                                                                                                                                             |
| ------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------ |
| `accumulate_tokens` | 스트리밍 모드에서 하나의 토큰만이 아닌 모든 누적된 토큰으로 후처리 모델을 호출하는 데 사용됩니다. 특정 토크나이저에는 이것이 필요할 수 있습니다. |

**추측적 디코딩**

BLS 모델은 추측적 디코딩을 지원합니다. 대상 및 초안 triton 모델은 `tensorrt_llm_model_name` `tensorrt_llm_draft_model_name` 매개변수로 설정됩니다. 추측적 디코딩은 요청에서 `num_draft_tokens`를 설정하여 수행됩니다. 로짓 비교 추측적 디코딩을 사용하기 위해 `use_draft_logits`를 설정할 수 있습니다. 추측적 디코딩을 사용할 때는 `return_generation_logits` 및 `return_context_logits`가 지원되지 않는다는 점에 유의하세요. 또한 현재는 추측적 디코딩에서 배치 크기가 1보다 큰 요청이 지원되지 않습니다.

| 이름                            | 설명                                        |
| ------------------------------- | ------------------------------------------- |
| `tensorrt_llm_model_name`       | 사용할 TensorRT-LLM 모델의 이름입니다.      |
| `tensorrt_llm_draft_model_name` | 사용할 TensorRT-LLM 초안 모델의 이름입니다. |

#### 모델 입력 및 출력

다음은 `tensorrt_llm` 및 `tensorrt_llm_bls` 모델의 입력 및 출력 텐서 목록입니다.

##### 공통 입력

| 이름                          | 형태 | 타입                         | 설명                                                                    |
| ----------------------------- | ---- | ---------------------------- | ----------------------------------------------------------------------- |
| `end_id`                      | [1]  | `int32`                      | 종료 토큰 ID입니다. 지정하지 않으면 기본값은 -1입니다                   |
| `pad_id`                      | [1]  | `int32`                      | 패딩 토큰 ID입니다                                                      |
| `temperature`                 | [1]  | `float32`                    | 샘플링 구성 매개변수: `temperature`                                     |
| `repetition_penalty`          | [1]  | `float`                      | 샘플링 구성 매개변수: `repetitionPenalty`                               |
| `min_length`                  | [1]  | `int32_t`                    | 샘플링 구성 매개변수: `minLength`                                       |
| `presence_penalty`            | [1]  | `float`                      | 샘플링 구성 매개변수: `presencePenalty`                                 |
| `frequency_penalty`           | [1]  | `float`                      | 샘플링 구성 매개변수: `frequencyPenalty`                                |
| `random_seed`                 | [1]  | `uint64_t`                   | 샘플링 구성 매개변수: `randomSeed`                                      |
| `return_log_probs`            | [1]  | `bool`                       | `true`일 때 출력에 로그 확률을 포함합니다                               |
| `return_context_logits`       | [1]  | `bool`                       | `true`일 때 출력에 컨텍스트 로짓을 포함합니다                           |
| `return_generation_logits`    | [1]  | `bool`                       | `true`일 때 출력에 생성 로짓을 포함합니다                               |
| `num_return_sequences`        | [1]  | `int32_t`                    | 요청당 생성된 시퀀스의 수입니다. (기본값=1)                             |
| `beam_width`                  | [1]  | `int32_t`                    | 이 요청의 빔 너비입니다. 탐욕적 샘플링의 경우 1로 설정합니다 (기본값=1) |
| `prompt_embedding_table`      | [1]  | `float16` (모델 데이터 타입) | P-튜닝 프롬프트 임베딩 테이블입니다                                     |
| `prompt_vocab_size`           | [1]  | `int32`                      | P-튜닝 프롬프트 어휘 크기입니다                                         |
| `return_kv_cache_reuse_stats` | [1]  | `bool`                       | `true`일 때 출력에 kv 캐시 재사용 통계를 포함합니다                     |

다음의 lora 입력은 `tensorrt_llm` 및 `tensorrt_llm_bls` 모델 모두에 대한 것입니다. 입력은 `tensorrt_llm` 모델을 통해 전달되며 `tensorrt_llm_bls` 모델은 `tensorrt_llm` 모델의 입력을 참조합니다.

| 이름           | 형태                                       | 타입                       | 설명                                                                                                                                                                                                                                                                                                                                                                       |
| -------------- | ------------------------------------------ | -------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `lora_task_id` | [1]                                        | `uint64`                   | 주어진 LoRA의 고유한 작업 ID입니다. 특정 LoRA로 처음 추론을 수행하려면 `lora_task_id`, `lora_weights` 및 `lora_config`가 모두 제공되어야 합니다. LoRA는 캐시되므로 동일한 작업에 대한 후속 요청에는 `lora_task_id`만 필요합니다. 캐시가 가득 차면 새로운 것을 위한 공간을 만들기 위해 가장 오래된 LoRA가 제거됩니다. `lora_task_id`가 캐시되지 않은 경우 오류가 반환됩니다 |
| `lora_weights` | [num_lora_modules_layers, D x Hi + Ho x D] | `float` (모델 데이터 타입) | LoRA 어댑터의 가중치입니다. 자세한 내용은 구성 파일을 참조하세요.                                                                                                                                                                                                                                                                                                          |
| `lora_config`  | [num_lora_modules_layers, 3]               | `int32t`                   | 모듈 식별자입니다. 자세한 내용은 구성 파일을 참조하세요.                                                                                                                                                                                                                                                                                                                   |

##### 공통 출력

| 이름                          | 형태                              | 타입    | 설명                                                                                                                                                       |
| ----------------------------- | --------------------------------- | ------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `cum_log_probs`               | [-1]                              | `float` | 각 출력에 대한 누적 확률                                                                                                                                   |
| `output_log_probs`            | [beam_width, -1]                  | `float` | 각 출력에 대한 로그 확률                                                                                                                                   |
| `context_logits`              | [-1, vocab_size]                  | `float` | 입력에 대한 문맥 로짓                                                                                                                                      |
| `generation_logits`           | [beam_width, seq_len, vocab_size] | `float` | 각 출력에 대한 생성 로짓                                                                                                                                   |
| `batch_index`                 | [1]                               | `int32` | 배치 인덱스                                                                                                                                                |
| `kv_cache_alloc_new_blocks`   | [1]                               | `int32` | KV 캐시 재사용 메트릭. 요청당 새로 할당된 블록 수. 선택적 입력 `return_kv_cache_reuse_stats`를 `true`로 설정하여 출력에 `kv_cache_alloc_new_blocks`를 포함 |
| `kv_cache_reused_blocks`      | [1]                               | `int32` | KV 캐시 재사용 메트릭. 요청당 재사용된 블록 수. 선택적 입력 `return_kv_cache_reuse_stats`를 `true`로 설정하여 출력에 `kv_cache_reused_blocks`를 포함       |
| `kv_cache_alloc_total_blocks` | [1]                               | `int32` | KV 캐시 재사용 메트릭. 요청당 총 할당된 블록 수. 선택적 입력 `return_kv_cache_reuse_stats`를 `true`로 설정하여 출력에 `kv_cache_alloc_total_blocks`를 포함 |

##### tensorrt_llm 모델의 고유 입력

| 이름                         | 형태     | 타입      | 설명                                                      |
| ---------------------------- | -------- | --------- | --------------------------------------------------------- |
| `input_ids`                  | [-1]     | `int32`   | 입력 토큰 ID                                              |
| `input_lengths`              | [1]      | `int32`   | 입력 길이                                                 |
| `request_output_len`         | [1]      | `int32`   | 요청된 출력 길이                                          |
| `draft_input_ids`            | [-1]     | `int32`   | 초안 입력 ID                                              |
| `decoder_input_ids`          | [-1]     | `int32`   | 디코더 입력 ID                                            |
| `decoder_input_lengths`      | [1]      | `int32`   | 디코더 입력 길이                                          |
| `draft_logits`               | [-1, -1] | `float32` | 초안 로짓                                                 |
| `draft_acceptance_threshold` | [1]      | `float32` | 초안 수용 임계값                                          |
| `stop_words_list`            | [2, -1]  | `int32`   | 중지 단어 목록                                            |
| `bad_words_list`             | [2, -1]  | `int32`   | 불용 단어 목록                                            |
| `embedding_bias`             | [-1]     | `string`  | 임베딩 편향 단어                                          |
| `runtime_top_k`              | [1]      | `int32`   | 런타임 top-k 샘플링을 위한 top-k 값                       |
| `runtime_top_p`              | [1]      | `float32` | 런타임 top-p 샘플링을 위한 top-p 값                       |
| `runtime_top_p_min`          | [1]      | `float32` | 런타임 top-p 샘플링을 위한 최소값                         |
| `runtime_top_p_decay`        | [1]      | `float32` | 런타임 top-p 샘플링을 위한 감쇠값                         |
| `runtime_top_p_reset_ids`    | [1]      | `int32`   | 런타임 top-p 샘플링을 위한 리셋 ID                        |
| `len_penalty`                | [1]      | `float32` | 빔 서치에서 긴 시퀀스를 패널티하는 방법 제어 (기본값=0.f) |
| `early_stopping`             | [1]      | `bool`    | 조기 종료 활성화                                          |
| `beam_search_diversity_rate` | [1]      | `float32` | 빔 서치 다양성 비율                                       |
| `stop`                       | [1]      | `bool`    | 중지 플래그                                               |
| `streaming`                  | [1]      | `bool`    | 스트리밍 활성화                                           |

##### tensorrt_llm 모델의 고유 출력

| 이름              | 형태     | 타입    | 설명         |
| ----------------- | -------- | ------- | ------------ |
| `output_ids`      | [-1, -1] | `int32` | 출력 토큰 ID |
| `sequence_length` | [-1]     | `int32` | 시퀀스 길이  |

##### tensorrt_llm_bls 모델의 고유 입력

| 이름                     | 형태                | 타입      | 설명                                                                                                    |
| ------------------------ | ------------------- | --------- | ------------------------------------------------------------------------------------------------------- |
| `text_input`             | [-1]                | `string`  | 프롬프트 텍스트                                                                                         |
| `decoder_text_input`     | [1]                 | `string`  | 디코더 입력 텍스트                                                                                      |
| `image_input`            | [3, 224, 224]       | `float16` | 입력 이미지                                                                                             |
| `max_tokens`             | [-1]                | `int32`   | 생성할 토큰 수                                                                                          |
| `bad_words`              | [2, num_bad_words]  | `int32`   | 불용 단어 목록                                                                                          |
| `stop_words`             | [2, num_stop_words] | `int32`   | 중지 단어 목록                                                                                          |
| `top_k`                  | [1]                 | `int32`   | 샘플링 설정 파라미터: `topK`                                                                            |
| `top_p`                  | [1]                 | `float32` | 샘플링 설정 파라미터: `topP`                                                                            |
| `length_penalty`         | [1]                 | `float32` | 샘플링 설정 파라미터: `lengthPenalty`                                                                   |
| `stream`                 | [1]                 | `bool`    | `true`일 때 토큰이 생성되는 대로 스트리밍. `false`일 때 전체 생성이 완료된 후에만 반환 (기본값=`false`) |
| `embedding_bias_words`   | [-1]                | `string`  | 임베딩 편향 단어                                                                                        |
| `embedding_bias_weights` | [-1]                | `float32` | 임베딩 편향 가중치                                                                                      |
| `num_draft_tokens`       | [1]                 | `int32`   | 추측적 디코딩 중 초안 모델에서 가져올 토큰 수                                                           |
| `use_draft_logits`       | [1]                 | `bool`    | 추측적 디코딩 중 로짓 비교 사용                                                                         |

##### tensorrt_llm_bls 모델의 고유 출력

| 이름          | 형태 | 타입     | 설명        |
| ------------- | ---- | -------- | ----------- |
| `text_output` | [-1] | `string` | 텍스트 출력 |

## Python 백엔드

> 백엔드 구성에 필요한 Python 라이브러리가 있을 경우 Triton Server 배포 시 설치해야 합니다. 요청 주시면 처리하겠습니다. (처리자: 이연호 프로)

Python 백엔드를 사용하기 위해서는 ==다음과 같은 구조의 Python 파일을 생성==해야 합니다:

```python
import triton_python_backend_utils as pb_utils
```

> `pb_utils`는 의존성 주입을 통해 불러옵니다. 최상단에서 `from triton_python_backend_utils import InferenceResponse` 등으로 불러올 경우 오류가 발생합니다.

> **pb_utils 파일** > [python_backend/src/resources/triton_python_backend_utils.py at main · triton-inference-server/python_backend](https://github.com/triton-inference-server/python_backend/blob/main/src/resources/triton_python_backend_utils.py)

```python
class TritonPythonModel:
    """Your Python model must use the same class name. Every Python model
    that is created must have "TritonPythonModel" as the class name.
    """

    @staticmethod
    def auto_complete_config(auto_complete_model_config):
        """`auto_complete_config` is called only once when loading the model
        assuming the server was not started with
        `--disable-auto-complete-config`. Implementing this function is
        optional. No implementation of `auto_complete_config` will do nothing.
        This function can be used to set `max_batch_size`, `input` and `output`
        properties of the model using `set_max_batch_size`, `add_input`, and
        `add_output`. These properties will allow Triton to load the model with
        minimal model configuration in absence of a configuration file. This
        function returns the `pb_utils.ModelConfig` object with these
        properties. You can use the `as_dict` function to gain read-only access
        to the `pb_utils.ModelConfig` object. The `pb_utils.ModelConfig` object
        being returned from here will be used as the final configuration for
        the model.

        Note: The Python interpreter used to invoke this function will be
        destroyed upon returning from this function and as a result none of the
        objects created here will be available in the `initialize`, `execute`,
        or `finalize` functions.

        Parameters
        ----------
        auto_complete_model_config : pb_utils.ModelConfig
          An object containing the existing model configuration. You can build
          upon the configuration given by this object when setting the
          properties for this model.

        Returns
        -------
        pb_utils.ModelConfig
          An object containing the auto-completed model configuration
        """
        inputs = [{
            'name': 'INPUT0',
            'data_type': 'TYPE_FP32',
            'dims': [4],
            # this parameter will set `INPUT0 as an optional input`
            'optional': True
        }, {
            'name': 'INPUT1',
            'data_type': 'TYPE_FP32',
            'dims': [4]
        }]
        outputs = [{
            'name': 'OUTPUT0',
            'data_type': 'TYPE_FP32',
            'dims': [4]
        }, {
            'name': 'OUTPUT1',
            'data_type': 'TYPE_FP32',
            'dims': [4]
        }]

        # Demonstrate the usage of `as_dict`, `add_input`, `add_output`,
        # `set_max_batch_size`, and `set_dynamic_batching` functions.
        # Store the model configuration as a dictionary.
        config = auto_complete_model_config.as_dict()
        input_names = []
        output_names = []
        for input in config['input']:
            input_names.append(input['name'])
        for output in config['output']:
            output_names.append(output['name'])

        for input in inputs:
            # The name checking here is only for demonstrating the usage of
            # `as_dict` function. `add_input` will check for conflicts and
            # raise errors if an input with the same name already exists in
            # the configuration but has different data_type or dims property.
            if input['name'] not in input_names:
                auto_complete_model_config.add_input(input)
        for output in outputs:
            # The name checking here is only for demonstrating the usage of
            # `as_dict` function. `add_output` will check for conflicts and
            # raise errors if an output with the same name already exists in
            # the configuration but has different data_type or dims property.
            if output['name'] not in output_names:
                auto_complete_model_config.add_output(output)

        auto_complete_model_config.set_max_batch_size(0)

        # To enable a dynamic batcher with default settings, you can use
        # auto_complete_model_config set_dynamic_batching() function. It is
        # commented in this example because the max_batch_size is zero.
        #
        # auto_complete_model_config.set_dynamic_batching()

        return auto_complete_model_config

    def initialize(self, args):
        """`initialize` is called only once when the model is being loaded.
        Implementing `initialize` function is optional. This function allows
        the model to initialize any state associated with this model.

        Parameters
        ----------
        args : dict
          Both keys and values are strings. The dictionary keys and values are:
          * model_config: A JSON string containing the model configuration
          * model_instance_kind: A string containing model instance kind
          * model_instance_device_id: A string containing model instance device
            ID
          * model_repository: Model repository path
          * model_version: Model version
          * model_name: Model name
        """
        print('Initialized...')

    def execute(self, requests):
        """`execute` must be implemented in every Python model. `execute`
        function receives a list of pb_utils.InferenceRequest as the only
        argument. This function is called when an inference is requested
        for this model.

        Parameters
        ----------
        requests : list
          A list of pb_utils.InferenceRequest

        Returns
        -------
        list
          A list of pb_utils.InferenceResponse. The length of this list must
          be the same as `requests`
        """

        responses = []

        # Every Python backend must iterate through list of requests and create
        # an instance of pb_utils.InferenceResponse class for each of them.
        # Reusing the same pb_utils.InferenceResponse object for multiple
        # requests may result in segmentation faults. You should avoid storing
        # any of the input Tensors in the class attributes as they will be
        # overridden in subsequent inference requests. You can make a copy of
        # the underlying NumPy array and store it if it is required.
        for request in requests:
            # Perform inference on the request and append it to responses
            # list...

        # You must return a list of pb_utils.InferenceResponse. Length
        # of this list must match the length of `requests` list.
        return responses

    def finalize(self):
        """`finalize` is called only once when the model is being unloaded.
        Implementing `finalize` function is optional. This function allows
        the model to perform any necessary clean ups before exit.
        """
        print('Cleaning up...')
```

모든 Python 백엔드는 4가지 주요 함수를 구현할 수 있습니다:

### auto_complete_config

`auto_complete_config`는 모델을 로드할 때 서버가 `--disable-auto-complete-config` 옵션으로 시작되지 않았다고 가정하면 단 한 번만 호출됩니다.

이 함수의 구현은 선택사항입니다. `auto_complete_config`를 구현하지 않으면 아무 작업도 수행되지 않습니다. 이 함수는 `set_max_batch_size`, `set_dynamic_batching`, `add_input`, `add_output`을 사용하여 `max_batch_size`, `dynamic_batching`, `input`, `output` 속성을 설정하는데 사용될 수 있습니다. 이러한 속성들은 설정 파일이 없을 때 Triton이 최소한의 모델 설정으로 모델을 로드할 수 있게 해줍니다. 이 함수는 이러한 속성들이 포함된 `pb_utils.ModelConfig` 객체를 반환합니다. as_dict 함수를 사용하여 `pb_utils.ModelConfig` 객체에 읽기 전용으로 접근할 수 있습니다. 여기서 반환되는 `pb_utils.ModelConfig` 객체는 모델의 최종 설정으로 사용됩니다.

최소 속성 외에도 `set_model_transaction_policy`를 사용하여 `auto_complete_config`를 통해 `model_transaction_policy`를 설정할 수 있습니다. 예를 들면:

```python
import triton_python_backend_utils as pb_utils

class TritonPythonModel:
    @staticmethod
    def auto_complete_config(auto_complete_model_config):
      …
      transaction_policy = {"decoupled": True}
      auto_complete_model_config.set_model_transaction_policy(transaction_policy)
      …
```

> 이 함수를 호출하는데 사용된 Python 인터프리터는 함수가 반환될 때 소멸되며, 그 결과 여기서 생성된 객체들은 initialize, execute 또는 finalize 함수에서 사용할 수 없습니다.

### initialize

`initialize`는 ==모델이 로드될 때 한 번 호출==됩니다. `initialize` 구현은 ==선택사항==입니다. `initialize`를 통해 실행 전에 필요한 초기화 작업을 수행할 수 있습니다. `initialize` 함수에서는 `args` 변수가 제공됩니다. `args`는 Python 딕셔너리입니다. 이 Python 딕셔너리의 키와 값은 모두 문자열입니다. args 딕셔너리에서 사용 가능한 키와 그 설명은 아래 표와 같습니다:

| 키                       | 설명                                             |
| ------------------------ | ------------------------------------------------ |
| model_config             | 모델 설정이 포함된 JSON 문자열 (`config.pbtxt`)  |
| model_instance_kind      | 모델 인스턴스 종류가 포함된 문자열               |
| model_instance_device_id | 모델 인스턴스 장치 ID가 포함된 문자열            |
| model_repository         | 모델 저장소 경로 (`/mnt/models/{모델 디렉토리}`) |
| model_version            | 모델 버전                                        |
| model_name               | 모델 이름                                        |

### execute

`execute` 함수는 추론 요청이 있을 때마다 호출됩니다. 모든 Python 모델은 `execute` 함수를 구현해야 합니다. `execute` 함수에서는 `InferenceRequest` 객체 리스트가 제공됩니다. 이 함수를 구현하는 방식에는 두 가지가 있습니다. 선택하는 방식은 디커플드 응답을 이 모델에서 반환하기를 원하는지 여부에 따라 달라집니다.

#### 기본 모드

이것은 모델을 구현하는 ==가장 일반적인 방법==이며 `execute` 함수가 <u>요청당 정확히 하나의 응답을 반환</u>해야 합니다. 이 모드에서는 `execute` 함수가 requests와 <u>동일한 길이의 </u>`InferenceResponse`<u> 객체 리스트를 반환</u>해야 합니다. 이 모드의 작업 흐름은 다음과 같습니다:

- `execute` 함수는 ==길이가 N인 배열==로 `pb_utils.InferenceRequest` 배치를 받습니다.
- `pb_utils.InferenceRequest`에 대한 ==추론을 수행==하고 해당하는 `pb_utils.InferenceResponse`를 ==응답 리스트에 추가==합니다.
- ==응답 리스트를 반환==합니다.
  - ==반환되는 응답 리스트의 길이는 N==이어야 합니다.
  - 리스트의 ==각 요소는 요청 배열의 해당 요소에 대한 응답==이어야 합니다.
  - ==각 요소는 응답을 포함==해야 합니다(응답은 출력 텐서나 오류일 수 있음); <u>요소는 </u>`None`<u>일 수 없습니다.</u>

Triton은 이러한 응답 리스트 요구사항이 충족되는지 확인하고, 충족되지 않으면 모든 추론 요청에 대해 오류 응답을 반환합니다. `execute` 함수에서 반환되면 함수에 전달된 `InferenceRequest` 객체와 관련된 모든 텐서 데이터가 삭제되므로, Python 모델은 `InferenceRequest` ==객체를 보관해서는 안 됩니다.==

24.06부터 모델은 디커플드 모드에서 설명된 것처럼 `InferenceResponseSender`를 사용하여 응답을 보낼 수 있습니다. 모델이 기본 모드이므로 요청당 정확히 하나의 응답을 보내야 합니다. `pb_utils.TRITONSERVER_RESPONSE_COMPLETE_FINAL` 플래그는 응답과 함께 보내거나 나중에 플래그만 포함된 응답으로 보내야 합니다.

#### 오류 처리

요청 중 하나에 오류가 있는 경우, `TritonError`<u> 객체를 사용하여 해당 요청에 대한 오류 메시지를 설정</u>할 수 있습니다. 다음은 `InferenceResponse` 객체에 오류를 설정하는 예시입니다:

```python
import triton_python_backend_utils as pb_utils

class TritonPythonModel:
    …

    def execute(self, requests):
        responses = []

        for request in requests:
            if an_error_occurred:
              # 오류가 발생한 경우 InferenceResponse에 "output_tensors"를 전달할
              # 필요가 없습니다. 이 경우 전달된 "output_tensors"는 무시됩니다.
              responses.append(pb_utils.InferenceResponse(
                error=pb_utils.TritonError("오류가 발생했습니다")))

        return responses
```

23.09부터 `pb_utils.TritonError`는 두 번째 매개변수로 선택적 Triton 오류 코드를 사용하여 생성될 수 있습니다. 예를 들면:

```python
pb_utils.TritonError("파일을 찾을 수 없습니다", pb_utils.TritonError.NOT_FOUND)
```

코드가 지정되지 않으면 기본적으로 `pb_utils.TritonError.INTERNAL`이 사용됩니다.

지원되는 오류 코드:

- `pb_utils.TritonError.UNKNOWN`
- `pb_utils.TritonError.INTERNAL`
- `pb_utils.TritonError.NOT_FOUND`
- `pb_utils.TritonError.INVALID_ARG`
- `pb_utils.TritonError.UNAVAILABLE`
- `pb_utils.TritonError.UNSUPPORTED`
- `pb_utils.TritonError.ALREADY_EXISTS`
- `pb_utils.TritonError.CANCELLED` (23.10부터)

> 오류 코드 사례 예시
>
> - INVALID_ARG:
>   - 입력 텐서 누락
>   - 빈 텐서 입력
>   - UTF-8 디코딩 오류
>   - 입력 길이 초과
> - UNAVAILABLE:
>   - 모델 미로드
>   - GPU 메모리 부족
>   - 리소스 관련 문제
> - UNSUPPORTED:
>   - 지원하지 않는 모델 작업
>   - 구현되지 않은 기능 호출
> - INTERNAL:
>   - 토크나이저 내부 오류
>   - 모델 추론 실패
>   - 출력 디코딩 오류
>   - 응답 생성 오류
> - UNKNOWN:
>   - 예상치 못한 기타 오류
>   - 오류 코드가 지정되지 않은 경우

#### 요청 취소 처리

클라이언트가 실행 중에 하나 이상의 요청을 취소할 수 있습니다. 23.10 버전부터 `request.is_cancelled()`<u>를 사용하여 요청이 취소되었는지 여부를 확인</u>할 수 있습니다. 예시:

```python
import triton_python_backend_utils as pb_utils

class TritonPythonModel:
    …

    def execute(self, requests):
        responses = []

        for request in requests:
            if request.is_cancelled():
                responses.append(pb_utils.InferenceResponse(
                    error=pb_utils.TritonError("Message", pb_utils.TritonError.CANCELLED)))
            else:
                …

        return responses
```

요청 취소 확인은 선택 사항이지만, 응답이 더 이상 필요하지 않은 경우 실행을 조기에 종료할 수 있는 전략적인 요청 실행 단계에서 취소 여부를 확인하는 것이 좋습니다.

#### 분리 모드

이 모드에서는 사용자가 하나의 요청에 대해 여러 응답을 보내거나 응답을 전혀 보내지 않을 수 있습니다. 모델은 <u>요청 배치가 실행되는 순서와 관계없이 응답을 순서에 상관없이 보낼 수</u>도 있습니다. 이러한 모델을 **분리된(decoupled) 모델**이라고 합니다. 이 모드를 사용하려면 모델 구성에서 거래 정책을 분리(decoupled)로 설정해야 합니다.

분리 모드에서는 모델이 요청당 하나의 `InferenceResponseSender` 객체를 사용하여 해당 요청에 대한 여러 응답을 계속 생성하고 전송할 수 있습니다. 이 모드의 워크플로우는 다음과 같습니다:

- `execute` 함수는 길이가 N인 배열로 `pb_utils.InferenceRequest` 배치를 받습니다.
- 각 `pb_utils.InferenceRequest`를 반복하면서 다음 단계를 수행합니다:

  1. `InferenceRequest.get_response_sender()`를 사용하여 `InferenceRequest`에 대한 `InferenceResponseSender` 객체를 가져옵니다.
  2. 전송할 `pb_utils.InferenceResponse`를 생성하고 채웁니다.
  3. `InferenceResponseSender.send()`를 사용하여 위의 응답을 전송합니다. 마지막 요청인 경우 `pb_utils.TRITONSERVER_RESPONSE_COMPLETE_FINAL`을 플래그로 전달합니다. 그렇지 않으면 다음 요청을 보내기 위해 1단계로 계속 진행합니다.

- 이 모드에서 `execute` 함수의 반환값은 `None`이어야 합니다.

위와 유사하게, 요청 중 하나에 오류가 있는 경우 `TritonError` 객체를 사용하여 해당 특정 요청에 대한 오류 메시지를 설정할 수 있습니다. `pb_utils.InferenceResponse` 객체에 오류를 설정한 후 `InferenceResponseSender.send()`를 사용하여 오류가 포함된 응답을 사용자에게 전송합니다.

23.10 버전부터 `response_sender.is_cancelled()`를 사용하여 `InferenceResponseSender` 객체에서 직접 요청 취소를 확인할 수 있습니다. 요청이 취소되더라도 응답 끝에 `TRITONSERVER_RESPONSE_COMPLETE_FINAL` 플래그를 전송해야 합니다.

##### 사용 사례

분리 모드는 강력하며 다양한 사용 사례를 지원합니다:

- 모델이 요청에 대한 응답을 전혀 보내지 않아야 하는 경우, 응답 없이 플래그 매개변수를 `pb_utils.TRITONSERVER_RESPONSE_COMPLETE_FINAL`로 설정하여 `InferenceResponseSender.send()`를 호출합니다.
- 모델은 요청을 받은 순서와 다르게 응답을 보낼 수도 있습니다.
- 요청 데이터와 `InferenceResponseSender` 객체를 모델의 별도 스레드로 전달할 수 있습니다. 이는 메인 호출자 스레드가 `execute` 함수를 종료할 수 있으며, 모델이 `InferenceResponseSender` 객체를 보유하고 있는 한 계속해서 응답을 생성할 수 있음을 의미합니다.

##### 비동기 실행

24.04 버전부터 분리된 Python 모델에서 `async def execute(self, requests):`가 지원됩니다. 이 코루틴은 동일한 모델 인스턴스에서 실행되는 요청과 공유되는 `AsyncIO` 이벤트 루프에 의해 실행됩니다. 현재 요청이 대기 중인 동안 모델 인스턴스에 대한 다음 요청이 실행을 시작할 수 있습니다.

이는 `AsyncIO`를 통해 요청을 동시에 실행할 수 있기 때문에, 대부분의 <u>시간을 대기하는 데 소비하는 모델의 인스턴스 수를 최소화</u>하는 데 유용합니다. <u>동시성을 최대한 활용하기 위해서는 비동기 실행 함수가 대기하는 동안(예: 네트워크를 통한 다운로드) 이벤트 루프가 진행되는 것을 차단하지 않는 것이 중요</u>합니다.

> - 모델은 실행 중인 이벤트 루프를 수정해서는 안 됩니다. 예기치 않은 문제가 발생할 수 있습니다.
> - 서버/백엔드는 모델 인스턴스에 의해 이벤트 루프에 추가되는 요청 수를 제어하지 않습니다.

#### 요청 재스케줄링

23.11 버전부터 Python 백엔드는 요청 재스케줄링을 지원합니다. 요청 객체에서 `set_release_flags` 함수를 `pb_utils.TRITONSERVER_REQUEST_RELEASE_RESCHEDULE` 플래그와 함께 호출하여 <u>요청을 향후 배치에서 추가 실행하도록 재스케줄링</u>할 수 있습니다. 이 기능은 반복적인 시퀀스를 처리하는 데 유용합니다.

요청 재스케줄링 API를 사용하려면 모델 구성에서 반복적 시퀀스 배치를 활성화하도록 구성해야 합니다:

```yaml
sequence_batching {
  iterative_sequence : true
}
```

분리되지 않은 모델의 경우 각 요청에 대해 하나의 응답만 가능합니다. 재스케줄링된 요청이 원래 요청과 동일하므로 재스케줄링된 요청에 대해 None 객체를 응답 목록에 추가해야 합니다. 예시:

```python
import triton_python_backend_utils as pb_utils

class TritonPythonModel:
    …

    def execute(self, requests):
        responses = []

        for request in requests:
            # 첫 번째 요청을 명시적으로 재스케줄링
            if self.idx == 0:
                request.set_release_flags(
                    pb_utils.TRITONSERVER_REQUEST_RELEASE_RESCHEDULE
                )
                responses.append(None)
                self.idx += 1
            else:
                responses.append(inference_response)

        return responses
```

분리된 모델의 경우 `execute` 함수에서 반환하기 전에 요청을 재스케줄링해야 합니다. 다음은 요청 재스케줄링을 사용하는 분리된 모델의 예시입니다. 이 모델은 1개의 입력 텐서(INT32 `[1]` 입력, "IN"이라는 이름)를 받아 입력 텐서와 동일한 형태의 "OUT" 출력 텐서를 생성합니다. 입력 값은 생성할 총 응답 수를 나타내고 출력 값은 남은 응답 수를 나타냅니다. 예를 들어, 요청 입력 값이 2인 경우 모델은:

- 값 1을 가진 응답을 전송합니다.
- RESCHEDULE 플래그로 요청을 해제합니다.
- 동일한 요청에 대해 실행할 때, 값 0을 가진 마지막 응답을 전송합니다.
- ALL 플래그로 요청을 해제합니다.

```python
import triton_python_backend_utils as pb_utils

class TritonPythonModel:
    ...

    def execute(self, requests):
        responses = []

        for request in requests:
            in_input = pb_utils.get_input_tensor_by_name(request, "IN").as_numpy()

            if self.reset_flag:
                self.remaining_response = in_input[0]
                self.reset_flag = False

            response_sender = request.get_response_sender()

            self.remaining_response -= 1

            out_output = pb_utils.Tensor(
                "OUT", np.array([self.remaining_response], np.int32)
            )
            response = pb_utils.InferenceResponse(output_tensors=[out_output])

            if self.remaining_response <= 0:
                response_sender.send(
                    response, flags=pb_utils.TRITONSERVER_RESPONSE_COMPLETE_FINAL
                )
                self.reset_flag = True
            else:
                request.set_release_flags(
                    pb_utils.TRITONSERVER_REQUEST_RELEASE_RESCHEDULE
                )
                response_sender.send(response)

        return None
```

### `finalize`

`finalize` 구현은 ==선택사항==입니다. 이 함수를 통해 모델이 Triton 서버에서 <u>언로드되기 전에 필요한 정리 작업을 수행</u>할 수 있습니다.

> 입력값을 더하고 빼는 Python 모델의 모든 함수 구현의 완전한 예시를 [add_sub example](https://github.com/triton-inference-server/python_backend/blob/main/examples/add_sub/model.py)에서 확인할 수 있습니다. 필요한 모든 함수를 구현한 후에는 이 파일을 `model.py`로 저장해야 합니다.
