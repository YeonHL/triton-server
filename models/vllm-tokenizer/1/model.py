# model.py
import os
from typing import Dict, List

import json
from huggingface_hub import login
import numpy as np
import triton_python_backend_utils as pb_utils
from transformers import AutoTokenizer


class TritonPythonModel:
    def initialize(self, args: Dict[str, str]) -> None:
        """`initialize`는 모델이 로드될 때 한 번만 호출됩니다.
        `initialize` 함수 구현은 선택 사항입니다. 이 함수를 통해
        모델은 해당 모델과 관련된 모든 상태를 초기화할 수 있습니다.

        매개변수 (Parameters)
        ----------
        args : dict
            키와 값 모두 문자열입니다. 딕셔너리 키와 값은 다음과 같습니다:
            * model_config: 모델 구성이 포함된 JSON 문자열
            * model_instance_kind: 모델 인스턴스 종류가 포함된 문자열
            * model_instance_device_id: 모델 인스턴스 장치 ID가 포함된 문자열
            * model_repository: 모델 저장소 경로
            * model_version: 모델 버전
            * model_name: 모델 이름
        """
        self.logger = pb_utils.Logger

        try:
            # model_config에서 parameters 가져오기
            model_config = json.loads(args["model_config"])
            parameters = model_config.get("parameters", {})

            hugging_face_token: str = os.getenv("HF_TOKEN") or parameters.get(
                "HUGGING_FACE_TOKEN", {}
            ).get("string_value", "")

            if not hugging_face_token:
                raise ValueError("Hugging Face token is not provided.")

            # Initialize
            login(token=hugging_face_token)

            model_repo: str = parameters.get("MODEL_REPO", {}).get("string_value", "")

            self.tokenizer = AutoTokenizer.from_pretrained(
                model_repo, trust_remote_code=True
            )

        except Exception as e:
            self.logger.log_error(f"Initialization failed: {str(e)}")
            raise e

    def execute(self, requests) -> "List[List[pb_utils.Tensor]]":
        """`execute`는 모든 Python 모델에서 구현되어야 합니다.
        `execute` 함수는 pb_utils.InferenceRequest의 리스트를
        유일한 인자로 받습니다. 이 함수는 이 모델에 대한
        추론이 요청될 때 호출됩니다.

        매개변수 (Parameters)
        ----------
        requests : list
            pb_utils.InferenceRequest의 리스트

        반환값 (Returns)
        -------
        list
            pb_utils.InferenceResponse의 리스트. 이 리스트의 길이는
            'requests'와 동일해야 합니다
        """
        responses = []

        try:
            for request in requests:
                # Get all input tensors
                text_input_tensor = pb_utils.get_input_tensor_by_name(
                    request, "text_input"
                )

                text_input = []
                for text in text_input_tensor.as_numpy():
                    if isinstance(text, bytes):
                        # bytes를 문자열로 디코딩
                        text_input.append(text.decode("utf-8"))
                    elif isinstance(text, str):
                        text_input.append(text)
                    else:
                        text_input.append(str(text))

                self.logger.log_info(f"Tokenizing {len(text_input)} texts")

                encoded_input = self.tokenizer(
                    text_input, return_tensors=None
                )
                input_ids = encoded_input["input_ids"]

                # Create output tensor for text_output
                input_ids_tensor = pb_utils.Tensor(
                    "input_ids", np.array(input_ids, dtype=np.int64)
                )

                output_tensors = [input_ids_tensor]

                # Create and append the inference response
                inference_response = pb_utils.InferenceResponse(
                    output_tensors=output_tensors
                )
                responses.append(inference_response)

            return responses
        except Exception as e:
            self.logger.log_error(f"Execution failed: {str(e)}")
            raise e

    def finalize(self):
        """`finalize`는 모델이 언로드될 때 한 번만 호출됩니다.
        `finalize` 함수를 구현하는 것은 선택 사항입니다.
        이 함수를 통해 모델은 종료하기 전에 필요한
        정리 작업을 수행할 수 있습니다.
        """
        self.tokenizer = None
