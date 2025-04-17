# model.py
import json
import os
from typing import Dict, List, Union, TYPE_CHECKING

import torch
import numpy as np
import triton_python_backend_utils as pb_utils  # type: ignore (Triton Server 환경에서 실행됨을 가정합니다.)
from huggingface_hub import login
from transformers import AutoTokenizer, AutoModelForTextEncoding

if TYPE_CHECKING:
    from transformers.tokenization_utils_base import BatchEncoding


class TritonPythonModel:
    max_new_tokens: int

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
            self.logger.log_info("Initializing model")
            # model_config에서 parameters 가져오기
            model_config = json.loads(args["model_config"])
            parameters = model_config.get("parameters", {})
            hugging_face_token: str = os.getenv("HF_TOKEN") or parameters.get(
                "HUGGING_FACE_TOKEN", {}
            ).get("string_value", "")

            if not hugging_face_token:
                raise ValueError("Hugging Face token is not provided.")

            model_repo: str = parameters.get("MODEL_REPO", {}).get("string_value", "")
            self.max_new_tokens: int = int(
                parameters.get("MAX_NEW_TOKENS", {}).get("string_value", 8192)
            )

            # Initialize
            login(token=hugging_face_token)

            self.tokenizer = AutoTokenizer.from_pretrained(model_repo)
            self.model = AutoModelForTextEncoding.from_pretrained(model_repo).eval()
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
            # for loop for batch requests
            for request in requests:
                # 입력 텐서 가져오기
                input_tensor = pb_utils.get_input_tensor_by_name(
                    request, "input_text"
                ).as_numpy()
                if input_tensor is None:
                    raise ValueError("input_text tensor not found")

                # dtype에 따른 처리
                input_text = []
                for text in input_tensor:
                    if isinstance(text, bytes):
                        # bytes를 문자열로 디코딩
                        input_text.append(text.decode("utf-8"))
                    elif isinstance(text, str):
                        input_text.append(text)
                    else:
                        input_text.append(str(text))

                # 토크나이징
                tokenized_inputs = self._tokenize(input_text)

                # 모델에 입력 전달 및 추론 수행
                embeddings_tensor = self._encode(tokenized_inputs)

                # 임베딩 후처리
                processed_embeddings = self._process(embeddings_tensor)

                # 출력 텐서 생성
                output_tensor = pb_utils.Tensor(
                    "sentence_embedding", processed_embeddings.astype(np.float32)
                )

                # inference response 생성
                inference_response = pb_utils.InferenceResponse(
                    output_tensors=[output_tensor]
                )
                responses.append(inference_response)

            return responses
        except Exception as e:
            self.logger.log_error(f"Execution failed: {str(e)}")
            raise e

    def _tokenize(self, input_text: list[str]) -> Dict:
        """텍스트를 토크나이징하고 인코딩합니다.

        Parameters
        ----------
        input_text : List[str]
            인코딩할 텍스트 리스트

        Returns
        -------
        Dict
            토크나이징된 입력값을 포함하는 딕셔너리
        """
        try:
            self.logger.log_info(f"Tokenizing {len(input_text)} texts")
            return self.tokenizer(
                input_text,
                padding=True,
                truncation=True,
                max_length=self.max_new_tokens,
                return_tensors="pt",
            )  # type: ignore (initialize가 선행 실행됨을 가정합니다.)
        except Exception as e:
            self.logger.error(f"Tokenizing failed: {str(e)}")
            raise e

    def _encode(
        self,
        tokenized_inputs: Union[Dict[str, "torch.Tensor"], "BatchEncoding"],
    ) -> "torch.Tensor":
        """토크나이즈된 입력을 인코딩하여 임베딩을 생성합니다.

        Parameters
        ----------
        tokenized_inputs : Union[Dict[str, Tensor], BatchEncoding]
            토크나이저가 생성한 입력값 딕셔너리 또는 BatchEncoding 객체

        Returns
        -------
        torch.Tensor
            [CLS] 토큰의 임베딩 벡터
        """
        self.logger.log_info("Encoding inputs")
        try:
            with torch.no_grad():
                outputs = self.model(**tokenized_inputs)  # type: ignore (initialize가 선행 실행됨을 가정합니다.)
                return outputs.last_hidden_state[:, 0]  # [CLS] 토큰 임베딩 추출
        except Exception as e:
            self.logger.log_error(f"Encoding failed: {str(e)}")
            raise e

    def _process(self, embeddings_tensor) -> "np.ndarray":
        """모델 출력을 후처리합니다.

        Parameters
        ----------
        model_output : torch.Tensor
            모델의 출력 텐서

        Returns
        -------
        np.ndarray
            후처리된 임베딩
        """
        self.logger.log_info("Processing inputs")
        try:
            # 텐서를 numpy 배열로 변환
            embeddings = embeddings_tensor.detach().cpu().numpy()

            # L2 정규화 수행
            embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

            return embeddings
        except Exception as e:
            self.logger.log_error(f"Processing failed: {str(e)}")
            raise e

    def finalize(self):
        """`finalize`는 모델이 언로드될 때 한 번만 호출됩니다.
        `finalize` 함수를 구현하는 것은 선택 사항입니다.
        이 함수를 통해 모델은 종료하기 전에 필요한
        정리 작업을 수행할 수 있습니다.
        """
        self.model = None
        self.tokenizer = None
