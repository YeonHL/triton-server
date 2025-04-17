# model.py
import os
from typing import Dict, List

from PIL import Image
import json
from huggingface_hub import login
import numpy as np
import triton_python_backend_utils as pb_utils
from transformers import PaliGemmaProcessor

import cv2


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
            self.image_size: int = int(
                parameters.get("IMAGE_SIZE", {}).get("string_value", "")
            )

            self.processor = PaliGemmaProcessor.from_pretrained(
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
                image_tensor = pb_utils.get_input_tensor_by_name(request, "image")
                stream_tensor = pb_utils.get_input_tensor_by_name(request, "stream")
                sampling_parameters_tensor = pb_utils.get_input_tensor_by_name(
                    request, "sampling_parameters"
                )
                exclude_input_in_output_tensor = pb_utils.get_input_tensor_by_name(
                    request, "exclude_input_in_output"
                )
                return_finish_reason_tensor = pb_utils.get_input_tensor_by_name(
                    request, "return_finish_reason"
                )
                return_cumulative_logprob_tensor = pb_utils.get_input_tensor_by_name(
                    request, "return_cumulative_logprob"
                )
                return_logprobs_tensor = pb_utils.get_input_tensor_by_name(
                    request, "return_logprobs"
                )
                return_num_input_tokens_tensor = pb_utils.get_input_tensor_by_name(
                    request, "return_num_input_tokens"
                )
                return_num_output_tokens_tensor = pb_utils.get_input_tensor_by_name(
                    request, "return_num_output_tokens"
                )

                model_inputs = self.processor(text=prompt, images=image, return_tensors="pt").to(torch.bfloat16)

                # Process text_input to get chat messages
                if text_input_tensor is not None:
                    text_input: str = (
                        text_input_tensor.as_numpy().tolist()[0].decode("utf-8")
                    )

                    # Create output tensor for text_output
                    text_output_tensor = pb_utils.Tensor(
                        "text_output", np.array(text_input, dtype=np.object_)
                    )
                else:
                    # If no text_input is provided, return empty result
                    text_output_tensor = pb_utils.Tensor(
                        "text_output", np.array([""], dtype=np.object_)
                    )



                output_tensors = [text_output_tensor]

                image_array = self.processor.image_processor(
                    images=image_tensor.as_numpy(), return_tensors="np"
                )["pixel_values"]
                pb_utils.Tensor("image", image_array)

                output_tensors.append(pb_utils.Tensor("image"))

                # Add other outputs (passing through input values)
                if stream_tensor is not None:
                    output_tensors.append(
                        pb_utils.Tensor("stream", stream_tensor.as_numpy())
                    )

                if sampling_parameters_tensor is not None:
                    output_tensors.append(
                        pb_utils.Tensor(
                            "sampling_parameters", sampling_parameters_tensor.as_numpy()
                        )
                    )

                if exclude_input_in_output_tensor is not None:
                    output_tensors.append(
                        pb_utils.Tensor(
                            "exclude_input_in_output",
                            exclude_input_in_output_tensor.as_numpy(),
                        )
                    )

                if return_finish_reason_tensor is not None:
                    output_tensors.append(
                        pb_utils.Tensor(
                            "return_finish_reason",
                            return_finish_reason_tensor.as_numpy(),
                        )
                    )

                if return_cumulative_logprob_tensor is not None:
                    output_tensors.append(
                        pb_utils.Tensor(
                            "return_cumulative_logprob",
                            return_cumulative_logprob_tensor.as_numpy(),
                        )
                    )

                if return_logprobs_tensor is not None:
                    output_tensors.append(
                        pb_utils.Tensor(
                            "return_logprobs", return_logprobs_tensor.as_numpy()
                        )
                    )

                if return_num_input_tokens_tensor is not None:
                    output_tensors.append(
                        pb_utils.Tensor(
                            "return_num_input_tokens",
                            return_num_input_tokens_tensor.as_numpy(),
                        )
                    )

                if return_num_output_tokens_tensor is not None:
                    output_tensors.append(
                        pb_utils.Tensor(
                            "return_num_output_tokens",
                            return_num_output_tokens_tensor.as_numpy(),
                        )
                    )

                # Create and append the inference response
                inference_response = pb_utils.InferenceResponse(
                    output_tensors=output_tensors
                )
                responses.append(inference_response)

            return responses
        except Exception as e:
            self.logger.log_error(f"Execution failed: {str(e)}")
            raise e

    def _process_tmp(self, image, prompt: str, extract_line_flag=False):
        """
        하나의 이미지 경로만 읽고 처리하는 함수.
        이미지 경로를 읽고 numpy로 변환 후 process_and_infer 호출
        # ISSUE
        - extract_line으로 한 경우, 라인 위/아래/앞/뒤에 검정 픽셀 제거해야 함. 아닌경우 글자로 인식
        Args:
            image (numpy.ndarray): OpenCV 이미지 배열. BGR 형식의 3채널 (H, W, 3)
            extract_line_flag (bool) : 입력 이미지의 라인 추출 후, 라인 별 결과내기
        Returns:
            padded_image_np (numpy) : 리사이징, 패딩이 적용된 numpy 이미지
        """
        image_np = np.array(image)

        if extract_line_flag:
            results = []
            line_img_list = self._extract_lines_with_margin(image)
            for line_img in line_img_list:
                # 텍스트만
                result = self._process_image_tmp(line_img)
                results.append(result)
            return "\n".join(results)

        result: str = self._process_image_tmp(image_np)
        return result

    def _extract_lines_with_margin(self, image, margin=1):
        """
        이미지에서 텍스트 라인을 분리하고 라인 간 여백을 추가.
        Args:
            image (numpy.ndarray): OpenCV 이미지 배열. BGR 형식의 3채널 (H, W, 3)
            margin (int): 라인 위아래에 추가할 여백 크기 (픽셀 단위).
        Returns:
            list: 잘린 라인 이미지 리스트.
        """
        # 1. 이진화 (Thresholding)
        _, binary_image = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY_INV)

        # 2. 수평 프로젝션 생성
        height, width = binary_image.shape
        horizontal_projection = np.sum(binary_image, axis=1)

        # 3. 라인 경계 찾기
        line_boundaries = []
        in_line = False
        for i, value in enumerate(horizontal_projection):
            if value > 0 and not in_line:
                in_line = True
                start = max(0, i - margin)  # 시작 지점에 margin 추가
            elif value == 0 and in_line:
                in_line = False
                end = min(height, i + margin)  # 끝 지점에 margin 추가
                line_boundaries.append((start, end))

        # 4. 라인별 이미지 추출
        line_images = []
        for start, end in line_boundaries:
            line_image = image[start:end, :]  # 각 라인 자르기
            line_images.append(line_image)

        return line_images

    def _process_image_tmp(self, image_np, use_prompt=True):
        """
        특정 폴더 내의 모든 이미지에 대해 처리 및 추론을 수행하고,
        결과를 딕셔너리 형태로 반환
        Args:
            image_np (numpy.ndarray): 이미지 Numpy 배열.
            use_prompt (bool): 프롬프트 사용.
            # ISSUE : 프롬프트 사용 시, 결과에 프롬프트가 같이 나오고 결과가 나쁘게 나옴. -> 기본값 : False로 둠.
        Returns:
            results (dict): process함수의 결과를 filename을 key로 가지는 dict으로 반환
        """
        image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
        padded_image_np = self._resize_and_pad(image_np)

        # 흑백 이미지를 RGB로 변환 (H, W) → (H, W, 3)
        padded_image_rgb = np.stack([padded_image_np] * 3, axis=-1)

        # PIL 이미지로 변환
        paded_image_pil = Image.fromarray(padded_image_rgb.astype(np.uint8))

        # 이미지 전처리
        if use_prompt:
            # ISSUE : <image> 뒤에 PROMPT를 추가로 넣었을 때 답변에 같이 나옴.
            text_with_token = "<image>"  # 하나의 이미지가 있을 경우
            inputs = self.processor(
                text=text_with_token, images=paded_image_pil, return_tensors="np"
            )
        else:
            inputs = self.processor(images=paded_image_pil, return_tensors="np")
        # 모델 추론
        # 결과 디코딩
        result = self.tokenizer.decode(output[0], skip_special_tokens=True)
        return result

    def _resize_and_pad(self, image_np, resize_flag=True, padding_flag=True):
        """
        입력 이미지 리사이즈 및 padding 추가
        리사이즈, padding을 하지 않고 OCR 모델에 바로 입력으로 줄 경우,
        OCR 모델이 정해진 모델 입력 크기에 따라 이미지 사이즈를 조정함
        Args:
            image_np (numpy): OpenCV 이미지 배열
            resize_flag (bool): True시 리사이징, 이미지 비율을 유지한 채로 모델 입력(self.output_size)에 맞게 확대/축소
            padding_flag (bool): True시 이미지 비율을 바꾸지 않고 padding만 추가해 이미지를 모델 입력 사이즈에 맞춤
        Returns:
            padded_image_np (numpy) : 리사이징, 패딩이 적용된 numpy 이미지
        """

        # 1. 이미지 리사이즈 (INTER_LANCZOS4 보간법 사용)
        h, w = image_np.shape  # 그레이스케일 (H, W)
        if resize_flag:
            # 긴 변을 image_size로 맞추고 비율 유지
            scale_factor = self.image_size / max(h, w)
            new_h, new_w = int(h * scale_factor), int(w * scale_factor)
            resized_image_np = cv2.resize(
                image_np, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4
            )
        else:
            new_h, new_w = h, w  # 기존 크기 유지
            resized_image_np = image_np

        # 2. 패딩 추가
        if padding_flag:
            # 패딩 계산 (중앙 정렬)
            pad_top = (self.image_size - new_h) // 2
            pad_bottom = self.image_size - new_h - pad_top
            pad_left = (self.image_size - new_w) // 2
            pad_right = self.image_size - new_w - pad_left

            # 패딩 적용 (배경 흰색)
            padded_image_np = np.pad(
                resized_image_np,
                ((pad_top, pad_bottom), (pad_left, pad_right)),
                mode="constant",
                constant_values=255,
            )
        else:
            padded_image_np = resized_image_np
        return padded_image_np

    def finalize(self):
        """`finalize`는 모델이 언로드될 때 한 번만 호출됩니다.
        `finalize` 함수를 구현하는 것은 선택 사항입니다.
        이 함수를 통해 모델은 종료하기 전에 필요한
        정리 작업을 수행할 수 있습니다.
        """
        self.processor = None
