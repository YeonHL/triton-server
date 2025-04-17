from typing import Optional
from pydantic import BaseModel, Field

from src.common.type import Tensor


class InferRequest(BaseModel):
    """추론 요청 모델, inputs 외에는 선택입니다."""

    request_id: Optional[str] = Field(
        default=None,
        description="요청 ID, 입력할 경우 응답에서 함께 반환합니다.",
        examples=["1234"],
    )
    model_version: Optional[str] = Field(
        default=None, description="모델 버전", examples=["2"]
    )
    inputs: list[Tensor] = Field(
        default=[],
        description="모델 입력",
        examples=[
            [
                {
                    "name": "text_input",
                    "datatype": "BYTES",
                    "shape": [2, 1],
                    "data": [
                        [
                            "너는 유용하고 친절하게 도움을 주는 AI 비서야. 외국에서 유명한 한국 요리를 5개 알려줘"
                        ],
                        ["안녕하세요"],
                    ],
                }
            ]
        ],
    )
    outputs: Optional[list[Tensor]] = Field(
        default=None,
        description="모델 출력, 지정하지 않으면 모델이 생성한 모든 출력이 기본 설정을 사용하여 반환",
        examples=[[{"name": "sentence_embedding"}]],
    )
    parameters: Optional[dict] = Field(
        default=None,
        description="추론 요청에 대한 추가 매개변수",
        examples=[
            {
                "max_tokens": 1024,
            }
        ],
    )


class InferResponse(BaseModel):
    """추론 응답 모델"""

    model_name: str = Field(description="모델 이름", examples=["bge-m3"])
    model_version: str = Field(description="모델 버전", examples=["2"])
    id: Optional[str] = Field(
        default=None,
        description="요청 ID, 입력할 경우 응답에서 함께 반환합니다.",
        examples=["1234"],
    )
    parameters: Optional[dict] = Field(
        default=None,
        description="추론 관련 매개변수",
        examples=[
            {
                "sequence_start": {"bool_param": False},
                "sequence_end": {"bool_param": False},
                "sequence_id": {"int64_param": "0"},
            },
        ],
    )
    outputs: list[Tensor] = Field(
        default=[],
        description="모델 출력",
        examples=[
            [
                {
                    "name": "sentence_embedding",
                    "datatype": "FP32",
                    "shape": [2, 1024],
                    "data": [
                        [
                            0.01634637825191021,
                            -0.00414966931566596,
                            -0.03891320526599884,
                            -0.018098587170243263,
                            "...",
                        ],
                        [
                            -0.021015631034970284,
                            0.016271047294139862,
                            -0.03314371034502983,
                            -0.0366581454873085,
                            "...",
                        ],
                    ],
                }
            ]
        ],
    )
