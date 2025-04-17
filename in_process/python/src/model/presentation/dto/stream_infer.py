import json
from typing import Optional

from pydantic import BaseModel, Field

from src.common.type import Tensor


class StreamInferRequest(BaseModel):
    """추론 요청 모델, inputs 외에는 선택입니다."""

    request_id: Optional[str] = Field(
        default=None,
        description="요청 ID, 입력할 경우 응답에서 함께 반환합니다.",
        examples=["1234"],
    )
    model_version: Optional[str] = Field(
        default=None, description="모델 버전", examples=["1"]
    )
    inputs: list[Tensor] = Field(
        default=[],
        description="모델 입력",
        examples=[
            [
                {
                    "name": "text_input",
                    "datatype": "BYTES",
                    "shape": [1, 1],
                    "data": [
                        [
                            f"""{json.dumps([
                                {"role": "user", "content": "Hello, how are you?"},
                                {"role": "assistant", "content": "I'm doing great. How can I help you today?"},
                                {"role": "user", "content": "I'd like to show off how chat templating works!"},
                            ])}"""
                        ]
                    ],
                },
                {
                    "name": "stream",
                    "datatype": "BOOL",
                    "shape": [1, 1],
                    "data": [[True]],
                },
            ]
        ],
    )
    outputs: Optional[list[Tensor]] = Field(
        default=None,
        description="모델 출력, 지정하지 않으면 모델이 생성한 모든 출력이 기본 설정을 사용하여 반환",
        examples=[[{"name": "text_output"}]],
    )
    parameters: Optional[dict] = Field(
        default=None,
        description="추론 요청에 대한 추가 매개변수",
        examples=[
            {
                "temperature": 0.8,
                "top_p": 0.9,
                "presence_penalty": 0.8,
                "frequency_penalty": 0.8,
                "repetition_penalty": 1.2,
                "max_tokens": 1024,
            }
        ],
    )
