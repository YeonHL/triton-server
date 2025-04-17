from typing import Optional
from pydantic import BaseModel, Field


class GenerateEmbeddingRequest(BaseModel):
    """임베딩 생성 요청 모델"""

    model_name: str = Field(description="모델 이름", examples=["bge-m3"])
    text_inputs: list[str] = Field(
        description="입력 텍스트", examples=[["안녕하세요", "반갑습니다.", "Hello"]]
    )
    parameters: Optional[dict] = Field(
        default=None,
        description="추론 요청에 대한 추가 매개변수, Sampling Parameter를 입력하세요. 모델에서 사용 가능한 값은 전달되고 이외에는 무시합니다.",
        examples=[
            {
                "max_tokens": 1024,
            }
        ],
    )
