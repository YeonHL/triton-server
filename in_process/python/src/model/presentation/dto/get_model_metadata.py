from typing import Optional
from pydantic import BaseModel, Field
from src.common.type import Tensor


class GetModelMetadataResponse(BaseModel):
    """모델 메타데이터 조회 응답"""

    name: str = Field(description="모델 이름", examples=["bge-m3"])
    versions: Optional[list[str]] = Field(
        default_factory=list, description="사용 가능한 모델 버전 목록", examples=[["2"]]
    )
    platform: str = Field(description="모델의 프레임워크/백엔드", examples=["ensemble"])
    inputs: list[Tensor] = Field(
        description="모델 요구 입력 텐서 목록",
        examples=[[{"name": "text_input", "datatype": "BYTES", "shape": [-1, -1]}]],
    )
    outputs: list[Tensor] = Field(
        description="모델 출력 결과 텐서 목록",
        examples=[
            [{"name": "sentence_embedding", "datatype": "FP32", "shape": [-1, 1024]}]
        ],
    )
