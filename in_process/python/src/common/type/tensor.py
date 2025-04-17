from typing import Optional
from pydantic import BaseModel, Field


class Tensor(BaseModel):
    """텐서 모델"""

    name: str = Field(default="", description="텐서 이름", examples=["text"])
    datatype: str = Field(
        default="", description="텐서 데이터 타입", examples=["BYTES"]
    )
    shape: list[int] = Field(
        default=[],
        description="텐서 형태, 맨 앞은 Triton 서버 내에서 처리하는 배치 사이즈입니다.",
        examples=[-1, -1],
    )
    data: Optional[list] = Field(
        default=None,
        description="텐서 데이터",
    )
