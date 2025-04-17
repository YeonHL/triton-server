from pydantic import BaseModel, Field


class GetReadyModelListResponse(BaseModel):
    """로그 설정 조회 응답 모델"""

    model_list: list[str] = Field(
        description="추론 준비된 모델 리스트",
        examples=[["bge-m3"]],
    )
