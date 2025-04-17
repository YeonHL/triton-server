from pydantic import BaseModel, Field
from typing import Optional


class ModelInfo(BaseModel):
    """모델 정보"""

    name: str = Field(description="모델 이름")
    version: Optional[str] = Field(default=None, description="모델 버전")
    state: Optional[str] = Field(default=None, description="모델 상태")
    reason: Optional[str] = Field(
        default=None, description="모델 상태 이유 (UNAVAILABLE일 경우)"
    )


class GetModelRepositoryIndexResponse(BaseModel):
    """로그 설정 조회 응답 모델"""

    model_list: list[ModelInfo] = Field(
        description="모델 리스트",
        examples=[
            {
                "model_list": [
                    {"name": "bge-m3", "version": "2", "state": "READY"},
                    {"name": "gemma-2-27b"},
                    {
                        "name": "llama-3.1-8b",
                        "version": "1",
                        "state": "UNAVAILABLE",
                        "reason": "unloaded",
                    },
                    {"name": "paligemma-3b"},
                    {"name": "vgt"},
                ]
            }
        ],
    )
