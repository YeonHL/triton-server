from pydantic import BaseModel, Field


class GetServerMetadataResponse(BaseModel):
    """서버 정보 요청 응답 모델"""

    name: str = Field(description="서버의 설명적 이름", examples=["triton"])
    version: str = Field(description="서버 버전", examples=["2.56.0"])
    extensions: list[str] = Field(
        description="서버가 지원하는 확장",
        examples=[
            [
                "classification",
                "sequence",
                "model_repository",
                "model_repository(unload_dependents)",
                "schedule_policy",
                "model_configuration",
                "system_shared_memory",
                "cuda_shared_memory",
                "binary_tensor_data",
                "parameters",
                "statistics",
                "trace",
                "logging",
            ]
        ],
    )
