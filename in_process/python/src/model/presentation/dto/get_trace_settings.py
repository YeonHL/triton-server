from pydantic import BaseModel, Field


class GetTraceSettingsResponse(BaseModel):
    """트레이스 설정 조회 응답 모델"""

    trace_level: list[str] = Field(
        description="트레이스 레벨 (OFF, TIMESTAMPS, TENSORS 등)", examples=[["OFF"]]
    )
    trace_rate: str = Field(
        description="트레이스 샘플링 비율 (초당)", examples=["1000"]
    )
    trace_count: str = Field(
        description="최대 트레이스 수 (-1은 무제한)", examples=["-1"]
    )
    log_frequency: str = Field(description="로그 출력 빈도")
    trace_file: str = Field(
        description="트레이스 파일 경로. 비어있으면 콘솔로 출력",
        default="",
        examples=[""],
    )
    trace_mode: str = Field(description="트레이스 모드", examples=["triton"])
