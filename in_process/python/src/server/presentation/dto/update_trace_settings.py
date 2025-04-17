from typing import Optional
from pydantic import BaseModel, Field


class UpdateTraceSettingsRequest(BaseModel):
    """트레이스 설정 변경 요청 모델"""

    trace_level: Optional[list[str]] = Field(
        default=None,
        description="트레이스 레벨, `trace_file`이 없으면 설정 불가 (OFF, TIMESTAMPS, TENSORS 등)",
        examples=[["OFF"]],
    )
    trace_rate: Optional[int] = Field(
        default=None, description="트레이스 샘플링 비율 (초당)", examples=[1000]
    )
    trace_count: Optional[int] = Field(
        default=None, description="최대 트레이스 수 (-1은 무제한)", examples=[-1]
    )
    log_frequency: Optional[int] = Field(
        default=None, description="로그 출력 빈도", examples=[0]
    )

    def model_dump(self, **kwargs):
        data = super().model_dump(**kwargs)

        # 정수 필드를 문자열로 변환
        for field in ["trace_rate", "trace_count", "log_frequency"]:
            if data.get(field) is not None:
                data[field] = str(data[field])
        return data


class UpdateTraceSettingsResponse(BaseModel):
    """트레이스 설정 변경 응답 모델"""

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
