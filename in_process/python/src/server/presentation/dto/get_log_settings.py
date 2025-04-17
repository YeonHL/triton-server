from pydantic import BaseModel, Field


class GetLogSettingsResponse(BaseModel):
    """로그 설정 조회 응답 모델"""

    log_file: str = Field(
        description="로그 출력이 저장될 로그 파일 위치. 비어 있으면 로그 출력이 콘솔로 스트리밍됩니다.",
        examples=[""],
    )
    log_info: bool = Field(
        description="INFO 레벨 메시지 로깅 활성화 여부", examples=[True]
    )
    log_warning: bool = Field(
        description="WARNING 레벨 메시지 로깅 활성화 여부", examples=[True]
    )
    log_error: bool = Field(
        description="ERROR 레벨 메시지 로깅 활성화 여부", examples=[True]
    )
    log_verbose_level: int = Field(
        description="상세 로깅 레벨 (0: 비활성화, 1: 레벨 1 메시지, 2: 레벨 2 이하 메시지)",
        examples=[0],
    )
    log_format: str = Field(
        description="로그 메시지 형식 (default 또는 ISO8601)", examples=["default"]
    )
