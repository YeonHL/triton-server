from pydantic import BaseModel, Field


class TimingStats(BaseModel):
    """시간 통계"""

    count: int = Field(default=0, description="발생 횟수")
    ns: int = Field(default=0, description="소요 시간(나노초)")


class InferenceStats(BaseModel):
    """추론 단계별 통계"""

    success: TimingStats = Field(default_factory=TimingStats, description="성공 통계")
    fail: TimingStats = Field(default_factory=TimingStats, description="실패 통계")
    queue: TimingStats = Field(default_factory=TimingStats, description="대기열 통계")
    compute_input: TimingStats = Field(
        default_factory=TimingStats, description="입력 처리 통계"
    )
    compute_infer: TimingStats = Field(
        default_factory=TimingStats, description="추론 처리 통계"
    )
    compute_output: TimingStats = Field(
        default_factory=TimingStats, description="출력 처리 통계"
    )
    cache_hit: TimingStats = Field(
        default_factory=TimingStats, description="캐시 적중 통계"
    )
    cache_miss: TimingStats = Field(
        default_factory=TimingStats, description="캐시 미스 통계"
    )


class ModelStats(BaseModel):
    """개별 모델 통계"""

    name: str = Field(description="모델 이름", examples=["bge-m3"])
    version: str = Field(description="모델 버전", examples=["2"])
    last_inference: int = Field(
        default=0, description="마지막 추론 시간", examples=[1744225523196]
    )
    inference_count: int = Field(default=0, description="총 추론 횟수", examples=[2])
    execution_count: int = Field(default=0, description="총 실행 횟수", examples=[1])
    inference_stats: InferenceStats = Field(description="추론 상세 통계")
    response_stats: dict = Field(
        default_factory=dict, description="응답 통계", examples=[{}]
    )
    batch_stats: list[dict] = Field(
        default_factory=list,
        description="배치 처리 통계",
        examples=[
            [
                {
                    "batch_size": "1",
                    "compute_input": {"count": "1", "ns": "555123"},
                    "compute_infer": {"count": "1", "ns": "5353018"},
                    "compute_output": {"count": "1", "ns": "66136"},
                }
            ]
        ],
    )
    memory_usage: list = Field(
        default_factory=list, description="메모리 사용량", examples=[]
    )


class GetInferenceStatisticsResponse(BaseModel):
    """모델 추론 통계 조회 응답"""

    model_stats: list[ModelStats] = Field(description="추론 통계")
