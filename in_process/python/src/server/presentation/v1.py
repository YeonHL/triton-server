from fastapi import APIRouter, HTTPException, Request, status
from tritonclient.utils import InferenceServerException

from src.common.api import triton_grpc_client as triton
from src.server.presentation.dto import (
    GetLogSettingsResponse,
    GetServerMetadataResponse,
    GetTraceSettingsResponse,
    UpdateLogSettingsRequest,
    UpdateLogSettingsResponse,
    UpdateTraceSettingsRequest,
    UpdateTraceSettingsResponse,
)

server_router = APIRouter(
    prefix="/server",
    tags=["server"],
)


@server_router.get(path="")
async def get_server_metadata(request: Request) -> GetServerMetadataResponse:
    """추론 서버에 연결하여 메타데이터를 가져옵니다."""
    parameters: dict = {
        key: value
        for key, value in {
            "headers": dict(request.headers),
            "client_timeout": 5.0,
        }.items()
        if value
    }  # 파라미터 처리

    return GetServerMetadataResponse(
        **triton.get_server_metadata(**parameters)
    )  # Triton Server 요청


@server_router.get(path="/health")
async def is_server_live(request: Request) -> bool:
    """Triton 서버에 연결하여 활성화 상태를 확인합니다.

    서버가 해당 주소에 존재하지 않을 때 `False`를 반환하는 것이 아니라
    예외(Exception)를 발생시킵니다. 이는 "서버가 살아있지 않다(not live)"라는 상태와
    "서버에 연결할 수 없다(unreachable)"라는 상태를 구분하기 위함입니다.
    """
    parameters: dict = {
        key: value
        for key, value in {
            "headers": dict(request.headers),
            "client_timeout": 5.0,
        }.items()
        if value
    }  # 파라미터 처리

    return triton.is_server_live(**parameters)  # Triton Server 요청


@server_router.get(path="/health/ready")
async def is_server_ready(request: Request) -> bool:
    """Triton 서버에 연결하여 추론 준비 상태를 확인합니다.

    모델 로드 등으로 인해 서버가 준비되지 않은 경우 False를 반환합니다.
    """
    parameters: dict = {
        key: value
        for key, value in {
            "headers": dict(request.headers),
            "client_timeout": 5.0,
        }.items()
        if value
    }  # 파라미터 처리

    return triton.is_server_ready(**parameters)  # Triton Server 요청


@server_router.get(path="/logging")
async def get_log_settings(request: Request) -> GetLogSettingsResponse:
    """Triton Server의 전역 로그 설정을 조회합니다."""
    parameters: dict = {
        key: value
        for key, value in {
            "headers": dict(request.headers),
            "query_params": dict(request.query_params),
            "client_timeout": 5.0,
        }.items()
        if value
    }  # 파라미터 처리

    return GetLogSettingsResponse(
        **triton.get_log_settings(**parameters)
    )  # Triton Server 요청


@server_router.put(path="/logging")
async def update_log_settings(
    request: Request, settings: UpdateLogSettingsRequest
) -> UpdateLogSettingsResponse:
    """Triton Server의 전역 로그 설정을 업데이트합니다."""
    parameters: dict = {
        key: value
        for key, value in {
            "settings": settings.model_dump(exclude_none=True),
            "headers": dict(request.headers),
            "query_params": dict(request.query_params),
            "client_timeout": 5.0,
        }.items()
        if value
    }  # 파라미터 처리

    return UpdateLogSettingsResponse(
        **triton.update_log_settings(**parameters)
    )  # Triton Server 요청


@server_router.get(path="/trace")
async def get_trace_settings(
    request: Request,
) -> GetTraceSettingsResponse:
    """Triton Server의 전역 트레이스 설정을 조회합니다."""
    parameters: dict = {
        key: value
        for key, value in {
            "headers": dict(request.headers),
            "query_params": dict(request.query_params),
            "client_timeout": 5.0,
        }.items()
        if value
    }  # 파라미터 처리

    try:
        return GetTraceSettingsResponse(
            **triton.get_trace_settings(**parameters)
        )  # Triton Server 요청
    except InferenceServerException:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND)


@server_router.put(path="/trace")
async def update_trace_settings(
    request: Request,
    settings: UpdateTraceSettingsRequest,
) -> UpdateTraceSettingsResponse:
    """Triton Server의 전역 트레이스 설정을 업데이트합니다."""
    parameters: dict = {
        key: value
        for key, value in {
            "settings": settings.model_dump(exclude_none=True),
            "headers": dict(request.headers),
            "query_params": dict(request.query_params),
            "client_timeout": 5.0,
        }.items()
        if value
    }  # 파라미터 처리

    return UpdateTraceSettingsResponse(
        **triton.update_trace_settings(**parameters)
    )  # Triton Server 요청
