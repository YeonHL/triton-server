from typing import Annotated, Optional
from fastapi import (
    APIRouter,
    BackgroundTasks,
    Path,
    Query,
    Request,
    HTTPException,
    status,
)
from fastapi.responses import StreamingResponse
from tritonclient.utils import InferenceServerException
from src.common.api import triton_grpc_client as triton
from src.model.presentation.dto import (
    GetInferenceStatisticsResponse,
    GetModelMetadataResponse,
    GetTraceSettingsResponse,
    InferRequest,
    InferResponse,
    StreamInferRequest,
    UpdateTraceSettingsRequest,
    UpdateTraceSettingsResponse,
)

model_router = APIRouter(
    prefix="",
    tags=["model"],
)


@model_router.get(path="/{model_name}", response_model_exclude_none=True)
async def get_model_metadata(
    request: Request,
    model_name: Annotated[str, Path(description="모델 이름")],
    model_version: Annotated[Optional[str], Query(description="모델 버전")] = "",
) -> GetModelMetadataResponse:
    """모델에 대한 정보를 조회합니다.

    모델이 추론 준비 상태가 아닌 경우 오류가 발생합니다.
    """
    parameters: dict = {
        key: value
        for key, value in {
            "model_name": model_name,
            "model_version": model_version,
            "headers": dict(request.headers),
            "client_timeout": 5.0,
        }.items()
        if value
    }  # 파라미터 처리

    try:
        model_metadata = triton.get_model_metadata(**parameters)  # Triton Server 요청
    except InferenceServerException:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND)
    return GetModelMetadataResponse(**model_metadata)


@model_router.post(path="/{model_name}", status_code=status.HTTP_202_ACCEPTED)
async def load_model(
    request: Request,
    background_tasks: BackgroundTasks,
    model_name: Annotated[str, Path(description="모델 이름")],
) -> None:
    """모델 로드를 요청합니다. null을 반환하고 백그라운드에서 요청을 수행합니다."""
    parameters: dict = {
        key: value
        for key, value in {
            "model_name": model_name,
            "headers": dict(request.headers),
        }.items()
        if value
    }  # 파라미터 처리

    try:
        if triton.is_server_live(dict(request.headers)):
            background_tasks.add_task(
                triton.load_model, **parameters
            )  # 응답 후 요청 수행
        else:
            raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE)
    except InferenceServerException:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR)


@model_router.delete(path="/{model_name}", status_code=status.HTTP_202_ACCEPTED)
async def unload_model(
    request: Request,
    background_tasks: BackgroundTasks,
    model_name: Annotated[str, Path(description="모델 이름")],
) -> None:
    """모델 언로드를 요청합니다. null을 반환하고 백그라운드에서 요청을 수행합니다."""
    parameters: dict = {
        key: value
        for key, value in {
            "model_name": model_name,
            "headers": dict(request.headers),
            "unload_dependents": True,
        }.items()
        if value
    }  # 파라미터 처리
    try:
        if triton.is_server_live(dict(request.headers)):
            background_tasks.add_task(
                triton.unload_model, **parameters
            )  # 응답 후 요청 수행
        else:
            raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE)
    except InferenceServerException:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR)


@model_router.get(path="/{model_name}/ready")
async def is_model_ready(
    request: Request,
    model_name: Annotated[str, Path(description="모델 이름")],
    model_version: Annotated[Optional[str], Query(description="모델 버전")] = "",
) -> bool:
    """모델이 추론 준비 상태인지 확인합니다."""
    parameters: dict = {
        key: value
        for key, value in {
            "model_name": model_name,
            "model_version": model_version,
            "headers": dict(request.headers),
            "client_timeout": 5.0,
        }.items()
        if value
    }  # 파라미터 처리

    try:
        return triton.is_model_ready(**parameters)  # Triton Server 요청
    except InferenceServerException:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND)


@model_router.get(path="/{model_name}/details")
async def get_model_config(
    request: Request,
    model_name: Annotated[str, Path(description="모델 이름")],
    model_version: Annotated[Optional[str], Query(description="모델 버전")] = "",
) -> dict:
    """모델에 대한 세부 정보를 조회합니다.

    모델이 추론 준비 상태가 아닌 경우 오류가 발생합니다.
    """
    parameters: dict = {
        key: value
        for key, value in {
            "model_name": model_name,
            "model_version": model_version,
            "headers": dict(request.headers),
            "client_timeout": 5.0,
        }.items()
        if value
    }  # 파라미터 처리

    try:
        return triton.get_model_config(**parameters)  # Triton Server 요청
    except InferenceServerException:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND)


@model_router.get(path="/{model_name}/trace")
async def get_trace_settings(
    request: Request,
    model_name: Annotated[str, Path(description="모델 이름")],
) -> GetTraceSettingsResponse:
    """모델의 트레이스 설정을 조회합니다.

    모델이 추론 준비 상태가 아닌 경우 오류가 발생합니다.
    """
    parameters: dict = {
        key: value
        for key, value in {
            "model_name": model_name,
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


@model_router.put(path="/{model_name}/trace")
async def update_trace_settings(
    request: Request,
    model_name: Annotated[str, Path(description="모델 이름")],
    settings: UpdateTraceSettingsRequest,
) -> UpdateTraceSettingsResponse:
    """모델의 트레이스 설정을 업데이트합니다.

    모델이 추론 준비 상태가 아닌 경우 오류가 발생합니다.
    """
    parameters: dict = {
        key: value
        for key, value in {
            "model_name": model_name,
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


@model_router.get(path="/{model_name}/inference")
async def get_inference_statistics(
    request: Request,
    model_name: Annotated[str, Path(description="모델 이름")],
    model_version: Annotated[Optional[str], Query(description="모델 버전")] = "",
) -> GetInferenceStatisticsResponse:
    """모델의 추론 통계를 조회합니다.

    모델이 추론 준비 상태가 아닌 경우 오류가 발생합니다.
    """
    parameters: dict = {
        key: value
        for key, value in {
            "model_name": model_name,
            "model_version": model_version,
            "headers": dict(request.headers),
            "query_params": dict(request.query_params),
            "client_timeout": 5.0,
        }.items()
        if value
    }  # 파라미터 처리

    try:
        return GetInferenceStatisticsResponse(
            **triton.get_inference_statistics(**parameters)
        )  # Triton Server 요청
    except InferenceServerException:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND)


@model_router.post(path="/{model_name}/inference")
async def infer(
    request: Request,
    model_name: Annotated[str, Path(description="모델 이름")],
    infer_request_list: list[InferRequest],
) -> list[InferResponse]:
    response = []
    for infer_request in infer_request_list:
        parameters: dict = {
            key: value
            for key, value in {
                "model_name": model_name,
                "model_version": infer_request.model_version,
                "inputs": infer_request.inputs,
                "outputs": infer_request.outputs,
                "parameters": infer_request.parameters,
                "request_id": infer_request.request_id,
                "headers": dict(request.headers),
                "client_timeout": 60.0,
            }.items()
            if value
        }  # 파라미터 처리

        # TODO: 비동기로 한번에 요청해서 결과 모두 받은 후 반환하기
        # TODO: 비동기 전환 후, stream 플래그로 스트리밍 처리 구현
        infer_result = triton.infer(**parameters)
        response.append(infer_result)
    return [InferResponse(**message) for message in response]


@model_router.post(path="/{model_name}/inference/stream")
async def async_stream_infer(
    request: Request,
    model_name: Annotated[str, Path(description="모델 이름")],
    infer_request: StreamInferRequest,
) -> StreamingResponse:
    parameters: dict = {
        key: value
        for key, value in {
            "model_name": model_name,
            "model_version": infer_request.model_version,
            "inputs": infer_request.inputs,
            "outputs": infer_request.outputs,
            "parameters": infer_request.parameters,
            "request_id": infer_request.request_id,
            "headers": dict(request.headers),
            "timeout": 60,
        }.items()
        if value
    }  # 파라미터 처리
    return StreamingResponse(
        content=triton.stream_infer(**parameters), media_type="text/event-stream"
    )
