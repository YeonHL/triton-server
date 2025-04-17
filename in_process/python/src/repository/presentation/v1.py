from fastapi import APIRouter, Request
import re

from src.common.api import triton_grpc_client as triton
from src.repository.presentation.dto import (
    GetModelRepositoryIndexResponse,
    GetReadyModelListResponse,
)

repository_router = APIRouter(
    prefix="/repository",
    tags=["repository"],
)

# get_model_repository_index에서 사용하는 변수
exclude_words = ["token", "process", "model"]
exclude_pattern = r"(" + "|".join(exclude_words) + r")"


@repository_router.get(path="", response_model_exclude_none=True)
async def get_model_repository_index(
    request: Request,
) -> GetModelRepositoryIndexResponse:
    """모델 저장소의 인덱스를 가져옵니다."""
    parameters: dict = {
        key: value
        for key, value in {
            "headers": dict(request.headers),
            "client_timeout": 5.0,
        }.items()
        if value
    }  # 파라미터 처리

    model_repository_index = triton.get_model_repository_index(
        **parameters
    )  # Triton Server 요청

    # 인덱스 처리
    model_list = []

    for model_info in model_repository_index:
        model_name = model_info.get("name", "")

        # 예외 단어가 있을 경우 결과에 미포함
        if re.search(exclude_pattern, model_name, re.IGNORECASE):
            continue

        model_list.append(model_info)

    return GetModelRepositoryIndexResponse(model_list=model_list)


@repository_router.get(path="/ready")
async def get_ready_model_list(request: Request):
    """추론 준비된 모델 목록을 가져옵니다."""
    parameters: dict = {
        key: value
        for key, value in {
            "headers": dict(request.headers),
            "client_timeout": 5.0,
        }.items()
        if value
    }  # 파라미터 처리
    model_repository_index = triton.get_model_repository_index(
        **parameters
    )  # Triton Server 요청

    # 인덱스 처리
    model_list = []

    for model_info in model_repository_index:
        model_name = model_info.get("name", "")

        # 예외 단어가 있을 경우 결과에 미포함
        if re.search(exclude_pattern, model_name, re.IGNORECASE):
            continue

        model_state: str = model_info.get("state", "")

        # 모델이 추론 준비 상태인 경우 포함
        if model_state.upper() == "READY":
            model_list.append(model_name)

    return GetReadyModelListResponse(model_list=model_list)
