from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse

from src.common.api import triton_grpc_client as triton
from src.generate.presentation.dto import GenerateChatRequest, GenerateEmbeddingRequest
from src.common.type import Tensor


generate_router = APIRouter(
    prefix="/generate",
    tags=["generate"],
)


@generate_router.post(path="/chat")
async def generate_chat(request: Request, generate_request: GenerateChatRequest):
    """채팅을 생성합니다.

    스트리밍 형태 (SSE)로 반환하며 한번에 하나의 텍스트만 요청할 수 있습니다.
    """
    inputs = [
        Tensor(
            name="text_input",
            datatype="BYTES",
            shape=[1, 1],
            data=[[generate_request.text_input]],
        ),
        Tensor(name="stream", datatype="BOOL", shape=[1, 1], data=[[True]]),
    ]

    if generate_request.image:
        inputs.append(
            Tensor(
                name="image",
                datatype="BYTES",
                shape=[1, 1],
                data=[[generate_request.image]],
            )
        )

    parameters: dict = {
        key: value
        for key, value in {
            "model_name": generate_request.model_name,
            "inputs": inputs,
            "outputs": [Tensor(name="text_output")],
            "parameters": generate_request.parameters,
            "headers": dict(request.headers),
            "timeout": 60,
        }.items()
        if value
    }  # 파라미터 처리

    return StreamingResponse(
        content=triton.stream_infer(**parameters), media_type="text/event-stream"
    )


@generate_router.post(path="/embedding")
async def generate_embedding(
    request: Request, generate_request: GenerateEmbeddingRequest
) -> list[list[float]]:
    """임베딩 생성 요청

    여러 텍스트를 전달할 수 있습니다. 각각의 임베딩 결과를 반환합니다.
    """
    inputs = [
        Tensor(
            name="text_input",
            datatype="BYTES",
            shape=[len(generate_request.text_inputs), 1],
            data=[[text] for text in generate_request.text_inputs],
        )
    ]
    parameters: dict = {
        key: value
        for key, value in {
            "model_name": generate_request.model_name,
            "inputs": inputs,
            "outputs": [Tensor(name="sentence_embedding")],
            "parameters": generate_request.parameters,
            "headers": dict(request.headers),
            "client_timeout": 60.0,
        }.items()
        if value
    }  # 파라미터 처리

    return triton.infer(**parameters).get("outputs", [{"data": None}])[0].get("data")
