from src.core.api.triton_server import create_client
from src.common.settings import settings

triton_grpc_client = create_client(protocol="grpc", **settings.triton.grpc.model_dump())
triton_http_client = create_client(protocol="http", **settings.triton.http.model_dump())
