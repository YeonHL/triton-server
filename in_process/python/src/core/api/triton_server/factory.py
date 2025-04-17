"""Triton Inference Server 클라이언트 팩토리 모듈입니다.
이 모듈은 Triton Inference Server에 연결하기 위한 클라이언트를 생성하는
팩토리 패턴을 구현합니다. HTTP 또는 gRPC 프로토콜을 사용하여 서버와
통신할 수 있는 적절한 클라이언트 인스턴스를 반환합니다.

Functions:
    create_client: 프로토콜 타입에 맞는 Triton 클라이언트 인스턴스를 생성합니다.

Classes:
    Protocol: 클라이언트 프로토콜 타입을 정의하는 Enum 클래스

Examples:
    >>> from src.core.api.triton_server.client_factory import create_client, Protocol
    >>>
    >>> # HTTP 프로토콜 클라이언트 생성
    >>> http_client = create_client(
    ...     url="localhost:8000",
    ...     protocol=Protocol.HTTP,
    ...     verbose=True
    ... )
    >>>
    >>> # gRPC 프로토콜 클라이언트 생성
    >>> grpc_client = create_client(
    ...     url="localhost:8001",
    ...     protocol=Protocol.GRPC,
    ...     verbose=True,
    ...     ssl=True,
    ...     root_certificates="/path/to/cert.pem"
    ... )
"""

from enum import Enum
from typing import Any, Callable, Optional, Union

from src.core.api.triton_server.grpc_client import GRPCTritonClient
from src.core.api.triton_server.http_client import HTTPTritonClient


class Protocol(str, Enum):
    """클라이언트 프로토콜 타입을 정의하는 Enum 클래스입니다.

    Attributes:
        HTTP: HTTP 프로토콜 타입
        GRPC: gRPC 프로토콜 타입
    """

    HTTP = "http"
    GRPC = "grpc"


def create_client(
    url: str = "localhost:8000",
    protocol: str = "grpc",
    verbose: bool = False,
    ssl: bool = False,
    # GRPC 전용 파라미터
    root_certificates: Optional[str] = None,
    private_key: Optional[str] = None,
    certificate_chain: Optional[str] = None,
    creds: Optional[Any] = None,
    keepalive_options: Optional[Any] = None,
    channel_args: Optional[list[tuple]] = None,
    # HTTP 전용 파라미터
    concurrency: int = 1,
    connection_timeout: float = 60.0,
    network_timeout: float = 60.0,
    max_greenlets: Optional[int] = None,
    ssl_options: Optional[dict[str, Any]] = None,
    ssl_context_factory: Optional[Callable] = None,
    insecure: bool = False,
) -> Union["HTTPTritonClient", "GRPCTritonClient"]:
    """프로토콜 타입에 맞는 Triton 클라이언트 인스턴스를 생성합니다.

    Args:
        protocol (str): 클라이언트 프로토콜 타입 ("http" 또는 "grpc", 기본값: "grpc")
        **kwargs: 클라이언트 타입별 설정 파라미터

    공통 파라미터:
        url (str): 서버 URL (예: 'localhost:8001')
        verbose (bool): 상세 로그 출력 여부 (기본값: False)
        ssl (bool): SSL 암호화 사용 여부 (기본값: False)

    GRPC 전용 파라미터:
        root_certificates (str): PEM 인코딩된 루트 인증서 파일 경로
        private_key (str): PEM 인코딩된 프라이빗 키 파일 경로
        certificate_chain (str): PEM 인코딩된 인증서 체인 파일 경로
        creds (grpc.ChannelCredentials): GRPC 채널 인증 객체
        keepalive_options (KeepAliveOptions): GRPC KeepAlive 설정
        channel_args (List[Tuple]): GRPC 채널 인자 리스트

    HTTP 전용 파라미터:
        concurrency (int): 클라이언트 연결 수 (기본값: 1)
        connection_timeout (float): 연결 타임아웃 (기본값: 60.0초)
        network_timeout (float): 네트워크 타임아웃 (기본값: 60.0초)
        max_greenlets (int): 비동기 처리용 최대 그린렛 수
        ssl_options (dict): SSL 소켓 옵션 딕셔너리
        ssl_context_factory (callable): SSL 컨텍스트 생성 함수
        insecure (bool): 호스트명 인증서 검증 비활성화 (기본값: False)

    Raises:
        ValueError: 지원하지 않는 protocol이 전달된 경우

    Returns:
        Union[HttpTritonClient, GrpcTritonClient]: 생성된 클라이언트 인스턴스

    Notes:
        - GRPC가 기본 클라이언트 타입입니다.
        - SSL 관련 옵션은 ssl=True일 때만 적용됩니다.
        - HTTP 클라이언트는 기본적으로 동기식 작동이며, concurrency로 병렬 처리를 조절할 수 있습니다.
        - GRPC 클라이언트는 기본적으로 비동기식으로 작동합니다.
    """
    try:
        protocol_enum = Protocol(protocol.lower())
    except ValueError:
        raise ValueError(
            f"Unsupported protocol: {protocol}. "
            f"Usable: {[t.value for t in Protocol]}"
        )

    # 공통 파라미터를 딕셔너리로 구성
    params = {}
    param_dict = {"url": url, "verbose": verbose, "ssl": ssl}
    params.update({k: v for k, v in param_dict.items() if v is not None})

    # HTTP
    if protocol_enum == Protocol.HTTP:
        # HTTP 전용 파라미터
        http_param_dict = {
            "concurrency": concurrency,
            "connection_timeout": connection_timeout,
            "network_timeout": network_timeout,
            "max_greenlets": max_greenlets,
            "ssl_options": ssl_options,
            "ssl_context_factory": ssl_context_factory,
            "insecure": insecure,
        }

        # None이 아닌 값만 params에 추가
        params.update({k: v for k, v in http_param_dict.items() if v is not None})

        return HTTPTritonClient(**params)

    # GRPC
    grpc_param_dict = {
        "root_certificates": root_certificates,
        "private_key": private_key,
        "certificate_chain": certificate_chain,
        "creds": creds,
        "keepalive_options": keepalive_options,
        "channel_args": channel_args,
    }

    # None이 아닌 값만 params에 추가
    params.update({k: v for k, v in grpc_param_dict.items() if v is not None})
    return GRPCTritonClient(**params)
