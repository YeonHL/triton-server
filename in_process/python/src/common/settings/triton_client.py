from enum import Enum
from typing import Callable, Optional

from pydantic import BaseModel, Field


class HostAddress(str, Enum):
    """호스트 주소를 정의하는 Enum 클래스입니다.

    Attributes:
        LOCAL: 로컬 서버 IP
        CURRENT: 현재 서버 IP
    """

    LOCAL = "localhost"
    CURRENT = "0.0.0.0"

    def socket_address(self, port: int) -> str:
        return f"{self.value}:{port}"

    def http_url(self, port: int) -> str:
        return f"http://{self.value}:{port}"

    def https_url(self, port: int) -> str:
        return f"https://{self.value}:{port}"


class BaseTritonClientSettings(BaseModel):
    """Triton Inference Server 클라이언트의 기본 설정을 관리하는 베이스 클래스입니다.

    모든 프로토콜 타입(HTTP, gRPC)에서 공통적으로 사용되는 설정값들을 정의합니다.

    Attributes:
        url (str): Inference 서버 URL (예: 'localhost:8001')
        verbose (bool): 상세 로그 출력 여부
        ssl (bool): SSL 암호화 사용 여부
    """

    url: str = Field(..., description="Inference 서버 URL")
    verbose: bool = Field(default=False, description="상세 로그 출력 여부")
    ssl: bool = Field(default=False, description="SSL 암호화 사용 여부")


class HTTPTritonClientSettings(BaseTritonClientSettings):
    """HTTP 프로토콜을 사용하는 Triton 클라이언트의 설정을 관리하는 클래스입니다.

    Attributes:
        verbose (bool): 상세 로그 출력 여부
        ssl (bool): SSL 암호화 사용 여부
        concurrency (int): 클라이언트 연결 수 (기본값: 1)
        connection_timeout (float): 연결 타임아웃 (기본값: 60.0초)
        network_timeout (float): 네트워크 타임아웃 (기본값: 60.0초)
        max_greenlets (int): 비동기 처리용 최대 그린렛 수
        ssl_options (dict): SSL 소켓 옵션 딕셔너리
        ssl_context_factory (callable): SSL 컨텍스트 생성 함수
        insecure (bool): 호스트명 인증서 검증 비활성화 (기본값: False)

    Example:
        >>> http_settings = HTTPTritonClientSettings(
        ...     url="localhost:8000",
        ...     concurrency=4,
        ...     connection_timeout=30.0
        ... )
    """

    concurrency: int = Field(default=1, description="클라이언트 연결 수")
    connection_timeout: float = Field(default=60.0, description="연결 타임아웃 (초)")
    network_timeout: float = Field(default=60.0, description="네트워크 타임아웃 (초)")
    max_greenlets: Optional[int] = Field(
        default=None, description="비동기 처리용 최대 그린렛 수"
    )
    ssl_options: Optional[dict] = Field(
        default=None, description="SSL 소켓 옵션 딕셔너리"
    )
    ssl_context_factory: Optional[Callable] = Field(
        default=None, description="SSL 컨텍스트 생성 함수"
    )
    insecure: bool = Field(default=False, description="호스트명 인증서 검증 비활성화")


class KeepAliveOptions(BaseModel):
    """GRPC KeepAlive 옵션을 관리하는 클래스입니다.

    GRPC 연결의 KeepAlive 관련 매개변수들을 캡슐화합니다.
    자세한 내용은 https://github.com/grpc/grpc/blob/master/doc/keepalive.md 를 참조하세요.

    Attributes:
        keepalive_time_ms (int): 전송 계층에서 keepalive ping을 보내는 주기(밀리초)
            기본값은 None으로, GRPC의 기본값(INT32_MAX)이 사용됩니다.

        keepalive_timeout_ms (int): keepalive ping의 응답을 기다리는 시간(밀리초)
            이 시간 내에 응답을 받지 못하면 연결이 종료됩니다.
            기본값은 None으로, GRPC의 기본값(20000, 20초)이 사용됩니다.

        keepalive_permit_without_calls (bool): 활성 호출이 없을 때도 ping 허용 여부
            True로 설정하면 진행 중인 호출이 없어도 ping을 보낼 수 있습니다.
            기본값은 False입니다.

        http2_max_pings_without_data (int): 데이터/헤더 프레임 없이 보낼 수 있는 최대 ping 수
            이 제한을 초과하면 GRPC Core는 더 이상 ping을 보내지 않습니다.
            0으로 설정하면 제한 없이 ping을 보낼 수 있습니다.
            기본값은 2입니다.

    Example:
        >>> options = KeepAliveOptions(
        ...     keepalive_time_ms=30000,      # 30초마다 ping
        ...     keepalive_timeout_ms=10000,    # 10초 타임아웃
        ...     keepalive_permit_without_calls=True
        ... )
    """

    keepalive_time_ms: Optional[int] = Field(
        default=None, description="전송 계층에서 keepalive ping을 보내는 주기(밀리초)"
    )
    keepalive_timeout_ms: Optional[int] = Field(
        default=None, description="keepalive ping의 응답을 기다리는 시간(밀리초)"
    )
    keepalive_permit_without_calls: bool = Field(
        default=False, description="활성 호출이 없을 때도 ping 허용 여부"
    )
    http2_max_pings_without_data: int = Field(
        default=2, description="데이터/헤더 프레임 없이 보낼 수 있는 최대 ping 수"
    )


class GRPCTritonClientSettings(BaseTritonClientSettings):
    """gRPC 프로토콜을 사용하는 Triton 클라이언트의 설정을 관리하는 클래스입니다.

    Attributes:
        verbose (bool): 상세 로그 출력 여부
        ssl (bool): SSL 암호화 사용 여부
        root_certificates (str): PEM 인코딩된 루트 인증서 파일 경로
        private_key (str): PEM 인코딩된 프라이빗 키 파일 경로
        certificate_chain (str): PEM 인코딩된 인증서 체인 파일 경로
        keepalive_options (KeepAliveOptions): GRPC KeepAlive 설정
        channel_args (list[tuple]): GRPC 채널 인자 리스트

    Example:
        >>> grpc_settings = GRPCTritonClientSettings(
        ...     url="localhost:8001",
        ...     keepalive_options=KeepAliveOptions(keepalive_time_ms=30000)
        ... )
    """

    root_certificates: Optional[str] = Field(
        default=None, description="PEM 인코딩된 루트 인증서 파일 경로"
    )
    private_key: Optional[str] = Field(
        default=None, description="PEM 인코딩된 프라이빗 키 파일 경로"
    )
    certificate_chain: Optional[str] = Field(
        default=None, description="PEM 인코딩된 인증서 체인 파일 경로"
    )
    keepalive_options: Optional[KeepAliveOptions] = Field(
        default=None, description="GRPC KeepAlive 옵션"
    )
    channel_args: Optional[list[tuple]] = Field(
        default=None, description="GRPC 채널 인자 리스트"
    )


class TritonSettings(BaseModel):
    """Triton Inference Server 클라이언트 설정을 관리하는 클래스입니다.

    Attributes:
        http (HTTPTritonClientSettings): HTTP 프로토콜 설정
        grpc (GRPCTritonClientSettings): gRPC 프로토콜 설정
    """

    http: HTTPTritonClientSettings = Field(
        default=HTTPTritonClientSettings(url=HostAddress.CURRENT.socket_address(18201)),
        description="HTTP Triton 클라이언트 설정",
    )
    grpc: GRPCTritonClientSettings = Field(
        default=GRPCTritonClientSettings(url=HostAddress.CURRENT.socket_address(18202)),
        description="GRPC Triton 클라이언트 설정",
    )
