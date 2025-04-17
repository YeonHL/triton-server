"""Triton 추론 서버와 HTTP 통신을 위한 클라이언트 모듈입니다.
이 모듈은 NVIDIA Triton Inference Server와 통신하기 위한 HTTP 클라이언트를 제공합니다.
HTTPTritonClient 클래스를 포함하며, 이 클래스는 서버 상태 확인, 모델 관리,
그리고 기본 InferenceServerClient의 모든 기능에 접근하는 기능을 제공합니다.

Classes:
    HTTPTritonClient: Triton 추론 서버와 HTTP 통신을 위한 유틸리티 클래스

Dependencies:
    - tritonclient

Examples:
    >>> from src.core.api.triton_server.http_client import HTTPTritonClient
    >>>
    >>> # HTTPTritonClient 인스턴스 생성
    >>> client = HTTPTritonClient(url="localhost:8000")
    >>>
    >>> # 서버 상태 확인
    >>> is_healthy, message = client.check_server_health()
    >>> print(f"Server health: {is_healthy}, Message: {message}")
    >>>
    >>> # 특정 모델 상태 확인
    >>> model_ready, status_msg = client.check_model_status("my_model")
    >>> print(f"Model status: {status_msg}")
    >>>
    >>> # 기본 클라이언트 메서드 직접 사용
    >>> metadata = client.get_model_metadata("my_model")

"""

from typing import Any

from tritonclient.http import InferenceServerClient as HTTPClient

from src.core.api.triton_server.base import BaseTritonClient


class HTTPTritonClient(BaseTritonClient):
    """Triton 추론 서버와 HTTP 통신을 위한 유틸리티 클래스입니다.
    이 클래스는 Triton Inference Server와 HTTP 프로토콜을 통해 통신하며,
    서버 상태 확인, 모델 관리, 그리고 기본 InferenceServerClient의
    모든 기능에 접근하는 기능을 제공합니다.

    Attributes:
        client (HTTPClient): 기본 Triton InferenceServerClient 인스턴스

    Methods:
        update_settings: 클라이언트 설정을 업데이트합니다.
        check_server_health: 서버의 상태를 확인합니다.
        check_model_status: 특정 모델의 상태를 확인합니다.
        get_log_settings: 서버 로그 설정을 반환합니다.
    """

    def __init__(
        self,
        url,
        verbose=False,
        concurrency=1,
        connection_timeout=60.0,
        network_timeout=60.0,
        max_greenlets=None,
        ssl=False,
        ssl_options=None,
        ssl_context_factory=None,
        insecure=False,
    ):
        """Triton 서버와 통신하기 위한 클라이언트를 초기화합니다.

        Args:
            url (str): 서버 URL (예: 'localhost:8001')
            verbose (bool): 상세 로그 출력 여부 (기본값: False)
            concurrency (int): 클라이언트 연결 수 (기본값: 1)
            connection_timeout (float): 연결 타임아웃 (기본값: 60.0초)
            network_timeout (float): 네트워크 타임아웃 (기본값: 60.0초)
            max_greenlets (int): 비동기 처리용 최대 그린렛 수
            ssl (bool): SSL 암호화 사용 여부 (기본값: False)
            ssl_options (dict): SSL 소켓 옵션 딕셔너리
            ssl_context_factory (callable): SSL 컨텍스트 생성 함수
            insecure (bool): 호스트명 인증서 검증 비활성화 (기본값: False)

        Raises:
            ValueError: 지원하지 않는 protocol이 전달된 경우

        Notes:
            - SSL 관련 옵션은 ssl=True일 때만 적용됩니다.
            - HTTP 클라이언트는 기본적으로 동기식 작동이며, concurrency로 병렬 처리를 조절할 수 있습니다.
        """
        # 설정 저장
        self._settings = {
            "url": url,
            "verbose": verbose,
            "concurrency": concurrency,
            "connection_timeout": connection_timeout,
            "network_timeout": network_timeout,
            "max_greenlets": max_greenlets,
            "ssl": ssl,
            "ssl_options": ssl_options,
            "ssl_context_factory": ssl_context_factory,
            "insecure": insecure,
        }

        # 클라이언트 생성 및 설정
        self._client = HTTPClient(**self._settings)

    def __getattr__(self, name: str) -> Any:
        """클라이언트 메서드에 대한 직접 접근을 지원합니다."""
        return getattr(self._client, name)

    @property
    def client(self) -> HTTPClient:
        """기본 InferenceServerClient 인스턴스에 대한 직접 접근을 제공합니다."""
        return self._client

    def close(self):
        """클라이언트 리소스를 정리합니다.

        연결을 종료하고 할당된 리소스를 해제합니다.
        컨텍스트 매니저 종료 시 자동으로 호출됩니다.

        Notes:
            - 내부 GRPC 클라이언트 연결을 종료합니다.
            - 할당된 리소스를 해제합니다.
        """

        if hasattr(self, "_client") and self._client is not None:
            try:
                # GRPC 클라이언트 종료 메서드 호출
                if hasattr(self._client, "close"):
                    self._client.close()
            except Exception as e:
                # 종료 중 발생한 예외 기록 (선택적으로 로깅 가능)
                if self._settings.get("verbose", False):
                    raise e
            finally:
                # 클라이언트 참조 해제하여 메모리 정리 지원
                self._client = None

    def update_settings(self, **kwargs) -> None:
        """클라이언트 설정을 업데이트합니다.

        Args:
            url (str): 서버 URL (예: 'localhost:8001')
            verbose (bool): 상세 로그 출력 여부 (기본값: False)
            concurrency (int): 클라이언트 연결 수 (기본값: 1)
            connection_timeout (float): 연결 타임아웃 (기본값: 60.0초)
            network_timeout (float): 네트워크 타임아웃 (기본값: 60.0초)
            max_greenlets (int): 비동기 처리용 최대 그린렛 수
            ssl (bool): SSL 암호화 사용 여부 (기본값: False)
            ssl_options (dict): SSL 소켓 옵션 딕셔너리
            ssl_context_factory (callable): SSL 컨텍스트 생성 함수
            insecure (bool): 호스트명 인증서 검증 비활성화 (기본값: False)

        Raises:
            TypeError: 제공하지 않는 설정값을 업데이트 시도할 경우
        """
        # 기존 설정에 새로운 설정 업데이트
        self._settings.update(kwargs)

        # 클라이언트 재생성
        self._client = HTTPClient(**self._settings)

    def check_server_health(self) -> tuple[bool, str]:
        try:
            if not self.is_server_live():
                return False, "Server is not live"
            if not self.is_server_ready():
                return False, "Server is not ready"
            return True, "Server is healthy"
        except Exception as e:
            return False, f"Server health check failed: {str(e)}"

    def check_model_status(self, model_name: str) -> tuple[bool, str]:
        try:
            if self.client.is_model_ready(model_name):
                return True, f"Model {model_name} is ready"

            server_status, message = self.check_server_health()

            if server_status:
                return False, f"Model {model_name} is not ready"

            return False, message

        except Exception as e:
            return False, f"Model status check failed: {str(e)}"
