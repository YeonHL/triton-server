"""Triton 서버와 통신을 위한 유틸리티 모듈입니다.
이 모듈은 Triton 추론 서버와 통신하기 위한 기본 인터페이스를 제공합니다.
BaseTritonClient 추상 클래스를 포함하며, 이 클래스는 서버 상태 확인,
모델 목록 조회, 모델 입출력 정보 조회 등의 기능을 정의합니다.

Classes:
    BaseTritonClient: Triton 서버와 통신하기 위한 기본 클라이언트 인터페이스

Examples:
    >>> from triton_client import ConcreteTritonClient  # BaseTritonClient 구현체 예시
    >>>
    >>> # TritonClient 인스턴스 생성
    >>> client = ConcreteTritonClient(
    …     url="localhost:8001",
    …     verbose=True
    … )
    …
    >>> # 서버 상태 확인
    >>> status, message = client.check_server_health()
    …
    >>> # 모델 목록 조회
    >>> models = client.get_model_list(loaded=True)
    …
    >>> # 모델 입출력 정보 조회
    >>> model_io = client.get_model_io()
    …
    >>> # 클라이언트 설정 업데이트
    >>> client.update_settings(url="localhost:8002", verbose=False)
"""

from abc import ABC, abstractmethod
from typing import Any


class BaseTritonClient(ABC):
    """Triton 서버와 통신을 위한 유틸리티 모듈입니다.
    이 모듈은 Triton 추론 서버와 통신하기 위한 기본 인터페이스를 제공합니다.
    BaseTritonClient 추상 클래스를 포함하며, 이 클래스는 서버 상태 확인,
    모델 목록 조회, 모델 입출력 정보 조회 등의 기능을 정의합니다.

    Classes:
        BaseTritonClient: Triton 서버와 통신하기 위한 기본 클라이언트 인터페이스

    Examples:
        >>> from triton_client import ConcreteTritonClient  # BaseTritonClient 구현체 예시
        >>>
        >>> # TritonClient 인스턴스 생성
        >>> client = ConcreteTritonClient(
        …     url="localhost:8001",
        …     verbose=True
        … )
        …
        >>> # 서버 상태 확인
        >>> status, message = client.check_server_health()
        …
        >>> # 모델 목록 조회
        >>> models = client.get_model_list(loaded=True)
        …
        >>> # 모델 입출력 정보 조회
        >>> model_io = client.get_model_io()
        …
        >>> # 클라이언트 설정 업데이트
        >>> client.update_settings(url="localhost:8002", verbose=False)

    Note:
        이 모듈을 사용하기 위해서는 Triton 추론 서버가 실행 중이어야 합니다.
    """

    def __init__(self, url: str, **kwargs):
        """기본 클라이언트를 초기화합니다.

        Args:
            url (str): 서버 URL (예: 'localhost:8001')
            **kwargs: 추가 설정 파라미터
        """
        self.url = url
        self._settings = kwargs

    def __enter__(self):
        """with 문을 통한 컨텍스트 관리를 위한 진입점.

        Returns:
            BaseTritonClient: 자기 자신(클라이언트 인스턴스)
        """
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """with 문 컨텍스트 종료 시 호출되는 메서드.

        컨텍스트 종료 시 필요한 리소스 정리 작업을 수행합니다.

        Args:
            exc_type: 예외 타입 (예외 발생 시)
            exc_val: 예외 값 (예외 발생 시)
            exc_tb: 예외 트레이스백 (예외 발생 시)

        Returns:
            bool: 예외를 처리했는지 여부 (True 반환 시 예외 전파 방지)
        """
        self.close()
        return False  # 발생한 예외를 상위로 전파

    def __deepcopy__(self):
        return self.__class__(**self.settings)

    @abstractmethod
    def close(self):
        """클라이언트 리소스를 정리합니다.

        연결을 종료하고 할당된 리소스를 해제합니다.
        컨텍스트 매니저 종료 시 자동으로 호출됩니다.
        """

    @property
    def settings(self) -> dict[str, Any]:
        """현재 클라이언트 설정을 반환합니다."""
        return self._settings.copy()

    @abstractmethod
    def update_settings(self, **kwargs) -> None:
        """클라이언트 설정을 업데이트합니다.

        기존 설정값에 새로운 설정값을 업데이트하고 클라이언트를 재생성합니다.
        변경된 설정은 이 클라이언트 인스턴스를 사용하는 모든 곳에 즉시 반영됩니다.

        Args:
            **kwargs: 업데이트할 설정값들
                url (str): 서버 URL
                verbose (bool): 상세 로그 출력 여부
                등 클라이언트 생성자에서 지원하는 모든 파라미터

        Raises:
            ValueError: 지원하지 않는 protocol이 전달된 경우
        """

    @abstractmethod
    def check_server_health(self) -> tuple[bool, str]:
        """
        서버의 활성화 및 로드 완료 여부를 확인합니다.

        Returns:
            tuple[bool, str]: 서버 상태 및 메시지.
                            첫 번째 요소는 서버가 정상인지 여부(True/False),
                            두 번째 요소는 상태 메시지
        """

    @abstractmethod
    def check_model_status(self) -> tuple[bool, str]:
        """
        모델의 로드 및 추론 준비 여부를 확인합니다.

        Returns:
            tuple[bool, str]: 모델 상태 및 메시지.
                            첫 번째 요소는 모델이 로드되고 사용 가능한지 여부(True/False),
                            두 번째 요소는 상태 메시지
        """
