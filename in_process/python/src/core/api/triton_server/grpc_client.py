"""Triton 추론 서버와 gRPC 통신을 위한 클라이언트 모듈입니다.
이 모듈은 NVIDIA Triton Inference Server와 통신하기 위한 gRPC 클라이언트를 제공합니다.
GRPCTritonClient 클래스를 포함하며, 이 클래스는 서버 상태 확인, 모델 관리,
그리고 기본 InferenceServerClient의 모든 기능에 접근하는 기능을 제공합니다.

Classes:
    GRPCTritonClient: Triton Inference Server와 GRPC 통신을 위한 클라이언트 클래스

Dependencies:
    - tritonclient

Examples:
    >>> from src.core.api.triton_server.grpc_client import GRPCTritonClient
    >>>
    >>> # GRPCTritonClient 인스턴스 생성
    >>> client = GRPCTritonClient(
    …     url="localhost:8001",
    …     verbose=True
    … )
    …
    >>> # 서버 메타데이터 조회
    >>> metadata = client.get_server_metadata()
    …
    >>> # 서버 상태 확인
    >>> status, message = client.check_server_health()
    …
    >>> # 모델 메타데이터 조회
    >>> model_metadata = client.get_model_metadata("model_name")
"""

import json
import queue
import threading
from typing import Any, Optional, TYPE_CHECKING
from functools import partial

import numpy as np
from tritonclient.grpc import InferenceServerClient as GRPCClient
from tritonclient.grpc import InferInput, InferRequestedOutput
from tritonclient.utils import triton_to_np_dtype

from src.core.api.triton_server.base import BaseTritonClient

if TYPE_CHECKING:
    from tritonclient.grpc import InferResult


class GRPCTritonClient(BaseTritonClient):
    """Triton Inference Server와 GRPC 통신을 위한 클라이언트 클래스입니다.

    이 클래스는 GRPC 프로토콜을 사용하여 Triton Inference Server와 통신하며,
    서버 메타데이터, 모델 정보 조회, 상태 확인 등 다양한 기능을 제공합니다.

    Attributes:
        client: 기본 InferenceServerClient 인스턴스에 대한 접근자
        settings (dict): 클라이언트 설정 정보를 저장하는 딕셔너리

    Methods:
        update_settings: 클라이언트 설정을 업데이트합니다
        get_server_metadata: 추론 서버의 메타데이터를 가져옵니다
        get_model_repository_index: 모델 저장소 컨텐츠의 인덱스를 가져옵니다
        check_server_health: 서버 상태를 확인합니다
        check_model_status: 특정 모델의 상태를 확인합니다
        get_model_metadata: 특정 모델의 메타데이터를 가져옵니다
        get_model_config: 특정 모델의 설정을 가져옵니다
        get_inference_statistics: 추론 통계를 가져옵니다
        get_log_settings: 서버의 로그 설정을 가져옵니다
        get_trace_settings: 서버의 트레이스 설정을 가져옵니다
    """

    def __init__(
        self,
        url,
        verbose=False,
        ssl=False,
        root_certificates=None,
        private_key=None,
        certificate_chain=None,
        creds=None,
        keepalive_options=None,
        channel_args=None,
    ):
        """Triton 서버와 통신하기 위한 클라이언트를 초기화합니다.

        Args:
            url (str): 서버 URL (예: 'localhost:8001')
            verbose (bool): 상세 로그 출력 여부 (기본값: False)
            ssl (bool): SSL 암호화 사용 여부 (기본값: False)
            root_certificates (str): PEM 인코딩된 루트 인증서 파일 경로
            private_key (str): PEM 인코딩된 프라이빗 키 파일 경로
            certificate_chain (str): PEM 인코딩된 인증서 체인 파일 경로
            creds (grpc.ChannelCredentials): GRPC 채널 인증 객체
            keepalive_options (KeepAliveOptions): GRPC KeepAlive 설정
            channel_args (List[Tuple]): GRPC 채널 인자 리스트

        Raises:
            ValueError: 지원하지 않는 protocol이 전달된 경우

        Notes:
            - SSL 관련 옵션은 ssl=True일 때만 적용됩니다.
            - GRPC 클라이언트는 기본적으로 비동기식으로 작동합니다.
        """
        # 설정 저장
        self._settings = {
            "url": url,
            "verbose": verbose,
            "ssl": ssl,
            "root_certificates": root_certificates,
            "private_key": private_key,
            "certificate_chain": certificate_chain,
            "creds": creds,
            "keepalive_options": keepalive_options,
            "channel_args": channel_args,
        }

        # 클라이언트 생성 및 설정
        self._client: GRPCClient = GRPCClient(**self._settings)

    def __getattr__(self, name: str) -> Any:
        """클라이언트 메서드에 대한 직접 접근을 지원합니다."""
        return getattr(self._client, name)

    @property
    def client(self) -> GRPCClient:
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

    def update_settings(self, **kwargs) -> None:
        """클라이언트 설정을 업데이트합니다.

        Args:
            url (str): 서버 URL (예: 'localhost:8001')
            verbose (bool): 상세 로그 출력 여부 (기본값: False)
            ssl (bool): SSL 암호화 사용 여부 (기본값: False)
            root_certificates (str): PEM 인코딩된 루트 인증서 파일 경로
            private_key (str): PEM 인코딩된 프라이빗 키 파일 경로
            certificate_chain (str): PEM 인코딩된 인증서 체인 파일 경로
            creds (grpc.ChannelCredentials): GRPC 채널 인증 객체
            keepalive_options (KeepAliveOptions): GRPC KeepAlive 설정
            channel_args (List[Tuple]): GRPC 채널 인자 리스트

        Raises:
            TypeError: 제공하지 않는 설정값을 업데이트 시도할 경우
        """
        # 기존 설정에 새로운 설정 업데이트
        self._settings.update(kwargs)

        # 클라이언트 재생성
        self._client = GRPCClient(**self._settings)

    def get_server_metadata(
        self, headers=None, as_json=True, client_timeout=None
    ) -> dict:
        """추론 서버에 연결하여 메타데이터를 가져옵니다.

        Args
        ----------
        headers: dict
            요청에 포함할 추가 HTTP 헤더를 지정하는 선택적 딕셔너리입니다.
        as_json : bool
            True이면 서버 메타데이터를 json 딕셔너리로 반환하고,
            False이면 protobuf 메시지로 반환합니다. 기본값은 False입니다.
            반환되는 json은 protobuf 메시지에서 MessageToJson을 사용하여 생성되며,
            결과적으로 int64 값은 문자열로 표현됩니다.
            필요한 경우 이러한 문자열을 다시 int64 값으로 변환하는 것은 호출자의 책임입니다.
        client_timeout: float
            요청이 허용되는 최대 종단간 시간(초)입니다.
            지정된 시간이 경과하면 클라이언트는 요청을 중단하고
            "Deadline Exceeded" 메시지와 함께 InferenceServerException을 발생시킵니다.
            기본값은 None이며, 이는 클라이언트가 서버의 응답을 기다린다는 의미입니다.

        Returns
        -------
        dict 또는 protobuf message
            메타데이터를 포함하는 JSON 딕셔너리 또는 ServerMetadataResponse 메시지

        Raises
        ------
        InferenceServerException
            서버 메타데이터를 가져올 수 없거나 시간 초과된 경우 발생

        Example
        ------
        ```
        {'extensions': ['classification',
            'sequence',
            'model_repository',
            'model_repository(unload_dependents)',
            'schedule_policy',
            'model_configuration',
            'system_shared_memory',
            'cuda_shared_memory',
            'binary_tensor_data',
            'parameters',
            'statistics',
            'trace',
            'logging'],
        'name': 'triton',
        'version': '2.54.0'}
        ```
        """
        metadata = self._client.get_server_metadata(
            headers=headers, as_json=as_json, client_timeout=client_timeout
        )

        if not metadata:
            raise ValueError("Failed to load server metadata.")

        return metadata

    def get_model_repository_index(
        self, headers=None, as_json=True, client_timeout=None
    ) -> list[dict]:
        """모델 저장소 컨텐츠의 인덱스를 가져옵니다.

        Args:
            headers (dict): 요청에 포함할 추가 HTTP 헤더를 지정하는 선택적 딕셔너리입니다.
            as_json (bool): True이면 모델 저장소 인덱스를 json 딕셔너리로 반환하고,
                그렇지 않으면 protobuf 메시지로 반환합니다.
                기본값은 True입니다.
                반환되는 json은 MessageToJson을 사용하여 protobuf 메시지에서
                생성되므로 int64 값은 문자열로 표현됩니다. 필요에 따라
                이러한 문자열을 int64 값으로 다시 변환하는 것은 호출자의 책임입니다.
            client_timeout (float): 요청이 허용되는 최대 end-to-end 시간(초)입니다.
                지정된 시간이 경과하면 클라이언트는 요청을 중단하고
                "Deadline Exceeded" 메시지와 함께 InferenceServerException을 발생시킵니다.
                기본값은 None으로, 클라이언트가 서버의 응답을 기다립니다.

        Returns:
            list[dict] 또는 protobuf 메시지: 모델 저장소 인덱스를 담고 있는
                JSON 딕셔너리 또는 RepositoryIndexResponse 메시지입니다.

        Example:
            ```
            [{'name': 'bge-m3'},
            {'name': 'bge-m3-model'},
            {'name': 'bge-m3-preprocess'},
            {'name': 'gemma-2-27b'},
            {'name': 'llama-3.1-8b', 'state': 'READY', 'version': '2'},
            {'name': 'paligemma-3b'},
            {'name': 'paligemma-3b-test'},
            {'name': 'vgt'},
            {'name': 'vgt-model'},
            {'name': 'vgt-post-processing'},
            {'name': 'vgt-pre-processing'}]
            ```
        """
        model_repository_index = self._client.get_model_repository_index(
            headers=headers,
            as_json=as_json,
            client_timeout=client_timeout,
        )

        if model_repository_index:
            return model_repository_index.get("models")

        return []

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
            if self.is_model_ready(model_name):
                return True, f"Model {model_name} is ready"

            server_status, message = self.check_server_health()

            if server_status:
                return False, f"Model {model_name} is not ready"

            return False, message

        except Exception as e:
            return False, f"Model status check failed: {str(e)}"

    def get_model_metadata(
        self,
        model_name,
        model_version="",
        headers=None,
        as_json=True,
        client_timeout=None,
    ) -> dict:
        """추론 서버에 연결하여 지정된 모델의 메타데이터를 가져옵니다.

        Args:
            model_name (str): 모델의 이름
            model_version (str): 메타데이터를 가져올 모델의 버전입니다.
                기본값은 빈 문자열이며, 이 경우 서버가 모델과 내부 정책에 따라
                버전을 선택합니다.
            headers (dict): 요청에 포함할 추가 HTTP 헤더를 지정하는
                선택적 딕셔너리입니다.
            as_json (bool): True이면 모델 메타데이터를 json 딕셔너리로 반환하고,
                그렇지 않으면 protobuf 메시지로 반환합니다.
                기본값은 True입니다.
                반환되는 json은 MessageToJson을 사용하여 protobuf 메시지에서
                생성되므로 int64 값은 문자열로 표현됩니다. 필요에 따라
                이러한 문자열을 int64 값으로 다시 변환하는 것은 호출자의 책임입니다.
            client_timeout (float): 요청이 허용되는 최대 end-to-end 시간(초)입니다.
                지정된 시간이 경과하면 클라이언트는 요청을 중단하고
                "Deadline Exceeded" 메시지와 함께 InferenceServerException을 발생시킵니다.
                기본값은 None으로, 클라이언트가 서버의 응답을 기다립니다.

        Returns:
            dict 또는 protobuf 메시지: 메타데이터를 담고 있는
                JSON 딕셔너리 또는 ModelMetadataResponse 메시지입니다.

        Raises:
            InferenceServerException: 모델 메타데이터를 가져올 수 없거나
                시간이 초과된 경우 발생합니다.

        Examples:
            ```
            {'inputs': [{'datatype': 'BYTES', 'name': 'text_input', 'shape': ['1']},
                        {'datatype': 'BYTES', 'name': 'image', 'shape': ['-1']},
                        {'datatype': 'BOOL', 'name': 'stream', 'shape': ['1']},
                        {'datatype': 'BYTES',
                        'name': 'sampling_parameters',
                        'shape': ['1']},
                        {'datatype': 'BOOL',
                        'name': 'exclude_input_in_output',
                        'shape': ['1']},
                        {'datatype': 'BOOL',
                        'name': 'return_finish_reason',
                        'shape': ['1']},
                        {'datatype': 'BOOL',
                        'name': 'return_cumulative_logprob',
                        'shape': ['1']},
                        {'datatype': 'BOOL', 'name': 'return_logprobs', 'shape': ['1']},
                        {'datatype': 'BOOL',
                        'name': 'return_num_input_tokens',
                        'shape': ['1']},
                        {'datatype': 'BOOL',
                        'name': 'return_num_output_tokens',
                        'shape': ['1']}],
            'name': 'llama-3.1-8b',
            'outputs': [{'datatype': 'BYTES', 'name': 'text_output', 'shape': ['-1']},
                        {'datatype': 'BYTES', 'name': 'finish_reason', 'shape': ['-1']},
                        {'datatype': 'FP32',
                        'name': 'cumulative_logprob',
                        'shape': ['-1']},
                        {'datatype': 'BYTES', 'name': 'logprobs', 'shape': ['-1']},
                        {'datatype': 'UINT32', 'name': 'num_input_tokens', 'shape': ['1']},
                        {'datatype': 'UINT32',
                        'name': 'num_output_tokens',
                        'shape': ['-1']}],
            'platform': 'vllm',
            'versions': ['2']}
            ```
        """
        metadata = self._client.get_model_metadata(
            model_name=model_name,
            model_version=model_version,
            headers=headers,
            as_json=as_json,
            client_timeout=client_timeout,
        )

        if not metadata:
            raise ValueError("Failed to load model metadata.")

        return metadata

    def get_model_config(
        self,
        model_name,
        model_version="",
        headers=None,
        as_json=True,
        client_timeout=None,
    ):
        """추론 서버에 연결하여 지정된 모델의 설정을 가져옵니다.

        Args:
            model_name (str): 모델 이름
            model_version (str): 설정을 가져올 모델 버전. 기본값은 빈 문자열로,
                서버가 모델과 내부 정책에 기반하여 버전을 선택합니다.
            headers (dict): 요청에 포함할 추가 HTTP 헤더를 지정하는 선택적 딕셔너리입니다.
            as_json (bool): True이면 설정을 json 딕셔너리로 반환하고,
                그렇지 않으면 protobuf 메시지로 반환합니다.
                기본값은 True입니다.
                반환되는 json은 MessageToJson을 사용하여 protobuf 메시지에서
                생성되므로 int64 값은 문자열로 표현됩니다. 필요에 따라
                이러한 문자열을 int64 값으로 다시 변환하는 것은 호출자의 책임입니다.
            client_timeout (float): 요청이 허용되는 최대 end-to-end 시간(초)입니다.
                지정된 시간이 경과하면 클라이언트는 요청을 중단하고
                "Deadline Exceeded" 메시지와 함께 InferenceServerException을 발생시킵니다.
                기본값은 None으로, 클라이언트가 서버의 응답을 기다립니다.

        Returns:
            dict 또는 protobuf 메시지: 모델 설정을 담고 있는
                JSON 딕셔너리 또는 ModelConfigResponse 메시지입니다.

        Raises:
            InferenceServerException: 모델 설정을 가져올 수 없거나 시간이 초과된 경우 발생합니다.

        Examples:
            ```
            {'backend': 'vllm',
            'input': [{'data_type': 'TYPE_STRING', 'dims': ['1'], 'name': 'text_input'},
                    {'data_type': 'TYPE_STRING',
                        'dims': ['-1'],
                        'name': 'image',
                        'optional': True},
                    {'data_type': 'TYPE_BOOL',
                        'dims': ['1'],
                        'name': 'stream',
                        'optional': True},
                    {'data_type': 'TYPE_STRING',
                        'dims': ['1'],
                        'name': 'sampling_parameters',
                        'optional': True},
                    {'data_type': 'TYPE_BOOL',
                        'dims': ['1'],
                        'name': 'exclude_input_in_output',
                        'optional': True},
                    {'data_type': 'TYPE_BOOL',
                        'dims': ['1'],
                        'name': 'return_finish_reason',
                        'optional': True},
                    {'data_type': 'TYPE_BOOL',
                        'dims': ['1'],
                        'name': 'return_cumulative_logprob',
                        'optional': True},
                    {'data_type': 'TYPE_BOOL',
                        'dims': ['1'],
                        'name': 'return_logprobs',
                        'optional': True},
                    {'data_type': 'TYPE_BOOL',
                        'dims': ['1'],
                        'name': 'return_num_input_tokens',
                        'optional': True},
                    {'data_type': 'TYPE_BOOL',
                        'dims': ['1'],
                        'name': 'return_num_output_tokens',
                        'optional': True}],
            'instance_group': [{'count': 1,
                                'gpus': [0],
                                'kind': 'KIND_GPU',
                                'name': 'llama-3.1-8b'}],
            'model_transaction_policy': {'decoupled': True},
            'name': 'llama-3.1-8b',
            'optimization': {'input_pinned_memory': {'enable': True},
                            'output_pinned_memory': {'enable': True}},
            'output': [{'data_type': 'TYPE_STRING', 'dims': ['-1'], 'name': 'text_output'},
                        {'data_type': 'TYPE_STRING',
                        'dims': ['-1'],
                        'name': 'finish_reason'},
                        {'data_type': 'TYPE_FP32',
                        'dims': ['-1'],
                        'name': 'cumulative_logprob'},
                        {'data_type': 'TYPE_STRING', 'dims': ['-1'], 'name': 'logprobs'},
                        {'data_type': 'TYPE_UINT32',
                        'dims': ['1'],
                        'name': 'num_input_tokens'},
                        {'data_type': 'TYPE_UINT32',
                        'dims': ['-1'],
                        'name': 'num_output_tokens'}],
            'runtime': 'model.py',
            'version_policy': {'latest': {'num_versions': 1}}}
            ```
        """
        config = self._client.get_model_config(
            model_name=model_name,
            model_version=model_version,
            headers=headers,
            as_json=as_json,
            client_timeout=client_timeout,
        )

        if not config:
            raise ValueError("Failed to load model config.")

        return config.get("config", {})

    def get_inference_statistics(
        self,
        model_name="",
        model_version="",
        headers=None,
        as_json=True,
        client_timeout=None,
    ) -> dict:
        """지정된 모델 이름과 버전에 대한 추론 통계를 가져옵니다.

        Args:
            model_name (str): 통계를 가져올 모델의 이름입니다.
                기본값은 빈 문자열로, 이 경우 모든 모델의 통계가 반환됩니다.
            model_version (str): 추론 통계를 가져올 모델의 버전입니다.
                기본값은 빈 문자열로, 이 경우 서버는 사용 가능한 모든 모델 버전의
                통계를 반환합니다.
            headers (dict): 요청에 포함할 추가 HTTP 헤더를 지정하는 선택적 딕셔너리입니다.
            as_json (bool): True이면 추론 통계를 json 딕셔너리로 반환하고,
                그렇지 않으면 protobuf 메시지로 반환합니다.
                기본값은 True입니다.
                반환되는 json은 MessageToJson을 사용하여 protobuf 메시지에서
                생성되므로 int64 값은 문자열로 표현됩니다. 필요에 따라
                이러한 문자열을 int64 값으로 다시 변환하는 것은 호출자의 책임입니다.
            client_timeout (float): 요청이 허용되는 최대 end-to-end 시간(초)입니다.
                지정된 시간이 경과하면 클라이언트는 요청을 중단하고
                "Deadline Exceeded" 메시지와 함께 InferenceServerException을 발생시킵니다.
                기본값은 None으로, 클라이언트가 서버의 응답을 기다립니다.

        Returns:
            dict 또는 protobuf 메시지: 추론 통계를 담고 있는
                JSON 딕셔너리 또는 ModelInferenceStatistics 메시지입니다.

        Raises:
            InferenceServerException: 모델 추론 통계를 가져올 수 없거나 시간이 초과된 경우 발생합니다.

        Examples:
            ```
            {'model_stats': [{'batch_stats': [{'batch_size': '1',
                                            'compute_infer': {'count': '2',
                                                                'ns': '1767508'},
                                            'compute_input': {'count': '2',
                                                                'ns': '1307989'},
                                            'compute_output': {'count': '2',
                                                                'ns': '5321'}}],
                            'execution_count': '2',
                            'inference_count': '2',
                            'inference_stats': {'cache_hit': {},
                                                'cache_miss': {},
                                                'compute_infer': {'count': '2',
                                                                    'ns': '1767508'},
                                                'compute_input': {'count': '2',
                                                                    'ns': '1307989'},
                                                'compute_output': {'count': '2',
                                                                    'ns': '5321'},
                                                'fail': {},
                                                'queue': {'count': '2', 'ns': '345706'},
                                                'success': {'count': '2',
                                                            'ns': '3430883'}},
                            'last_inference': '1740559476221',
                            'name': 'llama-3.1-8b',
                            'version': '2'}]}
            ```

        """
        statistics = self._client.get_inference_statistics(
            model_name=model_name,
            model_version=model_version,
            headers=headers,
            as_json=as_json,
            client_timeout=client_timeout,
        )

        if not statistics:
            raise ValueError("Failed to load model infernece statistics.")

        return statistics

    def get_log_settings(self, headers=None, as_json=True, client_timeout=None) -> dict:
        """Triton Server의 전역 로그 설정을 조회합니다.

        Args:
            headers (dict): 요청에 포함할 추가 HTTP 헤더를 지정하는 선택적 딕셔너리입니다.
            as_json (bool): True이면 로그 설정을 json 딕셔너리로 반환하고,
                그렇지 않으면 protobuf 메시지로 반환합니다.
                기본값은 True입니다.
                반환되는 json은 MessageToJson을 사용하여 protobuf 메시지에서
                생성되므로 int64 값은 문자열로 표현됩니다. 필요에 따라
                이러한 문자열을 int64 값으로 다시 변환하는 것은 호출자의 책임입니다.
            client_timeout (float): 요청이 허용되는 최대 end-to-end 시간(초)입니다.
                지정된 시간이 경과하면 클라이언트는 요청을 중단하고
                "Deadline Exceeded" 메시지와 함께 InferenceServerException을 발생시킵니다.
                기본값은 None으로, 클라이언트가 서버의 응답을 기다립니다.

        Returns:
            dict 또는 protobuf 메시지: 로그 설정을 담고 있는
                JSON 딕셔너리 또는 LogSettingsResponse 메시지입니다.

        Raises:
            InferenceServerException: 로그 설정을 가져올 수 없거나 시간이 초과된 경우 발생합니다.

        Example:
        ```
        {'log_error': True,
        'log_file': '',
        'log_format': 'default',
        'log_info': True,
        'log_verbose_level': 0,
        'log_warning': True}
        ```
        """
        settings = {}
        settings_data = self._client.get_log_settings(
            headers=headers, as_json=as_json, client_timeout=client_timeout
        )

        if not settings_data:
            raise ValueError("Failed to load log settings.")

        for key, value_dict in settings_data.get("settings", {}).items():
            # Each value_dict has only one key-value pair
            # Get the first (and only) value from the inner dictionary
            param_value = next(iter(value_dict.values()))
            settings[key] = param_value

        return settings

    def update_log_settings(
        self, settings, headers=None, as_json=True, client_timeout=None
    ) -> dict:
        """Triton Server의 전역 로그 설정을 업데이트합니다.

        Args:
            settings (dict): 업데이트할 로그 설정을 담고 있는 JSON 딕셔너리입니다.
                예를 들어, {'log_error': True}와 같이 설정할 수 있습니다.
            headers (dict): 요청에 포함할 추가 HTTP 헤더를 지정하는 선택적 딕셔너리입니다.
            as_json (bool): True이면 로그 설정을 json 딕셔너리로 반환하고,
                그렇지 않으면 protobuf 메시지로 반환합니다.
                기본값은 True입니다.
                반환되는 json은 MessageToJson을 사용하여 protobuf 메시지에서
                생성되므로 int64 값은 문자열로 표현됩니다. 필요에 따라
                이러한 문자열을 int64 값으로 다시 변환하는 것은 호출자의 책임입니다.
            client_timeout (float): 요청이 허용되는 최대 end-to-end 시간(초)입니다.
                지정된 시간이 경과하면 클라이언트는 요청을 중단하고
                "Deadline Exceeded" 메시지와 함께 InferenceServerException을 발생시킵니다.
                기본값은 None으로, 클라이언트가 서버의 응답을 기다립니다.

        Returns:
            dict 또는 protobuf 메시지: 반영 후 로그 설정을 담고 있는
                JSON 딕셔너리 또는 LogSettingsResponse 메시지입니다.

        Raises:
            InferenceServerException: 로그 설정을 가져올 수 없거나 시간이 초과된 경우 발생합니다.

        Example:
        ```
        {'log_error': True,
        'log_file': '',
        'log_format': 'default',
        'log_info': True,
        'log_verbose_level': 0,
        'log_warning': True}
        ```
        """
        result = {}
        settings_data = self._client.update_log_settings(
            settings=settings,
            headers=headers,
            as_json=as_json,
            client_timeout=client_timeout,
        )

        if not settings_data:
            raise ValueError("Failed to load log settings.")

        for key, value_dict in settings_data.get("settings", {}).items():
            # Each value_dict has only one key-value pair
            # Get the first (and only) value from the inner dictionary
            param_value = next(iter(value_dict.values()))
            result[key] = param_value

        return result

    def get_trace_settings(
        self, model_name=None, headers=None, as_json=True, client_timeout=None
    ) -> dict:
        """Triton Server에서 지정된 모델 이름에 대한 트레이스 설정을 가져오거나,
        모델 이름이 주어지지 않은 경우 전역 트레이스 설정을 가져옵니다.

        Args:
            model_name (str): 트레이스 설정을 가져올 모델의 이름입니다.
                None 또는 빈 문자열을 지정하면 전역 트레이스 설정을 반환합니다.
                기본값은 None입니다.
            headers (dict): 요청에 포함할 추가 HTTP 헤더를 지정하는 선택적 딕셔너리입니다.
            as_json (bool): True이면 트레이스 설정을 json 딕셔너리로 반환하고,
                그렇지 않으면 protobuf 메시지로 반환합니다.
                기본값은 True입니다.
                반환되는 json은 MessageToJson을 사용하여 protobuf 메시지에서
                생성되므로 int64 값은 문자열로 표현됩니다. 필요에 따라
                이러한 문자열을 int64 값으로 다시 변환하는 것은 호출자의 책임입니다.
            client_timeout (float): 요청이 허용되는 최대 end-to-end 시간(초)입니다.
                지정된 시간이 경과하면 클라이언트는 요청을 중단하고
                "Deadline Exceeded" 메시지와 함께 InferenceServerException을 발생시킵니다.
                기본값은 None으로, 클라이언트가 서버의 응답을 기다립니다.

        Returns:
            dict 또는 protobuf 메시지: 트레이스 설정을 담고 있는
                JSON 딕셔너리 또는 TraceSettingResponse 메시지입니다.

        Raises:
            InferenceServerException: 트레이스 설정을 가져올 수 없거나 시간이 초과된 경우 발생합니다.

        Example:
        ```
        {'log_frequency': '0',
        'trace_count': '-1',
        'trace_file': '',
        'trace_level': ['OFF'],
        'trace_mode': 'triton',
        'trace_rate': '1000'}
        ```
        """
        settings = {}
        settings_data = self._client.get_trace_settings(
            model_name=model_name,
            headers=headers,
            as_json=as_json,
            client_timeout=client_timeout,
        )

        if not settings_data:
            raise ValueError("Failed to load trace settings.")

        for key, value_dict in settings_data.get("settings", {}).items():
            # Each value_dict has only one key-value pair
            # Get the first (and only) value from the inner dictionary
            param_value = next(iter(value_dict.values()))
            settings[key] = param_value[0] if not key == "trace_level" else param_value

        return settings

    def update_trace_settings(
        self,
        model_name: Optional[str] = None,
        settings: dict = {},
        headers=None,
        as_json=True,
        client_timeout=None,
    ) -> dict:
        """Triton Server의 트레이스 설정을 업데이트합니다.

        Args:
            model_name (str): 트레이스 설정을 업데이트할 모델의 이름입니다.
                None 또는 빈 문자열을 지정하면 전역 트레이스 설정을 업데이트합니다.
                기본값은 None입니다.
            settings (dict): 업데이트할 트레이스 설정을 담고 있는 JSON 딕셔너리입니다.
                예를 들어, {'trace_count': "-1"}와 같이 설정할 수 있습니다.
            headers (dict): 요청에 포함할 추가 HTTP 헤더를 지정하는 선택적 딕셔너리입니다.
            as_json (bool): True이면 트레이스 설정을 json 딕셔너리로 반환하고,
                그렇지 않으면 protobuf 메시지로 반환합니다.
                기본값은 True입니다.
                반환되는 json은 MessageToJson을 사용하여 protobuf 메시지에서
                생성되므로 int64 값은 문자열로 표현됩니다. 필요에 따라
                이러한 문자열을 int64 값으로 다시 변환하는 것은 호출자의 책임입니다.
            client_timeout (float): 요청이 허용되는 최대 end-to-end 시간(초)입니다.
                지정된 시간이 경과하면 클라이언트는 요청을 중단하고
                "Deadline Exceeded" 메시지와 함께 InferenceServerException을 발생시킵니다.
                기본값은 None으로, 클라이언트가 서버의 응답을 기다립니다.

        Returns:
            dict 또는 protobuf 메시지: 반영 후 트레이스 설정을 담고 있는
                JSON 딕셔너리 또는 LogSettingsResponse 메시지입니다.

        Raises:
            InferenceServerException: 트레이스 설정을 가져올 수 없거나 시간이 초과된 경우 발생합니다.
        """
        result = {}
        settings_data = self._client.update_trace_settings(
            model_name=model_name,
            settings=settings,
            headers=headers,
            as_json=as_json,
            client_timeout=client_timeout,
        )

        if not settings_data:
            raise ValueError("Failed to update trace settings.")

        for key, value_dict in settings_data.get("settings", {}).items():
            # Each value_dict has only one key-value pair
            # Get the first (and only) value from the inner dictionary
            param_value = next(iter(value_dict.values()))
            result[key] = param_value if key == "trace_level" else param_value[0]

        return result

    def _to_input_tensor(
        self, name: str, shape: list[int], datatype: str, data
    ) -> InferInput:
        tensor = InferInput(name=name, shape=shape, datatype=datatype)
        tensor.set_data_from_numpy(np.array(data, dtype=triton_to_np_dtype(datatype)))
        return tensor

    def _to_output_tensor(self, name: str) -> InferRequestedOutput:
        return InferRequestedOutput(name=name)

    def infer(
        self,
        model_name,
        inputs,
        model_version="",
        outputs=None,
        request_id="",
        sequence_id=0,
        sequence_start=False,
        sequence_end=False,
        priority=0,
        timeout=None,
        client_timeout=None,
        headers=None,
        compression_algorithm=None,
        parameters=None,
    ) -> dict:
        """Run synchronous inference using the supplied 'inputs' requesting
        the outputs specified by 'outputs'.

        Parameters
        ----------
        model_name: str
            The name of the model to run inference.
        inputs : list
        model_version : str
            The version of the model to run inference. The default value
            is an empty string which means then the server will choose
            a version based on the model and internal policy.
        outputs : list
            If not specified all outputs produced
            by the model will be returned using default settings.
        request_id : str
            Optional identifier for the request. If specified will be returned
            in the response. Default value is an empty string which means no
            request_id will be used.
        sequence_id : int
            The unique identifier for the sequence being represented by the
            object. Default value is 0 which means that the request does not
            belong to a sequence.
        sequence_start : bool
            Indicates whether the request being added marks the start of the
            sequence. Default value is False. This argument is ignored if
            'sequence_id' is 0.
        sequence_end : bool
            Indicates whether the request being added marks the end of the
            sequence. Default value is False. This argument is ignored if
            'sequence_id' is 0.
        priority : int
            Indicates the priority of the request. Priority value zero
            indicates that the default priority level should be used
            (i.e. same behavior as not specifying the priority parameter).
            Lower value priorities indicate higher priority levels. Thus
            the highest priority level is indicated by setting the parameter
            to 1, the next highest is 2, etc. If not provided, the server
            will handle the request using default setting for the model.
        timeout : int
            The timeout value for the request, in microseconds. If the request
            cannot be completed within the time the server can take a
            model-specific action such as terminating the request. If not
            provided, the server will handle the request using default setting
            for the model. This option is only respected by the model that is
            configured with dynamic batching. See here for more details:
            https://github.com/triton-inference-server/server/blob/main/docs/user_guide/model_configuration.md#dynamic-batcher
        client_timeout : float
            The maximum end-to-end time, in seconds, the request is allowed
            to take. The client will abort request and raise
            InferenceServerExeption with message "Deadline Exceeded" when the
            specified time elapses. The default value is None which means
            client will wait for the response from the server.
        headers : dict
            Optional dictionary specifying additional HTTP headers to include
            in the request.
        compression_algorithm : str
            Optional grpc compression algorithm to be used on client side.
            Currently supports "deflate", "gzip" and None. By default, no
            compression is used.
        parameters : dict
            Optional custom parameters to be included in the inference
            request.

        Returns
        -------
        InferResult
            The object holding the result of the inference.

        Raises
        ------
        InferenceServerException
            If server fails to perform inference.
        """
        inputs = [
            self._to_input_tensor(
                name=input_tensor.name,
                shape=input_tensor.shape,
                datatype=input_tensor.datatype,
                data=np.array(input_tensor.data, dtype=np.object_),
            )
            for input_tensor in inputs
        ]

        if outputs:
            outputs = [
                self._to_output_tensor(name=output_tensor.name)
                for output_tensor in outputs
            ]

        response: InferResult = self._client.infer(
            model_name=model_name,
            inputs=inputs,
            model_version=model_version,
            outputs=outputs,
            request_id=request_id,
            sequence_id=sequence_id,
            sequence_start=sequence_start,
            sequence_end=sequence_end,
            priority=priority,
            timeout=timeout,
            client_timeout=client_timeout,
            headers=headers,
            compression_algorithm=compression_algorithm,
            parameters=parameters,
        )  # type: ignore  (C 라이브러리에서 Raise를 수행합니다. None이 되는 경우는 없습니다.)
        result: dict = response.get_response(as_json=True)

        # 불필요한 키 제거
        if "raw_output_contents" in result:
            del result["raw_output_contents"]

        # 각 출력 텐서의 결과 저장
        for output_tensor in result.get("outputs", []):
            tensor_name = output_tensor.get("name", "")
            numpy_array = response.as_numpy(name=tensor_name)

            # result["outputs"]에서 해당하는 딕셔너리 찾기
            for output_item in result["outputs"]:
                if output_item["name"] == tensor_name:
                    output_item["data"] = numpy_array.tolist()
                    break

        return result

    def stream_infer(
        self,
        model_name,
        inputs,
        model_version="",
        outputs=None,
        request_id="",
        sequence_id=0,
        sequence_start=False,
        sequence_end=False,
        enable_empty_final_response=False,
        priority=0,
        timeout=None,
        headers=None,
        parameters=None,
    ):
        """Runs an asynchronous inference over gRPC bi-directional streaming
        API. A stream must be established with a call to start_stream()
        before calling this function. All the results will be provided to the
        callback function associated with the stream.

        Parameters
        ----------
        model_name: str
            The name of the model to run inference.
        inputs : list
            Each describing data for a input
            tensor required by the model.
        model_version: str
            The version of the model to run inference. The default value
            is an empty string which means then the server will choose
            a version based on the model and internal policy.
        outputs : list
            Each describing how the output
            data must be returned. If not specified all outputs produced
            by the model will be returned using default settings.
        request_id : str
            Optional identifier for the request. If specified will be returned
            in the response. Default value is an empty string which means no
            request_id will be used.
        sequence_id : int or str
            The unique identifier for the sequence being represented by the
            object.  A value of 0 or "" means that the request does not
            belong to a sequence. Default is 0.
        sequence_start: bool
            Indicates whether the request being added marks the start of the
            sequence. Default value is False. This argument is ignored if
            'sequence_id' is 0 or "".
        sequence_end: bool
            Indicates whether the request being added marks the end of the
            sequence. Default value is False. This argument is ignored if
            'sequence_id' is 0 or "".
        enable_empty_final_response: bool
            Indicates whether "empty" responses should be generated and sent
            back to the client from the server during streaming inference when
            they contain the TRITONSERVER_RESPONSE_COMPLETE_FINAL flag.
            This strictly relates to the case of models/backends that send
            flags-only responses (use TRITONBACKEND_ResponseFactorySendFlags(TRITONSERVER_RESPONSE_COMPLETE_FINAL)
            or InferenceResponseSender.send(flags=TRITONSERVER_RESPONSE_COMPLETE_FINAL))
            Currently, this only occurs for decoupled models, and can be
            used to communicate to the client when a request has received
            its final response from the model. If the backend sends the final
            flag along with a non-empty response, this arg is not needed.
            Default value is False.
        priority : int
            Indicates the priority of the request. Priority value zero
            indicates that the default priority level should be used
            (i.e. same behavior as not specifying the priority parameter).
            Lower value priorities indicate higher priority levels. Thus
            the highest priority level is indicated by setting the parameter
            to 1, the next highest is 2, etc. If not provided, the server
            will handle the request using default setting for the model.
        timeout : int
            The timeout value for the request, in microseconds. If the request
            cannot be completed within the time the server can take a
            model-specific action such as terminating the request. If not
            provided, the server will handle the request using default setting
            for the model. This does not stop the grpc stream itself and is only
            respected by the model that is configured with dynamic batching.
            See here for more details:
            https://github.com/triton-inference-server/server/blob/main/docs/user_guide/model_configuration.md#dynamic-batcher
        headers : dict
            Optional dictionary specifying additional HTTP headers to include
            in the request.
        parameters : dict
            Optional custom parameters to be included in the inference
            request.
        Raises
        ------
        InferenceServerException
            If server fails to issue inference.
        """
        stream_client = self.__deepcopy__()
        result_queue = queue.Queue()
        processing_complete = threading.Event()

        def callback(result, error):
            if error:
                # self.log.error(f"추론 중 오류 발생: {error.message()}")
                result_queue.put({"error": error.message()})
                processing_complete.set()
            else:
                stream: dict = result.get_response(as_json=True)
                text_output = result.as_numpy("text_output")[0][0].decode("utf-8")

                finish_flag = (
                    stream.get("parameters", {})
                    .get("triton_final_response", {})
                    .get("bool_param", None)
                )
                result_queue.put({"data": text_output})

                if finish_flag:
                    processing_complete.set()

        try:
            stream_client.start_stream(
                callback=partial(
                    callback,
                ),
                stream_timeout=timeout,
                headers=headers,
            )

            inputs = [
                self._to_input_tensor(
                    name=input_tensor.name,
                    shape=input_tensor.shape,
                    datatype=input_tensor.datatype,
                    data=input_tensor.data,
                )
                for input_tensor in inputs
            ]

            if outputs:
                outputs = [
                    self._to_output_tensor(name=output_tensor.name)
                    for output_tensor in outputs
                ]

            try:
                stream_client.async_stream_infer(
                    model_name=model_name,
                    inputs=inputs,
                    model_version=model_version,
                    outputs=outputs,
                    request_id=request_id,
                    sequence_id=sequence_id,
                    sequence_start=sequence_start,
                    sequence_end=sequence_end,
                    enable_empty_final_response=enable_empty_final_response,
                    priority=priority,
                    timeout=timeout,
                    parameters=parameters,
                )

                while not processing_complete.is_set():
                    try:
                        result = result_queue.get(timeout=0.1)
                        if "error" in result:
                            error_json = json.dumps(
                                {"error": result["error"]}, ensure_ascii=False
                            )
                            yield f"data: {error_json}\n\n"
                        else:
                            data_json = json.dumps(
                                {"text": result["data"]}, ensure_ascii=False
                            )
                            yield f"data: {data_json}\n\n"
                    except queue.Empty:
                        continue
            except Exception as e:
                raise e
        finally:
            stream_client.stop_stream()
