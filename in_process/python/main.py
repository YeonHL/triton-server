import argparse
import logging
import logging.config
import subprocess
import sys
import time

import uvicorn
from typing import Optional

from src.common.settings import log_settings

logging.config.dictConfig(log_settings)
logger = logging.getLogger()

# Triton 프로세스 코드
ERROR_EXIT_DELAY = 15
ERROR_CODE_FATAL = 255
ERROR_CODE_USAGE = 253
EXIT_SUCCESS = 0


def parse_arguments() -> argparse.Namespace:
    """커맨드 라인 인자를 파싱하여 서버 및 모델 설정값을 반환합니다.

    Returns:
        argparse.Namespace: 파싱된 커맨드 라인 인자들
            - fastapi 그룹:
                - dev (bool): 개발 모드 활성화 여부
                - port (int): 서버 포트 번호 (1024-65535)
                - workers (int): 워커 프로세스 개수
            - triton 그룹:
                - model (str): 모델 경로
                - dt (str): 텐서 데이터 타입 (bfloat16, float16, float32)
                - pp (int): 파이프라인 병렬화 수준
                - tp (int): 텐서 병렬화 수준
                - iso8601 (int): ISO8601 타임스탬프 형식 사용 여부
                - verbose (int): 상세 로그 출력 수준
                - engine (str): 추론 엔진 종류 (trtllm, vllm)

    Raises:
        SystemExit: 포트 번호가 유효하지 않은 경우 (1024-65535 범위를 벗어날 때)

    Example:
        >>> args = parse_arguments()
        >>> print(args.port)
        8000
        >>> print(args.engine)
        vllm
    """
    parser = argparse.ArgumentParser(
        description="Model Manager Python 서버 실행 스크립트",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # 서버 관련 인자 그룹
    fastapi_group = parser.add_argument_group(
        title="FastAPI", description="서버 설정 관련 인자"
    )
    fastapi_group.add_argument(
        "--dev", action="store_true", help="개발 모드로 실행 (localhost만 접근 가능)"
    )
    fastapi_group.add_argument(
        "--host", type=str, default="localhost", help="서버 주소"
    )
    fastapi_group.add_argument("--port", type=int, default=18200, help="서버 포트 번호")
    fastapi_group.add_argument(
        "--workers",
        type=int,
        default=1,
        help="워커 프로세스 개수 (개발 모드에서는 무시됨)",
    )

    # 모델 관련 인자 그룹
    triton_group = parser.add_argument_group(
        title="Triton Server", description="모델 및 추론 설정 관련 인자"
    )
    triton_group.add_argument(
        "--model-repository", type=str, default=None, help="모델 저장소 경로"
    )
    triton_group.add_argument(
        "--dt",
        type=str,
        default="float16",
        choices=["bfloat16", "float16", "float32"],
        help="텐서 데이터 타입",
    )
    triton_group.add_argument(
        "--pp", type=int, default=1, help="파이프라인 병렬화 수준"
    )
    triton_group.add_argument("--tp", type=int, default=1, help="텐서 병렬화 수준")
    triton_group.add_argument(
        "--iso8601", action="count", default=0, help="ISO8601 타임스탬프 형식 사용"
    )
    triton_group.add_argument(
        "--verbose", action="count", default=0, help="상세 로그 출력 수준"
    )
    triton_group.add_argument(
        "--engine",
        type=str,
        default="vllm",
        choices=["trtllm", "vllm"],
        help="추론 엔진 선택",
    )

    args = parser.parse_args()

    # 포트 번호 유효성 검사
    if not (1024 <= args.port <= 65535):
        parser.error("포트 번호는 1024에서 65535 사이여야 합니다.")

    return args


def execute_triton(args):
    world_size = args.tp * args.pp

    if world_size <= 0:
        raise Exception(
            "usage: Options --pp and --pp must both be equal to or greater than 1."
        )

    # Single GPU setups can start a tritonserver process directly.
    if world_size == 1:
        cmd_args = [
            "tritonserver",
            "--http-port=18201",
            "--grpc-port=18202",
            "--metrics-port=18203",
            "--allow-cpu-metrics=false",
            "--allow-gpu-metrics=false",
            "--allow-metrics=true",
            "--metrics-interval-ms=1000",
            f"--model-repository={args.model_repository}",
            "--model-control-mode=explicit",
            "--model-load-thread-count=2",
            "--strict-readiness=true",
        ]

        if args.verbose > 0:
            cmd_args += ["--log-verbose=1"]

        if args.iso8601 > 0:
            cmd_args += ["--log-format=ISO8601"]

    # Multi-GPU setups require a specialized command line which based on `mpirun`.
    else:
        cmd_args = ["mpirun", "--allow-run-as-root"]

        for i in range(world_size):
            if i != 0:
                cmd_args += [":"]

            cmd_args += [
                "-n",
                "1",
                "tritonserver",
                f"--id=rank{i}",
                f"--http-port={(18201 + i * 10)}",
                f"--grpc-port={(18202 + i * 10)}",
                f"--metrics-port={(18203 + i * 10)}",
                "--model-load-thread-count=2",
                f"--model-repository={args.model_directory}",
                "--disable-auto-complete-config",
                f"--backend-config=python,shm-region-prefix-name=rank{i}_",
            ]

            if i == 0:
                cmd_args += [
                    "--allow-cpu-metrics=false",
                    "--allow-gpu-metrics=false",
                    "--allow-metrics=true",
                    "--metrics-interval-ms=1000",
                ]

                if args.verbose > 0:
                    cmd_args += ["--log-verbose=1"]

                if args.iso8601 > 0:
                    cmd_args += ["--log-format=ISO8601"]

            else:
                cmd_args += [
                    "--allow-http=false",
                    "--allow-grpc=false",
                    "--allow-metrics=false",
                    "--log-info=false",
                    "--log-warning=false",
                    "--model-control-mode=explicit",
                    "--load-model=tensorrt_llm",
                ]

    logger.info(f"> {' '.join(cmd_args)}\n")

    # Run triton_cli to build the TRT-LLM engine + plan.
    return subprocess.Popen(cmd_args, stderr=sys.stderr, stdout=sys.stdout)


def execute_fastapi(args):
    config = {
        "app": "src.api:app",
        "host": args.host,
        "port": args.port,
        "reload": args.dev,
        "workers": 1 if args.dev else args.workers,
        "log_level": "debug" if args.dev else "info",
    }

    logger.info("\nFastAPI 서버 시작:")
    for key, value in config.items():
        logger.info(f"  {key}: {value}")

    try:
        uvicorn.run(**config)
    except Exception as e:
        logger.error(f"서버 실행 중 오류 발생: {e}")
        sys.exit(1)


def die(exception: Optional[Exception] = None):
    if exception is not None:
        logger.error(f"fatal: {exception}")

    logger.error(f"       Waiting {ERROR_EXIT_DELAY} second before exiting.")
    # Delay the process' termination to provide a small window for administrators to capture the logs before it exits and restarts.
    time.sleep(ERROR_EXIT_DELAY)

    exit(ERROR_CODE_USAGE)


def main():
    try:
        # Parse options provided.
        args = parse_arguments()

        execute_triton(args)
        execute_fastapi(args)

    except Exception as exception:
        die(exception)


if __name__ == "__main__":
    main()
