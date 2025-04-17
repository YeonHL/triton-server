from datetime import datetime, timedelta, timezone
import logging
import os
from pythonjsonlogger.orjson import OrjsonFormatter
from src.common.settings.app import settings

kst = timezone(timedelta(hours=9))


# Json Formatter
class CustomJsonFormatter(OrjsonFormatter):
    def process_log_record(self, log_record):
        # 기본 처리 수행
        log_record = super(CustomJsonFormatter, self).process_log_record(log_record)

        # 타임스탬프 처리
        timestamp = (
            datetime.strptime(log_record["asctime"], "%Y-%m-%d %H:%M:%S,%f")
            .replace(tzinfo=kst)
            .isoformat()
        )

        # JSON 구조 직접 변경
        result = {
            "@timestamp": timestamp,
            "message": log_record["message"],
            "service": {
                "name": settings.server.name,
                "version": settings.server.version,
            },
            "log": {
                "level": log_record["levelname"],
                "logger": log_record["name"],
            },
        }

        if "exc_info" in log_record:
            result["log"]["traceback"] = log_record["exc_info"]

        return result


# 필터 클래스
class LevelFilter:
    def __init__(self, level, above=True):
        self.level = level
        self.above = above  # True: 지정 레벨 이상만, False: 지정 레벨 미만만

    def filter(self, record):
        if self.above:
            return record.levelno >= self.level
        else:
            return record.levelno < self.level


log_path: str = os.path.abspath(os.path.join("/var", "log", f"triton-in-process-{datetime.now().strftime('%Y-%m-%d')}.log"))

log_settings: dict = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "standard": {"format": "%(asctime)s - %(levelname)s: %(message)s"},
        "error": {
            "format": "%(asctime)s - %(pathname) - %(lineno) - %(levelname)s: %(message)s"
        },
        "standard_json": {
            "()": CustomJsonFormatter,
            "format": "%(asctime)s %(message)s %(levelname)s %(name)s ",
        },
        "error_json": {
            "()": CustomJsonFormatter,
            "format": "%(asctime)s %(message)s %(levelname) %(name)s %(exc_info)s",
        },
    },
    "filters": {
        "error_and_above": {"()": LevelFilter, "level": logging.ERROR, "above": True},
        "below_error": {"()": LevelFilter, "level": logging.ERROR, "above": False},
    },
    "handlers": {
        "file_handler": {
            "class": "logging.FileHandler",
            "filename": log_path,
            "level": "INFO",
            "formatter": "standard_json",
        },
        "error_handler": {
            "class": "logging.StreamHandler",
            "level": "ERROR",
            "formatter": "error_json",
            "filters": ["error_and_above"],
        },
        "console_handler": {
            "class": "logging.StreamHandler",
            "level": "DEBUG",
            "formatter": "standard_json",
            "filters": ["below_error"],
        },
    },
    "root": {
        "level": "DEBUG",
        "handlers": ["console_handler", "error_handler"],
    },
}


# 사용 예시
if __name__ == "__main__":
    import logging.config

    # 로깅 설정 적용
    logging.config.dictConfig(log_settings)

    # 로거 가져오기
    logger = logging.getLogger()

    # 다양한 로그 레벨 테스트
    logger.debug("디버그 메시지 - console_handler만 처리")
    logger.info("정보 메시지 - file_handler와 console_handler가 처리")
    logger.warning("경고 메시지 - file_handler와 console_handler가 처리")
    logger.error("오류 메시지 - file_handler와 error_handler가 처리")
    logger.critical("심각한 오류 메시지 - file_handler와 error_handler가 처리")

    # 예외 테스트
    try:
        1 / 0
    except Exception as e:
        logger.exception(f"예외 발생 - {str(e)}")
