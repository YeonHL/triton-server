import logging
import logging.config

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.common.settings import log_settings, settings
from src.generate.presentation import generate_router
from src.model.presentation import model_router
from src.repository.presentation import repository_router
from src.server.presentation import server_router

logging.config.dictConfig(log_settings)


def create_app() -> FastAPI:
    """FastAPI 애플리케이션을 초기화하고 설정합니다."""

    app = FastAPI(
        title=settings.server.name,
        description=settings.server.description,
        version=settings.server.version,
        root_path=settings.server.prefix,
    )

    # CORS 미들웨어 설정
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # 기본 라우터 설정

    # 임시 코드, 제거 예정
    # 에러 핸들러
    @app.exception_handler(Exception)
    async def global_exception_handler(request, exc):
        return JSONResponse(
            status_code=500,
            content={"message": "내부 서버 오류가 발생했습니다", "detail": str(exc)},
        )

    # 임시 코드 종료

    # 라우터 등록 (위에 있는 라우터가 높은 우선 순위를 가집니다.)
    app.include_router(server_router)
    app.include_router(repository_router)
    app.include_router(generate_router)
    app.include_router(model_router)

    return app


# 애플리케이션 인스턴스 생성
app = create_app()
