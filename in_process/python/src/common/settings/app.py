from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from src.common.settings.triton_client import TritonSettings


class ServerSettings(BaseModel):
    """
    애플리케이션 서버 설정을 정의하는 클래스입니다.

    Attributes:
        name (str): 애플리케이션 이름
        version (str): 애플리케이션 버전
        description (str): 애플리케이션 설명
        prefix (str): 서버 API Prefix
    """

    name: str = Field(default="triton-inprocess-python", description="애플리케이션 이름")
    version: str = Field(default="0.1.0", description="애플리케이션 버전")
    description: str = Field(
        default="Triton in-process python server", description="애플리케이션 설명"
    )
    prefix: str = Field(default="/api/v1/model", description="서버 API Prefix")


class Settings(BaseSettings):
    """애플리케이션 설정을 관리하는 클래스입니다.

    환경 변수를 통한 설정 주입을 지원하며, 설정값들의 유효성 검사를 자동으로 수행합니다.

    Attributes:
        server (ServerSettings): 서버 설정
        triton (TritonSettings): Triton 서버 설정
    """

    server: ServerSettings = Field(
        default_factory=ServerSettings,
        description="서버 설정",
    )

    triton: TritonSettings = Field(
        default_factory=TritonSettings,
        description="Triton 서버 설정",
    )

    model_config = SettingsConfigDict(env_ignore_empty=True)


settings = Settings()

if __name__ == "__main__":
    from pprint import pprint

    pprint(settings.model_dump())
