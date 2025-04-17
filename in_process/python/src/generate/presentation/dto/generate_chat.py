from typing import Optional
from pydantic import BaseModel, Field
import json


class GenerateChatRequest(BaseModel):
    """채팅 생성 요청 모델"""

    model_name: str = Field(description="모델 이름", examples=["llama-3.1-8b"])
    text_input: str = Field(
        description="입력 텍스트, Chat 형태의 프롬프트를 JSON 문자열로 입력하세요.",
        examples=[
            f"""{json.dumps([
                    {"role": "user", "content": "Hello, how are you?"},
                    {"role": "assistant", "content": "I'm doing great. How can I help you today?"},
                    {"role": "user", "content": "I'd like to show off how chat templating works!"},
                ])}"""
        ],
    )
    image: Optional[str] = Field(
        default=None,
        description="Base64 인코딩된 이미지 문자열, multi-modal 모델이 아닌 경우 입력하면 오류가 발생합니다.",
        examples=[
            "iVBORw0KGgoAAAANSUhEUgAAAFoAAAAWCAIAAADPUBxCAAAAAXNSR0IArs4c6QAAAARnQU1BAACxjwv8YQUAAAAJcEhZcwAADsMAAA7DAcdvqGQAAAGYSURBVFhH7ZTRkcIwDETph3rSR0qgntRDIXSQkxRjS/aukwyZuXz4fWFZXq12gMc8z5/BF41jHXwZcQRGHIERR4DGsUyPwrSkqiAXz9c7HY7wfj2DgEOH1FfH9bPwaUsUGIfOcTbjEc7uGbosjlpoPw5z7tlLDcUB1F0Jzq6Lcg64LRqLBZOA+sa5OMxCVd5me40KEEc9VXE1NpvsgPU6XBMHH9qzevTbcTYOOQeAM9/ir6G+oS98Zy8OrsKDEuh/hxdTH/mshw3XwqfD+VpytVofS9kjvbIPiU4cZGl+I8A4BD8y/gShXbqD0MbRdrseJpUcubv8iFmKxoVNg4bB4+Dw2QCd3MTRFISi6qR8k5anJQzvxyE4qQxszFwUR492+1bC9UB9vbdq/pCKe3GEun+MoXE4g5FuHPqqAjVbW1EXydIG9PU+d5edskVmqa6Xp4wL47Ad2yvdBSlZPeGvD+yQWm4RB4M/YG4xx7txHHI4CHB7XRz2Amxi9k4o/RrHT3Tj4LD1wKuzRu8Zx78x4rgPIw7Huv4BpQfRS6JA7f4AAAAASUVORK5CYII="
        ],
    )
    parameters: Optional[dict] = Field(
        default=None,
        description="추론 요청에 대한 추가 매개변수, Sampling Parameter를 입력하세요. 모델에서 사용 가능한 값은 전달되고 이외에는 무시합니다.",
        examples=[
            {
                "temperature": 0.8,
                "top_p": 0.9,
                "presence_penalty": 0.8,
                "frequency_penalty": 0.8,
                "repetition_penalty": 1.2,
                "max_tokens": 1024,
            }
        ],
    )
