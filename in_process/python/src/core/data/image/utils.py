import base64

from PIL import Image
import requests
import io
from urllib.parse import urlparse


def load_pil_image(src: str, timeout: float = 0) -> Image.Image:
    parsed = urlparse(src)  # 문자열 구조 분석

    # 1. URL 형식 검사:
    #    - scheme (http, https 등)이 있고,
    #    - netloc (domain name, e.g., www.google.com)이 있으면 URL로 간주
    if parsed.scheme in ["http", "https"] and parsed.netloc:
        response = requests.get(src, stream=True, timeout=timeout)
        response.raise_for_status()  # HTTP 오류 발생 시 예외 발생

        image_bytes = io.BytesIO(
            response.content
        )  # 응답 콘텐츠(바이트)를 BytesIO 객체로 변환

        return Image.open(
            image_bytes
        )  # BytesIO 객체를 Image.open()에 전달하여 결과 반환

    # 2. 로컬 경로 간주:
    #    - 위 URL 조건에 맞지 않으면 로컬 경로로 간주합니다.
    #    - 이 방식은 'C:\path', '/home/user', 'relative/path', 심지어 'file:///' URI도
    #      Local Path로 분류합니다.
    #    - 실제 경로 존재 여부(os.path.exists)는 여기서 확인하지 않습니다.
    #      존재 여부 확인은 필요에 따라 별도로 수행해야 합니다.
    else:
        return Image.open(src)


def encode_base64_string(image_path: str, encoding: str = "utf-8") -> str:
    """
    이미지 파일을 읽어 Base64로 인코딩하고 UTF-8 문자열로 반환합니다.

    Args:
        image_path: 인코딩할 이미지 파일의 경로.

    Returns:
        Base64로 인코딩된 이미지 데이터의 UTF-8 문자열.

    Raises:
        FileNotFoundError: 지정된 경로에 이미지 파일이 없을 경우 발생합니다.
        IOError: 파일을 읽는 중 오류가 발생할 경우 (예: 권한 문제) 발생합니다.
        TypeError: image_path가 문자열이 아닐 경우 발생할 수 있습니다.
    """
    if not isinstance(image_path, str):
        raise TypeError(f"image_path must be a string, not {type(image_path)}")

    try:
        # 파일을 바이너리 읽기 모드('rb')로 엽니다.
        with open(image_path, "rb") as image_file:
            image_bytes = image_file.read()  # 파일의 전체 바이너리 내용을 읽습니다.

            base64_bytes = base64.b64encode(
                image_bytes
            )  # 읽어온 바이너리 데이터를 Base64로 인코딩합니다.
            base64_string = base64_bytes.decode(
                encoding=encoding
            )  # Base64로 인코딩된 bytes 객체를 UTF-8 문자열로 디코딩합니다.

            return base64_string

    except FileNotFoundError:
        print(f"오류: 이미지 파일을 찾을 수 없습니다 - '{image_path}'")
        raise  # 원본 예외를 다시 발생시켜 호출자가 처리하도록 함
    except IOError as e:
        print(f"오류: 파일을 읽는 중 에러 발생 '{image_path}': {e}")
        raise  # 원본 예외를 다시 발생시킴
    except Exception as e:
        # 예상치 못한 다른 오류 처리
        print(f"이미지 인코딩 중 예상치 못한 오류 발생: {e}")
        raise
