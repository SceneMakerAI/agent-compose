"""API 오류 계약 — 오류별 예외 클래스. 응답 변환은 app.py 의 전역 핸들러가 담당한다.

라우터·하위 계층은 예외를 raise 만 한다 — HTTP 상태·응답 조립을 호출 자리에 두지 않는다.
응답 형식은 {detail: {code, message, ...ctx}} 로 고정 — 호출자는 code 로 분기한다.
새 오류 = ApiError 상속 클래스 하나 (code·http_status·message 선언).
"""

from fastapi import status


class ApiError(Exception):
    """
    Summary:
        API 오류의 기반 클래스 — code(계약 식별자)·http_status·message 를 갖는다.
    Description:
        - ctx(v_id 등 부가 정보)는 응답 detail 에 그대로 실린다.
        - message 를 생성 시 넘기면 기본 메시지를 덮어쓴다.
    """

    code: str = "INTERNAL_ERROR"
    http_status: int = status.HTTP_500_INTERNAL_SERVER_ERROR
    message: str = "서버 내부 오류가 발생했습니다."

    def __init__(self, message: str | None = None, **ctx) -> None:
        if message is not None:
            self.message = message
        self.ctx = ctx
        super().__init__(self.message)

    def detail(self) -> dict:
        """응답 detail 본문을 조립한다 — 전역 핸들러가 호출."""
        return {"code": self.code, "message": self.message, **self.ctx}


class VideoNotFoundError(ApiError):
    """t_video 에 대상 v_id 가 없다."""

    code = "VIDEO_NOT_FOUND"
    http_status = status.HTTP_404_NOT_FOUND
    message = "영상이 없습니다."


class UnsupportedCategoryError(ApiError):
    """cate_id 에 등록된 도메인 플로우가 없다 — 편성 불가 카테고리."""

    code = "UNSUPPORTED_CATEGORY"
    http_status = status.HTTP_422_UNPROCESSABLE_CONTENT
    message = "지원하지 않는 카테고리입니다."


class ComposeNotFoundError(ApiError):
    """t_compose 에 대상 (v_id, comp_id) 가 없다."""

    code = "COMPOSE_NOT_FOUND"
    http_status = status.HTTP_404_NOT_FOUND
    message = "편성을 찾을 수 없습니다."
