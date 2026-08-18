"""render.payload 순수 함수 단위 테스트 — 네트워크·DB 없음."""

import pytest

from render.payload import SOURCE_FILE_NAME, build_request, inning_key, sec_to_hms


def _clip(scene_id: int, inning: str, start: int, end: int) -> dict:
    return {"scene_id": scene_id, "inning": inning, "start": start, "end": end}


class TestInningKey:
    def test_초_말(self):
        assert inning_key("3회 초") == "3_top"
        assert inning_key("3회 말") == "3_bot"

    def test_공백_유무_무관(self):
        assert inning_key("5회초") == "5_top"

    def test_연장_이닝(self):
        assert inning_key("10회 말") == "10_bot"

    def test_형식_밖은_ValueError(self):
        for bad in ("", "연장", "회 초", None):
            with pytest.raises(ValueError):
                inning_key(bad or "")


class TestSecToHms:
    def test_규격_표기(self):
        # worker-render 스펙 예시와 동일 표기 (hh:mm:ss.f)
        assert sec_to_hms(412.0) == "00:06:52.0"
        assert sec_to_hms(3742.5) == "01:02:22.5"
        assert sec_to_hms(0) == "00:00:00.0"


class TestBuildRequest:
    def test_이닝_그룹핑과_시간순_유지(self):
        clips = [_clip(16, "3회 초", 3773, 3801), _clip(17, "3회 초", 3900, 3910),
                 _clip(25, "4회 말", 5451, 5510)]
        req = build_request(202, 5, clips, bumper=True)
        assert list(req["innings"]) == ["3_top", "4_bot"]
        assert [seg["start_sec"] for seg in req["innings"]["3_top"]] == [3773.0, 3900.0]
        assert req["innings"]["4_bot"][0] == {
            "start_sec": 5451.0, "end_sec": 5510.0,
            "start_hms": "01:30:51.0", "end_hms": "01:31:50.0"}

    def test_고정_필드(self):
        req = build_request(202, 5, [_clip(1, "1회 초", 0, 10)], bumper=False)
        assert req["v_id"] == 202 and req["c_id"] == 5
        assert req["file_name"] == f"202/{SOURCE_FILE_NAME}"   # v_id 접두 상대 경로
        assert req["sync_yn"] is True and req["bumper"] is False

    def test_클립_0건은_ValueError(self):
        with pytest.raises(ValueError, match="0건"):
            build_request(202, 5, [], bumper=True)

    def test_이닝_결손_클립은_scene_id_나열_ValueError(self):
        clips = [_clip(1, "1회 초", 0, 10), _clip(7, "", 20, 30), _clip(9, "", 40, 50)]
        with pytest.raises(ValueError, match=r"\[7, 9\]"):
            build_request(202, 5, clips, bumper=True)
