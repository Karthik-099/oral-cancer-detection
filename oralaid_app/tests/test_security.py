import time

import pytest
from fastapi import HTTPException

import security as sec


def test_password_policy():
    assert sec.password_problems("short1")
    assert sec.password_problems("onlyletterslong")
    assert sec.password_problems("drsmith12345", "drsmith@clinic.example.com")
    assert sec.password_problems("Screening2026") == []


def test_password_hash_roundtrip():
    h = sec.hash_password("Screening2026")
    assert sec.verify_password("Screening2026", h)
    assert not sec.verify_password("wrong", h)
    assert not sec.verify_password("x", "not-a-hash")


def test_token_roundtrip_and_tamper():
    token = sec.create_token(42, 3)
    payload = sec.decode_token(token)
    assert payload["sub"] == "42" and payload["ver"] == 3
    with pytest.raises(HTTPException):
        sec.decode_token(token[:-2] + ("A" if token[-1] != "A" else "B"))


def test_signed_file_urls():
    url = sec.sign_file("abc_0_gradcam.jpg")
    params = dict(p.split("=") for p in url.split("?")[1].split("&"))
    assert sec.verify_file("abc_0_gradcam.jpg", int(params["exp"]), params["sig"])
    assert not sec.verify_file("other.jpg", int(params["exp"]), params["sig"])
    assert not sec.verify_file("abc_0_gradcam.jpg", int(time.time()) - 1, params["sig"])


def test_api_key_hashing():
    raw, prefix, digest = sec.new_api_key()
    assert raw.startswith("oak_") and raw.startswith(prefix)
    assert sec.hash_api_key(raw) == digest


@pytest.mark.parametrize("data,kind", [
    (b"\xff\xd8\xff\xe0rest", "jpeg"),
    (b"\x89PNG\r\n\x1a\nrest", "png"),
    (b"RIFF\x00\x00\x00\x00WEBPrest", "webp"),
    (b"<svg xmlns=", None),
    (b"%PDF-1.7", None),
    (b"MZ\x90\x00", None),
])
def test_sniff_image(data, kind):
    assert sec.sniff_image(data) == kind
