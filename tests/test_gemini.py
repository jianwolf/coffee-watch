from __future__ import annotations

from google.genai import errors

from coffee_watch.gemini import is_retryable_gemini_api_error


def test_retryable_gemini_api_error_detects_503():
    exc = errors.ServerError(
        503,
        {
            "error": {
                "code": 503,
                "message": "This model is currently experiencing high demand.",
                "status": "UNAVAILABLE",
            }
        },
        None,
    )
    assert is_retryable_gemini_api_error(exc)


def test_retryable_gemini_api_error_rejects_non_api_error():
    assert not is_retryable_gemini_api_error(RuntimeError("boom"))
