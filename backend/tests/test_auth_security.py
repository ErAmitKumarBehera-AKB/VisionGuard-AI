from backend.app.auth.security import token_for
from backend.app.config import settings


def test_token_generation_uses_safe_default_secret_when_empty():
    original_secret = settings.JWT_SECRET
    settings.JWT_SECRET = ""
    try:
        token = token_for({"_id": "test-user", "role": "ADMIN", "machine_id": None})
        assert isinstance(token, str)
        assert len(token) > 20
    finally:
        settings.JWT_SECRET = original_secret
