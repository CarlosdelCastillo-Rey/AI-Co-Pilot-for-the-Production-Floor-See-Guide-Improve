from __future__ import annotations

from datetime import datetime, timedelta, timezone

import bcrypt
import jwt
from fastapi import HTTPException

from vision_ops_alerting.config import settings

ALGORITHM = "HS256"


def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_password(password: str, password_hash: str) -> bool:
    return bcrypt.checkpw(password.encode("utf-8"), password_hash.encode("utf-8"))


def create_access_token(*, user_id: str, email: str, name: str) -> str:
    expires = datetime.now(timezone.utc) + timedelta(hours=settings.auth_token_hours)
    payload = {
        "sub": user_id,
        "email": email,
        "name": name,
        "exp": expires,
    }
    return jwt.encode(payload, settings.auth_secret, algorithm=ALGORITHM)


def decode_access_token(token: str) -> dict:
    try:
        return jwt.decode(token, settings.auth_secret, algorithms=[ALGORITHM])
    except jwt.PyJWTError as e:
        raise HTTPException(status_code=401, detail="Invalid or expired token") from e
