from __future__ import annotations

from fastapi import HTTPException
from sqlalchemy.orm import Session

from vision_ops_alerting.config import settings
from vision_ops_alerting.db.models import User, new_id
from vision_ops_alerting.services.auth_tokens import hash_password, verify_password


def user_to_dict(user: User) -> dict:
    return {
        "id": user.id,
        "email": user.email,
        "name": user.name,
        "role": user.role or "Supervisor",
        "createdAt": user.created_at.isoformat() if user.created_at else None,
    }


def get_user_by_email(db: Session, email: str) -> User | None:
    return db.query(User).filter(User.email == email.strip().lower()).first()


def get_user_by_id(db: Session, user_id: str) -> User | None:
    return db.get(User, user_id)


def create_user(
    db: Session,
    *,
    email: str,
    password: str,
    name: str,
    role: str = "Supervisor",
) -> User:
    normalized = email.strip().lower()
    if not normalized or "@" not in normalized:
        raise HTTPException(status_code=400, detail="Valid email is required")
    if len(password) < 6:
        raise HTTPException(status_code=400, detail="Password must be at least 6 characters")
    display_name = name.strip()
    if not display_name:
        raise HTTPException(status_code=400, detail="Name is required")
    display_role = role.strip() or "Supervisor"
    if get_user_by_email(db, normalized):
        raise HTTPException(status_code=409, detail="Email already registered")

    user = User(
        id=new_id("user"),
        email=normalized,
        name=display_name,
        role=display_role,
        password_hash=hash_password(password),
    )
    db.add(user)
    db.flush()
    return user


def authenticate_user(db: Session, email: str, password: str) -> User:
    user = get_user_by_email(db, email)
    if not user or not verify_password(password, user.password_hash):
        raise HTTPException(status_code=401, detail="Invalid email or password")
    return user


def ensure_default_admin(db: Session) -> None:
    if db.query(User).count() > 0:
        return
    create_user(
        db,
        email=settings.seed_admin_email,
        password=settings.seed_admin_password,
        name=settings.seed_admin_name,
        role=settings.seed_admin_role,
    )
