from __future__ import annotations

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from vision_ops_alerting.auth_deps import get_current_user
from vision_ops_alerting.db.models import User
from vision_ops_alerting.db.session import get_db
from vision_ops_alerting.services.auth_tokens import create_access_token
from vision_ops_alerting.services.users import authenticate_user, create_user, user_to_dict

router = APIRouter(prefix="/api/auth", tags=["auth"])


class RegisterBody(BaseModel):
    email: str = Field(min_length=3)
    password: str = Field(min_length=6)
    name: str = Field(min_length=1)
    role: str = Field(default="Supervisor", min_length=1)


class LoginBody(BaseModel):
    email: str = Field(min_length=3)
    password: str = Field(min_length=1)


class AuthResponse(BaseModel):
    token: str
    user: dict


@router.post("/register", response_model=AuthResponse, status_code=201)
def register(body: RegisterBody, db: Session = Depends(get_db)):
    user = create_user(
        db,
        email=body.email,
        password=body.password,
        name=body.name,
        role=body.role,
    )
    db.commit()
    token = create_access_token(user_id=user.id, email=user.email, name=user.name)
    return AuthResponse(token=token, user=user_to_dict(user))


@router.post("/login", response_model=AuthResponse)
def login(body: LoginBody, db: Session = Depends(get_db)):
    user = authenticate_user(db, body.email, body.password)
    token = create_access_token(user_id=user.id, email=user.email, name=user.name)
    return AuthResponse(token=token, user=user_to_dict(user))


@router.get("/me")
def me(user: User = Depends(get_current_user)):
    return user_to_dict(user)
