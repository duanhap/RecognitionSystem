# app/services/auth_service.py
from app.repositories.user_repository import UserRepository

class LoginService:
    def __init__(self, db):
        self.user_repo = UserRepository(db)

    def authenticate_user(self, username: str, password: str):
        user = self.user_repo.get_user_by_username(username)
        if not user:
            return None
        if user.password != password and not user.is_active and user.role != 'admin':  # chưa mã hoá password
            return None
        return user
