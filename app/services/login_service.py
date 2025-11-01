# app/services/auth_service.py
from app.repositories.user_repository import UserRepository

class LoginService:
    def __init__(self, db):
        self.user_repo = UserRepository(db)

    def authenticate_user(self, username: str, password: str):
        user = self.user_repo.get_user_by_username(username)
        if not user:
            return None
        # Kiểm tra mật khẩu và trạng thái
        if user.password != password:
            return None
        if user.status != "active":
            return None
        if user.role != "admin":
            return None
        return user

