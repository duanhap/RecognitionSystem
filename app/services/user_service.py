# app/services/user_service.py
from app.repositories.user_repository import UserRepository

class UserService:
    def __init__(self, db):
        self.user_repo = UserRepository(db)

    def get_user(self, user_id: int):
        return self.user_repo.get_user_by_id(user_id)

    def search_users(self, search: str = None):
        return self.user_repo.list_users(search)

    def change_status(self, user_id: int, status: str):
        return self.user_repo.update_status(user_id, status)

    def remove_user(self, user_id: int):
        return self.user_repo.delete_user(user_id)