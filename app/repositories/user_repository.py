# app/repositories/user_repository.py
from sqlalchemy.orm import Session
from app.model.user import User


class UserRepository:
    def __init__(self, db: Session):
        self.db = db
    def update_status(self, user_id: int, status: str) -> bool:
            user = self.get_user_by_id(user_id)
            if not user:
                return False
            user.status = status
            self.db.commit()
            self.db.refresh(user)
            return True
    def get_user_by_username(self, username: str):
        return self.db.query(User).filter(User.username == username).first()
    
    def get_user_by_id(self, user_id: int):
        return self.db.query(User).filter(User.id == user_id).first()

    def list_users(self, search: str = None):
        query = self.db.query(User).filter(User.role == "user")  # chỉ lấy role = 'user'
        if search:
            query = query.filter(User.username.ilike(f"%{search}%"))
        return query.all()


    


    def delete_user(self, user_id: int) -> bool:
        user = self.get_user_by_id(user_id)
        if not user:
            return False
        self.db.delete(user)
        self.db.commit()
        return True

