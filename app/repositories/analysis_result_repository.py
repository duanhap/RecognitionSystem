# app/repositories/analysis_result_repository.py
from typing import List, Optional
from sqlalchemy.orm import Session
from sqlalchemy import or_
from app.model.analysis_result import AnalysisResult
from app.model.media import Media
from app.model.user import User

class AnalysisResultRepository:
    def __init__(self, db: Session):
        self.db = db

    def get_result_by_id(self, result_id: int) -> Optional[AnalysisResult]:
        """Lấy kết quả theo ID với đầy đủ thông tin liên quan"""
        return self.db.query(AnalysisResult)\
            .join(Media, AnalysisResult.media_id == Media.id)\
            .join(User, Media.user_id == User.id)\
            .filter(AnalysisResult.id == result_id)\
            .first()

    def list_results(self, search: str = "") -> List[AnalysisResult]:
        """Lấy danh sách kết quả theo từ khóa tìm kiếm"""
        query = self.db.query(AnalysisResult)\
            .join(Media, AnalysisResult.media_id == Media.id)\
            .join(User, Media.user_id == User.id)\
            .order_by(AnalysisResult.created_at.desc())
        
        if search:
            query = query.filter(
                or_(
                    AnalysisResult.label.ilike(f"%{search}%"),
                    User.username.ilike(f"%{search}%"),
                    AnalysisResult.id.ilike(f"%{search}%")
                )
            )
        
        return query.all()

    def save_model_file(self, result_id: int, model_path: str) -> bool:
        """Lưu đường dẫn file model"""
        try:
            result = self.db.query(AnalysisResult).filter(AnalysisResult.id == result_id).first()
            if result:
                # Ở đây bạn có thể thêm trường model_path nếu cần
                # result.model_path = model_path
                self.db.commit()
                return True
            return False
        except Exception:
            self.db.rollback()
            return False