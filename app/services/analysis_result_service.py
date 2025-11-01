# app/services/analysis_result_service.py
from typing import List, Optional
from app.repositories.analysis_result_repository import AnalysisResultRepository
from app.model.analysis_result import AnalysisResult

class AnalysisResultService:
    def __init__(self, db):
        self.analysis_result_repo = AnalysisResultRepository(db)

    def get_analysis_result(self, result_id: int) -> Optional[AnalysisResult]:
        """Lấy thông tin chi tiết kết quả"""
        return self.analysis_result_repo.get_result_by_id(result_id)

    def search_analysis_results(self, search: str = "") -> List[AnalysisResult]:
        """Tìm kiếm kết quả"""
        return self.analysis_result_repo.list_results(search)

    def average_confidence(self, results: List[AnalysisResult]) -> float:
        """Tính điểm tin cậy trung bình"""
        if not results:
            return 0.0
        
        total_confidence = sum(result.confidence_score or 0 for result in results)
        return round((total_confidence / len(results)) * 100, 1)