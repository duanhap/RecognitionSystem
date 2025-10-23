from fastapi import APIRouter, Depends, Request
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, RedirectResponse
from sqlalchemy.orm import Session
from app.services.user_service import UserService
from app.core.database import get_db

templates = Jinja2Templates(directory="app/templates")

class ManageUserRouter:
    def __init__(self):
        self.router = APIRouter()
        # Trang chính
        self.router.add_api_route("/manage_user", self.manage_user_page, methods=["GET"], response_class=HTMLResponse)
        # Xóa user
        self.router.add_api_route("/manage_user/delete/{user_id}", self.delete_user, methods=["POST"])

    async def manage_user_page(self, request: Request, search: str = None, db: Session = Depends(get_db)):
        service = UserService(db)
        users = service.search_users(search)
        username = request.cookies.get("username")  # lấy từ cookie
        request.session["users_cache"] = [user.to_dict() for user in users]
        return templates.TemplateResponse("manage_user.html", {"request": request, "users": users, "search": search, "username": username})

    async def delete_user(self, request: Request, user_id: int, db: Session = Depends(get_db)):
        service = UserService(db)
        success = service.remove_user(user_id)
        if success:
            return RedirectResponse(url="/manage_user", status_code=303)
        else:
            return templates.TemplateResponse("error.html", {"request": request, "message": "User not found"})
# Export instance
manage_user_router = ManageUserRouter().router
