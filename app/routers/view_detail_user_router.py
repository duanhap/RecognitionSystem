from fastapi import APIRouter, Depends, Request, Form
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session
from fastapi.responses import HTMLResponse, RedirectResponse
from app.services.user_service import UserService
from app.core.database import get_db

templates = Jinja2Templates(directory="app/templates")

class ViewDetailUserRouter:
    def __init__(self):
        self.router = APIRouter()
        self.router.add_api_route("/view_user/{user_id}", self.view_user_page, methods=["GET"], response_class=HTMLResponse)
        self.router.add_api_route("/view_user/{user_id}/update_status", self.update_status, methods=["POST"])
        self.router.add_api_route("/view_user/{user_id}/delete", self.delete_user, methods=["POST"])

    async def view_user_page(self, request: Request, user_id: int, db: Session = Depends(get_db)):
        service = UserService(db)
        user = service.get_user(user_id)
        username = request.cookies.get("username")  # lấy từ cookie
        return templates.TemplateResponse("view_detail_user.html", {"request": request, "user": user, "username": username})

    async def update_status(self, request: Request, user_id: int, status: str = Form(...), db: Session = Depends(get_db)):
        service = UserService(db)
        success = service.change_status(user_id, status)  # bây giờ trả về boolean
        if success:
            # Cập nhật thành công → redirect về chi tiết user
            return RedirectResponse(url=f"/view_user/{user_id}", status_code=303)
        else:
            # Không tìm thấy user → hiển thị thông báo lỗi
            return templates.TemplateResponse(
                "view_detail_user.html",
                {
                    "request": request,
                    "user": service.get_user(user_id),  # có thể là None
                    "error": "User not found, cannot update status!"
                }
            )

    async def delete_user(self, request: Request, user_id: int, db: Session = Depends(get_db)):
        service = UserService(db)
        success = service.remove_user(user_id)
        if success:
            return RedirectResponse(url="/manage_user", status_code=303)
        else:
            return templates.TemplateResponse("error.html", {"request": request, "message": "User not found"})

# Export instance
view_detail_user_router = ViewDetailUserRouter().router
