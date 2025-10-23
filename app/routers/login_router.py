# app/routers/login_router.py
from fastapi import APIRouter, Depends, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session
from app.services.login_service import LoginService   # <--- sửa import
from app.core.database import get_db
from fastapi.responses import RedirectResponse


templates = Jinja2Templates(directory="app/templates")

class LoginRouter:
    def __init__(self):
        self.router = APIRouter()
        self.router.add_api_route("/login", self.login_page, methods=["GET"], response_class=HTMLResponse)
        self.router.add_api_route("/login", self.login, methods=["POST"])

    async def login_page(self, request: Request):
        return templates.TemplateResponse("login.html", {"request": request})

    async def login(self, request: Request, username: str = Form(...), password: str = Form(...), db: Session = Depends(get_db)):
        service = LoginService(db)   # <--- khởi tạo service
        user = service.authenticate_user(username, password)
        if user:
            # Đúng thì chuyển sang router Home (redirect)
            response = RedirectResponse(url="/home", status_code=303)
            response.set_cookie("username", user.username)  # hoặc session/token
            response.set_cookie("user_id", str(user.id))
            return response
        return templates.TemplateResponse("login.html", {"request": request, "error": "Sai username hoặc password!"})

# Export instance để main.py import
login_router = LoginRouter().router
