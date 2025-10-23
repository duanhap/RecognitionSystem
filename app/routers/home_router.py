from fastapi import APIRouter, Depends, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

templates = Jinja2Templates(directory="app/templates")


class HomeRouter:
    def __init__(self):
        self.router = APIRouter()
        self.router.add_api_route("/home", self.home_page, methods=["GET"], response_class=HTMLResponse)


    async def home_page(self, request: Request):
        username = request.cookies.get("username")  # lấy từ cookie
        return templates.TemplateResponse("home.html", {"request": request, "username": username})



home_router = HomeRouter().router
