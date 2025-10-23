from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from app.routers.manage_user_router import manage_user_router
from app.routers.login_router import login_router
from app.routers.home_router import home_router
from app.routers.view_detail_user_router import view_detail_user_router
from app.routers.manage_sample_router import manage_sample_router
from app.routers.add_multiple_samples_router import add_multiple_samples_router
from starlette.middleware.sessions import SessionMiddleware


app = FastAPI()
# Mount thư mục static
app.mount("/static", StaticFiles(directory="app/static"), name="static")
# ✅ Thêm SessionMiddleware
app.add_middleware(SessionMiddleware, secret_key="my_super_secret_key")
# Đăng ký router
app.include_router(login_router)
app.include_router(home_router)
app.include_router(manage_user_router)
app.include_router(manage_sample_router)
app.include_router(add_multiple_samples_router)
app.include_router(view_detail_user_router)

