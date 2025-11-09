# app/routers/identity_verification_router.py
from typing import List
from fastapi import APIRouter, Depends, Request, Form, UploadFile, File
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, RedirectResponse
from sqlalchemy.orm import Session
from app.core.database import get_db
from app.services.identity_service import IdentityService

templates = Jinja2Templates(directory="app/templates")

class IdentityVerificationRouter:
    def __init__(self):
        self.router = APIRouter()
        self.router.add_api_route("/identity-verification", self.identity_verification_page, methods=["GET"], response_class=HTMLResponse)
        self.router.add_api_route("/identity-verification/add-label", self.add_label_page, methods=["GET"], response_class=HTMLResponse)
        self.router.add_api_route("/identity-verification/add-label", self.add_label, methods=["POST"])
        self.router.add_api_route("/identity-verification/add-sample", self.add_sample_page, methods=["GET"], response_class=HTMLResponse)
        self.router.add_api_route("/identity-verification/add-sample", self.add_sample, methods=["POST"])
        self.router.add_api_route("/identity-verification/delete/{sample_id}", self.delete_sample, methods=["GET"])
        self.router.add_api_route("/identity-verification/clear-message", self.clear_message, methods=["GET"])

    async def identity_verification_page(self, request: Request, search: str = None, page: int = 1, db: Session = Depends(get_db)):
        service = IdentityService(db)
        username = request.cookies.get("username")
        
        samples, total, total_pages = service.get_identity_samples(search, page)
        identities = service.get_all_identities()
        
        return templates.TemplateResponse("identity_verification.html", {
            "request": request,
            "samples": samples,
            "identities": identities,
            "search": search,
            "page": page,
            "total_pages": total_pages,
            "username": username
        })

    async def add_label_page(self, request: Request, db: Session = Depends(get_db)):
        username = request.cookies.get("username")
        return templates.TemplateResponse("add_identity_label.html", {
            "request": request,
            "username": username
        })

    async def add_label(self, request: Request, name: str = Form(...), db: Session = Depends(get_db)):
        service = IdentityService(db)
        success = service.create_identity(name)
        if success:
            return RedirectResponse(url="/identity-verification", status_code=303)
        else:
            return templates.TemplateResponse("add_identity_label.html", {
                "request": request,
                "error": "Failed to create identity"
            })

    async def add_sample_page(self, request: Request, db: Session = Depends(get_db)):
        service = IdentityService(db)
        username = request.cookies.get("username")
        identities = service.get_all_identities()
        
        return templates.TemplateResponse("add_identity_sample.html", {
            "request": request,
            "identities": identities,
            "username": username
        })

    async def add_sample(self, request: Request, 
                        identity_id: int = Form(...),
                        file_type: str = Form(...),
                        description: str = Form(...),
                        files: List[UploadFile] = File(...),
                        db: Session = Depends(get_db)):
        user_id = request.cookies.get("user_id")
        service = IdentityService(db)
        # Xử lý từng file
        success_count = 0
        total_files = len(files)
        
        for file in files:
            success = service.add_identity_sample(identity_id, file_type, description, file, user_id)
            if success:
                success_count += 1
        
        if success_count > 0:
            # Redirect với thông báo thành công
            response = RedirectResponse(url="/identity-verification", status_code=303)
            if success_count < total_files:
                # Nếu có file upload thất bại
                response.set_cookie(key="upload_message", value=f"Successfully uploaded {success_count}/{total_files} files")
            else:
                response.set_cookie(key="upload_message", value=f"Successfully uploaded {success_count} files")
            return response
        else:
            identities = service.get_all_identities()
            return templates.TemplateResponse("add_identity_sample.html", {
                "request": request,
                "identities": identities,
                "error": f"Failed to upload all {total_files} files"
            })

    async def delete_sample(self, request: Request, sample_id: int, db: Session = Depends(get_db)):
        service = IdentityService(db)
        success = service.delete_sample(sample_id)
        if success:
            return RedirectResponse(url="/identity-verification", status_code=303)
        else:
            return templates.TemplateResponse("error.html", {
                "request": request,
                "message": "Failed to delete sample"
            })
    async def clear_message(self, request: Request):
        response = RedirectResponse(url="/identity-verification", status_code=303)
        response.delete_cookie("upload_message")
        return response

identity_verification_router = IdentityVerificationRouter().router