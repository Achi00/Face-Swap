import os
from typing import Union
import uvicorn
import tempfile
import shutil
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import JSONResponse
from predict import Predictor
import uvicorn
from pathlib import Path

app = FastAPI()
predictor = Predictor()
predictor.setup()

def save_upload_file_tmp(upload_file: UploadFile) -> str:
    """
    Save the uploaded file to a temporary file and return its file path.
    """
    # Use the file extension if available.
    suffix = ""
    if upload_file.filename and "." in upload_file.filename:
        suffix = "." + upload_file.filename.split('.')[-1]
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    with temp_file as buffer:
        shutil.copyfileobj(upload_file.file, buffer)
    return temp_file.name

@app.post("/predictions")
def create_prediction(
    operation: str = Form(...),
    source_image: str = Form(None),
    target_image: UploadFile = File(...),
    enhance_result: bool = Form(True)
):
    try:
        # Save uploaded files to temporary file paths.
        target_image_path = save_upload_file_tmp(target_image)
        # target_image_path = None
        # if target_image is not None:
        #     target_image_path = save_upload_file_tmp(target_image)
        
        # Call the predictor synchronously.
        result = predictor.predict(
            operation=operation,
            source_image=source_image,
            target_image=target_image_path,
            enhance_result=enhance_result
        )
        
        return JSONResponse(content=result)
    except Exception as e:
        return JSONResponse(
            status_code=400,
            content={"success": False, "error": str(e)}
        )

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    uvicorn.run(app, host="0.0.0.0", port=port)
