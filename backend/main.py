from fastapi import FastAPI, UploadFile, File, Form
import pandas as pd
from model_runner import run_models
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/upload/")
async def upload_file(file: UploadFile = File(...), target: str = Form(...)):
    df = pd.read_csv(file.file)

    if target not in df.columns:
        return {"error": "Target column not found"}

    results = run_models(df, target)

    return {
        "target": target,
        **results 
    }


@app.get("/download/")
def download_model():
    return FileResponse("best_model.pkl", filename="best_model.pkl")

from fastapi import Body
import pandas as pd

@app.post("/load_url/")
async def load_url(data: dict = Body(...)):
    url = data["url"]
    target = data["target"]

    df = pd.read_csv(url)

    results = run_models(df, target)

    return results