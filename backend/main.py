from fastapi import FastAPI, UploadFile, File, Form
import pandas as pd
from model_runner import run_models
from fastapi.middleware.cors import CORSMiddleware
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # allow all (for now)
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
        "results": results
    }