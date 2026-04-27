from fastapi import FastAPI, UploadFile, File, Form
import pandas as pd
from model_runner import run_models
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from fastapi import Body
import pandas as pd

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


@app.post("/load_url/")
async def load_url(data: dict = Body(...)):
    try:
        url = data.get("url")
        target = data.get("target")

        print("URL:", url)

        df = pd.read_csv(url)

        print("Columns:", df.columns)

        if target not in df.columns:
            return {"error": f"Target '{target}' not found"}

        results = run_models(df, target)
        return results

    except Exception as e:
        print("ERROR:", str(e))   
        return {"error": str(e)}