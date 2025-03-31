import zipfile
import pandas as pd
import os
import requests
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
from tempfile import TemporaryDirectory

load_dotenv()
AIPROXY_TOKEN = os.getenv("AIPROXY_TOKEN")

app = FastAPI()

def query_llm(question: str) -> str:
    try:
        headers = {
            "Authorization": f"Bearer {AIPROXY_TOKEN}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": "gpt-3.5-turbo",
            "messages": [{"role": "user", "content": question}],
            "temperature": 0
        }
        response = requests.post(
            "https://api.aiproxy.io/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=30
        )
        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"].strip()
        else:
            return f"LLM API error: {response.status_code}"
    except Exception as e:
        return f"LLM exception: {str(e)}"

@app.post("/")
async def solve_assignment(
    question: str = Form(...),
    file: UploadFile = File(None)
):
    try:
        if file:
            with TemporaryDirectory() as tmpdir:
                file_path = os.path.join(tmpdir, file.filename)
                with open(file_path, "wb") as f:
                    f.write(await file.read())

                if zipfile.is_zipfile(file_path):
                    with zipfile.ZipFile(file_path, 'r') as zip_ref:
                        zip_ref.extractall(tmpdir)
                        for name in zip_ref.namelist():
                            if name.endswith(".csv"):
                                df = pd.read_csv(os.path.join(tmpdir, name))
                                if "answer" in df.columns:
                                    return {"answer": str(df["answer"].iloc[0])}
                                else:
                                    return {"answer": "Could not find 'answer' column."}
                else:
                    return {"answer": "Uploaded file is not a zip file."}
        else:
            return {"answer": query_llm(question)}
    except Exception as e:
        return JSONResponse(status_code=500, content={"answer": f"Error: {str(e)}"})
