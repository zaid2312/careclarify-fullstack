import os

from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from ai_bill_analyzer import analyze_bill_file
from schemas import BillAnalysis

app = FastAPI(
    title="CareClarify API",
    description="Backend for CareClarify — AI-powered hospital bill auditing.",
    version="1.0.0",
)

origins = [o.strip() for o in os.getenv("FRONTEND_ORIGINS", "*").split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins if origins != ["*"] else ["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/api/analyze-bill", response_model=BillAnalysis)
async def analyze_bill(
    file: UploadFile | None = File(default=None),
    text: str | None = Form(default=None),
):
    """
    Upload a hospital bill (JPEG/PNG photo, PDF, or .txt) and get back an AI-audited
    breakdown of line items, fair-price estimates, and flagged findings.

    Alternatively (or in addition), pass `text` as pasted bill content.
    """
    return await analyze_bill_file(file, text)


# Serve the frontend (index.html, app.js, styles.css, logo.png) from the /app
# directory. Mounted last so API routes above take precedence.
app.mount(
    "/",
    StaticFiles(directory=os.path.dirname(os.path.abspath(__file__)), html=True),
    name="static",
)
