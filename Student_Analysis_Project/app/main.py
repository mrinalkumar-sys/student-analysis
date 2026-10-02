"""
FastAPI Web Application and REST API for Student Performance Analysis.
"""
from typing import Dict, Any, Optional
from pathlib import Path
import shutil
from fastapi import FastAPI, HTTPException, Query, UploadFile, File
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from starlette.requests import Request
from pydantic import BaseModel, Field

from src.config import BASE_DIR, DATASET_FILE, HOST, PORT, DEBUG
from src.data_loader import DataLoader
from src.stats_engine import StatsEngine
from src.inference import InferenceService
from src.train import train_all_models

# Initialize FastAPI
app = FastAPI(
    title="Student Analysis & Performance Prediction System",
    description="Interactive platform for exploratory data analysis, statistical hypothesis testing, and ML performance prediction based on Students_Performance.ipynb.",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static and Templates
STATIC_DIR = BASE_DIR / "app" / "static"
TEMPLATES_DIR = BASE_DIR / "app" / "templates"
STATIC_DIR.mkdir(parents=True, exist_ok=True)
TEMPLATES_DIR.mkdir(parents=True, exist_ok=True)

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

# Service singletons
data_loader = DataLoader()
stats_engine = StatsEngine(data_loader)
inference_service = InferenceService()

# Pydantic Request Models
class StudentPredictRequest(BaseModel):
    Gender: str = Field(default="female", description="female or male")
    EthnicGroup: Optional[str] = Field(default="group C", description="group A through group E")
    ParentEduc: Optional[str] = Field(default="some college", description="Parental education level")
    LunchType: str = Field(default="standard", description="standard or free/reduced")
    TestPrep: Optional[str] = Field(default="none", description="none or completed")
    ParentMaritalStatus: Optional[str] = Field(default="married", description="married, single, widowed, divorced")
    PracticeSport: Optional[str] = Field(default="regularly", description="regularly, sometimes, never")
    IsFirstChild: Optional[str] = Field(default="yes", description="yes or no")
    NrSiblings: Optional[float] = Field(default=2.0, description="Number of siblings (0-7)")
    TransportMeans: Optional[str] = Field(default="school_bus", description="school_bus or private")
    WklyStudyHours: Optional[str] = Field(default="5 - 10", description="< 5, 5 - 10, > 10")
    ReadingScore: Optional[float] = Field(default=70.0, description="Reading Score (0-100)")
    WritingScore: Optional[float] = Field(default=70.0, description="Writing Score (0-100)")

class HypothesisTestRequest(BaseModel):
    group_column: str = Field(default="LunchType", description="Demographic column to split by")
    group1_val: str = Field(default="standard", description="Value for group 1")
    group2_val: str = Field(default="free/reduced", description="Value for group 2")
    score_column: str = Field(default="MathScore", description="Target exam score (MathScore, ReadingScore, WritingScore)")
    alpha: float = Field(default=0.05, description="Significance level (default 0.05)")

# Routes
@app.get("/", response_class=HTMLResponse)
async def serve_dashboard(request: Request):
    """Serves the main interactive dashboard UI."""
    return templates.TemplateResponse(request=request, name="index.html")

@app.get("/api/health")
async def health_check():
    return {
        "status": "online",
        "app": "Student Performance Analysis & ML Platform",
        "version": "1.0.0",
        "dataset_loaded": data_loader.df is not None,
        "total_records": len(data_loader.df) if data_loader.df is not None else 0
    }

@app.get("/api/data/summary")
async def get_summary_statistics():
    """Returns mean, median, mode, std dev, and describe stats (Cell 9)."""
    try:
        return data_loader.get_summary_statistics()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/data/missing")
async def get_missing_values():
    """Returns missing values breakdown (Cell 4)."""
    try:
        return data_loader.get_missing_values()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/data/records")
async def get_records(
    page: int = Query(1, ge=1),
    page_size: int = Query(15, ge=1, le=100),
    search: Optional[str] = None
):
    """Returns searchable, paginated student records."""
    try:
        return data_loader.get_records(page=page, page_size=page_size, search=search)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/analytics/correlations")
async def get_correlations():
    """Returns correlation matrix for exam scores (Cell 10)."""
    try:
        return data_loader.get_correlation_matrix()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/analytics/demographics")
async def get_all_demographics():
    """Returns grouped analysis for all demographic factors."""
    try:
        return data_loader.get_all_demographic_analyses()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/analytics/demographics/{factor}")
async def get_single_demographic(factor: str):
    """Returns grouped performance breakdown for a single factor."""
    try:
        return data_loader.get_demographic_breakdown(factor)
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/analytics/hypothesis-test")
async def run_hypothesis_test(req: HypothesisTestRequest):
    """Runs independent two-sample t-test (Cell 53)."""
    try:
        result = stats_engine.run_two_sample_ttest(
            group_column=req.group_column,
            group1_val=req.group1_val,
            group2_val=req.group2_val,
            score_column=req.score_column,
            alpha=req.alpha
        )
        return result
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/predict/score")
async def predict_score(req: StudentPredictRequest):
    """Regression inference: predicts student exam score (Cell 58)."""
    try:
        input_data = req.model_dump()
        return inference_service.predict_score(input_data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/predict/grade")
async def predict_grade(req: StudentPredictRequest):
    """Classification inference: predicts academic grade letter (Cell 60)."""
    try:
        input_data = req.model_dump()
        return inference_service.predict_grade(input_data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/models/info")
async def get_models_info():
    """Returns performance metrics and feature importance of trained models."""
    try:
        return inference_service.get_models_info()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/data/upload")
async def upload_dataset(file: UploadFile = File(...)):
    """Allows uploading a new or updated CSV dataset."""
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are accepted.")
    
    try:
        with open(DATASET_FILE, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Reload data
        data_loader.load_data(reload=True)
        # Retrain models
        train_all_models(DATASET_FILE)
        inference_service.ensure_models_loaded()
        
        return {
            "message": "Dataset uploaded successfully and models retrained.",
            "filename": file.filename,
            "total_records": len(data_loader.df)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process file: {str(e)}")

@app.post("/api/models/retrain")
async def retrain_models():
    """Triggers retraining of regression and classification models."""
    try:
        summary = train_all_models(DATASET_FILE)
        inference_service.ensure_models_loaded()
        return {"status": "success", "summary": summary}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
