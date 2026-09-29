from typing import Dict, Any
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend.api.v1 import api_v1_router
from weather_core.config import get_settings

settings = get_settings()

app = FastAPI(
    title="SIH26078 AERIS Backend API",
    description=(
        "AI-Driven Spatio-Temporal Tracking of Extreme Weather Anomalies API. "
        "Provides REST endpoints for extreme weather event detection, trajectories, "
        "GeoJSON spatial layers, and analytical alerts."
    ),
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS Configuration for Frontend Integration (Next.js / React)
allowed_origins = [
    "http://localhost:3000",
    "http://localhost:3001",
    "http://localhost:8000",
    "http://127.0.0.1:3000",
    "http://127.0.0.1:3001",
    "http://127.0.0.1:8000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# Include API v1 Router
app.include_router(api_v1_router)


# Root Health Check (as required by prompt: GET /health)
@app.get("/health", summary="System Health Check", tags=["Health"])
def health_check() -> Dict[str, Any]:
    return {
        "status": "ok",
        "service": "AERIS",
        "version": "0.1.0-dev",
    }


# Exception Handlers shielding internal stack traces
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": True,
            "status_code": exc.status_code,
            "message": exc.detail,
        },
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={
            "error": True,
            "status_code": 500,
            "message": "Internal server processing error. Please check system logs.",
        },
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="127.0.0.1", port=8000, reload=True)
