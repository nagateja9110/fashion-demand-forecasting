import json

import xgboost as xgb
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from src import config, forecast
from .schemas import ForecastRequest, ForecastResponse

app = FastAPI(title="Fashion Demand Forecasting API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

_booster: xgb.Booster | None = None
_context: dict | None = None
_dashboard_data: dict | None = None


@app.on_event("startup")
def load_artifacts() -> None:
    global _booster, _context, _dashboard_data
    _booster = xgb.Booster()
    _booster.load_model(str(config.MODELS_DIR / "xgboost_model.json"))
    _context = forecast.load_forecast_context()
    with open(config.DASHBOARD_DATA_JSON) as f:
        _dashboard_data = json.load(f)


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.get("/api/metrics")
def metrics():
    return _dashboard_data["metrics"]


@app.get("/api/feature-importance")
def feature_importance():
    return _dashboard_data["feature_importance"]


@app.get("/api/actual-vs-predicted")
def actual_vs_predicted():
    return _dashboard_data["actual_vs_predicted"]


@app.get("/api/groups")
def groups():
    return _context["groups"]


@app.post("/api/forecast", response_model=ForecastResponse)
def forecast_endpoint(req: ForecastRequest):
    try:
        points = forecast.recursive_forecast(
            _context,
            _booster,
            category=req.category,
            sub_category=req.sub_category,
            horizon_days=req.horizon_days,
            avg_unit_price=req.avg_unit_price,
            discount_override_pct=req.discount_override_pct,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return ForecastResponse(category=req.category, sub_category=req.sub_category, points=points)


app.mount("/", StaticFiles(directory=str(config.ROOT_DIR / "static"), html=True), name="static")
