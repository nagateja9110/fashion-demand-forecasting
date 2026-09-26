from pydantic import BaseModel, Field


class ForecastRequest(BaseModel):
    category: str
    sub_category: str
    horizon_days: int = Field(default=14, ge=1, le=60)
    avg_unit_price: float | None = Field(default=None, gt=0)
    discount_override_pct: float | None = Field(default=None, ge=0, le=1)


class ForecastPoint(BaseModel):
    date: str
    predicted_units_sold: float
    planned_discount_pct: float
    avg_unit_price: float


class ForecastResponse(BaseModel):
    category: str
    sub_category: str
    points: list[ForecastPoint]
