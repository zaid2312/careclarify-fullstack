from typing import Literal
from pydantic import BaseModel, Field

Category = Literal[
    "Room Rent", "OT Fees", "Doctor Fees", "Consumables",
    "Pharmacy", "Special Services", "Diagnostics", "Implants", "Other",
]

Status = Literal[
    "Reasonable", "Overcharged", "Duplicate Billed",
    "Capping Violation", "Not Administered", "Verify Charge",
]

WarningType = Literal["danger", "warning"]


class BillItem(BaseModel):
    name: str
    original: float
    fair: float
    category: Category
    status: Status
    reason: str


class BillWarning(BaseModel):
    id: str
    type: WarningType
    title: str
    text: str
    action: str


class ChartData(BaseModel):
    room: float = 0
    ot: float = 0
    doctor: float = 0
    consumables: float = 0
    other: float = 0


class BillAnalysis(BaseModel):
    """Matches the object shape `renderAnalysis(bill)` expects in app.js."""
    title: str
    patient: str
    date: str
    billNo: str
    hospitalName: str
    totalOriginal: float
    totalFair: float
    overcharge: float
    items: list[BillItem]
    warnings: list[BillWarning]
    chartData: ChartData


class AnalyzeError(BaseModel):
    detail: str
