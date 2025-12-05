from datetime import date
from typing import Optional

from pydantic import BaseModel


# ---- FinancialInfo ----

class FinancialInfoBase(BaseModel):
    monthly_income: int
    credit_history: Optional[float] = None
    personal_risk: Optional[float] = None
    credit_score: Optional[float] = None
    monthly_rent: Optional[float] = None
    loan_amount: Optional[float] = None


class FinancialInfoCreate(FinancialInfoBase):
    pass


class FinancialInfo(FinancialInfoBase):
    id: int
    client_id: int

    class Config:
        orm_mode = True


# ---- Client ----

class ClientBase(BaseModel):
    last_name: str
    first_name: str
    age: int
    height_cm: Optional[float] = None
    weight_kg: Optional[float] = None
    sex: str
    sport_licence: str
    education_level: str
    region: str
    smoker: str
    is_french: str
    family_status: str
    account_created_at: date

    orientation_sexuelle: str
    nb_enfants: int
    quotient_caf: float


class ClientCreate(ClientBase):
    # on permet de créer en même temps le profil financier si on veut
    financial_info: Optional[FinancialInfoCreate] = None


class Client(ClientBase):
    id: int
    financial_info: Optional[FinancialInfo] = None

    class Config:
        orm_mode = True
