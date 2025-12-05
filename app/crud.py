from sqlalchemy.orm import Session

from . import models, schemas


# ---- Clients ----

def get_clients(db: Session, skip: int = 0, limit: int = 100):
    return db.query(models.Client).offset(skip).limit(limit).all()


def get_client(db: Session, client_id: int):
    return db.query(models.Client).filter(models.Client.id == client_id).first()


def create_client(db: Session, client: schemas.ClientCreate):
    db_client = models.Client(
        last_name=client.last_name,
        first_name=client.first_name,
        age=client.age,
        height_cm=client.height_cm,
        weight_kg=client.weight_kg,
        sex=client.sex,
        sport_licence=client.sport_licence,
        education_level=client.education_level,
        region=client.region,
        smoker=client.smoker,
        is_french=client.is_french,
        family_status=client.family_status,
        account_created_at=client.account_created_at,
    )

    if client.financial_info:
        db_fin = models.FinancialInfo(
            monthly_income=client.financial_info.monthly_income,
            credit_history=client.financial_info.credit_history,
            personal_risk=client.financial_info.personal_risk,
            credit_score=client.financial_info.credit_score,
            monthly_rent=client.financial_info.monthly_rent,
            loan_amount=client.financial_info.loan_amount,
        )
        db_client.financial_info = db_fin

    db.add(db_client)
    db.commit()
    db.refresh(db_client)
    return db_client


def delete_client(db: Session, client_id: int):
    client = get_client(db, client_id)
    if client:
        db.delete(client)
        db.commit()
    return client
