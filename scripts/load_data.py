import os
from datetime import datetime

import pandas as pd
from sqlalchemy.orm import Session

from app.database import SessionLocal, engine, Base
from app import models

CSV_PATH = os.path.join("data", "data-all.csv")  # adapte le nom si besoin


def parse_date(date_str: str):
    # Si ton CSV est déjà au format YYYY-MM-DD, ça marche.
    # Sinon, adapte le format ici.
    return datetime.fromisoformat(str(date_str)).date()


def load_csv_to_db():
    # 1) On repart de zéro : drop + create tables
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)

    df = pd.read_csv(CSV_PATH)

    db: Session = SessionLocal()
    try:
        for _, row in df.iterrows():
            # Gestion des NaN pour situation_familiale
            family_status = row["situation_familiale"]
            if pd.isna(family_status):
                family_status = "inconnue"

            client = models.Client(
                last_name=row["nom"],
                first_name=row["prenom"],
                age=int(row["age"]),
                height_cm=float(row["taille"]),
                weight_kg=float(row["poids"]),
                sex=row["sexe"],
                sport_licence=row["sport_licence"],
                education_level=row["niveau_etude"],
                region=row["region"],
                smoker=row["smoker"],
                is_french=row["nationalité_francaise"],
                family_status=str(family_status),
                account_created_at=parse_date(row["date_creation_compte"]),
            )

            fin = models.FinancialInfo(
                monthly_income=int(row["revenu_estime_mois"]),
                credit_history=(
                    float(row["historique_credits"])
                    if not pd.isna(row["historique_credits"])
                    else None
                ),
                personal_risk=(
                    float(row["risque_personnel"])
                    if not pd.isna(row["risque_personnel"])
                    else None
                ),
                credit_score=(
                    float(row["score_credit"])
                    if not pd.isna(row["score_credit"])
                    else None
                ),
                monthly_rent=(
                    float(row["loyer_mensuel"])
                    if not pd.isna(row["loyer_mensuel"])
                    else None
                ),
                loan_amount=(
                    float(row["montant_pret"])
                    if not pd.isna(row["montant_pret"])
                    else None
                ),
            )

            client.financial_info = fin
            db.add(client)

        db.commit()
        print("Import terminé sans erreur.")
    finally:
        db.close()


if __name__ == "__main__":
    load_csv_to_db()
