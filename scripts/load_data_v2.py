import os
from datetime import datetime

import pandas as pd
from sqlalchemy.orm import Session

from app.database import SessionLocal, engine, Base
from app import models

CSV_PATH = os.path.join("data", "data-all-complete.csv")  # adapte au vrai nom


def parse_date(date_str: str):
    return datetime.fromisoformat(str(date_str)).date()


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    # Casts de base
    df["nb_enfants"] = df["nb_enfants"].astype(int)
    df["quotient_caf"] = df["quotient_caf"].astype(float)

    # Outliers simples
    df.loc[df["nb_enfants"] < 0, "nb_enfants"] = 0
    df.loc[df["nb_enfants"] > 10, "nb_enfants"] = 10

    q_low = df["quotient_caf"].quantile(0.01)
    q_high = df["quotient_caf"].quantile(0.99)
    df["quotient_caf"] = df["quotient_caf"].clip(lower=q_low, upper=q_high)

    # Option : supprimer les lignes avec trop de NaN sur des colonnes clés
    df = df.dropna(
        subset=["score_credit", "historique_credits", "loyer_mensuel", "situation_familiale"]
    )

    return df


def load_csv_to_db():
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)

    df = pd.read_csv(CSV_PATH)
    df = clean_dataframe(df)

    db: Session = SessionLocal()
    try:
        for _, row in df.iterrows():
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
                family_status=row["situation_familiale"],
                account_created_at=parse_date(row["date_creation_compte"]),
                orientation_sexuelle=row["orientation_sexuelle"],
                nb_enfants=int(row["nb_enfants"]),
                quotient_caf=float(row["quotient_caf"]),
            )

            fin = models.FinancialInfo(
                monthly_income=int(row["revenu_estime_mois"]),
                credit_history=float(row["historique_credits"]),
                personal_risk=float(row["risque_personnel"]),
                credit_score=float(row["score_credit"]),
                monthly_rent=float(row["loyer_mensuel"]),
                loan_amount=float(row["montant_pret"]),
            )

            client.financial_info = fin
            db.add(client)

        db.commit()
        print("Import v2 terminé sans erreur.")
    finally:
        db.close()


if __name__ == "__main__":
    load_csv_to_db()

