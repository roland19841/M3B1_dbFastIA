from sqlalchemy import (
    Column,
    Integer,
    String,
    Float,
    Date,
    ForeignKey,
    UniqueConstraint,
)
from sqlalchemy.orm import relationship

from .database import Base


class Client(Base):
    __tablename__ = "clients"

    id = Column(Integer, primary_key=True, index=True)

    last_name = Column(String(100), nullable=False)   # nom
    first_name = Column(String(100), nullable=False)  # prenom
    age = Column(Integer, nullable=False)

    height_cm = Column(Float, nullable=True)  # taille
    weight_kg = Column(Float, nullable=True)  # poids
    sex = Column(String(1), nullable=False)   # 'H' ou 'F'

    sport_licence = Column(String(10), nullable=False)         # 'oui' / 'non'
    education_level = Column(String(50), nullable=False)       # niveau_etude
    region = Column(String(100), nullable=False)
    smoker = Column(String(10), nullable=False)                # 'oui' / 'non'
    is_french = Column(String(10), nullable=False)             # nationalité_francaise
    family_status = Column(String(50), nullable=False)         # situation_familiale
    account_created_at = Column(Date, nullable=False)          # date_creation_compte

    # Relation 1-1 avec FinancialInfo
    financial_info = relationship(
        "FinancialInfo",
        back_populates="client",
        uselist=False,
        cascade="all, delete-orphan",
    )


class FinancialInfo(Base):
    __tablename__ = "financial_infos"

    id = Column(Integer, primary_key=True, index=True)
    client_id = Column(Integer, ForeignKey("clients.id"), nullable=False, unique=True)

    monthly_income = Column(Integer, nullable=False)     # revenu_estime_mois
    credit_history = Column(Float, nullable=True)        # historique_credits
    personal_risk = Column(Float, nullable=True)         # risque_personnel
    credit_score = Column(Float, nullable=True)          # score_credit
    monthly_rent = Column(Float, nullable=True)          # loyer_mensuel
    loan_amount = Column(Float, nullable=True)           # montant_pret

    client = relationship("Client", back_populates="financial_info")

    __table_args__ = (
        UniqueConstraint("client_id", name="uq_financialinfo_client"),
    )
