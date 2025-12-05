from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

# URL de connexion à la base SQLite (fichier fastia.db à la racine du projet)
SQLALCHEMY_DATABASE_URL = "sqlite:///./fastia.db"

# Création de l'engine
engine = create_engine(
    SQLALCHEMY_DATABASE_URL,
    connect_args={"check_same_thread": False},  # nécessaire pour SQLite + threads
)

# Fabrique de sessions
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base pour les modèles ORM
Base = declarative_base()
