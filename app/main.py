from fastapi import FastAPI

from .database import Base, engine
from .routers import clients

# Création des tables
Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="FastIA API",
    description="API REST pour exposer les données nettoyées (clients & crédit)",
    version="0.1.0",
)

app.include_router(clients.router)
