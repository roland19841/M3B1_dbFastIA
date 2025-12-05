"""ajout colonnes socio-demo clients

Revision ID: 505a50ece355
Revises: 
Create Date: 2025-12-05 09:47:01.505697

"""
from alembic import op
import sqlalchemy as sa


# Révision générée par Alembic
revision = "ajout_colonnes_socio_demo"
down_revision = None  # à remplacer par l'id réel de la révision précédente
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "clients",
        sa.Column("orientation_sexuelle", sa.String(length=20), nullable=False, server_default="het"),
    )
    op.add_column(
        "clients",
        sa.Column("nb_enfants", sa.Integer(), nullable=False, server_default="0"),
    )
    op.add_column(
        "clients",
        sa.Column("quotient_caf", sa.Float(), nullable=False, server_default="0"),
    )


def downgrade() -> None:
    op.drop_column("clients", "quotient_caf")
    op.drop_column("clients", "nb_enfants")
    op.drop_column("clients", "orientation_sexuelle")
