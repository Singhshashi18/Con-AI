import os
from pathlib import Path
from flask import Flask
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
from dotenv import load_dotenv
from sqlalchemy import inspect, text


db = SQLAlchemy()


def ensure_persona_schema() -> None:
    inspector = inspect(db.engine)
    if "personas" not in inspector.get_table_names():
        return

    columns = {column["name"] for column in inspector.get_columns("personas")}
    migrations = {
        "avatar": "ALTER TABLE personas ADD COLUMN avatar VARCHAR(255) NOT NULL DEFAULT ''",
        "recommended_memory_strategy": (
            "ALTER TABLE personas ADD COLUMN recommended_memory_strategy "
            "VARCHAR(30) NOT NULL DEFAULT 'buffer'"
        ),
        "temperature": "ALTER TABLE personas ADD COLUMN temperature FLOAT NOT NULL DEFAULT 0.4",
        "domain_focus": (
            "ALTER TABLE personas ADD COLUMN domain_focus VARCHAR(30) NOT NULL DEFAULT 'general'"
        ),
    }

    for column, ddl in migrations.items():
        if column not in columns:
            db.session.execute(text(ddl))
    db.session.commit()


def create_app() -> Flask:
  
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")

    app = Flask(__name__)
    
 
    database_uri = os.getenv("DATABASE_URL", "sqlite:///conversational_ai.db")
    app.config["SQLALCHEMY_DATABASE_URI"] = database_uri
    app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

    CORS(app)
    db.init_app(app)

    with app.app_context():
        from . import models  
        from .routes import api_bp

        db.create_all()
        ensure_persona_schema()
        app.register_blueprint(api_bp, url_prefix="/api")

    return app
