from __future__ import annotations

from datetime import datetime, timezone

from . import db


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class Persona(db.Model):
    __tablename__ = "personas"

    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(120), nullable=False, unique=True)
    avatar = db.Column(db.String(255), nullable=False, default="")
    system_prompt = db.Column(db.Text, nullable=False)
    personality = db.Column(db.Text, nullable=False)
    default_memory_strategy = db.Column(db.String(30), nullable=False, default="buffer")
    recommended_memory_strategy = db.Column(db.String(30), nullable=False, default="buffer")
    temperature = db.Column(db.Float, nullable=False, default=0.4)
    domain_focus = db.Column(db.String(30), nullable=False, default="general")
    created_at = db.Column(db.DateTime(timezone=True), default=utc_now, nullable=False)

    conversations = db.relationship("Conversation", backref="persona", lazy=True)


class Conversation(db.Model):
    __tablename__ = "conversations"

    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(255), nullable=False)
    memory_strategy = db.Column(db.String(30), nullable=False, default="buffer")
    pinned = db.Column(db.Boolean, nullable=False, default=False)
    persona_id = db.Column(db.Integer, db.ForeignKey("personas.id"), nullable=True)
    created_at = db.Column(db.DateTime(timezone=True), default=utc_now, nullable=False)
    updated_at = db.Column(db.DateTime(timezone=True), default=utc_now, onupdate=utc_now, nullable=False)

    messages = db.relationship(
        "Message",
        backref="conversation",
        lazy=True,
        cascade="all, delete-orphan",
        order_by="Message.created_at",
    )
    entities = db.relationship(
        "EntityMemory",
        backref="conversation",
        lazy=True,
        cascade="all, delete-orphan",
    )
    edges = db.relationship(
        "KnowledgeEdge",
        backref="conversation",
        lazy=True,
        cascade="all, delete-orphan",
    )
    summary = db.relationship(
        "ConversationSummary",
        backref="conversation",
        uselist=False,
        lazy=True,
        cascade="all, delete-orphan",
    )


class Message(db.Model):
    __tablename__ = "messages"

    id = db.Column(db.Integer, primary_key=True)
    conversation_id = db.Column(db.Integer, db.ForeignKey("conversations.id"), nullable=False)
    role = db.Column(db.String(20), nullable=False)
    content = db.Column(db.Text, nullable=False)
    created_at = db.Column(db.DateTime(timezone=True), default=utc_now, nullable=False)


class EntityMemory(db.Model):
    __tablename__ = "entity_memories"

    id = db.Column(db.Integer, primary_key=True)
    conversation_id = db.Column(db.Integer, db.ForeignKey("conversations.id"), nullable=False)
    entity_name = db.Column(db.String(120), nullable=False)
    entity_type = db.Column(db.String(60), nullable=False, default="unknown")
    facts = db.Column(db.Text, nullable=False, default="")
    updated_at = db.Column(db.DateTime(timezone=True), default=utc_now, onupdate=utc_now, nullable=False)


class KnowledgeEdge(db.Model):
    __tablename__ = "knowledge_edges"

    id = db.Column(db.Integer, primary_key=True)
    conversation_id = db.Column(db.Integer, db.ForeignKey("conversations.id"), nullable=False)
    source = db.Column(db.String(120), nullable=False)
    relation = db.Column(db.String(120), nullable=False)
    target = db.Column(db.String(120), nullable=False)
    confidence = db.Column(db.Float, nullable=False, default=0.7)
    updated_at = db.Column(db.DateTime(timezone=True), default=utc_now, onupdate=utc_now, nullable=False)


class ConversationSummary(db.Model):
    __tablename__ = "conversation_summaries"

    id = db.Column(db.Integer, primary_key=True)
    conversation_id = db.Column(db.Integer, db.ForeignKey("conversations.id"), nullable=False, unique=True)
    summary = db.Column(db.Text, nullable=False, default="")
    updated_at = db.Column(db.DateTime(timezone=True), default=utc_now, onupdate=utc_now, nullable=False)


class ComparisonRun(db.Model):
    __tablename__ = "comparison_runs"

    id = db.Column(db.Integer, primary_key=True)
    flow_name = db.Column(db.String(120), nullable=False)
    script = db.Column(db.Text, nullable=False)
    results = db.Column(db.Text, nullable=False)
    created_at = db.Column(db.DateTime(timezone=True), default=utc_now, nullable=False)
