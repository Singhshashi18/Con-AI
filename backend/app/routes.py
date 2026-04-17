from __future__ import annotations

import json
import re
from io import BytesIO

from flask import Blueprint, Response, jsonify, request, stream_with_context
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

from . import db
from .chain_service import run_conversation_chain
from .llm import generate_gemini_reply, stream_gemini_reply
from .memory import (
    DOMAIN_FOCUS,
    MEMORY_STRATEGIES,
    build_memory_context,
    refresh_summary,
    sanitize_title,
    simulate_strategy_reply,
    update_entity_memory,
    update_knowledge_graph,
)
from .models import (
    ComparisonRun,
    Conversation,
    ConversationSummary,
    EntityMemory,
    KnowledgeEdge,
    Message,
    Persona,
)

api_bp = Blueprint("api", __name__)


def sse_event(payload: dict) -> str:
    return f"data: {json.dumps(payload)}\n\n"


def iter_word_chunks(text: str) -> list[str]:
    if not text:
        return []
    # Split into small visible chunks to keep progressive rendering smooth.
    return re.findall(r"\S+\s*|\s+", text)


def looks_generic_title(title: str) -> bool:
    normalized = (title or "").strip().lower()
    return normalized in {"new chat", "new conversation", "untitled conversation", ""}


def infer_conversation_title(message_text: str, persona: Persona | None) -> str:
    text = (message_text or "").strip()
    lowered = text.lower()

    keyword_rules = [
        ("code issue", ["bug", "error", "exception", "traceback", "debug", "fix", "code", "react", "flask", "python", "javascript", "frontend", "backend", "api"]),
        ("project discussion", ["project", "deadline", "roadmap", "launch", "feature", "milestone"]),
        ("business discussion", ["kpi", "revenue", "sales", "report", "metric", "analysis", "business"]),
        ("creative writing", ["story", "character", "plot", "chapter", "novel"]),
        ("study session", ["study", "learn", "concept", "exam", "question", "topic"]),
    ]

    for title, keywords in keyword_rules:
        if any(keyword in lowered for keyword in keywords):
            return title.title()

    words = [word for word in re.findall(r"[A-Za-z][A-Za-z0-9+.#-]+", text) if word.lower() not in {"i", "my", "the", "a", "an", "and", "or", "to", "for", "with"}]
    if persona and persona.domain_focus == "technical":
        prefix = "Coding"
    elif persona and persona.domain_focus == "creative":
        prefix = "Creative"
    elif persona and persona.domain_focus == "business":
        prefix = "Business"
    else:
        prefix = "Chat"

    if not words:
        return f"{prefix} Discussion"

    compact = " ".join(words[:4])
    return f"{prefix}: {compact}"[:80]


BUILTIN_PERSONAS = [
    {
        "name": "General Assistant",
        "avatar": "https://api.dicebear.com/9.x/shapes/svg?seed=general",
        "personality": "Helpful, clear, and balanced across tasks.",
        "system_prompt": "You are a reliable general assistant. Answer clearly and ask useful follow-up questions.",
        "default_memory_strategy": "buffer",
        "recommended_memory_strategy": "buffer",
        "temperature": 0.4,
        "domain_focus": "general",
    },
    {
        "name": "Code Helper",
        "avatar": "https://api.dicebear.com/9.x/shapes/svg?seed=code-helper",
        "personality": "Technical, precise, and implementation-focused.",
        "system_prompt": "You are a senior coding assistant. Track functions, files, classes, and debugging facts.",
        "default_memory_strategy": "entity",
        "recommended_memory_strategy": "entity",
        "temperature": 0.2,
        "domain_focus": "technical",
    },
    {
        "name": "Creative Writer",
        "avatar": "https://api.dicebear.com/9.x/shapes/svg?seed=creative-writer",
        "personality": "Imaginative, vivid, and story-driven.",
        "system_prompt": "You are a creative writing partner. Track characters, arcs, settings, and relationships.",
        "default_memory_strategy": "knowledge_graph",
        "recommended_memory_strategy": "knowledge_graph",
        "temperature": 0.9,
        "domain_focus": "creative",
    },
    {
        "name": "Business Analyst",
        "avatar": "https://api.dicebear.com/9.x/shapes/svg?seed=business-analyst",
        "personality": "Formal, structured, and KPI-oriented.",
        "system_prompt": "You are a business analyst. Track metrics, assumptions, and actionable recommendations.",
        "default_memory_strategy": "summary_entity",
        "recommended_memory_strategy": "summary_entity",
        "temperature": 0.35,
        "domain_focus": "business",
    },
    {
        "name": "Study Buddy",
        "avatar": "https://api.dicebear.com/9.x/shapes/svg?seed=study-buddy",
        "personality": "Educational, patient, and concept-first.",
        "system_prompt": "You are a study coach. Build concept maps and explain step-by-step with checkpoints.",
        "default_memory_strategy": "knowledge_graph",
        "recommended_memory_strategy": "knowledge_graph",
        "temperature": 0.45,
        "domain_focus": "general",
    },
]


def persona_to_dict(persona: Persona) -> dict:
    return {
        "id": persona.id,
        "name": persona.name,
        "avatar": persona.avatar,
        "system_prompt": persona.system_prompt,
        "personality": persona.personality,
        "default_memory_strategy": persona.default_memory_strategy,
        "recommended_memory_strategy": persona.recommended_memory_strategy,
        "temperature": persona.temperature,
        "domain_focus": persona.domain_focus,
        "created_at": persona.created_at.isoformat(),
    }


def message_to_dict(message: Message) -> dict:
    return {
        "id": message.id,
        "role": message.role,
        "content": message.content,
        "created_at": message.created_at.isoformat(),
    }


def conversation_to_dict(conversation: Conversation) -> dict:
    return {
        "id": conversation.id,
        "title": conversation.title,
        "memory_strategy": conversation.memory_strategy,
        "pinned": conversation.pinned,
        "persona_id": conversation.persona_id,
        "persona_name": conversation.persona.name if conversation.persona else None,
        "persona_avatar": conversation.persona.avatar if conversation.persona else "",
        "created_at": conversation.created_at.isoformat(),
        "updated_at": conversation.updated_at.isoformat(),
        "message_count": len(conversation.messages),
        "last_message": conversation.messages[-1].content if conversation.messages else "",
    }


def ensure_builtin_personas() -> None:
    existing_names = {persona.name for persona in Persona.query.all()}
    created = False
    for payload in BUILTIN_PERSONAS:
        if payload["name"] in existing_names:
            continue
        db.session.add(Persona(**payload))
        created = True
    if created:
        db.session.commit()


def validate_persona_payload(payload: dict, *, update: bool = False) -> tuple[dict, Response | None]:
    data: dict = {}

    if not update or "name" in payload:
        name = (payload.get("name") or "").strip()
        if not name:
            return {}, (jsonify({"error": "name is required"}), 400)
        data["name"] = name

    if "default_memory_strategy" in payload or not update:
        strategy = payload.get("default_memory_strategy", "buffer")
        if strategy not in MEMORY_STRATEGIES:
            return {}, (jsonify({"error": "Invalid memory strategy"}), 400)
        data["default_memory_strategy"] = strategy

    if "recommended_memory_strategy" in payload or not update:
        recommended = payload.get(
            "recommended_memory_strategy",
            data.get("default_memory_strategy", payload.get("default_memory_strategy", "buffer")),
        )
        if recommended not in MEMORY_STRATEGIES:
            return {}, (jsonify({"error": "Invalid recommended memory strategy"}), 400)
        data["recommended_memory_strategy"] = recommended

    if "domain_focus" in payload or not update:
        domain_focus = payload.get("domain_focus", "general")
        if domain_focus not in DOMAIN_FOCUS:
            return {}, (jsonify({"error": "Invalid domain focus"}), 400)
        data["domain_focus"] = domain_focus

    if "temperature" in payload or not update:
        try:
            temperature = float(payload.get("temperature", 0.4))
        except (TypeError, ValueError):
            return {}, (jsonify({"error": "temperature must be numeric"}), 400)
        if temperature < 0 or temperature > 1.5:
            return {}, (jsonify({"error": "temperature out of range (0.0 - 1.5)"}), 400)
        data["temperature"] = temperature

    if "system_prompt" in payload or not update:
        data["system_prompt"] = payload.get("system_prompt", "You are a helpful assistant.")

    if "personality" in payload or not update:
        data["personality"] = payload.get("personality", "Balanced and clear")

    if "avatar" in payload or not update:
        data["avatar"] = (payload.get("avatar") or "").strip()

    return data, None


@api_bp.before_app_request
def ensure_defaults() -> None:
    ensure_builtin_personas()


@api_bp.get("/health")
def health() -> Response:
    return jsonify({"status": "ok"})


@api_bp.get("/personas")
def list_personas() -> Response:
    personas = Persona.query.order_by(Persona.created_at.asc()).all()
    return jsonify([persona_to_dict(p) for p in personas])


@api_bp.post("/personas")
def create_persona() -> Response:
    payload = request.get_json(force=True)
    data, error_response = validate_persona_payload(payload, update=False)
    if error_response:
        return error_response

    persona = Persona(**data)
    db.session.add(persona)
    db.session.commit()
    return jsonify(persona_to_dict(persona)), 201


@api_bp.put("/personas/<int:persona_id>")
def update_persona(persona_id: int) -> Response:
    persona = Persona.query.get_or_404(persona_id)
    payload = request.get_json(force=True)
    data, error_response = validate_persona_payload(payload, update=True)
    if error_response:
        return error_response

    for key, value in data.items():
        setattr(persona, key, value)

    db.session.commit()
    return jsonify(persona_to_dict(persona))


@api_bp.delete("/personas/<int:persona_id>")
def delete_persona(persona_id: int) -> Response:
    persona = Persona.query.get_or_404(persona_id)
    if persona.name in {item["name"] for item in BUILTIN_PERSONAS}:
        return jsonify({"error": "Built-in personas cannot be deleted"}), 400
    db.session.delete(persona)
    db.session.commit()
    return jsonify({"deleted": True})


@api_bp.get("/conversations")
def list_conversations() -> Response:
    search = request.args.get("search", "").strip().lower()
    query = Conversation.query.order_by(Conversation.pinned.desc(), Conversation.updated_at.desc())
    conversations = query.all()

    if search:
        filtered = []
        for conv in conversations:
            in_title = search in conv.title.lower()
            in_messages = any(search in m.content.lower() for m in conv.messages)
            if in_title or in_messages:
                filtered.append(conv)
        conversations = filtered

    return jsonify([conversation_to_dict(c) for c in conversations])


@api_bp.post("/conversations")
def create_conversation() -> Response:
    payload = request.get_json(force=True)
    persona_id = payload.get("persona_id")
    persona = Persona.query.get(persona_id) if persona_id else None

    strategy = payload.get("memory_strategy") or (
        persona.default_memory_strategy if persona else "buffer"
    )
    if strategy not in MEMORY_STRATEGIES:
        return jsonify({"error": "Invalid memory strategy"}), 400

    conversation = Conversation(
        title=sanitize_title(payload.get("title", "New Conversation")),
        memory_strategy=strategy,
        persona_id=persona.id if persona else None,
        pinned=bool(payload.get("pinned", False)),
    )
    db.session.add(conversation)
    db.session.commit()
    return jsonify(conversation_to_dict(conversation)), 201


@api_bp.get("/conversations/<int:conversation_id>")
def get_conversation(conversation_id: int) -> Response:
    conversation = Conversation.query.get_or_404(conversation_id)
    return jsonify(conversation_to_dict(conversation))


@api_bp.patch("/conversations/<int:conversation_id>")
def patch_conversation(conversation_id: int) -> Response:
    conversation = Conversation.query.get_or_404(conversation_id)
    payload = request.get_json(force=True)

    if "title" in payload:
        conversation.title = sanitize_title(payload["title"])
    if "pinned" in payload:
        conversation.pinned = bool(payload["pinned"])
    if "memory_strategy" in payload:
        strategy = payload["memory_strategy"]
        if strategy not in MEMORY_STRATEGIES:
            return jsonify({"error": "Invalid memory strategy"}), 400
        conversation.memory_strategy = strategy
    if "persona_id" in payload:
        persona_id = payload["persona_id"]
        if persona_id is None:
            conversation.persona_id = None
        else:
            persona = Persona.query.get_or_404(persona_id)
            conversation.persona_id = persona.id

    db.session.commit()
    return jsonify(conversation_to_dict(conversation))


@api_bp.delete("/conversations/<int:conversation_id>")
def delete_conversation(conversation_id: int) -> Response:
    conversation = Conversation.query.get_or_404(conversation_id)
    db.session.delete(conversation)
    db.session.commit()
    return jsonify({"deleted": True})


@api_bp.get("/conversations/<int:conversation_id>/messages")
def list_messages(conversation_id: int) -> Response:
    messages = Message.query.filter_by(conversation_id=conversation_id).order_by(Message.created_at.asc()).all()
    return jsonify([message_to_dict(m) for m in messages])


@api_bp.get("/conversations/<int:conversation_id>/summary")
def get_summary(conversation_id: int) -> Response:
    Conversation.query.get_or_404(conversation_id)
    summary = ConversationSummary.query.filter_by(conversation_id=conversation_id).first()
    return jsonify(
        {
            "summary": summary.summary if summary else "No summary available yet.",
            "updated_at": summary.updated_at.isoformat() if summary else None,
        }
    )


def render_assistant_reply(conversation: Conversation, user_message: Message) -> str:
    reply_text, _metadata = run_conversation_chain(conversation, user_message)
    return reply_text


@api_bp.post("/conversations/<int:conversation_id>/chat")
def chat(conversation_id: int) -> Response:
    conversation = Conversation.query.get_or_404(conversation_id)
    payload = request.get_json(force=True)
    content = (payload.get("message") or "").strip()
    if not content:
        return jsonify({"error": "Message is required"}), 400

    user_msg = Message(conversation_id=conversation.id, role="user", content=content)
    db.session.add(user_msg)
    db.session.flush()

    if looks_generic_title(conversation.title):
        conversation.title = infer_conversation_title(content, conversation.persona)

    assistant_text = render_assistant_reply(conversation, user_msg)
    assistant_msg = Message(conversation_id=conversation.id, role="assistant", content=assistant_text)
    db.session.add(assistant_msg)
    db.session.flush()

    all_messages = Message.query.filter_by(conversation_id=conversation.id).order_by(Message.created_at.asc()).all()
    refresh_summary(conversation, all_messages)

    conversation.updated_at = assistant_msg.created_at
    db.session.commit()

    return jsonify(
        {
            "user": message_to_dict(user_msg),
            "assistant": message_to_dict(assistant_msg),
            "conversation": conversation_to_dict(conversation),
        }
    )


@api_bp.post("/conversations/<int:conversation_id>/chat/stream")
def chat_stream(conversation_id: int) -> Response:
    conversation = Conversation.query.get_or_404(conversation_id)
    payload = request.get_json(force=True)
    content = (payload.get("message") or "").strip()
    if not content:
        return jsonify({"error": "Message is required"}), 400

    user_msg = Message(conversation_id=conversation.id, role="user", content=content)
    db.session.add(user_msg)
    db.session.flush()

    if looks_generic_title(conversation.title):
        conversation.title = infer_conversation_title(content, conversation.persona)

    # Persist the user turn immediately so it is never lost if streaming aborts.
    db.session.commit()

    persona_header = conversation.persona.system_prompt if conversation.persona else "You are a helpful assistant."
    strategy = conversation.memory_strategy
    temperature = conversation.persona.temperature if conversation.persona else 0.4
    memory_context = build_memory_context(conversation, strategy)

    @stream_with_context
    def event_stream():
        assistant_parts: list[str] = []
        try:
            yield sse_event({"type": "start"})
            for chunk in stream_gemini_reply(
                persona_header=persona_header,
                strategy=strategy,
                memory_context=memory_context,
                user_input=content,
                temperature=temperature,
            ):
                if not chunk:
                    continue
                assistant_parts.append(chunk)
                for piece in iter_word_chunks(chunk):
                    yield sse_event({"type": "token", "token": piece})

            assistant_text = "".join(assistant_parts).strip()
            if not assistant_text:
                assistant_text = generate_gemini_reply(
                    persona_header=persona_header,
                    strategy=strategy,
                    memory_context=memory_context,
                    user_input=content,
                    temperature=temperature,
                ) or simulate_strategy_reply(strategy, content, [])

                if assistant_text:
                    for piece in iter_word_chunks(assistant_text):
                        yield sse_event({"type": "token", "token": piece})

            update_entity_memory(conversation, content)
            update_knowledge_graph(conversation, content)

            assistant_msg = Message(conversation_id=conversation.id, role="assistant", content=assistant_text)
            db.session.add(assistant_msg)
            db.session.flush()

            all_messages = Message.query.filter_by(conversation_id=conversation.id).order_by(Message.created_at.asc()).all()
            refresh_summary(conversation, all_messages)

            conversation.updated_at = assistant_msg.created_at
            db.session.commit()

            yield sse_event(
                {
                    "type": "done",
                    "user": message_to_dict(user_msg),
                    "assistant": message_to_dict(assistant_msg),
                    "conversation": conversation_to_dict(conversation),
                }
            )
        except Exception:
            db.session.rollback()
            yield sse_event({"type": "error", "error": "Streaming failed."})

    headers = {
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "X-Accel-Buffering": "no",
    }
    return Response(event_stream(), mimetype="text/event-stream", headers=headers)


@api_bp.get("/conversations/<int:conversation_id>/entity-memory")
def get_entity_memory(conversation_id: int) -> Response:
    Conversation.query.get_or_404(conversation_id)
    entities = EntityMemory.query.filter_by(conversation_id=conversation_id).order_by(EntityMemory.updated_at.desc()).all()
    return jsonify(
        [
            {
                "id": e.id,
                "entity_name": e.entity_name,
                "entity_type": e.entity_type,
                "facts": e.facts,
                "updated_at": e.updated_at.isoformat(),
            }
            for e in entities
        ]
    )


@api_bp.get("/conversations/<int:conversation_id>/knowledge-graph")
def get_knowledge_graph(conversation_id: int) -> Response:
    Conversation.query.get_or_404(conversation_id)
    edges = KnowledgeEdge.query.filter_by(conversation_id=conversation_id).order_by(KnowledgeEdge.updated_at.desc()).all()

    nodes = {}
    edge_payload = []
    for edge in edges:
        nodes[edge.source] = {"id": edge.source, "label": edge.source}
        nodes[edge.target] = {"id": edge.target, "label": edge.target}
        edge_payload.append(
            {
                "id": edge.id,
                "source": edge.source,
                "target": edge.target,
                "relation": edge.relation,
                "confidence": edge.confidence,
            }
        )

    return jsonify({"nodes": list(nodes.values()), "edges": edge_payload})


@api_bp.get("/conversations/<int:conversation_id>/export")
def export_conversation(conversation_id: int) -> Response:
    conversation = Conversation.query.get_or_404(conversation_id)
    messages = Message.query.filter_by(conversation_id=conversation.id).order_by(Message.created_at.asc()).all()

    markdown = [f"# {conversation.title}", ""]
    markdown.append(f"- Memory Strategy: {conversation.memory_strategy}")
    markdown.append(f"- Persona: {conversation.persona.name if conversation.persona else 'Default'}")
    markdown.append("")
    for message in messages:
        markdown.append(f"## {message.role.title()} ({message.created_at.isoformat()})")
        markdown.append(message.content)
        markdown.append("")
    markdown_text = "\n".join(markdown)

    fmt = request.args.get("format", "markdown")
    if fmt == "pdf":
        buffer = BytesIO()
        pdf = canvas.Canvas(buffer, pagesize=A4)
        _, height = A4
        y = height - 40
        for raw_line in markdown_text.splitlines():
            line = raw_line[:110]
            pdf.drawString(36, y, line)
            y -= 14
            if y < 40:
                pdf.showPage()
                y = height - 40
        pdf.save()
        buffer.seek(0)
        return Response(
            buffer.read(),
            mimetype="application/pdf",
            headers={
                "Content-Disposition": f"attachment; filename=conversation_{conversation.id}.pdf"
            },
        )

    return Response(
        markdown_text,
        mimetype="text/markdown",
        headers={"Content-Disposition": f"attachment; filename=conversation_{conversation.id}.md"},
    )


@api_bp.post("/compare-memory")
def compare_memory() -> Response:
    payload = request.get_json(force=True)
    script = payload.get("script", [])
    flow_name = payload.get("flow_name", "Comparison Flow")

    if not isinstance(script, list) or not all(isinstance(s, str) for s in script):
        return jsonify({"error": "script must be an array of user utterances"}), 400

    results: dict[str, list[dict]] = {}
    for strategy in MEMORY_STRATEGIES:
        history: list[str] = []
        run = []
        for item in script:
            reply = simulate_strategy_reply(strategy, item, history)
            history.append(item)
            history.append(reply)
            run.append({"user": item, "assistant": reply})
        results[strategy] = run

    comparison = ComparisonRun(
        flow_name=flow_name,
        script=json.dumps(script),
        results=json.dumps(results),
    )
    db.session.add(comparison)
    db.session.commit()

    return jsonify(
        {
            "comparison_id": comparison.id,
            "flow_name": flow_name,
            "results": results,
        }
    )
