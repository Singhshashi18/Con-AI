from __future__ import annotations

import re
from typing import Iterable

from .models import Conversation, ConversationSummary, EntityMemory, KnowledgeEdge, Message

MEMORY_STRATEGIES = ("buffer", "summary", "entity", "knowledge_graph", "summary_entity")
DOMAIN_FOCUS = ("general", "technical", "creative", "business")


def sanitize_title(raw: str) -> str:
    trimmed = (raw or "").strip()
    if not trimmed:
        return "Untitled Conversation"
    return trimmed[:120]


def infer_entity_type(entity_name: str) -> str:
    if re.search(r"(inc|corp|llc|ltd|technologies)$", entity_name, re.IGNORECASE):
        return "company"
    if len(entity_name.split()) >= 2:
        return "person"
    return "entity"


def extract_entities(text: str) -> list[str]:
    candidates = re.findall(r"\b[A-Z][a-z]+(?:\s[A-Z][a-z]+)?\b", text)
    ignore = {"I", "We", "The", "A", "An", "AI", "It", "Today", "Tomorrow"}
    entities = []
    for entity in candidates:
        if entity in ignore:
            continue
        if entity not in entities:
            entities.append(entity)
    return entities[:12]


def extract_edges(text: str) -> list[tuple[str, str, str, float]]:
    rules = [
        (r"([A-Z][a-z]+(?:\s[A-Z][a-z]+)?)\s+works at\s+([A-Z][\w\s]+)", "works_at", 0.9),
        (r"([A-Z][\w\s]+)\s+is a\s+([a-z][\w\s]+)", "is_a", 0.75),
        (r"([A-Z][a-z]+(?:\s[A-Z][a-z]+)?)\s+leads\s+([A-Z][\w\s]+)", "leads", 0.8),
        (r"([A-Z][\w\s]+)\s+partners with\s+([A-Z][\w\s]+)", "partners_with", 0.8),
    ]
    edges: list[tuple[str, str, str, float]] = []
    for pattern, relation, confidence in rules:
        for match in re.finditer(pattern, text):
            src = match.group(1).strip()
            tgt = match.group(2).strip().rstrip(".")
            edges.append((src, relation, tgt, confidence))
    return edges[:16]


def update_entity_memory(conversation: Conversation, message_text: str) -> None:
    entities = extract_entities(message_text)
    if not entities:
        return

    for name in entities:
        existing = EntityMemory.query.filter_by(
            conversation_id=conversation.id,
            entity_name=name,
        ).first()
        fact = f"Mentioned in conversation: {message_text[:140]}"
        if existing:
            if fact not in existing.facts:
                existing.facts = f"{existing.facts}\n- {fact}" if existing.facts else f"- {fact}"
        else:
            existing = EntityMemory(
                conversation_id=conversation.id,
                entity_name=name,
                entity_type=infer_entity_type(name),
                facts=f"- {fact}",
            )
            from . import db

            db.session.add(existing)


def update_knowledge_graph(conversation: Conversation, message_text: str) -> None:
    from . import db

    for source, relation, target, confidence in extract_edges(message_text):
        edge = KnowledgeEdge.query.filter_by(
            conversation_id=conversation.id,
            source=source,
            relation=relation,
            target=target,
        ).first()
        if edge:
            edge.confidence = max(edge.confidence, confidence)
        else:
            db.session.add(
                KnowledgeEdge(
                    conversation_id=conversation.id,
                    source=source,
                    relation=relation,
                    target=target,
                    confidence=confidence,
                )
            )


def refresh_summary(conversation: Conversation, messages: Iterable[Message]) -> None:
    from . import db

    sorted_messages = list(messages)[-8:]
    if not sorted_messages:
        return

    snippets = []
    for msg in sorted_messages:
        prefix = "User" if msg.role == "user" else "Assistant"
        snippets.append(f"{prefix}: {msg.content[:90]}")
    composed = " | ".join(snippets)
    summary_text = composed[:1200]

    existing = ConversationSummary.query.filter_by(conversation_id=conversation.id).first()
    if existing:
        existing.summary = summary_text
    else:
        db.session.add(ConversationSummary(conversation_id=conversation.id, summary=summary_text))




def build_cross_session_context(conversation: Conversation) -> str:
   
    other_sessions = (
        Conversation.query.filter(Conversation.id != conversation.id)
        .order_by(Conversation.updated_at.desc())
        .limit(4)
        .all()
    )
    if not other_sessions:
        return "Cross-Session Memory:\nNo prior sessions available."

    session_blocks: list[str] = []
    for session in other_sessions:
        title = sanitize_title(session.title or f"Conversation {session.id}")
        summary = ConversationSummary.query.filter_by(conversation_id=session.id).first()
        recap = (summary.summary[:240] if summary and summary.summary else "No summary yet.")

        turns = (
            Message.query.filter_by(conversation_id=session.id)
            .order_by(Message.created_at.desc())
            .limit(3)
            .all()
        )
        turns.reverse()
        turn_lines = [f"{m.role}: {m.content[:120]}" for m in turns]
        rendered_turns = "\n".join(turn_lines) if turn_lines else "No turns available."
        session_blocks.append(f"Session: {title}\nSummary: {recap}\nRecent Turns:\n{rendered_turns}")

    return "Cross-Session Memory:\n" + "\n\n".join(session_blocks)




def build_memory_context(conversation: Conversation, strategy: str) -> str:
    messages = Message.query.filter_by(conversation_id=conversation.id).order_by(Message.created_at.asc()).all()
    cross_session_block = build_cross_session_context(conversation)

    if strategy == "summary":
        summary = ConversationSummary.query.filter_by(conversation_id=conversation.id).first()
        recap = summary.summary if summary else "No summary yet."
        tail = messages[-4:]
        tail_text = "\n".join(f"{m.role}: {m.content}" for m in tail)
        return f"Summary Memory:\n{recap}\n\nRecent Turns:\n{tail_text}\n\n{cross_session_block}"

    if strategy == "entity":
        entities = EntityMemory.query.filter_by(conversation_id=conversation.id).all()
        entity_lines = [f"{e.entity_name} ({e.entity_type}): {e.facts[:180]}" for e in entities[:12]]
        recent = "\n".join(f"{m.role}: {m.content}" for m in messages[-6:])
        return (
            "Entity Memory:\n"
            + ("\n".join(entity_lines) if entity_lines else "No entities extracted yet.")
            + "\n\nRecent Turns:\n"
            + recent
            + "\n\n"
            + cross_session_block
        )

    if strategy == "knowledge_graph":
        edges = KnowledgeEdge.query.filter_by(conversation_id=conversation.id).all()
        edge_lines = [
            f"{e.source} -[{e.relation}/{e.confidence:.2f}]-> {e.target}"
            for e in edges[:20]
        ]
        recent = "\n".join(f"{m.role}: {m.content}" for m in messages[-6:])
        return (
            "Knowledge Graph Memory:\n"
            + ("\n".join(edge_lines) if edge_lines else "No relationships extracted yet.")
            + "\n\nRecent Turns:\n"
            + recent
            + "\n\n"
            + cross_session_block
        )

    if strategy == "summary_entity":
        summary = ConversationSummary.query.filter_by(conversation_id=conversation.id).first()
        recap = summary.summary if summary else "No summary yet."
        entities = EntityMemory.query.filter_by(conversation_id=conversation.id).all()
        entity_lines = [f"{e.entity_name} ({e.entity_type}): {e.facts[:120]}" for e in entities[:10]]
        recent = "\n".join(f"{m.role}: {m.content}" for m in messages[-5:])
        return (
            "Summary + Entity Hybrid Memory:\n"
            f"Summary: {recap}\n"
            + (
                "Entities:\n" + "\n".join(entity_lines)
                if entity_lines
                else "Entities:\nNo entities extracted yet."
            )
            + "\n\nRecent Turns:\n"
            + recent
            + "\n\n"
            + cross_session_block
        )

    # Buffer strategy keeps full ordered context with a practical cap.
    tail = messages[-20:]
    return (
        "Buffer Memory:\n"
        + "\n".join(f"{m.role}: {m.content}" for m in tail)
        + "\n\n"
        + cross_session_block
    )





def simulate_strategy_reply(strategy: str, user_input: str, history: list[str]) -> str:
    if strategy == "summary":
        gist = " | ".join(history[-4:])[:220]
        return f"[Summary] I am relying on condensed context: {gist}. You asked: {user_input}"
    if strategy == "entity":
        entities = extract_entities(" ".join(history + [user_input]))
        return f"[Entity] I tracked entities {', '.join(entities[:6]) or 'none yet'} while handling: {user_input}"
    if strategy == "knowledge_graph":
        edges = extract_edges(" ".join(history + [user_input]))
        edge_desc = "; ".join(f"{a}-{r}->{b}" for a, r, b, _ in edges[:4])
        return f"[Knowledge Graph] Relations considered: {edge_desc or 'none'}. Replying to: {user_input}"
    if strategy == "summary_entity":
        gist = " | ".join(history[-3:])[:180]
        entities = extract_entities(" ".join(history + [user_input]))
        return (
            "[Summary+Entity] Combined compact summary with tracked entities "
            f"({', '.join(entities[:5]) or 'none'}). Context gist: {gist}."
        )
    return f"[Buffer] I retained verbatim recent turns and answer your request: {user_input}"
