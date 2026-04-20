from __future__ import annotations

import re
from typing import Iterable

from .models import Conversation, ConversationSummary, EntityMemory, KnowledgeEdge, Message

MEMORY_STRATEGIES = ("buffer", "summary", "entity", "knowledge_graph", "summary_entity")
DOMAIN_FOCUS = ("general", "technical", "creative", "business")

_ENTITY_IGNORE = {
    "i", "we", "the", "a", "an", "ai", "it", "today", "tomorrow", "there", "here", "this", "that",
    "he", "she", "they", "you", "me", "my", "your", "our", "their", "and", "or", "but", "is", "are",
    "himself", "herself", "themselves",
}

_PRONOUNS = {"it", "he", "she", "they", "this", "that", "him", "her", "them", "himself", "herself", "themselves"}

_LEADING_NOISE = re.compile(r"^(?:and|or|but|so|then|also|at|in|from|to|with|for|of|on|the|a|an)\s+", re.IGNORECASE)
_TRAILING_NOISE = re.compile(r"\s+(?:and|or|but|so|then|also)$", re.IGNORECASE)

_ENTITY_BLOCKLIST = {
    # discourse / filler
    "additionally", "since", "because", "however", "therefore", "thus", "nice", "thanks", "thank", "okay", "ok",
    "got", "noted", "sounds", "sounds like", "information", "conversation", "story", "details", "current", "first",
    # generic nouns / sentence fragments
    "change", "changes", "job", "transition", "role", "group", "network", "position", "situation", "setup",
    # common verb / clause fragments that should never be entities
    "work", "works", "worked", "is", "are", "was", "were", "been", "be", "live", "lives", "located",
    "based", "lead", "leads", "partner", "partners", "joined", "moved", "and it is", "it is", "it was",
}


def sanitize_title(raw: str) -> str:
    trimmed = (raw or "").strip()
    if not trimmed:
        return "Untitled Conversation"
    return trimmed[:120]


def infer_entity_type(entity_name: str) -> str:
    lowered = (entity_name or "").strip().lower()
    if not lowered:
        return "entity"

    if re.search(r"\b(developer|engineer|manager|analyst|designer|architect|lead|consultant|specialist|intern)\b", lowered):
        return "role"

    if re.search(r"\b(delhi|ghaziabad|noida|gurgaon|bengaluru|bangalore|mumbai|pune|hyderabad|india|city|state|country)\b", lowered):
        return "location"

    if re.search(r"\b(inc|corp|llc|ltd|technologies?|technology|solutions?|systems?|labs?|team|department|hcl|infosys|google|microsoft)\b", lowered):
        return "organization"

    if re.search(r"\b(noida|ghaziabad|delhi|gurgaon|bengaluru|bangalore|mumbai|pune|hyderabad|sector\s*\d+)\b", lowered):
        return "location"

    if re.search(r"(inc|corp|llc|ltd|technologies)$", entity_name, re.IGNORECASE):
        return "organization"

    if len(entity_name.split()) >= 2:
        return "person"
    return "entity"


def _normalize_entity_name(raw: str) -> str:
    cleaned = (raw or "").strip().strip("\"'`.,;:!?()[]{}")
    cleaned = re.sub(r"\bteh\b", "the", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s+", " ", cleaned)
    cleaned = _LEADING_NOISE.sub("", cleaned)
    cleaned = _TRAILING_NOISE.sub("", cleaned)
    cleaned = " ".join(re.findall(r"[A-Za-z0-9&.+#-]+", cleaned))
    if not cleaned:
        return ""

    words = cleaned.split()
    words = [word for word in words if word.lower() not in {"himself", "herself", "themselves"}]
    if not words:
        return ""

    candidate_lower = " ".join(words).lower()
    if candidate_lower in _ENTITY_BLOCKLIST:
        return ""

    if any(token.lower() in _ENTITY_BLOCKLIST for token in words):
        return ""

    # Reject long fragments that look like sentence clauses rather than names.
    if len(words) >= 2 and any(word.lower() in {"and", "or", "but", "since", "because", "that", "which", "who", "whom"} for word in words):
        return ""

    if len(words) == 1 and words[0].lower() in _ENTITY_IGNORE.union(_PRONOUNS):
        return ""

    if words[0].lower() in _PRONOUNS:
        return ""

    if any(word.lower() in {"works", "leads", "partners", "located", "based", "lives", "resides"} for word in words):
        return ""

    if len(words) > 5:
        words = words[:5]

    # Strip leading filler again after truncation.
    while words and words[0].lower() in {"and", "or", "but", "so", "then", "also", "at", "in", "from", "to", "with", "for", "of", "on"}:
        words = words[1:]
    if not words:
        return ""

    if len(words) == 1 and words[0].lower() in _ENTITY_BLOCKLIST:
        return ""

    normalized_words: list[str] = []
    for word in words:
        if word.isupper() and len(word) <= 5:
            normalized_words.append(word)
        else:
            normalized_words.append(word.capitalize())
    return " ".join(normalized_words)


def _split_clauses(text: str) -> list[str]:
    # Split on punctuation and on conjunctions that usually introduce a new fact.
    base_parts = re.split(r"[\n\r]+|[.;!?]+", text)
    clauses: list[str] = []
    for part in base_parts:
        for clause in re.split(r"\s+\band\b\s+(?=(?:it|he|she|they|[A-Za-z])\b)", part, flags=re.IGNORECASE):
            candidate = clause.strip().strip(",")
            if candidate:
                clauses.append(candidate)
    return clauses


def extract_entities(text: str) -> list[str]:
    candidates: list[str] = []

    # Use extracted graph edges as high-confidence entity signals.
    for source, _relation, target, _confidence in extract_edges(text):
        candidates.extend([source, target])

    # Title-cased person/place/org names.
    candidates.extend(re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2}\b", text))

    # Acronyms/org names like HCL, IBM, TCS.
    candidates.extend(re.findall(r"\b[A-Z]{2,6}\b", text))

    # Common intro phrases for entities.
    candidates.extend(
        re.findall(
            r"\b(?:my\s+friend|friend|name\s+is|called)\s+([A-Za-z][A-Za-z0-9&.+#-]*(?:\s+[A-Za-z0-9&.+#-]+){0,2})",
            text,
            flags=re.IGNORECASE,
        )
    )

    # Capture name after explicit self-introduction forms.
    candidates.extend(
        re.findall(
            r"\b(?:i\s+am|i'm|my\s+name\s+is|i\s+am\s+called|this\s+is)\s+([A-Za-z][A-Za-z0-9&.+#-]*(?:\s+[A-Za-z0-9&.+#-]+){0,2})",
            text,
            flags=re.IGNORECASE,
        )
    )

    entities: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        entity = _normalize_entity_name(candidate)
        if not entity:
            continue
        key = entity.lower()
        if key in seen:
            continue
        if key in _ENTITY_BLOCKLIST:
            continue
        seen.add(key)
        entities.append(entity)
    return entities[:16]


def extract_edges(text: str) -> list[tuple[str, str, str, float]]:
    subject = r"([A-Za-z][A-Za-z0-9&.+#-]*(?:\s+[A-Za-z0-9&.+#-]+){0,2})"
    entity = r"([A-Za-z][A-Za-z0-9&.+#-]*(?:\s+[A-Za-z0-9&.+#-]+){0,4})"
    role = r"([A-Za-z][A-Za-z0-9&.+#-]*(?:\s+[A-Za-z0-9&.+#-]+){0,3})"

    # Keep ordered rules so pronoun resolution can use earlier relations.
    rules = [
        (rf"^{subject}\s+is\s+(?:a|an)\s+{role}\s+at\s+{entity}$", "is_a_at", 0.82),
        (rf"^{subject}\s+(?:works?|worked|wrked)\s+as\s+(?:a|an)\s+{role}\s+at\s+{entity}$", "worked_as_at", 0.84),
        (rf"^{subject}\s+works\s+at\s+{entity}$", "works_at", 0.9),
        (rf"^{subject}\s+work\s+at\s+{entity}$", "works_at", 0.9),
        (rf"^{subject}\s+worked\s+at\s+{entity}$", "works_at", 0.88),
        (rf"^{subject}\s+wrked\s+at\s+{entity}$", "works_at", 0.86),
        (rf"^{subject}\s+is\s+(?:a|an)\s+{entity}$", "is_a", 0.75),
        (rf"^{subject}\s+leads\s+{entity}$", "leads", 0.8),
        (rf"^{subject}\s+partners\s+with\s+{entity}$", "partners_with", 0.8),
        (rf"^{subject}\s+(?:is\s+)?located\s+in\s+{entity}$", "located_in", 0.78),
        (rf"^{subject}\s+which\s+is\s+located\s+in\s+{entity}$", "located_in", 0.78),
        (rf"^{subject}\s+which\s+is\s+based\s+in\s+{entity}$", "based_in", 0.76),
        (rf"^{subject}\s+(?:is\s+)?based\s+in\s+{entity}$", "based_in", 0.76),
        (rf"^{subject}\s+live\s+in\s+{entity}$", "lives_in", 0.75),
        (rf"^{subject}\s+lives\s+in\s+{entity}$", "lives_in", 0.75),
        (rf"^{subject}\s+resides\s+in\s+{entity}$", "lives_in", 0.75),
        (rf"^{subject}\s+moved\s+to\s+{entity}$", "moved_to", 0.72),
        (rf"^{subject}\s+joined\s+{entity}$", "joined", 0.72),
    ]
    edges: list[tuple[str, str, str, float]] = []
    seen: set[tuple[str, str, str]] = set()
    context_subject = ""
    context_org = ""

    def add_edge(src_raw: str, relation: str, tgt_raw: str, confidence: float) -> None:
        nonlocal context_subject, context_org

        src = _normalize_entity_name(src_raw)
        tgt = _normalize_entity_name(tgt_raw)
        if not tgt:
            return

        if src_raw.strip().lower() in _PRONOUNS and context_org:
            src = context_org
        elif src_raw.strip().lower() in _PRONOUNS and context_subject:
            src = context_subject

        if not src:
            return

        key = (src.lower(), relation, tgt.lower())
        if key in seen:
            return
        seen.add(key)
        edges.append((src, relation, tgt, confidence))

        context_subject = src
        if relation == "works_at":
            context_org = tgt

    for clause in _split_clauses(text):
        normalized_clause = clause.strip().strip(" ,")
        if not normalized_clause:
            continue

        # Handle chained relation in one clause: "X ... at Org which is located in Place"
        chained = re.search(
            rf"^{subject}\s+(?:works?|worked|wrked)\s+as\s+(?:a|an)\s+{role}\s+at\s+{entity}\s+which\s+is\s+located\s+in\s+{entity}$",
            normalized_clause,
            flags=re.IGNORECASE,
        )
        if chained:
            person = chained.group(1).strip()
            role_name = chained.group(2).strip()
            org = chained.group(3).strip()
            place = chained.group(4).strip()
            add_edge(person, "is_a", role_name, 0.75)
            add_edge(person, "works_at", org, 0.9)
            add_edge(org, "located_in", place, 0.78)
            continue

        for pattern, relation, confidence in rules:
            match = re.search(pattern, normalized_clause, flags=re.IGNORECASE)
            if not match:
                continue
            if relation == "is_a_at":
                add_edge(match.group(1).strip(), "is_a", match.group(2).strip().rstrip("."), 0.75)
                add_edge(match.group(1).strip(), "works_at", match.group(3).strip().rstrip("."), 0.9)
            elif relation == "worked_as_at":
                add_edge(match.group(1).strip(), "is_a", match.group(2).strip().rstrip("."), 0.75)
                add_edge(match.group(1).strip(), "works_at", match.group(3).strip().rstrip("."), 0.9)
            else:
                add_edge(match.group(1).strip(), relation, match.group(2).strip().rstrip("."), confidence)
            break

    return edges[:16]


def update_entity_memory(conversation: Conversation, message_text: str) -> None:
    entities = extract_entities(message_text)
    if not entities:
        return

    existing_rows = EntityMemory.query.filter_by(conversation_id=conversation.id).all()
    existing_by_key = {row.entity_name.lower(): row for row in existing_rows}

    for name in entities:
        existing = existing_by_key.get(name.lower())
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
            existing_by_key[name.lower()] = existing


def update_knowledge_graph(conversation: Conversation, message_text: str) -> None:
    from . import db

    existing_rows = KnowledgeEdge.query.filter_by(conversation_id=conversation.id).all()
    existing_by_key = {
        (row.source.lower(), row.relation, row.target.lower()): row
        for row in existing_rows
    }

    for source, relation, target, confidence in extract_edges(message_text):
        edge = existing_by_key.get((source.lower(), relation, target.lower()))
        if edge:
            edge.confidence = max(edge.confidence, confidence)
        else:
            edge = KnowledgeEdge(
                conversation_id=conversation.id,
                source=source,
                relation=relation,
                target=target,
                confidence=confidence,
            )
            db.session.add(edge)
            existing_by_key[(source.lower(), relation, target.lower())] = edge


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
