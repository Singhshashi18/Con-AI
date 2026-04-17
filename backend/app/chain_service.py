from __future__ import annotations

from .chain_builder import chain_builder
from .memory import refresh_summary, update_entity_memory, update_knowledge_graph
from .models import Conversation, Message


def run_conversation_chain(conversation: Conversation, user_message: Message) -> tuple[str, dict[str, object]]:
    artifacts = chain_builder.generate(conversation, user_message.content)

    update_entity_memory(conversation, user_message.content)
    update_knowledge_graph(conversation, user_message.content)

    context_messages = Message.query.filter_by(conversation_id=conversation.id).order_by(Message.created_at.asc()).all()
    refresh_summary(conversation, context_messages)

    metadata = {
        "intent": artifacts.intent,
        "draft": artifacts.draft,
        "context_block": artifacts.context_block,
        "entities": artifacts.entities,
        "edges": [
            {"source": source, "relation": relation, "target": target, "confidence": confidence}
            for source, relation, target, confidence in artifacts.edges
        ],
    }
    return artifacts.refined, metadata
