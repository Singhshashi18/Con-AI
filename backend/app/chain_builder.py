from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

try:
    from langchain_core.runnables import RunnableBranch, RunnableLambda, RunnableParallel, RunnablePassthrough
except Exception: 
    class _RunnableBase:
        def __or__(self, other: Any):
            return _RunnablePipeline(self, other)

        def invoke(self, value: Any) -> Any:
            raise NotImplementedError


    class _RunnableLambda(_RunnableBase):
        def __init__(self, func: Callable[[Any], Any]):
            self.func = func

        def invoke(self, value: Any) -> Any:
            return self.func(value)


    class _RunnableParallel(_RunnableBase):
        def __init__(self, **runnables: Any):
            self.runnables = runnables

        def invoke(self, value: Any) -> dict[str, Any]:
            result: dict[str, Any] = {}
            for key, runnable in self.runnables.items():
                result[key] = runnable.invoke(value) if hasattr(runnable, "invoke") else runnable(value)
            return result


    class _RunnableBranch(_RunnableBase):
        def __init__(self, *branches: Any):
            self.branches = branches

        def invoke(self, value: Any) -> Any:
            *pairs, default = self.branches
            for predicate, runnable in pairs:
                if predicate(value):
                    return runnable.invoke(value)
            return default.invoke(value)


    class _RunnableAssign(_RunnableBase):
        def __init__(self, base: Any, assignments: dict[str, Any]):
            self.base = base
            self.assignments = assignments

        def invoke(self, value: Any) -> dict[str, Any]:
            result = self.base.invoke(value) if hasattr(self.base, "invoke") else self.base(value)
            if not isinstance(result, dict):
                result = {"value": result}
            enriched = dict(result)
            for key, runnable in self.assignments.items():
                enriched[key] = runnable.invoke(enriched) if hasattr(runnable, "invoke") else runnable(enriched)
            return enriched


    class _RunnablePassthrough(_RunnableBase):
        @staticmethod
        def assign(**assignments: Any) -> _RunnableAssign:
            return _RunnableAssign(_RunnableLambda(lambda value: value), assignments)

        def invoke(self, value: Any) -> Any:
            return value


    class _RunnablePipeline(_RunnableBase):
        def __init__(self, left: Any, right: Any):
            self.left = left
            self.right = right

        def invoke(self, value: Any) -> Any:
            left_value = self.left.invoke(value) if hasattr(self.left, "invoke") else self.left(value)
            return self.right.invoke(left_value) if hasattr(self.right, "invoke") else self.right(left_value)


    RunnableLambda = _RunnableLambda
    RunnableParallel = _RunnableParallel
    RunnableBranch = _RunnableBranch
    RunnablePassthrough = _RunnablePassthrough

from .llm import generate_gemini_reply
from .memory import build_memory_context, extract_edges, extract_entities
from .models import Conversation


@dataclass
class ChainArtifacts:
    intent: str
    context_block: str
    prompt: str
    draft: str
    refined: str
    entities: list[str]
    edges: list[tuple[str, str, str, float]]


class ConversationChainBuilder:
    """LCEL pipeline for conversational generation.

    Pipeline shape:
    1. Parallel snapshot: intent classification, context build, entity extraction, KG extraction.
    2. Branching prompt selection: recall vs analysis vs creative vs general.
    3. Sequential generation: prompt -> draft -> refinement.
    """

    def __init__(self) -> None:
        self.snapshot_chain = RunnableLambda(self._snapshot_state)
        self.prompt_chain = RunnableBranch(
            (lambda state: state["intent"] == "recall", RunnableLambda(self._recall_prompt)),
            (lambda state: state["intent"] == "analysis", RunnableLambda(self._analysis_prompt)),
            (lambda state: state["intent"] == "creative", RunnableLambda(self._creative_prompt)),
            RunnableLambda(self._general_prompt),
        )
        self.chain = (
            self.snapshot_chain
            | RunnablePassthrough.assign(prompt=self.prompt_chain)
            | RunnableLambda(self._generate_artifacts)
        )

    def classify_intent(self, user_input: str) -> str:
        lowered = user_input.lower()
        if any(word in lowered for word in ["remember", "recall", "who is", "what did i say", "what do you know"]):
            return "recall"
        if any(word in lowered for word in ["analyze", "compare", "why", "summarize", "strategy"]):
            return "analysis"
        if any(word in lowered for word in ["story", "write", "character", "plot", "creative"]):
            return "creative"
        return "general"

    def _snapshot_state(self, payload: dict[str, Any]) -> dict[str, Any]:
        conversation = payload["conversation"]
        user_input = payload["user_input"]
        context_block = build_memory_context(conversation, conversation.memory_strategy)
        combined = f"{context_block}\n{user_input}"
        entities = extract_entities(combined)
        edges = extract_edges(combined)
        return {
            "conversation": conversation,
            "user_input": user_input,
            "intent": self.classify_intent(user_input),
            "context_block": context_block,
            "entities": entities,
            "edges": edges,
        }

    def _recall_prompt(self, state: dict[str, Any]) -> str:
        return self._build_prompt(state, "Prioritize exact stored facts, entity details, and prior commitments.")

    def _analysis_prompt(self, state: dict[str, Any]) -> str:
        return self._build_prompt(state, "Prioritize summary context, tradeoffs, and structured reasoning.")

    def _creative_prompt(self, state: dict[str, Any]) -> str:
        return self._build_prompt(state, "Prioritize imagination, relationships, and narrative continuity.")

    def _general_prompt(self, state: dict[str, Any]) -> str:
        return self._build_prompt(state, "Keep the response natural, direct, and conversational.")

    def _build_prompt(self, state: dict[str, Any], branch_hint: str) -> str:
        conversation = state["conversation"]
        persona = conversation.persona
        persona_name = persona.name if persona else "Default Assistant"
        domain = persona.domain_focus if persona else "general"
        system_prompt = persona.system_prompt if persona else "You are a helpful assistant."
        return (
            f"Persona: {persona_name}\n"
            f"Domain: {domain}\n"
            f"System: {system_prompt}\n"
            f"Intent: {state['intent']}\n"
            f"Branch Hint: {branch_hint}\n\n"
            f"Context Snapshot:\n{state['context_block']}\n\n"
            f"User: {state['user_input']}"
        )

    def _generate_artifacts(self, state: dict[str, Any]) -> ChainArtifacts:
        conversation = state["conversation"]
        prompt = state["prompt"]
        user_input = state["user_input"]
        entities = state["entities"]
        edges = state["edges"]
        intent = state["intent"]
        context_block = state["context_block"]

        draft = generate_gemini_reply(
            persona_header=prompt,
            strategy=conversation.memory_strategy,
            memory_context=context_block,
            user_input=user_input,
            temperature=conversation.persona.temperature if conversation.persona else 0.4,
        ) or self.fallback_reply(conversation, user_input, intent, entities, edges)

        refined = self.refine_response(draft, intent, entities, edges, conversation, user_input)
        return ChainArtifacts(
            intent=intent,
            context_block=context_block,
            prompt=prompt,
            draft=draft,
            refined=refined,
            entities=entities,
            edges=edges,
        )

    def fallback_reply(
        self,
        conversation: Conversation,
        user_input: str,
        intent: str,
        entities: list[str],
        edges: list[tuple[str, str, str, float]],
    ) -> str:
        persona = conversation.persona
        persona_name = persona.name if persona else "Assistant"
        entity_hint = ", ".join(entities[:5]) or "no new entities"
        edge_hint = "; ".join(f"{source}-{relation}->{target}" for source, relation, target, _ in edges[:3]) or "no new relationships"
        if intent == "recall":
            return f"{persona_name} remembers {entity_hint}. Relevant links: {edge_hint}. You asked: {user_input}"
        if intent == "analysis":
            return f"{persona_name} is analyzing the context with summary/entity focus. Key entities: {entity_hint}."
        if intent == "creative":
            return f"{persona_name} is continuing the narrative with {entity_hint} and relationships {edge_hint}."
        return f"{persona_name} replies: {user_input}"

    def refine_response(
        self,
        draft: str,
        intent: str,
        entities: list[str],
        edges: list[tuple[str, str, str, float]],
        conversation: Conversation,
        user_input: str,
    ) -> str:
        needs_entity_injection = intent in {"recall", "analysis"} and entities and not any(
            entity.lower() in draft.lower() for entity in entities[:3]
        )
        if needs_entity_injection:
            entity_clause = ", ".join(entities[:4])
            draft = f"{draft}\n\nEntity context: {entity_clause}"

        if intent == "recall" and edges and "relationship" not in draft.lower():
            relationship_clause = "; ".join(f"{source} {relation} {target}" for source, relation, target, _ in edges[:3])
            draft = f"{draft}\n\nRelationship context: {relationship_clause}"

        if len(draft) < 40:
            draft = self.fallback_reply(conversation, user_input, intent, entities, edges)

        return draft

    def generate(self, conversation: Conversation, user_input: str) -> ChainArtifacts:
        return self.chain.invoke({"conversation": conversation, "user_input": user_input})


chain_builder = ConversationChainBuilder()
