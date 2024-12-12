from typing import TypedDict, Annotated, Optional
from langgraph.graph.message import add_messages

class OverallState(TypedDict):
    user_input: dict
    concepts_to_evaluate: list
    lesson_doc: str

    blooms_state: str
    next: Optional[str]
    reflection_steps: Optional[int]
    scratchpad: Optional[dict]
    first_submission: Optional[str]

    current_knowledge_state: dict
    messages: Annotated[list, add_messages]  # General messages