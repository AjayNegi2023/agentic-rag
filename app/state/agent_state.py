from typing_extensions import TypedDict
from typing import Annotated, Sequence

from langchain_core.messages import BaseMessage
from langgraph.graph.messages import add_messages

class AgentState(TypedDict):
    # The add_messgages function defines how an update should be proceeded.
    # Default is to replace. add_messages says "append"
    messages : Annotated[Sequence[BaseMessage] , add_messages]

