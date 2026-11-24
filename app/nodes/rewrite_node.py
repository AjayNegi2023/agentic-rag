
from app.state.agent_state import AgentState
from app.llms.groq_llm import get_llm

from langchain_core.messages import HumanMessage


def rewrite(state: AgentState) -> dict:
    """
    Rewrite the user's query to improve retrieval relevance.
    """

    print("-------- Rewriting USER Query --------")

    question = state["messages"][0].content

    message = [
        HumanMessage(
            content=f"""
            Analyze the user's query and understand its underlying
            semantic meaning.

            User Query:
            ----------------
            {question}
            ----------------

            Rewrite the query to improve document retrieval.

            Return only the improved question.
            """
        )
    ]

    response = get_llm().invoke(message)

    return {"messages": [response]}
