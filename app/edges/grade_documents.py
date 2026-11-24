
from app.state.agent_state import AgentState
from app.llms.groq_llm import get_llm

from typing import Literal
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field


class Grade(BaseModel):
    """
    Binary score for the relevance of retrieved document chunks.
    """

    binary_score: str = Field(
        description="Relevance score: 'yes' or 'no'"
    )


GRADE_PROMPT = PromptTemplate(
    template="""
    You are a grader assessing the relevance of a retrieved document
    to a user question.

    Here is the retrieved document:
    ----------------
    {context}
    ----------------

    Here is the user question:
    {question}

    If the document contains keywords or semantic meaning related
    to the user question, grade it as relevant.

    Give a binary score: 'yes' or 'no'.
    Return only the structured score.
    """,
    input_variables=["context", "question"]
)


def grade_documents(state: AgentState) -> Literal["generate", "rewrite"]:
    """
    Route to 'generate' if retrieved documents are relevant,
    otherwise route to 'rewrite'.
    """

    print("------- Checking Document Relevance -------")

    chain = GRADE_PROMPT | get_llm().with_structured_output(Grade)

    messages = state["messages"]

    result = chain.invoke({
        "question": messages[0].content,
        "context": messages[-1].content
    })

    if result.binary_score.lower() == "yes":
        print("------- Decision: Documents Relevant -------")
        return "generate"

    print("------- Decision: Documents Not Relevant -------")
    return "rewrite"
