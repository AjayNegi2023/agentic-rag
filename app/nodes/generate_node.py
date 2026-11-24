from app.state.agent_state import AgentState
from app.llms.groq_llm import get_llm
from langchain_core.output_parsers import StrOutputParser
from langchain import hub


def generate(state: AgentState) -> dict:
    """
    Answer the question using the retrieved documents.
    """

    print("------Generate-------")

    messages = state["messages"]

    question = messages[0].content
    docs = messages[-1].content

    prompt = hub.pull("rlm/rag-prompt")

    rag_chain = prompt | get_llm() | StrOutputParser()

    response = rag_chain.invoke({
        "context": docs,
        "question": question
    })

    return {"messages": [response]}