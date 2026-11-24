from app.state.agent_state import AgentState
from app.llms.groq_llm import get_llm
from app.tools.retriever_tools import get_tools


def agent(state: AgentState ) ->dict:
    """
    This node function decides whether to call a retriever tool or answer directly.
    """ 
    print("-----------Call Agent-------------")
    model = get_llm().bind_tools(get_tools())
    response = model.invoke(state["messages"])
    return {"messages" : [response]}

