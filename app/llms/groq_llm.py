from langchain_groq  import ChatGroq

from app.config.settings import GROQ_MODEL

def get_llm() -> ChatGroq:
    return ChatGroq(model = GROQ_MODEL)


