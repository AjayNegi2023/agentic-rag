import os
from dotenv import load_dotenv

load_dotenv()

OPEN_API_KEY = os.getenv("OPEN_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

GROQ_MODEL = "qwen3.8-27b"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 100

LANGGRAPH_URLS = [
    "https://docs.langchain.com/oss/python/langgraph/overview",
    "https://docs.langchain.com/oss/python/langgraph/streaming",
    "https://docs.langchain.com/oss/python/langgraph/interrupts"
]

LANGCHAIN_URLS = [
    "https://docs.langchain.com/oss/python/langchain/overview",
    "https://docs.langchain.com/oss/python/langchain/tools",
    "https://docs.langchain.com/oss/python/langchain/streaming",
    "https://docs.langchain.com/oss/python/langchain/mcp/index"
]

