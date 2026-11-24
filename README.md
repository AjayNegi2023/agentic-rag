# Agentic RAG --- LangGraph-Powered Retrieval-Augmented AI Agent

An agentic Retrieval-Augmented Generation (RAG) application built with
**LangGraph**, **LangChain**, **Groq**, **OpenAI embeddings**, and
**FAISS**.

The project demonstrates how an AI agent can decide when to retrieve
external knowledge, evaluate retrieved documents, rewrite a query when
the results are not relevant, and generate a grounded response. It can
also answer questions directly when retrieval is not needed.

> **Project focus:** Agent orchestration, tool calling, retrieval,
> relevance grading, query rewriting, and grounded answer generation.

------------------------------------------------------------------------

## Table of Contents

-   [Overview](#overview)
-   [Key Capabilities](#key-capabilities)
-   [Architecture](#architecture)
-   [Workflow](#workflow)
-   [Technology Stack](#technology-stack)
-   [Project Structure](#project-structure)
-   [How the RAG Pipeline Works](#how-the-rag-pipeline-works)
-   [Getting Started](#getting-started)
-   [Configuration](#configuration)
-   [Run the Application](#run-the-application)
-   [Visualize the Graph](#visualize-the-graph)
-   [Example Behaviour](#example-behaviour)
-   [Engineering Notes](#engineering-notes)
-   [Potential Enhancements](#potential-enhancements)
-   [Author](#author)

------------------------------------------------------------------------

## Overview

Traditional RAG applications commonly follow a fixed sequence:

1.  Receive a question.
2.  Retrieve relevant documents.
3.  Pass the documents and question to an LLM.
4.  Generate an answer.

This project takes a more flexible, **agentic approach**. A LangGraph
workflow gives the agent control over whether to call a retrieval tool
or respond directly. When documents are retrieved, a grading step checks
their relevance. If the retrieved information is not relevant enough,
the question is rewritten and sent through the agent again.

This design demonstrates conditional execution and iterative reasoning
rather than relying on a single fixed retrieval chain.

## Key Capabilities

-   **Agent-based routing:** The LLM decides whether a retrieval tool is
    required or whether it can answer directly.
-   **Multiple knowledge sources:** Separate retriever tools are
    configured for LangChain and LangGraph documentation/blog content.
-   **Document ingestion:** Web pages are loaded using `WebBaseLoader`.
-   **Text chunking:** `RecursiveCharacterTextSplitter` divides source
    documents into smaller chunks.
-   **Vector search:** FAISS stores document embeddings and provides
    similarity-based retrieval.
-   **Semantic embeddings:** OpenAI embeddings are used to represent
    document chunks in vector space.
-   **Relevance grading:** A structured LLM output evaluates whether
    retrieved content is relevant to the question.
-   **Query rewriting:** If the retrieved documents are not relevant,
    the workflow rewrites the question and retries through the agent.
-   **Grounded generation:** The generation step uses retrieved context
    to produce an answer.
-   **Direct responses:** Questions that do not require the indexed
    knowledge can be answered without invoking retrieval.

## Architecture

The application is organized around a LangGraph state graph. The agent
uses retriever tools, while dedicated workflow nodes handle rewriting
and answer generation.

### Workflow Diagram

The following diagram is generated from the compiled LangGraph workflow:

![Agentic RAG LangGraph Workflow](docs/agent-workflow.png)

### Workflow Components

| Component | Responsibility |
|---|---|
| `START` | Entry point of the graph. |
| `agent` | Invokes the LLM with the available retrieval tools and decides whether to call a tool or answer directly. |
| `retrieve` | Executes the selected retriever tool through LangGraph's `ToolNode`. |
| `grade_documents` | Evaluates the relevance of retrieved content and selects the next route. This is a conditional routing function in the current graph, not a separately registered graph node. |
| `rewrite` | Rephrases the user's question when retrieved content is not relevant, then routes back to the agent. |
| `generate` | Produces an answer using the question and retrieved context. |
| `END` | Terminates the workflow after a direct answer or generated response. |

## Workflow

The graph follows two main paths.

### 1. Retrieval path

1.  The question enters the `agent` node.
2.  The agent decides to use a retrieval tool.
3.  The `retrieve` node executes the selected tool.
4.  `grade_documents` checks the retrieved content.
5.  If the content is relevant, the graph routes to `generate`.
6.  If the content is not relevant, the graph routes to `rewrite`.
7.  The rewritten question returns to `agent` for another attempt.

### 2. Direct-answer path

If the agent determines that retrieval is unnecessary, it responds
directly and the graph reaches `END`.

## Technology Stack

| Technology | Responsibility |
|---|---|
| Python | Application implementation. |
| LangGraph | Stateful agent workflow and conditional routing. |
| LangChain | LLM integration, prompts, document loading, and retrieval tools. |
| Groq | LLM inference through the configured chat model. |
| Qwen model | Chat model configured through Groq. |
| OpenAI Embeddings | Converts document chunks into vector representations. |
| FAISS | Vector storage and similarity search. |
| WebBaseLoader | Loads source content from web pages. |
| RecursiveCharacterTextSplitter | Splits documents into manageable chunks. |
| Pydantic | Structured output for document relevance grading. |
| python-dotenv | Loads environment configuration from `.env`. |

## Project Structure

The project is organized by responsibility. The exact filenames may vary
as the implementation evolves.

``` text
agentic-rag/
│
├── app/
│   ├── config/
│   │   └── settings.py          # Environment and application settings
│   │
│   ├── graphs/
│   │   └── agent_graph.py       # LangGraph workflow definition
│   │
│   ├── llms/
│   │   └── groq_llm.py          # Groq chat model initialization
│   │
│   ├── nodes/
│   │   ├── agent_node.py        # Agent and tool-calling logic
│   │   ├── rewrite_node.py      # Query rewriting
│   │   ├── generate_node.py     # Context-grounded answer generation
│   │   └── grade_node.py        # Retrieved-document relevance grading
│   │
│   ├── rag/
│   │   └── builder.py           # Web ingestion, chunking and FAISS retriever
│   │
│   ├── state/
│   │   └── agent_state.py       # Shared graph state definition
│   │
│   ├── tools/
│   │   └── retriever_tools.py   # Retriever tool definitions
│   │
│   ├── main.py                  # Application entry point
│   └── visualize_graph.py       # Exports the workflow diagram
│
├── docs/
│   └── agent-workflow.png       # LangGraph workflow image
│
├── .env                         # Local secrets and configuration (not committed)
├── .gitignore
├── requirements.txt
└── README.md
```

> Keep the structure in this section aligned with the actual files in
> your repository. If you have named a node or module differently,
> update the corresponding entry.

## How the RAG Pipeline Works

### 1. Document ingestion

Source URLs are loaded using `WebBaseLoader`. The loaded documents are
split into smaller chunks using `RecursiveCharacterTextSplitter`.

### 2. Embedding and indexing

Each chunk is converted into an embedding using `OpenAIEmbeddings`. The
embeddings and their associated document chunks are indexed in a FAISS
vector store.

### 3. Retriever tools

The application exposes separate retrieval tools for the configured
LangGraph and LangChain sources. The agent can select the tool that best
matches the user's question.

### 4. Relevance grading

After retrieval, an LLM evaluates whether the returned context is
relevant. The grading response uses a structured output schema with a
binary relevance decision.

### 5. Query rewriting

When the retrieved context is not relevant, the rewrite step improves
the original question. The updated query is sent back to the agent so it
can try retrieval again.

### 6. Answer generation

When relevant context is available, the generation step combines the
question and retrieved context with a RAG prompt. The LLM produces the
final response.

## Getting Started

### Prerequisites

-   Python 3.10 or newer (use a Python version compatible with the
    packages in your `requirements.txt`).
-   A Groq API key.
-   An OpenAI API key for embeddings.
-   Git (optional, for cloning the repository).

### 1. Clone the repository

``` bash
git clone <YOUR_REPOSITORY_URL>
cd agentic-rag
```

### 2. Create and activate a virtual environment

**Windows PowerShell**

``` powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**macOS / Linux**

``` bash
python -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

``` bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Ensure your dependencies include the packages used by the application,
such as `langgraph`, `langchain`, `langchain-community`,
`langchain-groq`, `langchain-openai`, `langchain-text-splitters`,
`faiss-cpu`, `beautifulsoup4`, `python-dotenv`, and `pydantic`.

## Configuration

Create a `.env` file in the project root. Add your own credentials and
values:

``` env
GROQ_API_KEY=your_groq_api_key
GROQ_MODEL=your_configured_groq_model
OPENAI_API_KEY=your_openai_api_key

USER_AGENT=agentic-rag/1.0
```

Configure the source URL lists and text-splitting settings in
`app/config/settings.py` according to the configuration format used by
your application. Typical settings include:

-   `LANGGRAPH_URLS` --- source URLs for LangGraph content.
-   `LANGCHAIN_URLS` --- source URLs for LangChain content.
-   `CHUNK_SIZE` --- maximum chunk size.
-   `CHUNK_OVERLAP` --- overlap between adjacent chunks.

**Security:** Never commit `.env` or expose API keys in source code,
screenshots, logs, or the README. Add `.env` to `.gitignore`. You may
commit a sanitized `.env.example` containing placeholders, but never
real credentials.

## Run the Application

From the repository root, with the virtual environment activated:

``` bash
python -m app.main
```

The application loads the configured web documents, builds the FAISS
vector stores, and runs the sample questions through the graph.

## Visualize the Graph

The workflow image can be regenerated from the compiled graph. In
`app/visualize_graph.py`:

``` python
from app.graphs.agent_graph import graph

png_data = graph.get_graph().draw_mermaid_png()

with open("docs/agent-workflow.png", "wb") as file:
    file.write(png_data)

print("Workflow diagram saved to docs/agent-workflow.png")
```

Run it from the repository root:

``` bash
python -m app.visualize_graph
```

The PNG is stored under `docs/` and is displayed above in this README.

## Example Behaviour

The current implementation has been tested with questions covering both
retrieval and direct-answer behaviour.

  -------------------------------------------------------------------------------------
  Example question                                  Expected workflow behaviour
  ------------------------------------------------- -----------------------------------
  `What is LangChain?`                              Agent selects a relevant retriever,
                                                    grades the retrieved content, and
                                                    generates an answer.

  `What is streaming in LangGraph?`                 Agent retrieves LangGraph-related
                                                    content and generates an answer
                                                    from the context.

  `Who is the current world champion in cricket?`   Agent can answer directly without
                                                    invoking the configured
                                                    documentation retrievers.
  -------------------------------------------------------------------------------------

The direct-answer example demonstrates that the agent is not forced to
retrieve documents for every question. The current configured knowledge
sources are focused on LangChain and LangGraph; a dedicated live
web-search tool is not described as part of the current implementation.

## Engineering Notes

-   **Separation of concerns:** Graph orchestration, state, model
    initialization, retrieval, tools, and node logic are separated into
    modules.
-   **Conditional execution:** The workflow uses conditional edges to
    choose between retrieval, rewriting, generation, and direct
    completion.
-   **Structured grading:** Pydantic-backed structured output makes the
    relevance decision explicit.
-   **Reusable retrievers:** Retriever construction is encapsulated so
    the agent can use multiple knowledge sources.
-   **Configuration management:** API credentials and runtime settings
    are supplied through environment variables.

## Potential Enhancements

The following are possible next steps, not claims about features already
implemented:

-   Add streaming with LangGraph's `stream` / `astream` APIs.
-   Add conversation persistence and session-based memory using a
    LangGraph checkpointer.
-   Add a maximum retry count or loop guard for repeated query
    rewriting.
-   Add a web-search tool for questions requiring current information.
-   Add automated tests for nodes, routing decisions, retrieval, and
    end-to-end graph execution.
-   Expose the agent through a FastAPI service.
-   Add observability and tracing for debugging and evaluation.
-   Add retrieval evaluation using metrics such as precision, recall,
    and answer faithfulness.

## Author

**Ajay Negi**

Full Stack Developer \| AI Agent & RAG Development
