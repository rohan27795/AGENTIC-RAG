# 🤖 Agentic-RAG: Multi-Agent Retrieval-Augmented Generation System

> An intelligent, multi-agent Q&A system powered by CrewAI, LangChain, and Groq — combining PDF-based vector retrieval with live web search to deliver accurate, hallucination-checked answers.

---

## 🚀 Overview

**Agentic-RAG** is a production-ready, agentic question-answering system that goes far beyond traditional RAG pipelines. Instead of a single retrieval-generation step, it orchestrates a **crew of specialized AI agents** — each responsible for a distinct stage of the pipeline: routing, retrieval, relevance grading, hallucination detection, and final answer validation.

### The Problem It Solves

Standard RAG pipelines suffer from:
- Retrieving irrelevant chunks and blindly passing them to the LLM
- No verification of whether the generated answer is actually grounded in retrieved content
- No fallback when local knowledge is insufficient

**Agentic-RAG** solves this by introducing agent-level checks at each step — ensuring the final answer is relevant, grounded, and trustworthy.

### Who Is It For?

- AI/ML engineers exploring agentic architectures
- Developers building document Q&A systems
- Researchers experimenting with LLM orchestration frameworks
- Anyone who wants to move beyond naive RAG into robust, production-grade pipelines

---

## ✨ Features

### 🧠 Multi-Agent Orchestration
- Five specialized CrewAI agents, each owning a distinct responsibility in the pipeline
- Sequential task execution with shared context passing between agents

### 🔀 Intelligent Query Routing
- Keyword-based router that dynamically decides between **local vector search** (PDF knowledge base) and **live web search**
- Easily extensible routing logic

### 📄 PDF Knowledge Base
- Automatically downloads and indexes the landmark *"Attention Is All You Need"* paper
- Powered by `PDFSearchTool` with HuggingFace embeddings (`BAAI/bge-small-en-v1.5`)

### 🌐 Live Web Search Fallback
- Integrates Tavily Search API for real-time, web-sourced answers when the PDF doesn't cover the topic

### 🔍 Hallucination Detection
- Dedicated `Hallucination Grader` agent cross-checks generated answers against retrieved context before surfacing them to the user

### ✅ Answer Quality Grading
- A final `Answer Grader` agent validates the relevance and completeness of the answer, triggering a web search fallback if needed

### 🖥️ Streamlit Web Interface
- Clean, user-friendly UI for asking questions and viewing AI-generated answers in real time

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| **Frontend** | Streamlit |
| **LLM Backend** | Groq API (`llama3-8b-8192`) via LangChain OpenAI-compatible interface |
| **Agent Orchestration** | CrewAI |
| **Vector Search** | CrewAI `PDFSearchTool` + HuggingFace Embeddings |
| **Web Search** | Tavily Search API (via LangChain) |
| **Embeddings** | `BAAI/bge-small-en-v1.5` (sentence-transformers) |
| **Environment Management** | Python `dotenv` |
| **HTTP** | Python `requests` (PDF download) |

---

## 🏗️ Architecture / How It Works

The system follows an **agentic pipeline** where each agent receives the output of the previous one as context:

```
User Question
      │
      ▼
┌─────────────────┐
│  Router Agent   │  ──► Decides: vectorstore OR web_search
└────────┬────────┘
         │
         ▼
┌─────────────────────┐
│  Retriever Agent    │  ──► Fetches from PDF vectorstore or Tavily web search
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  Grader Agent       │  ──► Checks if retrieved content is relevant to the question
└────────┬────────────┘
         │
         ▼
┌─────────────────────────┐
│  Hallucination Grader   │  ──► Verifies answer is grounded in retrieved content
└────────┬────────────────┘
         │
         ▼
┌─────────────────────┐
│  Answer Grader      │  ──► Final quality check; triggers fallback if needed
└────────┬────────────┘
         │
         ▼
   Final Answer ✅
```

### Data Flow

1. The user submits a question via the Streamlit UI.
2. The **Router Agent** inspects the question and routes it to either the local PDF vectorstore (for self-attention/transformer topics) or Tavily web search.
3. The **Retriever Agent** fetches relevant content from the selected source.
4. The **Grader Agent** filters out irrelevant or low-quality retrievals.
5. The **Hallucination Grader** ensures the generated answer is factually supported by the retrieved context.
6. The **Answer Grader** performs a final pass and either surfaces the answer or initiates a web search fallback.
7. The validated answer is displayed in the Streamlit interface.

---

## 📂 Folder Structure

```
agentic-rag/
│
├── app.py                        # Main application — agents, tasks, Streamlit UI
├── requirements.txt              # Python dependencies
├── .env                          # API keys (not committed — see .env.example)
├── attention_is_all_you_need.pdf # Auto-downloaded at runtime; PDF knowledge base
└── README.md                     # Project documentation
```

| File | Purpose |
|---|---|
| `app.py` | Core logic: LLM init, PDF download, agent/task definitions, CrewAI orchestration, Streamlit UI |
| `requirements.txt` | Pinned dependencies for reproducible installs |
| `.env` | Stores secret API keys (Groq, Tavily) — never commit this file |

---

## ⚙️ Installation & Setup

### Prerequisites

- Python 3.9+
- A [Groq API key](https://console.groq.com/)
- A [Tavily API key](https://app.tavily.com/)

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/agentic-rag.git
cd agentic-rag
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Set Up Environment Variables

Create a `.env` file in the project root:

```bash
cp .env.example .env
```

Then fill in your API keys (see [Environment Variables](#-environment-variables) below).

### 5. Run the Application

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`.

> **Note:** On first run, the app will automatically download the *"Attention Is All You Need"* PDF (~2MB) and build the vector index. This may take a minute.

---

## 🔑 Environment Variables

Create a `.env` file at the project root with the following keys:

```env
# .env.example

# Groq API Key — used to power the LLM (llama3-8b-8192)
# Get yours at: https://console.groq.com/
GROQ_API_KEY=your_groq_api_key_here

# Tavily Search API Key — used for live web search fallback
# Get yours at: https://app.tavily.com/
TAVILY_API_KEY=your_tavily_api_key_here
```

| Variable | Required | Description |
|---|---|---|
| `GROQ_API_KEY` | ✅ Yes | Authenticates requests to Groq's LLM API (free tier available) |
| `TAVILY_API_KEY` | ✅ Yes | Authenticates requests to Tavily's real-time web search API |

---

## 🧪 Usage

### Asking Questions

1. Launch the app with `streamlit run app.py`
2. Type your question in the input box
3. Click **"Get Answer"**
4. Watch the multi-agent pipeline reason through your question in the terminal (verbose mode), then display the final answer

### Example Questions

**Routed to PDF vectorstore (transformer paper topics):**
```
What is self-attention and how does it work?
How does the multi-head attention mechanism function?
What are the key components of the Transformer architecture?
```

**Routed to web search (general/current topics):**
```
What are the latest LLM benchmarks in 2024?
Who are the leading AI research labs today?
What is LangChain used for?
```

### Extending the Router

To route more topics to the vectorstore, update the `router_tool` function in `app.py`:

```python
@tool
def router_tool(question):
    vectorstore_keywords = ['self-attention', 'transformer', 'encoder', 'decoder', 'positional encoding']
    if any(kw in question.lower() for kw in vectorstore_keywords):
        return 'vectorstore'
    return 'web_search'
```

---

## 📸 Screenshots / Demo

> **Suggested screenshot locations:**
> - `assets/screenshot_ui.png` — the Streamlit question input and answer display
> - `assets/screenshot_terminal.png` — the verbose agent reasoning output in the terminal

To add screenshots, create an `assets/` folder and reference them here:

```markdown
![UI Screenshot](assets/screenshot_ui.png)
![Agent Pipeline Terminal](assets/screenshot_terminal.png)
```

---

## 🚧 Challenges & Learnings

### Challenges

**1. Agent Context Bleeding**
Getting CrewAI agents to pass only the relevant context (not the entire conversation history) between tasks required careful use of the `context` parameter in task definitions. Without this, downstream agents would receive too much noise.

**2. Hallucination Grading Subjectivity**
Defining what counts as "hallucination" for the grader agent is inherently ambiguous. The agent prompt had to be carefully tuned to distinguish between a well-supported claim and a plausible-sounding fabrication.

**3. PDFSearchTool Latency**
Building the vector index on first run (embedding the full Transformer paper) adds noticeable cold-start latency. Future iterations should persist the index to disk.

**4. Routing Logic Fragility**
The initial keyword-based router (`'self-attention' in question`) is brittle. Synonyms and paraphrases bypass it. A semantic router using embeddings would be more robust.

**5. Groq Rate Limits**
With five agents each making LLM calls, complex queries can hit Groq's rate limits on the free tier. Batching or caching intermediate results helps mitigate this.

### Learnings

- **CrewAI's agent abstraction** makes it remarkably easy to decompose complex pipelines into single-responsibility units — a major win for maintainability.
- **LangChain's tool abstraction** enables seamless swapping of retrieval backends (vector DB, web, SQL) without changing agent logic.
- **Groq's inference speed** (`llama3-8b-8192` at ~800 tokens/sec) makes rapid multi-agent pipelines practical in ways that weren't possible with hosted OpenAI endpoints alone.
- **HuggingFace embeddings** (`BAAI/bge-small-en-v1.5`) offer a strong quality-to-cost ratio for local embedding — no API key required.

---

## 🔮 Future Improvements

| Enhancement | Description |
|---|---|
| 🗂️ **Persistent Vector Index** | Cache the PDF vector index to disk (ChromaDB/FAISS) to eliminate cold-start latency on repeat runs |
| 🔀 **Semantic Router** | Replace keyword matching with an embedding-based semantic router for more accurate, intent-aware routing |
| 📚 **Multi-Document Support** | Allow users to upload their own PDFs via the Streamlit UI and dynamically index them |
| 🔄 **Retry & Fallback Logic** | Add explicit retry loops when the hallucination grader rejects an answer, before falling back to web search |
| 📊 **Agent Reasoning Visualization** | Display each agent's chain-of-thought in the UI (expandable sections) for transparency |
| 💾 **Conversation History** | Maintain session-level conversation history for multi-turn Q&A |
| 🧪 **Evaluation Suite** | Build a benchmark dataset of questions with ground-truth answers to measure end-to-end pipeline accuracy |
| 🐳 **Dockerization** | Package the app in a Docker container for one-command deployment |

---

## 🤝 Contributing

Contributions are welcome! Here's how to get started:

1. **Fork** the repository
2. **Create a feature branch**: `git checkout -b feature/your-feature-name`
3. **Make your changes** and add tests where applicable
4. **Commit** with a clear message: `git commit -m "feat: add semantic router"`
5. **Push** to your branch: `git push origin feature/your-feature-name`
6. **Open a Pull Request** — describe what you changed and why

### Code Style

- Follow PEP 8 for Python code
- Keep agent definitions and task definitions in clearly separated sections
- Add docstrings to any new tools or utility functions

### Reporting Issues

Please open a GitHub Issue with:
- A clear description of the bug or feature request
- Steps to reproduce (for bugs)
- Expected vs. actual behavior

---

## 📜 License

This project is licensed under the **MIT License**.

```
MIT License

Copyright (c) 2024

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
```

---

<div align="center">

**Built with ❤️ using [CrewAI](https://github.com/joaomdmoura/crewAI) · [LangChain](https://langchain.com/) · [Groq](https://groq.com/) · [Streamlit](https://streamlit.io/)**

</div>
