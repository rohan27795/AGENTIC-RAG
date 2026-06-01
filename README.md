🤖 Agentic RAG with CrewAI
A multi-agent, self-correcting Retrieval-Augmented Generation (RAG) pipeline powered by CrewAI, LangChain, Groq (Llama 3.3 70B), and Hugging Face embeddings — featuring intelligent routing, hallucination filtering, and hybrid PDF + web search retrieval.

🚀 Overview
Traditional RAG pipelines are static: they retrieve, generate, and output — with no mechanism to catch bad retrievals, hallucinated answers, or irrelevant responses. This project solves that problem by building an agentic, self-correcting RAG system using a crew of specialized AI agents that collaborate, critique, and verify each other's work before surfacing a final answer.

The pipeline is demonstrated on the landmark ML paper "Attention Is All You Need" (Vaswani et al., 2017), enabling intelligent Q&A over the document while seamlessly falling back to live web search for out-of-scope queries.

Who is this for?

ML engineers exploring advanced RAG architectures
Developers building production-grade AI pipelines with multi-agent orchestration
Researchers who want reliable, hallucination-resistant document Q&A systems
✨ Features
🧠 Multi-Agent Orchestration
5 specialized CrewAI agents, each with a clearly defined role, goal, and backstory
Sequential task execution with shared context passed between agents
No agent can delegate to another, ensuring clean role separation
🔀 Intelligent Query Routing
A dedicated Router Agent classifies every incoming question
Routes to the local PDF vectorstore (RAG) for domain-specific questions (e.g., self-attention, transformers)
Routes to Tavily web search for general or real-time knowledge queries
📄 PDF-Grounded Retrieval (RAG)
Downloads and indexes the "Attention Is All You Need" PDF automatically at startup
Uses PDFSearchTool with semantic search backed by Groq's Llama 3.3 70B and Hugging Face embeddings (BAAI/bge-small-en-v1.5)
Retrieval is grounded in the actual document, not model memory
🌐 Live Web Search Fallback
Integrates Tavily Search API for queries outside the PDF's scope
Returns top-3 web results, summarized and grounded by the Retriever Agent
✅ Multi-Stage Answer Validation
Grader Agent: Checks if retrieved content is actually relevant to the question
Hallucination Grader: Verifies the generated answer is factually grounded
Answer Grader: Final quality gate — triggers a web search fallback if the answer fails validation
⚡ Fast Inference with Groq
Powered by Llama 3.3 70B Versatile via Groq's ultra-low-latency inference API
Temperature set to 0.1 for highly deterministic, factual responses
🛠️ Tech Stack
Layer	Technology
LLM	Llama 3.3 70B Versatile (via Groq API)
Agent Framework	CrewAI 0.28.8
LLM Orchestration	LangChain Community 0.0.29
Embeddings	Hugging Face BAAI/bge-small-en-v1.5 (via sentence-transformers)
PDF RAG	CrewAI Tools PDFSearchTool
Web Search	Tavily Search API (via LangChain integration)
Environment Config	python-dotenv
Language	Python 3.10+
🏗️ Architecture / How It Works
The system implements a Corrective RAG (CRAG) pattern — a self-healing pipeline that validates its own outputs and falls back gracefully when retrieval or generation quality is insufficient.

High-Level Data Flow
User Question
      │
      ▼
┌─────────────────┐
│  Router Agent   │──── keyword analysis ────► 'vectorstore' or 'websearch'
└─────────────────┘
      │
      ▼
┌─────────────────────┐
│  Retriever Agent    │
│  ├─ RAG (PDF)       │◄── PDFSearchTool (Groq + HuggingFace embeddings)
│  └─ Web Search      │◄── Tavily Search API
└─────────────────────┘
      │
      ▼
┌─────────────────┐
│  Grader Agent   │──── relevance check ────► 'yes' / 'no'
└─────────────────┘
      │
      ▼
┌──────────────────────┐
│ Hallucination Grader │──── factual grounding ────► 'yes' / 'no'
└──────────────────────┘
      │
      ▼
┌──────────────────┐
│  Answer Grader   │──── final validation + optional web fallback
└──────────────────┘
      │
      ▼
  Final Answer
Agent Roles & Responsibilities
Agent	Role	Key Responsibility
Router_Agent	Traffic Director	Routes question to vectorstore or web search
Retriever_Agent	Information Retriever	Fetches content from the chosen source
Grader_agent	Relevance Judge	Filters out irrelevant retrieved content
hallucination_grader	Fact Checker	Ensures the answer is grounded in real facts
answer_grader	Final Validator	Confirms answer quality; triggers fallback if needed
Router Logic
The router_tool currently uses keyword matching ('self-attention' → vectorstore) as a fast heuristic, with the Router Agent's LLM reasoning as a broader fallback for nuanced classification.

📂 Folder Structure
agentic-rag-crewai/
│
├── app.py                        # Main application — agents, tasks, crew definition
├── requirements.txt              # Python dependencies
├── .env                          # Environment variables (not committed)
├── .env.example                  # Template for environment setup
│
└── attenstion_is_all_you_need.pdf  # Auto-downloaded at runtime (Attention Is All You Need)
Note: The PDF is downloaded programmatically from NeurIPS proceedings on every run. The local file is used for vectorstore indexing and semantic search.

⚙️ Installation & Setup
Prerequisites
Python 3.10 or higher
A Groq API key (free tier available)
A Tavily API key (free tier available)
Steps
1. Clone the repository

git clone https://github.com/rohan27795/AGENTIC-RAG.git
cd agentic-rag-crewai
2. Create and activate a virtual environment

python -m venv venv
source venv/bin/activate        # macOS/Linux
venv\Scripts\activate           # Windows
3. Install dependencies

pip install -r requirements.txt
4. Configure environment variables

cp .env.example .env
# Edit .env and add your API keys
5. Run the pipeline

python app.py
On first run, the pipeline will:

Download the "Attention Is All You Need" PDF (~3MB)
Build the local vectorstore index (this may take 30–60 seconds)
Execute two example queries and print the full agent trace + final answers
🔑 Environment Variables
Create a .env file in the project root with the following:

# .env.example

# Groq API Key — used for Llama 3.3 70B inference
# Get yours at: https://console.groq.com/
GROQ_API_KEY=your_groq_api_key_here

# Tavily API Key — used for live web search fallback
# Get yours at: https://app.tavily.com/
TAVILY_API_KEY=your_tavily_api_key_here
Variable	Required	Description
GROQ_API_KEY	✅ Yes	Authenticates with Groq's inference API for Llama 3.3 70B
TAVILY_API_KEY	✅ Yes	Enables Tavily web search for out-of-scope queries
🧪 Usage
Default Queries (as configured in __main__)
The app ships with two demonstration queries that test both pipeline branches:

# Query 1 — Routed to the PDF vectorstore (RAG path)
inputs = {"question": "Tell me about self-attention mechanism in Transformers?"}

# Query 2 — Routed to web search
inputs = {"question": "Tell me about LLMs using web_search?"}
Running Custom Queries
Modify the inputs dict in app.py to ask any question:

inputs = {"question": "What is the purpose of positional encoding in the Transformer?"}
result = rag_crew.kickoff(inputs=inputs)
print(result)
Understanding the Output
With verbose=True set on the Crew, you'll see a full trace of each agent's reasoning:

[Router Agent] Routing question to: vectorstore
[Retriever Agent] Searching vectorstore for: self-attention mechanism...
[Grader Agent] Retrieved content is relevant: yes
[Hallucination Grader] Answer is grounded in facts: yes
[Answer Grader] Final answer: The self-attention mechanism allows each token...
📸 Screenshots / Demo
📌 Suggested screenshots to add:

Terminal output showing the full multi-agent trace for a vectorstore query
Terminal output for a web search query showing the Tavily fallback
A diagram of the agent pipeline (the architecture diagram above works well)
To capture output for documentation:

python app.py 2>&1 | tee demo_output.txt
🚧 Challenges & Learnings
1. Balancing Router Precision vs. Recall
The keyword-based router_tool is fast but brittle — questions phrased differently about the same concept can be misrouted. The LLM-backed Router Agent adds reasoning capacity, but prompt tuning was necessary to keep routing deterministic.

2. Vectorstore Cold Start Latency
Building the Chroma/FAISS index from the PDF on every run adds significant startup time. A production version would persist the vectorstore to disk and only rebuild when the source document changes.

3. CrewAI Context Passing
Configuring the context parameter between tasks — ensuring the grader receives the retriever's output, and the hallucination grader receives the grader's verdict — required careful task dependency mapping to avoid context bleed.

4. Embedding Model Selection
Choosing BAAI/bge-small-en-v1.5 balanced retrieval quality with inference speed. Larger models improved accuracy on nuanced technical questions but added unacceptable latency for an interactive use case.

5. Hallucination in Chained Agents
When any agent in the chain produces a low-confidence output, errors can compound downstream. The multi-stage grading approach (relevance → hallucination → answer quality) was specifically designed to catch failures at each stage rather than at the end.

🔮 Future Improvements
Persist the vectorstore to disk (Chroma or FAISS) to eliminate cold-start indexing on every run
Expand the document corpus — support ingestion of multiple PDFs or entire research paper collections
Streaming output — stream agent responses token-by-token for a more interactive experience
Add a web UI — wrap the pipeline in a FastAPI backend + React frontend with real-time agent trace visualization
Smarter routing — replace the keyword heuristic with a small fine-tuned classifier or embedding similarity score
Evaluation harness — integrate RAGAS or TruLens to benchmark retrieval precision, answer faithfulness, and hallucination rates
Memory between sessions — add long-term memory so the agent crew can recall past Q&A sessions
Support more LLM providers — abstract the LLM layer to swap between Groq, OpenAI, Anthropic, or local Ollama models
🤝 Contributing
Contributions are welcome! Here's how to get started:

Fork the repository
Create a branch for your feature: git checkout -b feature/smarter-router
Make your changes and write clear commit messages
Test your changes end-to-end with at least one vectorstore query and one web search query
Open a pull request with a description of what you changed and why
Please follow these guidelines:

Keep agent roles and responsibilities clearly separated
Add comments when modifying the task chain or agent prompts
Do not commit .env files or the downloaded PDF
