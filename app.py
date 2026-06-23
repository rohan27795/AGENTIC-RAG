import os
import streamlit as st
from langchain_openai import ChatOpenAI
from crewai_tools import PDFSearchTool
from langchain_community.tools.tavily_search import TavilySearchResults
from crewai_tools import tool
from crewai import Crew, Task, Agent
import requests
from dotenv import load_dotenv

# Load API Keys from .env file
load_dotenv()
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
os.environ['TAVILY_API_KEY'] = os.getenv("TAVILY_API_KEY")

# Initialize LLM — gemma2-9b-it has 15,000 TPM (much higher than llama-3.1-8b-instant's 6,000)
llm = ChatOpenAI(
    openai_api_base="https://api.groq.com/openai/v1",
    openai_api_key=os.environ['GROQ_API_KEY'],
    model_name="llama-3.1-8b-instant",
    temperature=0.1,
    max_tokens=500,
)

# Download and process PDF (only if not already downloaded)
pdf_path = "attention_is_all_you_need.pdf"
if not os.path.exists(pdf_path):
    pdf_url = "https://proceedings.neurips.cc/paper_files/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf"
    response = requests.get(pdf_url)
    with open(pdf_path, "wb") as file:
        file.write(response.content)

rag_tool = PDFSearchTool(
    pdf=pdf_path,
    config=dict(
        llm=dict(
            provider="groq",
            config=dict(
                model="llama-3.1-8b-instant",
            ),
        ),
        embedder=dict(
            provider="huggingface",
            config=dict(
                model="BAAI/bge-small-en-v1.5",
            ),
        ),
    )
)


@tool
def router_tool(question: str) -> str:
    """
    Routes the user question to either 'vectorstore' or 'web_search'.
    If the question is about transformers, attention, encoder, decoder, or neural network architecture, route to vectorstore.
    Otherwise, route to web_search.
    """
    q_lower = question.lower()
    keywords = ['self-attention', 'transformer', 'attention', 'encoder', 'decoder',
                'multi-head', 'architecture', 'nlp', 'neural', 'bert', 'gpt', 'model']
    if any(keyword in q_lower for keyword in keywords):
        return 'vectorstore'
    else:
        return 'web_search'


@tool
def web_search_tool(query: str) -> str:
    """
    Searches the web for information when the question is not covered by the PDF vectorstore.
    """
    try:
        tavily = TavilySearchResults(k=3)
        result = tavily.run(query)
        if result:
            return str(result)
    except Exception:
        pass

    try:
        from duckduckgo_search import DDGS
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=3))
            if results:
                return "\n\n".join([f"Title: {r['title']}\nSnippet: {r['body']}" for r in results])
    except Exception:
        pass

    return f"Could not retrieve web results. Please answer based on your training knowledge about: {query}"


# Define Agents — max_iter=3 prevents runaway loops that burn tokens
Router_Agent = Agent(
    role='Router',
    goal='Route user question to a vectorstore or web search',
    backstory='You are an expert router. You use the router_tool to decide routing and immediately return the result.',
    verbose=True,
    allow_delegation=False,
    max_iter=3,
    llm=llm
)

Retriever_Agent = Agent(
    role="Retriever",
    goal="Search the PDF vectorstore for the answer to the user's question and return the relevant content",
    backstory="You search the vectorstore using the exact user question and return the most relevant excerpts.",
    verbose=True,
    allow_delegation=False,
    max_iter=3,
    llm=llm
)

Grader_agent = Agent(
    role='Relevance Grader',
    goal='Determine if the retrieved content answers the user question',
    backstory="You assess whether the retrieved content is relevant and useful to answer the user's question.",
    verbose=True,
    allow_delegation=False,
    max_iter=2,
    llm=llm
)

hallucination_grader = Agent(
    role="Hallucination Grader",
    goal="Check if the answer is grounded in the retrieved content",
    backstory="You verify that the answer is factually supported by the retrieved documents. If the content is relevant to the question, approve it.",
    verbose=True,
    allow_delegation=False,
    max_iter=2,
    llm=llm
)

answer_grader = Agent(
    role="Final Answer Agent",
    goal="Produce a concise, accurate final answer to the user's question",
    backstory="You synthesize the retrieved information and grading results to give the user a clear, direct answer.",
    verbose=True,
    allow_delegation=False,
    max_iter=3,
    llm=llm
)


# Define Tasks — all tasks now include {question} so agents know what to answer
router_task = Task(
    description=(
        "The user asked: '{question}'\n"
        "Use the router_tool with this exact question to decide the routing: 'vectorstore' or 'web_search'.\n"
        "Return only the routing decision."
    ),
    expected_output="Either 'vectorstore' or 'web_search'.",
    agent=Router_Agent,
    tools=[router_tool]
)

retriever_task = Task(
    description=(
        "The user asked: '{question}'\n"
        "Use the PDF search tool to search for information about this question in the vectorstore.\n"
        "Search using the user's actual question as the query. Return the relevant content found."
    ),
    expected_output="Relevant excerpts from the PDF that help answer the user's question.",
    agent=Retriever_Agent,
    context=[router_task],
    tools=[rag_tool]
)

grader_task = Task(
    description=(
        "The user asked: '{question}'\n"
        "Review the retrieved content from the previous step.\n"
        "Decide if the content is relevant and useful to answer the question.\n"
        "Respond with 'relevant' or 'not relevant' and a brief reason."
    ),
    expected_output="'relevant' or 'not relevant' with a brief explanation.",
    agent=Grader_agent,
    context=[retriever_task]
)

hallucination_task = Task(
    description=(
        "The user asked: '{question}'\n"
        "Review the retrieved content and the relevance grade.\n"
        "If the content is relevant to the question, respond with 'supported'.\n"
        "If the content is completely off-topic, respond with 'not supported'.\n"
        "Be lenient — if the content is about the same topic, mark it as 'supported'."
    ),
    expected_output="'supported' or 'not supported' with a brief reason.",
    agent=hallucination_grader,
    context=[grader_task]
)

answer_task = Task(
    description=(
        "The user asked: '{question}'\n"
        "Using all the retrieved and graded information from previous steps, provide a clear, concise answer.\n"
        "If the hallucination grader said 'supported', summarize the answer from the retrieved content.\n"
        "If 'not supported', use the web_search_tool to find the answer.\n"
        "Always give a direct, helpful answer to the user's question."
    ),
    expected_output="A clear, concise answer to the user's question.",
    context=[hallucination_task],
    agent=answer_grader,
    tools=[web_search_tool]
)

# max_rpm=5 prevents hitting rate limits by slowing down LLM calls
rag_crew = Crew(
    agents=[Router_Agent, Retriever_Agent, Grader_agent, hallucination_grader, answer_grader],
    tasks=[router_task, retriever_task, grader_task, hallucination_task, answer_task],
    verbose=True,
    max_rpm=5
)

# Streamlit UI
st.set_page_config(page_title="RAG-based Q&A", layout="wide")
st.title("📖 RAG-based Question Answering System")
st.markdown("Type your question below and get AI-powered answers from the **Attention is All You Need** paper!")

question = st.text_input("Enter your question:", placeholder="e.g. What is the main advantage of transformer architecture?")
if st.button("Get Answer", type="primary"):
    if question:
        with st.spinner("🤖 Agents are working on your answer..."):
            inputs = {"question": question}
            result = rag_crew.kickoff(inputs=inputs)
        st.subheader("✅ Answer:")
        st.write(result)
    else:
        st.warning("Please enter a question!")

st.markdown("---")
st.markdown("**Built with LangChain, CrewAI, and Streamlit** | Powered by Groq (gemma2-9b-it)")
