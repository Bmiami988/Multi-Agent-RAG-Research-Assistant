import streamlit as st
import os
import json
import uuid
import datetime
import numpy as np
import faiss

from groq import Groq
from serpapi import GoogleSearch
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader
from docx import Document

from dotenv import load_dotenv
import os

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY")

# -------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------

st.set_page_config(
    page_title="Autonomous Multi-Agent AI System",
    layout="wide"
)

st.title("Autonomous Multi-Agent AI Task Executor")
st.markdown(
    "Tool Selection → Research → Plan → Write → Critique → Improve"
)


# -------------------------------------------------
# SIDEBAR CONFIGURATION
# -------------------------------------------------

st.sidebar.header("Configuration")

if GROQ_API_KEY and SERPAPI_API_KEY:
    st.sidebar.success("API Keys Loaded from .env")
else:
    st.sidebar.error("API Keys Missing. Check your .env file.")

iterations = st.sidebar.slider(
    "Critique Iterations",
    min_value=0,
    max_value=3,
    value=1
)

uploaded_files = st.sidebar.file_uploader(
    "Upload PDF / TXT / DOCX for RAG",
    accept_multiple_files=True
)


# -------------------------------------------------
# INITIALIZATION
# -------------------------------------------------

if GROQ_API_KEY:
    client = Groq(api_key=GROQ_API_KEY)
else:
    client = None

embedder = SentenceTransformer("all-MiniLM-L6-v2")
dimension = 384
index = faiss.IndexFlatL2(dimension)
documents = []


# -------------------------------------------------
# UTILITY FUNCTIONS
# -------------------------------------------------

def call_llm(system_prompt, user_prompt, temperature=0.3):
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=temperature
    )
    return response.choices[0].message.content


def load_file(file):
    if file.name.endswith(".pdf"):
        reader = PdfReader(file)
        return "\n".join([p.extract_text() or "" for p in reader.pages])

    elif file.name.endswith(".docx"):
        doc = Document(file)
        return "\n".join([p.text for p in doc.paragraphs])

    elif file.name.endswith(".txt"):
        return file.read().decode("utf-8")

    return ""


def add_to_vector_store(text, chunk_size=500):
    chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
    embeddings = embedder.encode(chunks)
    index.add(np.array(embeddings))
    documents.extend(chunks)


def retrieve(query, top_k=5):
    if len(documents) == 0:
        return []
    query_embedding = embedder.encode([query])
    D, I = index.search(np.array(query_embedding), top_k)
    return [documents[i] for i in I[0] if i < len(documents)]


def serpapi_search(query):
    params = {
        "q": query,
        "engine": "google",
        "api_key": SERPAPI_API_KEY,
        "num": 5
    }

    search = GoogleSearch(params)
    results = search.get_dict()

    snippets = []

    if "organic_results" in results:
        for r in results["organic_results"][:5]:
            snippets.append(
                f"{r.get('title','')}\n"
                f"{r.get('snippet','')}\n"
                f"Source: {r.get('link','')}\n"
            )

    return "\n".join(snippets)


# -------------------------------------------------
# MEMORY LOGGER
# -------------------------------------------------

class MemoryLogger:
    def __init__(self):
        self.session_id = str(uuid.uuid4())
        self.logs = []

    def log(self, agent, input_data, output_data):
        self.logs.append({
            "timestamp": str(datetime.datetime.utcnow()),
            "session_id": self.session_id,
            "agent": agent,
            "input": input_data,
            "output": output_data
        })


# -------------------------------------------------
# AUTONOMOUS TOOL SELECTOR AGENT
# -------------------------------------------------

def tool_selector_agent(task, memory):

    system_prompt = """
    You are an AI orchestration controller.

    Decide which tools are needed for the task.

    Available tools:
    - web_search
    - rag
    - none

    Return ONLY valid JSON in this format:

    {
        "use_web_search": true/false,
        "use_rag": true/false,
        "reasoning": "short explanation"
    }
    """

    decision_raw = call_llm(system_prompt, task)

    try:
        decision = json.loads(decision_raw)
    except:
        decision = {
            "use_web_search": True,
            "use_rag": True,
            "reasoning": "Fallback due to parsing error."
        }

    memory.log("Tool Selector Agent", task, decision)

    return decision


# -------------------------------------------------
# AGENTS
# -------------------------------------------------

def research_agent(task, tool_decision, memory):

    web_data = ""
    rag_data = ""

    if tool_decision.get("use_web_search"):
        web_data = serpapi_search(task)

    if tool_decision.get("use_rag"):
        rag_data = "\n".join(retrieve(task))

    combined_context = f"""
    WEB SEARCH DATA:
    {web_data}

    RAG DATA:
    {rag_data}
    """

    system_prompt = """
    You are a professional research analyst.
    Use the provided context to generate structured insights:
    - Trends
    - Risks
    - Opportunities
    - Data-backed observations
    """

    research_output = call_llm(system_prompt, combined_context)

    memory.log("Research Agent", combined_context, research_output)

    return research_output


def planning_agent(research, memory):
    system_prompt = "Create a structured professional report outline."
    outline = call_llm(system_prompt, research)
    memory.log("Planning Agent", research, outline)
    return outline


def writing_agent(task, outline, research, memory):

    system_prompt = "Write a detailed executive-level report."

    prompt = f"""
    TASK:
    {task}

    OUTLINE:
    {outline}

    RESEARCH:
    {research}
    """

    report = call_llm(system_prompt, prompt, temperature=0.4)

    memory.log("Writing Agent", prompt, report)

    return report


def critic_agent(report, memory):
    system_prompt = """
    Critique the report.
    Identify weaknesses and missing analysis.
    Provide actionable improvement suggestions.
    """

    critique = call_llm(system_prompt, report)

    memory.log("Critic Agent", report, critique)

    return critique


def improve_agent(report, critique, memory):
    system_prompt = "Improve the report using the critique."
    improved = call_llm(system_prompt, report + "\n\n" + critique)

    memory.log("Improvement Agent", critique, improved)

    return improved


# -------------------------------------------------
# MAIN EXECUTION
# -------------------------------------------------

task = st.text_area("Enter Task", height=150)

if st.button("Run Autonomous Multi-Agent System"):


    memory = MemoryLogger()

    # Load uploaded documents into RAG
    if uploaded_files:
        for file in uploaded_files:
            text = load_file(file)
            add_to_vector_store(text)

    with st.spinner("Selecting Tools..."):
        tool_decision = tool_selector_agent(task, memory)

    st.subheader("Tool Selection Decision")
    st.json(tool_decision)

    with st.spinner("Running Research Agent..."):
        research = research_agent(task, tool_decision, memory)

    with st.spinner("Planning..."):
        outline = planning_agent(research, memory)

    with st.spinner("Writing Report..."):
        report = writing_agent(task, outline, research, memory)

    for i in range(iterations):
        with st.spinner(f"Critique Iteration {i+1}..."):
            critique = critic_agent(report, memory)
            report = improve_agent(report, critique, memory)

    st.success("Execution Complete")

    col1, col2 = st.columns(2)

    with col1:
        with st.expander("Research Output"):
            st.write(research)

        with st.expander("Outline"):
            st.write(outline)

    with col2:
        st.subheader("Final Report")
        st.write(report)

    st.download_button(
        "Download Memory Log (JSON)",
        data=json.dumps(memory.logs, indent=4),
        file_name="agent_memory.json",
        mime="application/json"
    )
