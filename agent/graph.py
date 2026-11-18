# agent/graph.py
# ─────────────────────────────────────────────
# Yeh agent ka BRAIN hai — LangGraph se bana hai
#
# Socho ek flowchart hai:
#   User sawaal poochhe → Agent soche → Tool chalaye → Jawab de
#
# LangGraph issi flowchart ko code mein likhne deta hai
# ─────────────────────────────────────────────

import os
import time
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage

from agent.state import AgentState
from agent.tools import load_and_describe, analyze_data, create_chart

# Load environment variables
load_dotenv()


# ==============================================
# 1) LLM Setup
# ==============================================

groq_key = os.getenv("GROQ_API_KEY")
google_key = os.getenv("GOOGLE_API_KEY")

if groq_key and groq_key.strip():
    # Fast inference with Groq — gpt-oss-120b creates accurate tool calls
    llm = ChatGroq(
        model="openai/gpt-oss-120b",
        groq_api_key=groq_key.strip(),
        temperature=0,
        max_tokens=2048,
    )
elif google_key and google_key.strip():
    # Google Gemini 3.8 Flash Fallback
    llm = ChatGoogleGenerativeAI(
        model="gemini-3.8-flash",
        google_api_key=google_key.strip(),
        temperature=0,
    )
else:
    raise ValueError("Neither GROQ_API_KEY nor GOOGLE_API_KEY found in .env")

# Bind analytical tools to the model
tools = [load_and_describe, analyze_data, create_chart]
llm_with_tools = llm.bind_tools(tools)


# ══════════════════════════════════════════════
# 2) NODES — Graph ke "stations"
# ══════════════════════════════════════════════
# Socho train ki tarah — har node ek station hai
# Agent ek station se doosre station pe jaata hai

def agent_node(state: AgentState):
    """
    Node 1: AGENT — Sochta hai kya karna hai.
    
    Yeh node:
    - User ka message padhta hai
    - Sochta hai: "Kya mujhe koi tool chalana chahiye?"
    - Agar haan → tool call karta hai
    - Agar nahi → seedha jawab deta hai
    """

    file_path = state.get("file_path", "sample_data/sales.csv")

    # System message defining agent role and instructions
    system = SystemMessage(content=f"""
    You are an expert Data Analyst AI agent. Your responsibilities:
    1. Inspect and analyze the user's dataset thoroughly.
    2. Provide clear, accurate numerical insights, comparisons, and aggregations.
    3. Generate visualizations (charts) whenever requested or helpful.
    4. Provide your answers in professional, clear, and structured English. Avoid using emojis.
    
    IMPORTANT WORKFLOW: You MUST call ALL necessary tools in sequence to fully answer the user's question.
    - If the user asks for analysis AND a chart, call BOTH analyze_data AND create_chart — do NOT stop after just loading data.
    - When comparing averages, use aggregation='mean' in create_chart.
    - Always begin by loading and inspecting the data structure using load_and_describe if not yet inspected, then run appropriate analysis or charting tools.
    The current active dataset file path is: {file_path}
    CRITICAL: When calling load_and_describe, analyze_data, or create_chart, always pass the exact dataset file path "{file_path}" for the file_path parameter. Do not pass an html or output file path as the file_path argument.
    """)

    # Pass recent messages to LLM to preserve context without exceeding rate limits
    recent_messages = state.get("messages", [])[-8:]

    # Retry logic for rate limit errors (Groq free tier has low TPM)
    for attempt in range(3):
        try:
            response = llm_with_tools.invoke([system] + recent_messages)
            break
        except Exception as e:
            if "rate_limit" in str(e).lower() or "429" in str(e):
                wait_time = 12 * (attempt + 1)  # 12s, 24s, 36s
                print(f"Rate limit hit, waiting {wait_time}s before retry {attempt + 1}/3...")
                time.sleep(wait_time)
            else:
                raise e
    else:
        raise Exception("Rate limit exceeded after 3 retries. Please wait a minute and try again.")

    # Return only the new response; LangGraph's operator.add appends it to state["messages"]
    return {"messages": [response]}


# Tool Node — jab agent tool call kare toh yeh chalti hai
tool_node = ToolNode(tools)         # LangGraph ka built-in tool runner


# ══════════════════════════════════════════════
# 3) DECISION — Agent ko aage kahan jaana hai?
# ══════════════════════════════════════════════

def should_continue(state: AgentState):
    """
    Yeh function decide karta hai:
    - Kya agent ne koi tool call ki?  → "tools" pe jao
    - Ya seedha jawab de diya?        → "end" pe jao (khatam)
    
    Jaise ek chauraha (crossroad):
      ← Tool call hai? → tools node pe jao
      → Nahi hai?      → kaam khatam, jawab de do
    """
    last_message = state["messages"][-1]     # sabse last message dekho

    # Agar message mein tool_calls hain → matlab agent tool use karna chahta hai
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"          # tools node pe jao
    else:
        return END              # khatam — jawab de do


# ══════════════════════════════════════════════
# 4) GRAPH BUILD — Sab jodho!
# ══════════════════════════════════════════════
# Ab sab nodes ko ek flowchart mein jodhte hain:
#
#   START → agent → (decision) → tools → agent → ... → END
#                        ↓
#                       END

def build_graph():
    """
    Poora graph (flowchart) banao aur return karo.
    
    Flow:
    [Start] → [Agent sochta hai] → [Tool call hai?]
                                       ↓ Haan         ↓ Nahi
                                   [Tool chalo]      [END - jawab do]
                                       ↓
                                   [Agent phir sochta hai] → ...
    """

    # Step A: Khaali graph banao (AgentState memory ke saath)
    graph = StateGraph(AgentState)

    # Step B: Nodes (stations) add karo
    graph.add_node("agent", agent_node)     # station 1: agent sochta hai
    graph.add_node("tools", tool_node)      # station 2: tools chalti hain

    # Step C: Starting point set karo
    graph.set_entry_point("agent")          # train "agent" station se shuru hogi

    # Step D: Connections (edges) banao
    graph.add_conditional_edges(
        "agent",                # agent station se
        should_continue,        # yeh function decide karega kidhar jaana hai
        {
            "tools": "tools",   # agar "tools" bole → tools station pe jao
            END: END,           # agar "END" bole → khatam
        }
    )

    # Step E: Tools ke baad wapas agent pe aao (loop!)
    graph.add_edge("tools", "agent")        # tool chalne ke baad → agent phir sochega

    # Step F: Graph compile karo (tayyar karo)
    return graph.compile()
