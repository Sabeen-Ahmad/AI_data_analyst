import os
import time
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from langchain_core.messages import HumanMessage, AIMessage

from agent.graph import build_graph

# Page configuration
st.set_page_config(
    page_title="AI Data Analyst Agent",
    layout="wide"
)

# Helper function to extract clean text from agent response
def extract_text(content):
    if isinstance(content, str):
        return content
    elif isinstance(content, list):
        texts = []
        for part in content:
            if isinstance(part, dict) and "text" in part:
                texts.append(part["text"])
            elif isinstance(part, str):
                texts.append(part)
        return "\n".join(texts)
    return str(content)

# Initialize Session State
if "graph" not in st.session_state:
    st.session_state.graph = build_graph()

if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar: Data Source & Settings
with st.sidebar:
    st.header("Dataset Configuration")
    
    uploaded_file = st.file_uploader("Upload CSV file:", type=["csv"])
    
    if uploaded_file is not None:
        file_path = os.path.join("sample_data", uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.success(f"File uploaded: {uploaded_file.name}")
    else:
        file_path = "sample_data/sales.csv"
        st.info("Active dataset: `sample_data/sales.csv`")
        
    # Data preview
    if os.path.exists(file_path):
        try:
            df = pd.read_csv(file_path)
        except UnicodeDecodeError:
            df = pd.read_csv(file_path, encoding="latin1")
            
        st.write(f"**Rows:** {len(df):,} | **Columns:** {len(df.columns)}")
        with st.expander("Data Preview (Top Rows)"):
            st.dataframe(df.head(), use_container_width=True)

    st.markdown("---")
    st.markdown("### Suggested Prompts")
    sample_queries = [
        "Which product generated the highest sales? Create a chart.",
        "Which region produced the highest revenue?",
        "Show the monthly revenue trend with a line chart.",
        "Provide a summary of total sales and key metrics."
    ]
    for q in sample_queries:
        if st.button(q, key=q):
            st.session_state.quick_prompt = q

    st.markdown("---")
    if st.button("Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# Main UI
st.title("AI Data Analyst Agent")
st.caption("Autonomous data exploration, insights, and visualization powered by LangGraph, Gemini, and Plotly.")

# Display Chat History
for msg in st.session_state.messages:
    if isinstance(msg, HumanMessage):
        with st.chat_message("user"):
            st.markdown(extract_text(msg.content))
    elif isinstance(msg, AIMessage):
        with st.chat_message("assistant"):
            st.markdown(extract_text(msg.content))

# Check for quick prompt from sidebar button
prompt = None
if "quick_prompt" in st.session_state and st.session_state.quick_prompt:
    prompt = st.session_state.quick_prompt
    st.session_state.quick_prompt = None

chat_input = st.chat_input("Ask any question regarding your dataset...")
if chat_input:
    prompt = chat_input

# Handle User Query
if prompt:
    # 1. Show user message
    st.session_state.messages.append(HumanMessage(content=prompt))
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. Run Agent
    with st.chat_message("assistant"):
        with st.spinner("Analyzing dataset and generating insights..."):
            chart_file = "chart_output.html"
            chart_mtime_before = os.path.getmtime(chart_file) if os.path.exists(chart_file) else 0

            # Invoke Graph
            result = st.session_state.graph.invoke({
                "messages": st.session_state.messages,
                "file_path": file_path,
                "data_summary": "",
                "analysis_result": "",
                "chart_path": "",
            })

            # Get final AI message
            final_ai_msg = result["messages"][-1]
            response_text = extract_text(final_ai_msg.content)
            st.markdown(response_text)
            st.session_state.messages.append(final_ai_msg)

            # Check if a new chart was created
            if os.path.exists(chart_file):
                chart_mtime_after = os.path.getmtime(chart_file)
                if chart_mtime_after > chart_mtime_before or "chart" in prompt.lower():
                    with open(chart_file, "r", encoding="utf-8") as f:
                        html_data = f.read()
                    st.markdown("### Interactive Visualization")
                    components.html(html_data, height=520, scrolling=True)
