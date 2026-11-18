# 📊 AI Data Analyst Agent

An autonomous data analytics assistant built with **LangGraph**, **Groq LLM**, **Plotly**, and **Streamlit**.

Instead of having LLMs hallucinate calculations, this agent uses a **ReAct (Reason + Act)** loop to inspect datasets, run verified **Pandas** statistical queries, and generate interactive **Plotly** visualizations.

---

## 🚀 Key Features

- **Autonomous Tool Routing**: Decides dynamically whether to inspect structure, calculate statistics, or generate charts.
- **Hallucination-Free Analysis**: Numerical operations and aggregations are executed via deterministic Python/Pandas functions.
- **Dynamic Visualizations**: Automatically chooses appropriate chart types (bar, line, pie, scatter) and aggregations (`sum`, `mean`, `count`).
- **Interactive Web UI**: Clean Streamlit dashboard with file upload, dataset preview, and interactive Plotly chart rendering.
- **Optimized Token Efficiency**: Safe state pruning and rate-limit handling built specifically for production agentic workflows.

---

## 🛠️ Architecture

```
User Query + CSV Dataset
          │
          ▼
   Streamlit Interface (app.py)
          │
          ▼
   LangGraph Agent (agent/graph.py)
   ├── State Management (agent/state.py)
   └── ReAct Loop with Groq (gpt-oss-120b)
          │
          ├──> Tool 1: load_and_describe (Data shape, types, missing values)
          ├──> Tool 2: analyze_data (Summary stats, aggregations, group-bys)
          └──> Tool 3: create_chart (Interactive Plotly visualizations)
          │
          ▼
Verified Analytical Insights + Interactive Chart
```

---

## 📦 Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Sabeen-Ahmad/ai-data-analyst.git
   cd ai-data-analyst
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python -m venv .venv
   # Windows
   .venv\Scripts\activate
   # Linux / macOS
   source .venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure API Keys:**
   Create a `.env` file in the root directory:
   ```env
   GROQ_API_KEY=your_groq_api_key_here
   GOOGLE_API_KEY=your_google_api_key_here
   ```

5. **Run the application:**
   ```bash
   streamlit run app.py
   ```

---

## 📊 Example Queries

- *"Show total Profit by Sub-Category. Create a bar chart."*
- *"Which Category generated the highest sales?"*
- *"Compare the average math score between students who completed test preparation vs none."*

---

## 👩‍💻 Author

**Sabeen Ahmad**  
MS Artificial Intelligence @ NUST SEECS  
[LinkedIn](https://linkedin.com/in/sabeen-ahmad-khan) • [GitHub](https://github.com/Sabeen-Ahmad)
