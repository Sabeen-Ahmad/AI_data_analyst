# agent/tools.py
# ---------------------------------------------
# Agent Tools:
#   Tool 1: Load and describe CSV dataset
#   Tool 2: Analyze data using pandas
#   Tool 3: Create visualizations using plotly
# ---------------------------------------------

import os
import pandas as pd
import plotly.express as px
from langchain_core.tools import tool


def _resolve_csv(file_path: str) -> str:
    """Safely resolves the input path to an actual valid CSV file."""
    if file_path and os.path.exists(file_path) and file_path.lower().endswith(".csv"):
        return file_path
    
    # Check default project paths
    candidates = [
        "sample_data/SampleSuperstore.csv",
        "sample_data/sales.csv"
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
            
    # Search for any CSV file inside sample_data
    if os.path.exists("sample_data"):
        for f in os.listdir("sample_data"):
            if f.lower().endswith(".csv"):
                return os.path.join("sample_data", f)
                
    return file_path


# ==============================================
# TOOL 1: Load and Describe Dataset
# ==============================================

@tool
def load_and_describe(file_path: str) -> str:
    """Loads a CSV file and provides a structural summary: rows, columns, data types, and head rows."""
    actual_path = _resolve_csv(file_path)
    try:
        df = pd.read_csv(actual_path)
    except UnicodeDecodeError:
        df = pd.read_csv(actual_path, encoding="latin1")

    summary = []
    summary.append(f"Active File: {actual_path}")
    summary.append(f"Total Rows: {len(df)}")
    summary.append(f"Total Columns: {len(df.columns)}")
    summary.append("\nColumn Names & Types:")

    for col in df.columns:
        dtype = df[col].dtype
        nulls = df[col].isnull().sum()
        summary.append(f" - {col} ({dtype}) - {nulls} missing values")

    summary.append("\nFirst 3 Rows:")
    summary.append(df.head(3).to_string())

    summary.append("\nBasic Statistics:")
    summary.append(df.describe().to_string())

    return "\n".join(summary)


# ==============================================
# TOOL 2: Analyze Data
# ==============================================

@tool
def analyze_data(file_path: str, question: str) -> str:
    """Analyzes the dataset to answer user queries with aggregations and statistics."""
    actual_path = _resolve_csv(file_path)
    try:
        df = pd.read_csv(actual_path)
    except UnicodeDecodeError:
        df = pd.read_csv(actual_path, encoding="latin1")

    result = []
    result.append(f"Analysis for query: {question}\n")

    # Numeric summary
    for col in df.select_dtypes(include="number").columns:
        result.append(f"Metric: {col}")
        result.append(f"   Average : {df[col].mean():.2f}")
        result.append(f"   Total   : {df[col].sum():.2f}")
        result.append(f"   Minimum : {df[col].min():.2f}")
        result.append(f"   Maximum : {df[col].max():.2f}")

    # Categorical summary
    for col in df.select_dtypes(include="object").columns:
        result.append(f"\nCategorical: {col} - Top Values:")
        counts = df[col].value_counts().head(5)
        for val, count in counts.items():
            result.append(f"   {val}: {count}")

    # Group-by analysis
    cat_cols = df.select_dtypes(include="object").columns.tolist()
    num_cols = df.select_dtypes(include="number").columns.tolist()

    if cat_cols and num_cols:
        group_col = cat_cols[0]
        num_col = num_cols[0]
        grouped = df.groupby(group_col)[num_col].sum().sort_values(ascending=False)
        result.append(f"\nAggregation: {num_col} grouped by {group_col}:")
        for name, val in grouped.items():
            result.append(f"   {name}: {val:,.2f}")

    return "\n".join(result)


# ==============================================
# TOOL 3: Create Chart
# ==============================================

@tool
def create_chart(file_path: str, chart_type: str, x_column: str, y_column: str, aggregation: str = "sum") -> str:
    """Creates an interactive Plotly chart and saves it as an HTML file.
    file_path: MUST be the input CSV dataset path (e.g. 'sample_data/SampleSuperstore.csv').
    chart_type: 'bar', 'pie', 'line', or 'scatter'.
    x_column: Name of the category/dimension column (e.g. 'Ship Mode', 'Category', 'Region').
    y_column: Name of the numeric metric column (e.g. 'Sales', 'Profit', 'Quantity').
    aggregation: 'sum', 'mean', 'count', 'min', or 'max'. Use 'mean' for average comparisons. Defaults to 'sum'.
    """
    actual_path = _resolve_csv(file_path)
    try:
        df = pd.read_csv(actual_path)
    except UnicodeDecodeError:
        df = pd.read_csv(actual_path, encoding="latin1")

    chart_type_lower = chart_type.lower()

    # Match case-insensitive column names if needed
    col_map = {col.lower(): col for col in df.columns}
    real_x = col_map.get(x_column.lower(), x_column)
    real_y = col_map.get(y_column.lower(), y_column)

    # Aggregate repetitive category rows before plotting to avoid thousands of overlapping slices
    agg_func = aggregation.lower() if aggregation else "sum"
    if agg_func not in ("sum", "mean", "count", "min", "max"):
        agg_func = "sum"

    if real_x in df.columns and real_y in df.columns:
        if pd.api.types.is_numeric_dtype(df[real_y]):
            plot_df = df.groupby(real_x, as_index=False)[real_y].agg(agg_func)
        else:
            plot_df = df.copy()
    else:
        plot_df = df.copy()
        real_x, real_y = x_column, y_column

    # Limit to top 15 categories for clean visual presentation
    if len(plot_df) > 15 and chart_type_lower in ["bar", "pie"]:
        plot_df = plot_df.sort_values(by=real_y, ascending=False).head(15)

    color_palette = px.colors.qualitative.Safe
    agg_label = agg_func.capitalize()

    if chart_type_lower == "pie":
        fig = px.pie(
            plot_df,
            names=real_x,
            values=real_y,
            title=f"{agg_label} of {real_y} by {real_x}",
            color_discrete_sequence=color_palette,
            template="plotly_white"
        )
        fig.update_traces(textposition='inside', textinfo='percent+label', marker=dict(line=dict(color='#ffffff', width=1.5)))
    elif chart_type_lower == "line":
        fig = px.line(
            plot_df,
            x=real_x,
            y=real_y,
            title=f"{agg_label} of {real_y} over {real_x}",
            markers=True,
            template="plotly_white"
        )
    elif chart_type_lower == "scatter":
        fig = px.scatter(
            df,
            x=real_x,
            y=real_y,
            title=f"{real_x} vs {real_y}",
            template="plotly_white"
        )
    else:
        # Default bar chart
        fig = px.bar(
            plot_df,
            x=real_x,
            y=real_y,
            title=f"{agg_label} of {real_y} by {real_x}",
            color=real_x,
            color_discrete_sequence=color_palette,
            template="plotly_white"
        )

    chart_file = "chart_output.html"
    fig.write_html(chart_file)

    return f"Chart saved to {chart_file} (Type: {chart_type}, Aggregation: {agg_func}, X: {real_x}, Y: {real_y})"
