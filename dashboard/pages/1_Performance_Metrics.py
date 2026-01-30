import streamlit as st
from datetime import datetime
import plotly.express as px
import pandas as pd

from utils.data_loader import load_run_list, load_results_for_run
from utils.style import SIDEBAR_STYLE

st.set_page_config(page_title="Performance Metrics")
st.markdown(SIDEBAR_STYLE, unsafe_allow_html=True)


def create_results_table(results):
    """Create a DataFrame from test results"""
    rows = []

    for test in results:  # Results is a list of test cases
        row = {
            "Image": test["fileUrl"],
            "OCR Model": test["ocrModel"],
            "Extraction Model": test["extractionModel"],
            "Levenshtein Score": test.get("levenshteinDistance", 0),
            "JSON Accuracy": test.get("jsonAccuracy", 0),
            "Total Cost": test.get("usage", {}).get("totalCost", 0),
            "Duration (ms)": test.get("usage", {}).get("duration", 0),
            "Metadata": test.get("metadata", {}),
        }
        rows.append(row)

    return pd.DataFrame(rows)


def create_model_comparison_table(results):
    """Create a DataFrame comparing different model combinations"""
    model_stats = {}

    for test in results:
        if "error" in test and test["error"]:
            continue

        model_key = (
            f"{test['extractionModel']} (IMG2JSON)"
            if test.get("directImageExtraction", False)
            else f"{test['ocrModel']} → {test['extractionModel']}"
        )
        if model_key not in model_stats:
            model_stats[model_key] = {
                "count": 0,
                "json_accuracy": 0,
                "text_accuracy": 0,
                "total_cost": 0.0,
                "ocr_cost": 0.0,
                "extraction_cost": 0.0,
                "ocr_latency": 0,
                "extraction_latency": 0,
                "extraction_count": 0,
                "ocr_input_tokens": 0,
                "ocr_output_tokens": 0,
                "extraction_input_tokens": 0,
                "extraction_output_tokens": 0,
            }

        stats = model_stats[model_key]
        stats["count"] += 1
        stats["text_accuracy"] += test.get("levenshteinDistance", 0) or 0
        # Ensure None values are converted to 0.0
        totalCost = test.get("usage", {}).get("totalCost")
        stats["total_cost"] += 0.0 if totalCost is None else totalCost
        usage = test.get("usage", {})

        # Handle possible None values in ocr cost
        ocrCost = usage.get("ocr", {}).get("totalCost")
        stats["ocr_cost"] += 0.0 if ocrCost is None else ocrCost

        stats["ocr_latency"] += usage.get("ocr", {}).get("duration", 0) / 1000
        stats["ocr_input_tokens"] += usage.get("ocr", {}).get("inputTokens", 0)
        stats["ocr_output_tokens"] += usage.get("ocr", {}).get("outputTokens", 0)

        # Add token counting
        if usage.get("extraction"):
            stats["extraction_input_tokens"] += usage.get("extraction", {}).get(
                "inputTokens", 0
            )
            stats["extraction_output_tokens"] += usage.get("extraction", {}).get(
                "outputTokens", 0
            )

        # Only add JSON accuracy and extraction stats if extraction was performed
        if "jsonAccuracy" in test and usage.get("extraction"):
            stats["extraction_count"] += 1
            stats["json_accuracy"] += test["jsonAccuracy"]
            # Handle possible None values in extraction cost
            extractionCost = usage.get("extraction", {}).get("totalCost")
            stats["extraction_cost"] += (
                0.0 if extractionCost is None else extractionCost
            )
            stats["extraction_latency"] += (
                usage.get("extraction", {}).get("duration", 0) / 1000
            )

    # Calculate averages
    for stats in model_stats.values():
        stats["text_accuracy"] /= stats["count"]
        stats["ocr_latency"] /= stats["count"]
        stats["ocr_cost"] /= stats["count"]
        stats["total_cost"] /= stats["count"]
        stats["ocr_input_tokens"] /= stats["count"]
        stats["ocr_output_tokens"] /= stats["count"]

        # Calculate extraction-related averages only if there were extractions
        if stats["extraction_count"] > 0:
            stats["json_accuracy"] /= stats["extraction_count"]
            stats["extraction_latency"] /= stats["extraction_count"]
            stats["extraction_cost"] /= stats["extraction_count"]
            stats["extraction_input_tokens"] /= stats["extraction_count"]
            stats["extraction_output_tokens"] /= stats["extraction_count"]

    # Convert to DataFrame
    df = pd.DataFrame.from_dict(model_stats, orient="index")
    df.index.name = "Model Combination"
    return df


def create_accuracy_comparison_charts(results):
    """Create separate DataFrames for JSON and Text accuracy comparisons"""
    model_accuracies = {}

    for test in results:
        if "error" in test and test["error"]:
            continue

        model_key = (
            f"{test['extractionModel']} (IMG2JSON)"
            if test.get("directImageExtraction", False)
            else f"{test['ocrModel']} → {test['extractionModel']}"
        )
        if model_key not in model_accuracies:
            model_accuracies[model_key] = {
                "count": 0,
                "json_accuracy": 0,
                "text_similarity": 0,
                "total_matched_items": 0,
                "total_items": 0,
                "extraction_count": 0,
            }

        stats = model_accuracies[model_key]
        stats["count"] += 1
        stats["text_similarity"] += test.get("levenshteinDistance", 0) or 0

        # Handle JSON accuracy if present
        if "jsonAccuracy" in test:
            stats["extraction_count"] += 1
            stats["json_accuracy"] += test["jsonAccuracy"] or 0

    # Calculate final averages
    for stats in model_accuracies.values():
        stats["text_similarity"] /= stats["count"]

        # Calculate JSON accuracy only if there were extractions
        if stats["extraction_count"] > 0:
            stats["json_accuracy"] /= stats["extraction_count"]
        else:
            stats["json_accuracy"] = 0

    # Create DataFrames
    json_df = pd.DataFrame(
        {
            "Model": model_accuracies.keys(),
            "JSON Accuracy": [
                stats["json_accuracy"] for stats in model_accuracies.values()
            ],
        }
    ).set_index("Model")

    text_df = pd.DataFrame(
        {
            "Model": model_accuracies.keys(),
            "Text Similarity": [
                stats["text_similarity"] for stats in model_accuracies.values()
            ],
        }
    ).set_index("Model")

    return json_df, text_df


def main():
    st.title("Performance Metrics")

    # Load only the list of runs initially
    runs = load_run_list()

    if not runs:
        st.warning("No benchmark runs found.")
        return

    # Create columns for the header section
    col1, col2 = st.columns([2, 3])

    with col1:
        # Create a dropdown to select the test run
        selected_timestamp = st.selectbox(
            "Select Test Run",
            [run["timestamp"] for run in runs],
            format_func=lambda x: datetime.strptime(x, "%Y-%m-%d-%H-%M-%S").strftime(
                "%Y-%m-%d %H:%M:%S"
            ),
        )

    # Load the detailed results only when a run is selected
    run_data = load_results_for_run(selected_timestamp)

    with col2:
        st.markdown('<div style="margin-top: 24px;">', unsafe_allow_html=True)
        with st.expander("Run Details", expanded=True):
            if run_data.get("run_by"):
                st.markdown(f"**Run By:** {run_data['run_by']}")
            if run_data.get("description"):
                st.markdown(f"**Description:** {run_data['description']}")
            st.markdown(f"**Total # of documents:** {run_data['total_documents']}")
            st.markdown(f"**Status:** {run_data['status'].title()}")
            st.markdown(f"**Created:** {run_data['created_at']}")
            if run_data.get("completed_at"):
                st.markdown(f"**Completed:** {run_data['completed_at']}")

    results = run_data["results"]

    st.header("Evaluation Metrics by Model")
    json_df, text_df = create_accuracy_comparison_charts(results)
    fig1 = px.bar(
        json_df.reset_index().sort_values("JSON Accuracy", ascending=False),
        x="Model",
        y="JSON Accuracy",
        title="JSON Accuracy by Model",
        height=600,
        color_discrete_sequence=["#636EFA"],
    )
    fig1.update_layout(showlegend=False)
    fig1.update_traces(texttemplate="%{y:.1%}", textposition="outside")
    st.plotly_chart(fig1)

    fig2 = px.bar(
        text_df.reset_index().sort_values("Text Similarity", ascending=False),
        x="Model",
        y="Text Similarity",
        title="Text Similarity by Model",
        height=600,
        color_discrete_sequence=["#636EFA"],
    )
    fig2.update_layout(showlegend=False)
    fig2.update_traces(texttemplate="%{y:.1%}", textposition="outside")
    st.plotly_chart(fig2)

    # Model Statistics Table
    st.header("Model Performance Statistics")
    model_stats = create_model_comparison_table(results)
    st.dataframe(
        model_stats.style.format(
            {
                "json_accuracy": "{:.2%}",
                "text_accuracy": "{:.2%}",
                "avg_latency": "{:.2f} s",
                "total_cost": "${:.4f}",
                "count": "{:.0f}",
            }
        )
    )

    # Cost and Latency Charts
    st.header("Cost and Latency Analysis")

    # Cost per document chart
    cost_df = pd.DataFrame(model_stats["total_cost"] * 1000).reset_index()
    cost_df.columns = ["Model", "Cost per 1,000 Pages"]
    fig4 = px.bar(
        cost_df.sort_values("Cost per 1,000 Pages", ascending=True),
        x="Model",
        y="Cost per 1,000 Pages",
        title="Cost per 1,000 Pages by Model Combination",
        height=600,
        color_discrete_sequence=["#EE553B"],
    )
    fig4.update_layout(showlegend=False)
    fig4.update_traces(texttemplate="$%{y:.2f}", textposition="outside")
    st.plotly_chart(fig4)

    # Create stacked bar chart for cost breakdown per document
    cost_breakdown_df = pd.DataFrame(
        {
            "Model": model_stats.index,
            "OCR": model_stats["ocr_cost"] * 1000,
            "Extraction": model_stats["extraction_cost"] * 1000,
        }
    )

    # Calculate cost per 1k documents for sorting
    cost_breakdown_df["Total"] = (
        cost_breakdown_df["OCR"] + cost_breakdown_df["Extraction"]
    )
    fig_cost = px.bar(
        cost_breakdown_df.sort_values("Total", ascending=True),
        x="Model",
        y=["OCR", "Extraction"],
        title="Cost per 1,000 Pages Breakdown by Model Combination (OCR + Extraction)",
        height=600,
        color_discrete_sequence=["#636EFA", "#EF553B"],
    )
    fig_cost.update_layout(
        barmode="stack",
        showlegend=True,
        legend_title="Phase",
        yaxis=dict(
            title="Cost per 1,000 Pages (USD)",
            range=[
                0,
                cost_breakdown_df["Total"].max() * 1.2,
            ],
        ),
    )
    fig_cost.update_traces(texttemplate="$%{y:.2f}", textposition="inside")
    st.plotly_chart(fig_cost)

    # Create stacked bar chart for latency
    latency_df = pd.DataFrame(
        {
            "Model": model_stats.index,
            "OCR": model_stats["ocr_latency"],
            "Extraction": model_stats["extraction_latency"],
        }
    )

    # Calculate total latency for labels
    latency_df["Total"] = latency_df.get("OCR", 0) + latency_df.get("Extraction", 0)
    fig5 = px.bar(
        latency_df.sort_values("Total", ascending=True),
        x="Model",
        y=["OCR", "Extraction"],
        title="Latency by Model Combination (OCR + Extraction)",
        height=600,
        color_discrete_sequence=["#636EFA", "#EF553B"],
    )
    fig5.update_layout(
        barmode="stack",
        showlegend=True,
        legend_title="Phase",
        yaxis=dict(
            range=[
                0,
                latency_df["Total"].max() * 1.2,
            ]  # Set y-axis range to 120% of max value
        ),
    )
    fig5.update_traces(texttemplate="%{y:.2f}s", textposition="inside")
    st.plotly_chart(fig5)

    # Total latency chart
    total_latency_df = pd.DataFrame(
        {
            "Model": model_stats.index,
            "Total Latency": model_stats["ocr_latency"]
            + model_stats["extraction_latency"],
        }
    )
    fig6 = px.bar(
        total_latency_df.sort_values("Total Latency", ascending=True),
        x="Model",
        y="Total Latency",
        title="Total Latency by Model Combination",
        height=600,
        color_discrete_sequence=["#636EFA"],
    )
    fig6.update_layout(showlegend=False)
    fig6.update_traces(texttemplate="%{y:.2f}s", textposition="outside")
    st.plotly_chart(fig6)

    # Add new token usage chart at the bottom
    st.header("Token Usage Analysis")
    token_df = pd.DataFrame(
        {
            "Model": model_stats.index,
            "Input Tokens": model_stats["ocr_input_tokens"],
            "Output Tokens": model_stats["ocr_output_tokens"],
            "Extraction Input Tokens": model_stats["extraction_input_tokens"],
            "Extraction Output Tokens": model_stats["extraction_output_tokens"],
        }
    )

    # Calculate total tokens for sorting
    token_df["Total"] = (
        token_df["Input Tokens"]
        + token_df["Output Tokens"]
        + token_df["Extraction Input Tokens"]
        + token_df["Extraction Output Tokens"]
    )

    fig_tokens = px.bar(
        token_df.sort_values("Total", ascending=True),
        x="Model",
        y=[
            "Input Tokens",
            "Output Tokens",
            "Extraction Input Tokens",
            "Extraction Output Tokens",
        ],
        title="Average Token Usage per Page by Model Combination",
        height=600,
        color_discrete_sequence=["#636EFA", "#EF553B", "#7B83FB", "#F76D57"],
    )

    fig_tokens.update_layout(
        barmode="stack",
        showlegend=True,
        legend_title="Token Type",
        yaxis=dict(
            title="Number of Tokens",
            range=[0, token_df["Total"].max() * 1.2],
        ),
    )
    fig_tokens.update_traces(texttemplate="%{y:.0f}", textposition="inside")
    st.plotly_chart(fig_tokens)

    # Detailed Results Table
    st.header("Test Results")
    df = create_results_table(results)
    st.dataframe(df)


if __name__ == "__main__":
    main()
