import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sentence_transformers import SentenceTransformer, util
import torch
from io import StringIO

# Config
CSV_PATH = "15-01-2026-16-38community_545.csv"

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', None)

import os
os.makedirs("outputs/plots", exist_ok=True)

from jinja2 import Environment, FileSystemLoader

# HTML Report Generator
def generate_html_report(insights):
    env = Environment(loader=FileSystemLoader("templates"))
    template = env.get_template("report_template.html")

    html = template.render(**insights)

    with open("outputs/report.html", "w", encoding="utf-8") as f:
        f.write(html)

def save_boxplot(df, num_cols):
    plt.figure(figsize=(12, 8))
    sns.boxplot(data=df[num_cols])
    plt.xticks(rotation=45)
    plt.title("Metric Distribution Overview")
    plt.tight_layout()
    plt.savefig("outputs/plots/metric_boxplot.png")
    plt.close()


def save_histogram(df, col, filename, title):
    plt.figure(figsize=(10, 4))
    sns.histplot(df[col], kde=True, bins=20)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(f"outputs/plots/{filename}")
    plt.close()

# Insight generator
def generate_insights(df):
    insights = {}

    insights["precision_mean"] = round(df["Precision"].mean(), 2)
    insights["recall_mean"] = round(df["Recall"].mean(), 2)
    insights["reranker_mean"] = round(df["Reranker Score"].mean(), 2)

    insights["answer_relevancy_mean"] = round(df["answer_relevancy"].mean(), 2)
    insights["answer_correctness_mean"] = round(df["answer_correctness"].mean(), 2)


    # HALLUCINATION ANALYSIS
    hallucination_rows = df[
        (df["answer_relevancy"] > 0.8) & 
        (df["answer_correctness"] < 0.4)
    ]
    
    insights["hallucination_count"] = len(hallucination_rows)
    
    insights["hallucination_percentage"] = round(
        (insights["hallucination_count"] / len(df)) * 100, 2
    )

    # sample hallucination rows
    insights["hallucination_samples"] = (
        hallucination_rows[["user_input", "response", "reference", 
                            "answer_relevancy", "answer_correctness"]]
        .head(25)
        .to_html(classes="dataframe", index=False, border=0)
    )

    # Average metrics for hallucination cases
    if len(hallucination_rows) > 0:
        insights["hallucination_avg_relevancy"] = round(
            hallucination_rows["answer_relevancy"].mean(), 3
        )
        insights["hallucination_avg_correctness"] = round(
            hallucination_rows["answer_correctness"].mean(), 3
        )
    else:
        insights["hallucination_avg_relevancy"] = "N/A"
        insights["hallucination_avg_correctness"] = "N/A"

    # MISUNDERSTANDINGS ANALYSIS
    misunderstanding_rows = df[
        (df["answer_relevancy"] < 0.4) & 
        (df["answer_correctness"] > 0.8)
    ]

    insights["misunderstanding_count"] = len(misunderstanding_rows)
    insights["misunderstanding_percentage"] = round(
        (insights["misunderstanding_count"] / len(df)) * 100, 2
    )

    # Sample misunderstanding cases
    insights["misunderstanding_samples"] = (
        misunderstanding_rows[["user_input", "response", "reference",
                               "answer_relevancy", "answer_correctness"]]
        .head(5)
        .to_html(classes="dataframe", index=False, border=0)
    )

    # Average metrics
    if len(misunderstanding_rows) > 0:
        insights["misunderstanding_avg_relevancy"] = round(
            misunderstanding_rows["answer_relevancy"].mean(), 3
        )
        insights["misunderstanding_avg_correctness"] = round(
            misunderstanding_rows["answer_correctness"].mean(), 3
        )
    else:
        insights["misunderstanding_avg_relevancy"] = "N/A"
        insights["misunderstanding_avg_correctness"] = "N/A"
    
    # EDGE CASE ANALYSIS
    # Edge Case 1: Perfect recall but poor precision
    ec1 = df[(df["Recall"] == 1) & (df["Precision"] < 0.3)]
    insights["ec1_count"] = len(ec1)
    insights["ec1_description"] = "Perfect recall (Recall = 1.0) but poor precision (Precision < 0.3). Correct doc retrieved but many irrelevant docs also retrieved."
    insights["ec1_threshold"] = "Recall = 1.0 AND Precision < 0.3"

    # Edge Case 2: Perfect recall but poor reranker score
    ec2 = df[(df["Recall"] == 1) & (df["Reranker Score"] < 0.33)]
    insights["ec2_count"] = len(ec2)
    insights["ec2_description"] = "Correct document retrieved (Recall = 1.0) but ranked poorly by reranker (Reranker Score < 0.33)"
    insights["ec2_threshold"] = "Recall = 1.0 AND Reranker Score < 0.33"

    # Edge Case 3: Good retrieval but poor correctness
    ec3 = df[
        (df["Precision"] >= 0.5) & 
        (df["Recall"] == 1) & 
        (df["answer_correctness"] < 0.6)
    ]
    insights["ec3_count"] = len(ec3)
    insights["ec3_description"] = "Good retrieval metrics (Precision ≥ 0.5, Recall = 1.0) but model still generates incorrect answers (answer_correctness < 0.6)"
    insights["ec3_threshold"] = "Precision ≥ 0.5 AND Recall = 1.0 AND answer_correctness < 0.6"

    # Edge Case 4: Poor retrieval but good correctness
    ec4 = df[
        (df["Precision"] < 0.3) & 
        (df["Recall"] == 0) & 
        (df["answer_correctness"] > 0.7)
    ]
    insights["ec4_count"] = len(ec4)
    insights["ec4_description"] = "Failed to retrieve correct document (Recall = 0, Precision < 0.3) but answer is still correct (answer_correctness > 0.7) - possibly from model's pre-training"
    insights["ec4_threshold"] = "Precision < 0.3 AND Recall = 0 AND answer_correctness > 0.7"

    # Create summary table with thresholds
    edge_case_summary = pd.DataFrame({
        "Edge Case": ["EC1: High Recall, Low Precision", 
                      "EC2: High Recall, Low Reranker", 
                      "EC3: Good Retrieval, Low Correctness",
                      "EC4: Poor Retrieval, High Correctness"],
        "Count": [insights["ec1_count"], insights["ec2_count"], 
                  insights["ec3_count"], insights["ec4_count"]],
        "Thresholds": [insights["ec1_threshold"], insights["ec2_threshold"],
                       insights["ec3_threshold"], insights["ec4_threshold"]],
        "Description": [insights["ec1_description"], insights["ec2_description"],
                       insights["ec3_description"], insights["ec4_description"]]
    })

    insights["edge_case_summary_table"] = edge_case_summary.to_html(
        classes="dataframe", index=False, border=0
    )

    # Sample rows for each edge case 
    insights["ec1_samples"] = ec1[["user_input", "Precision", "Recall", 
                                    "Reranker Score", "answer_correctness"]].head(3).to_html(
        classes="dataframe", index=False, border=0
    )

    insights["ec2_samples"] = ec2[["user_input", "Precision", "Recall", 
                                    "Reranker Score", "answer_correctness"]].head(3).to_html(
        classes="dataframe", index=False, border=0
    )

    insights["ec3_samples"] = ec3[["user_input", "Precision", "Recall", 
                                    "Reranker Score", "answer_correctness"]].head(3).to_html(
        classes="dataframe", index=False, border=0
    )

    insights["ec4_samples"] = ec4[["user_input", "Precision", "Recall", 
                                    "Reranker Score", "answer_correctness"]].head(3).to_html(
        classes="dataframe", index=False, border=0
    )
    
    # RERANKER SCORE ANALYSIS
    reranker_summary = df.groupby("Reranker Score")[[
        "Precision", "Recall", "answer_correctness", "answer_relevancy"
    ]].mean().assign(
        count=df.groupby("Reranker Score").size()
    )
    
    insights["reranker_summary_table"] = reranker_summary.to_html(
        classes="dataframe", border=0
    )

    insights["data_preview"] = (
        df.head(5)
        .to_html(
            classes="dataframe",
            index=False,
            border=0
        )
    )

    insights["plots"] = [
    {"title": "Metric Distribution", "file": "metric_boxplot.png"},
    {"title": "Answer Relevancy Distribution (KDE)", "file": "relevancy_dist.png"},
    {"title": "Answer Relevancy Buckets (Pie)", "file": "relevancy_pie.png"},
    {"title": "Answer Relevancy Buckets (Bar)", "file": "relevancy_barplot.png"},  
    {"title": "Answer Correctness Distribution (KDE)", "file": "correctness_dist.png"},
    {"title": "Answer Correctness Buckets (Pie)", "file": "correctness_pie.png"},
    {"title": "Answer Correctness Buckets (Bar)", "file": "correctness_barplot.png"}, 
    {"title": "Metric Quality Buckets", "file": "metric_bucket_summary.png"},
    ]

    return insights
    
import io

def generate_eda_tables(df, num_cols):
    eda = {}

    # df.info()
    buffer = io.StringIO()
    df.info(buf=buffer)
    eda["df_info"] = buffer.getvalue()

    # Zero counts
    eda["zero_counts"] = (df[num_cols] == 0).sum().to_frame("Zero Count").to_html()

    # Null values
    null_counts_data = []
    for col in num_cols:
        null_count = df[col].isna().sum()
        null_pct = (null_count / len(df)) * 100
        null_counts_data.append({
            "Column": col,
            "Null Count": null_count,
            "Percentage": f"{null_pct:.2f}%"
        })
    
    null_df = pd.DataFrame(null_counts_data)
    eda["null_counts_table"] = null_df.to_html(classes="dataframe", index=False, border=0)
    
    # Keep specific answer_correctness count for backward compatibility
    eda["null_correctness"] = int(df["answer_correctness"].isna().sum())

    null_rows = df[df[num_cols].isna().any(axis=1)]
    
    if len(null_rows) > 0:
        # Select columns to display (you can adjust this)
        display_cols = ["user_input"] + num_cols  # Show question + all numeric cols
        
        # Limit to first 20 rows to keep report manageable
        eda["null_rows_sample"] = null_rows[display_cols].head(20).to_html(
            classes="dataframe", index=False, border=0
        )
        eda["total_null_rows"] = len(null_rows)
        eda["has_null_rows"] = True
    else:
        eda["null_rows_sample"] = "<p style='color: green;'>✅ No rows with null values found!</p>"
        eda["total_null_rows"] = 0
        eda["has_null_rows"] = False

    # Describe
    eda["describe"] = df[num_cols].describe().to_html()

    # Rows with null correctness (minimized later in HTML)
    eda["null_correctness_rows"] = (
        df[df["answer_correctness"].isna()]
        .head()
        .to_html(index=False)
    )

    return eda

# SentenceTransformer
model = SentenceTransformer("all-MiniLM-L6-v2")

# def compute_cosine_similarity(row):
#     emb1 = model.encode(str(row["response"]), convert_to_tensor=True)
#     emb2 = model.encode(str(row["reference"]), convert_to_tensor=True)
#     return util.cos_sim(emb1, emb2).item()

# Main
def main():
    # Load Data
    df = pd.read_csv(CSV_PATH)
    df = df.dropna(how="all")
    
    df["Recall"] = df["Recall"].astype(float)
    # Numeric columns
    num_cols = [
        "answer_relevancy",
        "answer_correctness",
        "Precision",
        "Recall",
        "Reranker Score",
        "F1 Score"
    ]
    
    # Zero & Null checks
    zero_counts = (df[num_cols] == 0).sum()
    print("Zero counts:\n", zero_counts)

    print("Null answer_correctness:", df["answer_correctness"].isna().sum())
    print(df[num_cols].describe())

    
    # Boxplot overview

    # plt.figure(figsize=(12, 8))
    # sns.boxplot(data=df[num_cols])
    # plt.xticks(rotation=45)
    # plt.title("Metric Distribution Overview")
    # plt.show()

    save_boxplot(df, num_cols)

    # Recall is almost always 1, indicating the correct source is usually retrieved.
    # Precision has high variance, showing many cases of noisy or extra retrievals.
    # Reranker score and F1 score vary widely.

    
    # Relevancy vs Correctness
    
    print(df[["answer_relevancy", "answer_correctness"]].describe())

    hallucinations = ((df["answer_relevancy"] > 0.8) & (df["answer_correctness"] < 0.4)).sum()
    print("Hallucinations:", hallucinations)

    misunderstandings = ((df["answer_relevancy"] < 0.4) & (df["answer_correctness"] > 0.8)).sum()
    print("Misunderstandings:", misunderstandings)

    
    # Bucketing
    df["relevancy_bucket"] = pd.cut(
        df["answer_relevancy"],
        bins=[-0.1, 0.3, 0.6, 0.9, 1.01],
        labels=["Low", "Medium", "High", "Excellent"]
    )
    df["correctness_bucket"] = pd.cut(
        df["answer_correctness"],
        bins=[-0.1, 0.3, 0.6, 0.9, 1.01],
        labels=["Low", "Medium", "High", "Excellent"]
    )

    plt.figure(figsize=(6, 6))
    bucket_counts = df["relevancy_bucket"].value_counts()
    labels_with_counts = [f'{label}\n({count})' for label, count in zip(bucket_counts.index, bucket_counts.values)]
    plt.pie(
        bucket_counts,
        labels=labels_with_counts,
        autopct="%1.1f%%",
        startangle=90
    )
    plt.title("Answer Relevancy Distribution")
    plt.ylabel("")
    plt.tight_layout()
    plt.savefig("outputs/plots/relevancy_pie.png")
    plt.close()

    
    plt.figure(figsize=(6, 6))
    bucket_counts = df["correctness_bucket"].value_counts()
    labels_with_counts = [f'{label}\n({count})' for label, count in zip(bucket_counts.index, bucket_counts.values)]
    plt.pie(
        bucket_counts,
        labels=labels_with_counts,
        autopct="%1.1f%%",
        startangle=90
    )
    plt.title("Answer Correctness Distribution")
    plt.ylabel("")
    plt.tight_layout()
    plt.savefig("outputs/plots/correctness_pie.png")
    plt.close()

    def save_barplot_with_freq(df, col, bucket_col, filename, title):
    
        plt.figure(figsize=(10, 6))
        
        # Count the buckets
        bucket_counts = df[bucket_col].value_counts().sort_index()
        
        # Create bar plot
        ax = sns.countplot(data=df, x=bucket_col, order=bucket_counts.index)
        
        # Add count labels on top of bars
        for container in ax.containers:
            ax.bar_label(container, fmt='%d')
        
        plt.title(title)
        plt.xlabel("Range")
        plt.ylabel("Frequency")
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(f"outputs/plots/{filename}")
        plt.close()

    

    # Bar plots for relevancy and correctness
    save_barplot_with_freq(
        df, 
        "answer_relevancy", 
        "relevancy_bucket",
        "relevancy_barplot.png",
        "Answer Relevancy Distribution (Frequency)"
    )
    save_barplot_with_freq(
        df,
        "answer_correctness",
        "correctness_bucket", 
        "correctness_barplot.png",
        "Answer Correctness Distribution (Frequency)"
    )
    save_histogram(
        df,
        "answer_relevancy",
        "relevancy_dist.png",
        "Answer Relevancy Distribution"
    )
    save_histogram(
        df,
        "answer_correctness",
        "correctness_dist.png",
        "Answer Correctness Distribution"
    )
    # Metric thresholds
    metrics = [
        "Precision", "Recall", "Reranker Score",
        "F1 Score", "answer_relevancy", "answer_correctness"
    ]
    high_counts = (df[metrics] > 0.8).sum()
    low_counts = (df[metrics] < 0.3).sum()

    print("High (>0.8):\n", high_counts)
    print("Low (<0.3):\n", low_counts)

    # Summary table    
    summary = pd.DataFrame({
        "High (>0.8)": (df[metrics] > 0.8).sum(),
        "Mid (0.3–0.8)": df[metrics].apply(lambda c: c.between(0.3, 0.8).sum()),
        "Low (<0.3)": (df[metrics] < 0.3).sum()
    })
    summary.plot(
        kind="bar",
        figsize=(12, 5),
        color=["green", "orange", "red"]
    )
    plt.title("High / Mid / Low Value Distribution Across Metrics")
    plt.ylabel("Count")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("outputs/plots/metric_bucket_summary.png")
    plt.close()

    
    # SentenceTransformer correctness
    
    # df["cosine_similarity"] = df.apply(compute_cosine_similarity, axis=1)
    # df["diff"] = df["answer_relevancy"] - df["cosine_similarity"]

    # Edge cases
    ec1 = df[(df["Recall"] == 1) & (df["Precision"] < 0.3)]
    ec2 = df[(df["Recall"] == 1) & (df["Reranker Score"] < 0.33)]
    ec4 = df[(df["Precision"] >= 0.5) & (df["Recall"] == 1) & (df["answer_correctness"] < 0.6)]
    ec5 = df[(df["Precision"] < 0.3) & (df["Recall"] == 0) & (df["answer_correctness"] > 0.7)]

    print("EC1:", len(ec1))
    print("EC2:", len(ec2))
    print("EC4:", len(ec4))
    print("EC5:", len(ec5))
    # Reranker summary
    
    reranker_summary = df.groupby("Reranker Score")[[
        "Precision", "Recall", "answer_correctness", "answer_relevancy"
    ]].mean().assign(
        count=df.groupby("Reranker Score").size()
    )
    print(reranker_summary)
    # HTML Report
    insights = generate_insights(df)
    eda_tables = generate_eda_tables(df, num_cols)
    insights.update(eda_tables)
    insights["csv_filename"] = CSV_PATH
    generate_html_report(insights)


    print("HTML report generated at outputs/report.html")
# Entry point
if __name__ == "__main__":
    main()
