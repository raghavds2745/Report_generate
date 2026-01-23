import os
import pandas as pd


METRICS_CSV_PATH = r"replay_files\20-01-2026-10-52community_545_metrics.csv"
RAGAS_CSV_PATH = r"replay_files\20-01-2026-10-52community_545_metrics_ragas.csv"

OUTPUT_DIR = "merged_csvs"
OUTPUT_FILE_NAME = "merged_metrics.csv"

def main():
    print(f"Reading metrics CSV: {METRICS_CSV_PATH}")
    df_metrics = pd.read_csv(METRICS_CSV_PATH, encoding="utf-8")

    print(f"Reading ragas CSV: {RAGAS_CSV_PATH}")
    df_ragas = pd.read_csv(RAGAS_CSV_PATH, encoding="utf-8")

    # Normalize column names
    df_ragas = df_ragas.rename(columns={
        "user_input": "Question",
        "expected_output": "reference",
        "response": "response"
    })
    ragas_cols = [
        "Question",
        "answer_relevancy",
        "answer_correctness"
    ]
    # ragas_cols = [
    #     "Question",
    #     "answer_relevancy",
    #     "answer_correctness",
    #     "answer_relevancy_reason",
    #     "answer_correctness_reason"
    # ]

    print("Merging dataframes on 'Question' ...")
    df_merged = pd.merge(
        df_metrics,
        df_ragas[ragas_cols],
        on="Question",
        how="inner"
    )

    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    output_path = os.path.join(OUTPUT_DIR, OUTPUT_FILE_NAME)
    df_merged.to_csv(output_path, index=False, encoding="utf-8")

    print(f"Merged CSV saved at: {output_path}")
    print(f"Final merged shape: {df_merged.shape}")


if __name__ == "__main__":
    main()
