import pandas as pd
import json
from pathlib import Path

curr_dir = Path(__file__).parent

models = [
    "Custom Naive",
    "Custom Multimodal",
    "LlamaParse",
    "LlamaParse Multimodal",
    "ChatGPT",
    "ChatGPT File Upload",
]
model_to_normalized_name = {model: f"Normalized {model}" for model in models}
normalized_name_to_model = {f"Normalized {model}": model for model in models}
models_normalized = [model_to_normalized_name[model] for model in models]


def extract_figure1_stats(df):
    grouped = df.groupby("Manufacturer")[models_normalized].mean()
    overall = df[models_normalized].mean().to_frame().T
    overall.index = ["Overall"]
    grouped = pd.concat([overall, grouped])
    grouped = grouped.rename(columns=normalized_name_to_model)
    grouped.index.name = "Document"

    # Convert to dictionary
    fig1_dict = {}
    for document in grouped.index:
        fig1_dict[document] = {}
        for model in models:
            fig1_dict[document][model] = round(grouped.loc[document, model], 3)
    return fig1_dict


def extract_figure2_stats(df, question_df):
    df = df.merge(
        question_df[["Question Type"]], left_on="Question", right_index=True, how="left"
    )
    grouped = df.groupby("Question Type")[models_normalized].mean()
    grouped = grouped.rename(columns=normalized_name_to_model)

    # Convert to dictionary
    fig2_dict = {}
    for qtype in grouped.index:
        fig2_dict[qtype] = {}
        for model in models:
            fig2_dict[qtype][model] = round(grouped.loc[qtype, model], 3)
    return fig2_dict


def main():
    df = pd.read_csv(curr_dir / "ProtocolDocRag-Results.csv", index_col=0)
    question_df = pd.read_csv(curr_dir / "ProtocolDocRag-Questions.csv", index_col=0)

    fig1_json = extract_figure1_stats(df)
    fig2_json = extract_figure2_stats(df, question_df)

    output = {"figure1": fig1_json, "figure2": fig2_json}

    outpath = curr_dir / "ProtocolDocRag-Stats.json"
    with open(outpath, "w") as f:
        json.dump(output, f, indent=2)

    print(f"Statistics exported to {outpath}")


if __name__ == "__main__":
    main()
