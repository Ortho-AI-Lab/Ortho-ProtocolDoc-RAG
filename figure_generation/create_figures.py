import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path


curr_dir = Path(__file__).parent

sns.set_theme(context="paper", style="whitegrid")


models = ["Naive", "Multimodal", "LlamaParse", "LlamaParse Multimodal", "ChatGPT (Document Upload)", "ChatGPT"]
model_to_normalized_name = {model: f"Normalized {model}" for model in models}
normalized_name_to_model = {f"Normalized {model}": model for model in models}
models_normalized = [model_to_normalized_name[model] for model in models]


def fig1():
    """Grouped bar plot of overall and document-level accuracies."""
    df = pd.read_csv(curr_dir / "ProtocolDocRag-Results.csv", index_col=0)
    grouped = df.groupby("Manufacturer")[models_normalized].mean()

    overall = df[models_normalized].mean().to_frame().T
    overall.index = ["Overall"]

    grouped = pd.concat([overall, grouped])
    grouped = grouped.rename(columns=normalized_name_to_model)
    grouped.index.name = "Document"

    melted = grouped.reset_index().melt(
        id_vars="Document", var_name="LLM System", value_name="Accuracy"
    )

    fig, ax = plt.subplots(figsize=(7.8, 3))
    barplot = sns.barplot(
        data=melted, x="Document", y="Accuracy", hue="LLM System", ax=ax,
        legend=False
    )
    ax.set_xlabel("")
    ax.grid(False)

    for container in barplot.containers:
        barplot.bar_label(
            container, fmt="%.3f", label_type="edge", padding=2, fontsize=6
        )

    ax.set_ylim(ax.get_ylim()[0], ax.get_ylim()[1] + 0.05)
    # ax.legend(
    #     title="LLM System",
    #     bbox_to_anchor=(1.01, 0.5),
    #     loc="center left",
    #     fontsize=7,
    #     title_fontsize=8,
    # )

    fig.tight_layout()
    fig.savefig(curr_dir / "figures" / "fig1.png", dpi=300)


def fig2():
    """Bar plot of overall accuracies, grouped by question type (overall + each category)."""

    df = pd.read_csv(curr_dir / "ProtocolDocRag-Results.csv", index_col=0)
    question_df = pd.read_csv(curr_dir / "ProtocolDocRag-Questions.csv", index_col=0)
    df = df.merge(
        question_df[["Question Type"]], left_on="Question", right_index=True, how="left"
    )
    grouped = df.groupby("Question Type")[models_normalized].mean()

    # overall = df[models_normalized].mean().to_frame().T
    # overall.index = ["Overall"]
    # grouped = pd.concat([overall, grouped])

    grouped = grouped.rename(columns=normalized_name_to_model)
    grouped.index.name = "Question Type"

    melted = grouped.reset_index().melt(
        id_vars="Question Type", var_name="LLM System", value_name="Accuracy"
    )

    fig, ax = plt.subplots(figsize=(7, 3))
    barplot = sns.barplot(
        data=melted, x="Question Type", y="Accuracy", hue="LLM System", ax=ax
    )

    ax.set_xlabel("")
    ax.set_ylabel("Accuracy")
    ax.grid(False)

    ax.set_ylim(ax.get_ylim()[0], ax.get_ylim()[1] + 0.05)

    for container in barplot.containers:
        barplot.bar_label(
            container, fmt="%.3f", label_type="edge", padding=2, fontsize=6
        )

    ax.legend(
        title="LLM System",
        bbox_to_anchor=(1.01, 0.5),
        loc="center left",
        fontsize=7,
        title_fontsize=8,
    )

    fig.tight_layout()
    fig.savefig(curr_dir / "figures" / "fig2.png", dpi=300)


if __name__ == "__main__":
    fig1()
    fig2()
