import anthropic
import base64
from pathlib import Path
import os
import dotenv
import time

curr_dir = Path(__file__).resolve().parent
dotenv.load_dotenv(curr_dir.parent / ".env")
client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

dir_path = curr_dir.parent
analysis_path = dir_path / "analysis"


def read_questions(questions_path: Path) -> list[str]:
    """Reads questions from a file and returns a list of questions."""
    with open(questions_path, "r") as f:
        questions = f.readlines()
    questions = [question.strip() for question in questions]
    print("Read", len(questions), "questions from", questions_path)
    return questions


def main():
    base_questions = read_questions(analysis_path / "questions.txt")
    file_stems = [
        # "Depuy_distal",
        # "OSSZimmer_Distal",
        "stryker_cut",
    ]

    for filestem in file_stems:
        print(f"\n\n\n\nWorking on {filestem}.\n\n")
        pdf_path = (
            curr_dir.parent / "agents_llamaindex" / "reference_docs" / f"{filestem}.pdf"
        )
        with open(pdf_path, "rb") as pdf_file:
            pdf_data = base64.standard_b64encode(pdf_file.read()).decode("utf-8")

        results_dir = analysis_path / "results_anthropic" / filestem
        results_dir.mkdir(parents=True, exist_ok=True)
        results_path = results_dir / "anthropic_answers.txt"
        if results_path.exists():
            results_path.unlink()
        results_path.touch()

        for question in base_questions:
            prompt = question
            message = client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=1024,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "document",
                                "source": {
                                    "type": "base64",
                                    "media_type": "application/pdf",
                                    "data": pdf_data,
                                },
                                "cache_control": {"type": "ephemeral"},
                            },
                            {"type": "text", "text": prompt},
                        ],
                    }
                ],
            )
            answer = message.content[0].text

            with open(results_dir / "anthropic_answers.txt", "a") as f:
                f.write(f"Question: {prompt}\n")
                f.write(f"Answer: {answer}\n\n\n")

            print(f"Question: {prompt}")
            print(f"Answer: {answer}\n\n\n")

            time.sleep(60)


if __name__ == "__main__":
    main()
