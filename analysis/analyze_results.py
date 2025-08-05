import sys
from pathlib import Path
import pandas as pd

curr_dir = Path(__file__).resolve().parent
sys.path.append(str(curr_dir.parent))
import re
from agents_llamaindex.llm.openai import build_chat_openai


model = build_chat_openai(
    model="o3-mini-2025-01-31",
    temperature=0.0,
)


# load the ground truth
ground_truth = pd.read_csv(curr_dir / "ground_truth" / "model_answers.csv", index_col=0)
# read the questions.txt
with open(curr_dir / "questions.txt", "r") as f:
    questions = f.readlines()
questions = [question.strip() for question in questions]


prompt = """\
You are a grader for an exam. The exam consists of free response questions \
that require a test taker to extract relevant information from a surgical protocol \
document. \
For each question, you will assign a score from 1 to 3 based on a grading rubric. \
Note: not every question will have relevant information in the protocol document to 
answer it.\

Here is the grading rubric:
1: Incorrect or hallucination.
2: Partially correct or incomplete answer.
3: Correct, or correctly identifies a limitation.

Here is the question:
{}

Here is the correct/model answer:
{}

Here is the respondent's answer:
{}

Respond with a score from 1 to 3 based on the grading rubric, followed by a comma, \
followed by a brief explanation of the score. \

Example:
Question: What is the purpose of the study?
Model answer: The purpose of the study is to evaluate the effectiveness of a new drug.
Respondent's answer: The study is about a new drug.
Response: 3, The respondent correctly identifies the purpose of the study.
"""


company_to_document_title = {
    "DePuy": "Orthogenesis Limb Preservation System Surgical Techniques",
    "Onkos": "Distal Femoral Replacement Surgical Technique: Passive Fixed Hinge Tibia Option",
    "Zimmer Biomet": "Segmental Distal Femoral Replacement Surgical Technique",
    "Stryker": "GMRS Distal Femur and Revision Baseplate Surgical Protocol",
}
file_stems = [
    "Depuy_distal",
    "OSSZimmer_Distal",
    "stryker_cut",
]
stem_to_company = {
    "Depuy_distal": "DePuy",
    "onko_distal": "Onkos",
    "OSSZimmer_Distal": "Zimmer Biomet",
    "stryker_cut": "Stryker",
}


curr_analysis_dir = curr_dir / "results_multiple_docs"
assert curr_analysis_dir.exists(), f"Directory {curr_analysis_dir} does not exist."


output_dir = curr_dir / "meta_results" / curr_analysis_dir.stem
output_dir.mkdir(exist_ok=True)

model_type = "llamaparse"


def parse_question_answer_file(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        text = f.read()

    # Split by QUESTION-ANSWER PAIR
    pairs = re.split(r"-{10,}", text.strip())

    results = []

    for pair in pairs:
        # Extract Question Num
        qnum_match = re.search(r"QUESTION-ANSWER PAIR\s+(\d+)", pair)
        if not qnum_match:
            continue
        qnum = int(qnum_match.group(1))

        # Extract Question
        question_match = re.search(r"QUESTION:\n(.*?)\nANSWER:", pair, re.DOTALL)
        question = question_match.group(1).strip() if question_match else ""

        # Extract Answer
        answer_match = re.search(r"ANSWER:\n(.*)", pair, re.DOTALL)
        answer = answer_match.group(1).strip() if answer_match else ""

        results.append({"Question Num": qnum, "Question": question, "Answer": answer})

    return results


def parse_question_answer_file_v2(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        text = f.read()

    # Split by 17 or more dashes (matches the separators like '-----------------')
    pairs = re.split(r"-{17,}", text.strip())

    results = []

    for idx, pair in enumerate(pairs, 0):
        # Extract Question
        question_match = re.search(r"Question:\s*(.*?)\s*Answer:", pair, re.DOTALL)
        question = question_match.group(1).strip() if question_match else ""

        # Extract Answer
        answer_match = re.search(r"Answer:\s*(.*)", pair, re.DOTALL)
        answer = answer_match.group(1).strip() if answer_match else ""

        if question and answer:
            results.append(
                {
                    "Question Num": idx,  # no +1 anymore
                    "Question": question,
                    "Answer": answer,
                }
            )

    assert len(results) == 28

    return results


def parse_question_answer_file_v3(filepath):
    df = pd.read_csv(filepath, sep=",")
    question_nums_list = df["Question Num"].tolist()
    questions_list = df["Question"].tolist()
    answers_list = df["Answer"].tolist()

    results = []
    for i in range(len(question_nums_list)):
        results.append(
            {
                "Question Num": question_nums_list[i],
                "Question": questions_list[i],
                "Answer": answers_list[i],
            }
        )
    assert len(results) == 28
    return results


for file_stem in file_stems:
    print(f"Processing {file_stem}...")
    company_name = stem_to_company[file_stem]
    document_title = company_to_document_title[company_name].replace(" ", "_")

    local_output_dir = output_dir / document_title
    local_output_dir.mkdir(exist_ok=True)

    if model_type == "chatgpt":
        orig_results_dir = (
            curr_analysis_dir
            / document_title.replace(" ", "_")
            / f"{model_type}_answers.csv"
        )
    else:
        orig_results_dir = (
            curr_analysis_dir
            / document_title.replace(" ", "_")
            / f"{model_type}_answers.txt"
        )
    assert orig_results_dir.exists(), f"Directory {orig_results_dir} does not exist."
    if model_type == "openai":
        results = parse_question_answer_file_v2(orig_results_dir)
    elif model_type == "chatgpt":
        results = parse_question_answer_file_v3(orig_results_dir)
    else:
        results = parse_question_answer_file(orig_results_dir)

    scores = []
    rationales = []
    question_nums = []
    questions = []
    correct_answers = []
    answers = []

    for result in results:
        question_num = int(result["Question Num"])
        correct_answer = str(ground_truth.loc[question_num, company_name])
        answer = result["Answer"]
        question = result["Question"]
        question = re.sub(r"\s+", " ", question).strip()
        correct_answer = re.sub(r"\s+", " ", correct_answer).strip()

        print(
            f"Question {question_num}: {question}\nAnswer: {answer}\nCorrect Answer: {correct_answer}"
        )

        correct_answers.append(correct_answer)
        answers.append(answer)
        questions.append(question)
        question_nums.append(question_num)

        output = str(
            model.complete(
                prompt=prompt.format(
                    question,
                    correct_answer,
                    answer,
                )
            )
        )
        output = output.split(",")
        score = int(output[0].strip())
        rationale = ",".join(output[1:]).strip()
        scores.append(score)
        rationales.append(rationale)
        print(f"Score: {score}, Rationale: {rationale}\n\n")
    # save the results
    results_df = pd.DataFrame(
        {
            "Question Num": question_nums,
            "Question": questions,
            "Correct Answer": correct_answers,
            "Answer": answers,
            "Score": scores,
            "Rationale": rationales,
        }
    )
    results_df.to_csv(
        output_dir / local_output_dir / f"{model_type}_results.csv", index=False
    )
