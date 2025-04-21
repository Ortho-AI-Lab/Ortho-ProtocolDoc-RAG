from llama_index.core.agent import FunctionCallingAgent
from typing import Literal
from llama_index.core.tools import QueryEngineTool
from pathlib import Path
import sys


dir_path = Path(__file__).resolve().parent.parent
sys.path.append(str(dir_path))


from agents_llamaindex.retrieval.populate_vector_store import (
    get_retriever_llamaparse_multiple_docs_multimodal,
    stem_to_company,
    company_to_document_title,
)
from agents_llamaindex.llm.openai import build_chat_openai


analysis_path = dir_path / "analysis"


agent_system_prompt = """\
You are a helpful assistant who can answer questions based on reference documents.
Use your tools to obtain information from the user-specified reference document.
Be concise. Do not answer in Markdown format, just plain text.
"""


retriever_dict = get_retriever_llamaparse_multiple_docs_multimodal()


def build_query_engine_tools() -> dict[str, QueryEngineTool]:
    output = {}
    for key in retriever_dict.keys():
        company = stem_to_company[key]
        title = company_to_document_title[company]
        query_engine_tool = QueryEngineTool.from_defaults(
            query_engine=retriever_dict[key],
            description="""\
Query engine for the document titled "{title}" from company "{company}".
Useful for running a natural language query against the document and get back a natural language response.
""".format(
                title=title, company=company
            ),
            name=f"query_engine_{key}",
        )
        output[key] = query_engine_tool
    return output


def build_agent_with_llamaparse_retriever():
    """Builds an agent that can retrieve the top three most similar nodes from the
    document and analyze them. Has no memory."""
    query_engine_tools = [tool for tool in build_query_engine_tools().values()]

    agent = FunctionCallingAgent.from_tools(
        tools=query_engine_tools,
        llm=build_chat_openai(model="gpt-4o-2024-08-06"),
        system_prompt=agent_system_prompt,
        verbose=True,
    )
    return agent


def read_questions(questions_path: Path) -> list[str]:
    """Reads questions from a file and returns a list of questions."""
    with open(questions_path, "r") as f:
        questions = f.readlines()
    questions = [question.strip() for question in questions]
    print("Read", len(questions), "questions from", questions_path)
    return questions


def answer_questions(agent: FunctionCallingAgent, questions: list[str]) -> list[str]:
    """Answers a list of questions using the agent.
    Returns a list of answers in the order the questions were asked.
    """
    answers = []
    for i, question in enumerate(questions):
        print("Answering question", i + 1, "of", len(questions))
        print("Question:", question)
        answer = str(agent.chat(question))
        answers.append(answer)
        print("Answer:", answer)
        print("\n\n")
        print("---------------------------------------------------\n\n")
    return answers


if __name__ == "__main__":
    base_questions = read_questions(analysis_path / "questions.txt")

    llamaparse_agent = build_agent_with_llamaparse_retriever()

    document_stems = sorted(list(retriever_dict.keys()))

    for document in document_stems:
        company_name = stem_to_company[document]
        document_title = company_to_document_title[company_name]

        document_title_wo_spaces = document_title.replace(" ", "_")

        results_dir = (
            analysis_path
            / "results_multiple_docs_multimodal"
            / document_title_wo_spaces
        )
        results_dir.mkdir(parents=True, exist_ok=True)

        question_prefix = f"Based on the document titled {document_title} from {company_name}, answer this question:\n\n"

        questions = [question_prefix + question for question in base_questions]

        llamaparse_answers = answer_questions(llamaparse_agent, questions)

        with open(results_dir / "llamaparse_answers.txt", "w") as f:
            for i, (question, answer) in enumerate(zip(questions, llamaparse_answers)):
                f.write("QUESTION-ANSWER PAIR " + str(i + 1) + "\n\n")

                f.write("QUESTION:\n")
                f.write(question + "\n\n")

                f.write("ANSWER:\n")
                f.write(answer + "\n\n\n\n")

                f.write("---------------------------------------------------\n\n\n\n")
