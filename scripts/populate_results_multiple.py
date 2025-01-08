from llama_index.core.agent import FunctionCallingAgent, ReActAgent
from typing import Literal
from pydantic import BaseModel, Field
from llama_index.core.tools import FunctionTool, QueryEngineTool
from pathlib import Path
import sys
from functools import partial


dir_path = Path(__file__).resolve().parent.parent
sys.path.append(str(dir_path))


from agents_llamaindex.retrieval.populate_vector_store import (
    get_vector_store_naive_multiple_docs,
    get_vector_store_llamaparse_multiple_docs,
    stem_to_company,
    company_to_document_title,
)
from agents_llamaindex.llm.openai import build_chat_openai


llamaparse_vector_store_dict = get_vector_store_llamaparse_multiple_docs()
naive_vector_store_dict = get_vector_store_naive_multiple_docs()
top_k = 3


analysis_path = dir_path / "analysis"


agent_system_prompt = """You are a helpful assistant who can answer questions based on reference documents.
Use your tools to obtain information from the user-specified reference document.
Be concise. Do not answer in Markdown format, just plain text.
"""


def build_query_engine_tools(
    type: Literal["llamaparse", "naive"]
) -> dict[str, QueryEngineTool]:
    output = {}
    for key in llamaparse_vector_store_dict.keys():
        company = stem_to_company[key]
        title = company_to_document_title[company]
        if type == "llamaparse":
            vector_index = llamaparse_vector_store_dict[key]
            query_engine_tool = QueryEngineTool.from_defaults(
                query_engine=vector_index.as_query_engine(llm=build_chat_openai()),
                name=f"query_engine_tool_llamaparse_{key}",
                description=f"A query engine for the document titled {title} from {company}.",
            )
        elif type == "naive":
            vector_index = naive_vector_store_dict[key]
            query_engine_tool = QueryEngineTool.from_defaults(
                query_engine=vector_index.as_query_engine(llm=build_chat_openai()),
                name=f"query_engine_tool_naive_{key}",
                description=f"A query engine for the document titled {title} from {company}.",
            )
        else:
            raise ValueError("Invalid type. Must be 'llamaparse' or 'naive'.")
        output[key] = query_engine_tool
    return output


def build_retriever_tools(
    type: Literal["llamaparse", "naive"]
) -> dict[str, FunctionTool]:
    class _RetrieverInput(BaseModel):
        query: str = Field(
            description="Natural language query to be used for retrieval."
        )

    output_dict = {}

    def retrieve_from_llamaparse(query, key):
        llamaparse_retriever = llamaparse_vector_store_dict[key].as_retriever(
            similarity_top_k=top_k
        )
        retrieved_nodes = llamaparse_retriever.retrieve(query)
        output = ""
        for i, node in enumerate(retrieved_nodes):
            output += f"Retrieved Node {i+1}:\n\n"
            output += f"{node.text}\n\n"
            output += f"{str(node.metadata)}"
            if i != len(retrieved_nodes) - 1:
                output += "\n\n\n"
        return output

    def retrieve_from_naive(query, key):
        naive_retriever = naive_vector_store_dict[key].as_retriever(
            similarity_top_k=top_k
        )
        retrieved_nodes = naive_retriever.retrieve(query)
        output = ""
        for i, node in enumerate(retrieved_nodes):
            output += f"Retrieved Node {i+1}:\n\n"
            output += f"{node.text}\n\n"
            output += f"{str(node.metadata)}"
            if i != len(retrieved_nodes) - 1:
                output += "\n\n\n"
        return output

    for key in llamaparse_vector_store_dict.keys():
        company = stem_to_company[key]
        title = company_to_document_title[company]
        if type == "llamaparse":
            retriever_tool = FunctionTool.from_defaults(
                fn=partial(retrieve_from_llamaparse, key=key),
                name=f"retriever_tool_llamaparse_{key}",
                description=f"Retrieves the top {top_k} most similar nodes from the document titled {title} by {company}. Returns the text of the nodes in the order of relevance. Use this tool for citing the source of the answer.",
                fn_schema=_RetrieverInput,
            )
        elif type == "naive":
            retriever_tool = FunctionTool.from_defaults(
                fn=partial(retrieve_from_naive, key=key),
                name=f"retriever_tool_naive_{key}",
                description=f"Retrieves the top {top_k} most similar nodes from the document titled {title} by {company}. Returns the text of the nodes in the order of relevance. Use this tool for citing the source of the answer.",
                fn_schema=_RetrieverInput,
            )
        else:
            raise ValueError("Invalid type. Must be 'llamaparse' or 'naive'.")
        output_dict[key] = retriever_tool
    return output_dict


def build_agent_with_llamaparse_retriever():
    """Builds an agent that can retrieve the top three most similar nodes from the
    document and analyze them. Has no memory."""
    retriever_tools = [tool for tool in build_retriever_tools("llamaparse").values()]
    # query_tools = [tool for tool in build_query_engine_tools("llamaparse").values()]
    query_tools = []

    agent = FunctionCallingAgent.from_tools(
        tools=retriever_tools + query_tools,
        llm=build_chat_openai(),
        system_prompt=agent_system_prompt,
        verbose=True,
    )
    return agent


def build_agent_with_naive_retriever():
    """Builds an agent that can retrieve the top three most similar nodes from the
    document and analyze them. Has no memory."""
    retriever_tools = [tool for tool in build_retriever_tools("naive").values()]
    # query_tools = [tool for tool in build_query_engine_tools("naive").values()]
    query_tools = []
    agent = FunctionCallingAgent.from_tools(
        tools=retriever_tools + query_tools,
        llm=build_chat_openai(),
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
    naive_agent = build_agent_with_naive_retriever()

    document_stems = sorted(list(llamaparse_vector_store_dict.keys()))

    for document in document_stems:
        company_name = stem_to_company[document]
        document_title = company_to_document_title[company_name]

        document_title_wo_spaces = document_title.replace(" ", "_")

        results_dir = analysis_path / "results_multiple_docs" / document_title_wo_spaces
        results_dir.mkdir(parents=True, exist_ok=True)

        question_prefix = f"Based on the document titled {document_title} from {company_name}, answer this question:\n\n"

        questions = [question_prefix + question for question in base_questions]

        # assert False

        llamaparse_answers = answer_questions(llamaparse_agent, questions)

        with open(results_dir / "llamaparse_answers.txt", "w") as f:
            for i, (question, answer) in enumerate(zip(questions, llamaparse_answers)):
                f.write("QUESTION-ANSWER PAIR " + str(i + 1) + "\n\n")

                f.write("QUESTION:\n")
                f.write(question + "\n\n")

                f.write("ANSWER:\n")
                f.write(answer + "\n\n\n\n")

                f.write("---------------------------------------------------\n\n\n\n")

        # naive_answers = answer_questions(naive_agent, questions)

        # with open(results_dir / "naive_answers.txt", "w") as f:
        #     for i, (question, answer) in enumerate(zip(questions, naive_answers)):
        #         f.write("QUESTION-ANSWER PAIR " + str(i + 1) + "\n\n")

        #         f.write("QUESTION:\n")
        #         f.write(question + "\n\n")

        #         f.write("ANSWER:\n")
        #         f.write(answer + "\n\n\n\n")

        #         f.write("---------------------------------------------------\n\n\n\n")
