from llama_index.core.agent import FunctionCallingAgent, ReActAgent
from typing import Literal
from pydantic import BaseModel, Field
from llama_index.core.tools import FunctionTool, QueryEngineTool
from pathlib import Path
import sys

dir_path = Path(__file__).resolve().parent.parent
sys.path.append(str(dir_path))


from agents_llamaindex.retrieval.populate_vector_store import (
    get_vector_store_llamaparse,
    get_vector_store_naive,
)
from agents_llamaindex.llm.openai import build_chat_openai


llamaparse_vector_store = get_vector_store_llamaparse()
naive_vector_store = get_vector_store_naive()


llamaparse_query_engine = llamaparse_vector_store.as_query_engine()
naive_query_engine = naive_vector_store.as_query_engine()


llamaparse_retriever = llamaparse_vector_store.as_retriever(similarity_top_k=3)
naive_retriever = naive_vector_store.as_retriever(similarity_top_k=3)


analysis_path = dir_path / "analysis"


agent_system_prompt = """You are a helpful assistant who 
can answer questions based on the reference documents.
Use your tools to obtain information from the reference documents.
Be very informative in your answers.
If you cannot find the answer based on the reference documents, you can say so.
Answer in English.
"""


def build_query_engine_tool(type: Literal["llamaparse", "naive"]):
    if type == "llamaparse":
        query_engine_tool = QueryEngineTool.from_defaults(
            query_engine=llamaparse_query_engine,
            name="query_engine_tool",
            description="A RAG engine for relevant medical documents.",
        )
    elif type == "naive":
        query_engine_tool = QueryEngineTool.from_defaults(
            query_engine=naive_query_engine,
            name="query_engine_tool",
            description="A RAG engine for relevant medical documents.",
        )
    else:
        raise ValueError("Invalid type. Must be 'llamaparse' or 'naive'.")
    return query_engine_tool


def build_retriever_tool(type: Literal["llamaparse", "naive"]):
    class _RetrieverInput(BaseModel):
        query: str = Field(
            description="Natural language query to be used for retrieval."
        )

    def retrieve_from_llamaparse(query):
        retrieved_nodes = llamaparse_retriever.retrieve(query)
        output = ""
        for i, node in enumerate(retrieved_nodes):
            output += f"Retrieved Node {i+1}:\n\n"
            output += f"{node.text}\n\n"
            output += f"{str(node.metadata)}"
            if i != len(retrieved_nodes) - 1:
                output += "\n\n\n"
        return output

    def retrieve_from_naive(query):
        retrieved_nodes = naive_retriever.retrieve(query)
        output = ""
        for i, node in enumerate(retrieved_nodes):
            output += f"Retrieved Node {i+1}:\n\n"
            output += f"{node.text}\n\n"
            output += f"{str(node.metadata)}"
            if i != len(retrieved_nodes) - 1:
                output += "\n\n\n"
        return output

    if type == "llamaparse":
        retriever_tool = FunctionTool.from_defaults(
            fn=retrieve_from_llamaparse,
            name="retriever_tool",
            description="""Retrieves the top three most similar nodes from the 
            document. Returns the text of the nodes in the order they were
            retrieved. Use this tool for citing the source of the answer.""",
            fn_schema=_RetrieverInput,
        )
    elif type == "naive":
        retriever_tool = FunctionTool.from_defaults(
            fn=retrieve_from_naive,
            name="retriever_tool",
            description="""Retrieves the top three most similar nodes from the 
            document. Returns the text of the nodes in the order they were
            retrieved. Use this tool for citing the source of the answer.""",
            fn_schema=_RetrieverInput,
        )
    else:
        raise ValueError("Invalid type. Must be 'llamaparse' or 'naive'.")
    return retriever_tool


def build_agent_with_llamaparse_retriever():
    """Builds an agent that can retrieve the top three most similar nodes from the
    document and analyze them. Has no memory."""
    retriever_tool = build_retriever_tool("llamaparse")
    query_tool = build_query_engine_tool("llamaparse")
    agent = ReActAgent.from_tools(
        tools=[query_tool, retriever_tool],
        llm=build_chat_openai(),
        system_prompt=agent_system_prompt,
    )
    return agent


def build_agent_with_naive_retriever():
    """Builds an agent that can retrieve the top three most similar nodes from the
    document and analyze them. Has no memory."""
    retriever_tool = build_retriever_tool("naive")
    query_tool = build_query_engine_tool("naive")
    agent = FunctionCallingAgent.from_tools(
        tools=[query_tool, retriever_tool],
        llm=build_chat_openai(),
        system_prompt=agent_system_prompt,
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
        answer = str(agent.chat(question))
        answers.append(answer)
    return answers


if __name__ == "__main__":
    base_questions = read_questions(analysis_path / "questions.txt")

    llamaparse_agent = build_agent_with_llamaparse_retriever()
    naive_agent = build_agent_with_naive_retriever()

    documents = [
        "DePuy Orthogenesis Limb Preservation System Surgical Techniques",
        "Onkos Distal Femoral Replacement Surgical Technique: Passive Fixed Hinge Tibia Option",
        "Zimmer Biomet Segmental Distal Femoral Replacement Surgical Technique",
        "Stryker Triathlon TS Femur and Revision Baseplate",
    ]

    for document in documents:
        document_wo_spaces = document.replace(" ", "_")

        results_dir = analysis_path / "results" / document_wo_spaces
        results_dir.mkdir(parents=True, exist_ok=True)

        question_prefix = (
            f"Based on the {document} reference document, answer this " + "question: "
        )

        questions = [question_prefix + question for question in base_questions]

        # llamaparse_answers = answer_questions(llamaparse_agent, questions)

        # with open(results_dir / "llamaparse_answers.txt", "w") as f:
        #     for i, (question, answer) in enumerate(zip(questions, llamaparse_answers)):
        #         f.write("QUESTION-ANSWER PAIR " + str(i + 1) + "\n\n")

        #         f.write("QUESTION:\n")
        #         f.write(question + "\n\n")

        #         f.write("ANSWER:\n")
        #         f.write(answer + "\n\n\n\n")

        #         f.write("---------------------------------------------------\n\n\n\n")

        naive_answers = answer_questions(naive_agent, questions)

        with open(results_dir / "naive_answers.txt", "w") as f:
            for i, (question, answer) in enumerate(zip(questions, naive_answers)):
                f.write("QUESTION-ANSWER PAIR " + str(i + 1) + "\n\n")

                f.write("QUESTION:\n")
                f.write(question + "\n\n")

                f.write("ANSWER:\n")
                f.write(answer + "\n\n\n\n")

                f.write("---------------------------------------------------\n\n\n\n")
