from llama_index.core.agent import (
    FunctionCallingAgent,
    FunctionCallingAgentWorker,
    AgentRunner,
)
from llama_index.core.memory import BaseMemory
from typing import Literal
from pydantic import BaseModel, Field
from llama_index.core.tools import FunctionTool
from pathlib import Path
import sys

dir_path = Path(__file__).resolve().parent.parent
sys.path.append(str(dir_path))


from src.retrieval.populate_vector_store import (
    get_vector_store_llamaparse,
    get_vector_store_naive,
)
from src.llm.openai import build_chat_openai


llamaparse_retriever = get_vector_store_llamaparse().as_retriever(similarity_top_k=3)
naive_retriever = get_vector_store_naive().as_retriever(similarity_top_k=3)


class NoMemory(BaseMemory):
    def put(self, message):
        pass

    def get(self, prompt=None):
        return []

    def reset(self):
        pass


analysis_path = dir_path / "analysis"


def build_retriever_tool(type: Literal["llamaparse", "naive"]):
    class _RetrieverInput(BaseModel):
        query: str = Field(
            description="Natural language query to be used for retrieval."
        )

    def retrieve_from_llamaparse(query):
        retrieved_nodes = llamaparse_retriever.retrieve(query)
        output = ""
        for i, node in enumerate(retrieved_nodes):
            output += f"Retrieved Node {i+1}:\n"
            output += f"{node.text}"
            if i != len(retrieved_nodes) - 1:
                output += "\n\n"
        return output

    def retrieve_from_naive(query):
        retrieved_nodes = naive_retriever.retrieve(query)
        output = ""
        for i, node in enumerate(retrieved_nodes):
            output += f"Retrieved Node {i+1}:\n"
            output += f"{node.text}"
            if i != len(retrieved_nodes) - 1:
                output += "\n\n"
        return output

    if type == "llamaparse":
        retriever_tool = FunctionTool.from_defaults(
            fn=retrieve_from_llamaparse,
            name="retriever_tool",
            description="""Retrieves the top three most similar nodes from the 
            document. Returns the text of the nodes in the order they were
            retrieved.""",
        )
    elif type == "naive":
        retriever_tool = FunctionTool.from_defaults(
            fn=retrieve_from_naive,
            name="retriever_tool",
            description="""Retrieves the top three most similar nodes from the 
            document. Returns the text of the nodes in the order they were
            retrieved.""",
        )
    else:
        raise ValueError("Invalid type. Must be 'llamaparse' or 'naive'.")
    return retriever_tool


def build_agent_with_llamaparse_retriever():
    """Builds an agent that can retrieve the top three most similar nodes from the
    document and analyze them. Has no memory."""
    retriever_tool = build_retriever_tool("llamaparse")
    agent = FunctionCallingAgent(
        tools=[retriever_tool],
        llm=build_chat_openai(),
        memory=NoMemory(),
    )
    return agent
