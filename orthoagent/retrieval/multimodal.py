from llama_index.core.query_engine import CustomQueryEngine
from llama_index.core import PromptTemplate
from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.multi_modal_llms.openai import OpenAIMultiModal
from llama_index.core.schema import ImageNode, TextNode, MetadataMode
from llama_index.core.response import Response
from typing import Optional
from llama_index.core import SimpleDirectoryReader
from llama_index.core import VectorStoreIndex
from llama_parse import LlamaParse
from llama_index.core.tools.query_engine import QueryEngineTool
from llama_index.core.tools.types import ToolMetadata
import nest_asyncio

from ..utils.find_key import find_key
from ..utils.constants import PACKAGE_DIR


nest_asyncio.apply()


multimodal_parser = LlamaParse(
    use_vendor_multimodal_model=True,
    vendor_multimodal_model_name="openai-gpt4o",
    vendor_multimodal_api_key=find_key("openai"),
)
file_extractor = {".pdf": multimodal_parser}
reader = SimpleDirectoryReader(
    input_dir=PACKAGE_DIR / "reference_docs", file_extractor=file_extractor
)
documents = reader.load_data()
index = VectorStoreIndex.from_documents(documents=documents)

retriever = index.as_retriever(retrieve_image_nodes=True)
QA_PROMPT = PromptTemplate(
    "We have provided context information below. \n"
    "---------------------\n"
    "{context_str}"
    "\n---------------------\n"
    "Given this information, please answer the question: {query_str}\n"
)


class MultimodalQueryEngine(CustomQueryEngine):
    """Custom multimodal Query Engine.

    Takes in a retriever to retrieve a set of document nodes.
    Also takes in a prompt template and multimodal model.
    """

    qa_prompt: PromptTemplate
    retriever: BaseRetriever
    multi_modal_llm: OpenAIMultiModal

    def __init__(self, qa_prompt: Optional[PromptTemplate] = None, **kwargs) -> None:
        """Initialize."""
        super().__init__(qa_prompt=qa_prompt or QA_PROMPT, **kwargs)

    def custom_query(self, query_str: str):
        # retrieve text nodes
        nodes = self.retriever.retrieve(query_str)
        img_nodes = [n for n in nodes if isinstance(n.node, ImageNode)]
        text_nodes = [n for n in nodes if isinstance(n.node, TextNode)]

        # create context string from text nodes, dump into the prompt
        context_str = "\\n\\n".join(
            [r.get_content(metadata_mode=MetadataMode.LLM) for r in nodes]
        )
        fmt_prompt = self.qa_prompt.format(context_str=context_str, query_str=query_str)

        # synthesize an answer from formatted text and images
        llm_response = self.multi_modal_llm.complete(
            prompt=fmt_prompt,
            image_documents=[n.node for n in img_nodes],
        )
        return Response(
            response=str(llm_response),
            source_nodes=nodes,
            metadata={"text_nodes": text_nodes, "image_nodes": img_nodes},
        )


query_engine = index.as_query_engine(similarity_top_k=3)

multimodal_query_engine = MultimodalQueryEngine(
    retriever=retriever,
    multi_modal_llm=OpenAIMultiModal(model="gpt-4o", api_key=find_key("openai")),
)

query_engine_tool = QueryEngineTool(
    query_engine=multimodal_query_engine,
    metadata=ToolMetadata(
        name="query_engine",
        description="A query engine that retrieves and "
        "synthesizes information from the reference documents.",
    ),
)
