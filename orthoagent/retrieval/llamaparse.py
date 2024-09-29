from llama_index.core import VectorStoreIndex
from llama_parse import LlamaParse
from llama_index.core.tools.query_engine import QueryEngineTool
from llama_index.core.tools.types import ToolMetadata
from llama_index.core.schema import TextNode
from llama_index.core.node_parser import MarkdownElementNodeParser
from llama_index.postprocessor.flag_embedding_reranker import FlagEmbeddingReranker
import nest_asyncio

from copy import deepcopy

from ..utils.constants import DATA_DIR
from ..utils.find_key import find_key
from ..llm.openai import build_chat_openai

nest_asyncio.apply()


parser = LlamaParse(
    api_key=find_key("llamacloud"),
    result_type="markdown",
    num_workers=4,
    verbose=True,
    language="en",
)

# pdf_urls = [str(path) for path in DATA_DIR.glob("*.pdf")]
pdf_urls = [
    "/Users/andrewyang/Desktop/research/AgenticRAG/orthoagent/reference_docs/OSSSegmentalDistalFemoralReplacementSurgicalTechnique01462GLBLenREV0316.pdf"
]


documents = parser.load_data(pdf_urls)

index = VectorStoreIndex([])


def get_page_nodes(docs, separator="\n---\n"):
    """Split each document into page node, by separator."""
    nodes = []
    for doc in docs:
        doc_chunks = doc.text.split(separator)
        for doc_chunk in doc_chunks:
            node = TextNode(
                text=doc_chunk,
                metadata=deepcopy(doc.metadata),
            )
            nodes.append(node)

    return nodes


page_nodes = get_page_nodes(documents)

node_parser = MarkdownElementNodeParser(llm=build_chat_openai(), num_workers=8)
nodes = node_parser.get_nodes_from_documents(documents)
base_nodes, objects = node_parser.get_nodes_and_objects(nodes)

recursive_index = VectorStoreIndex(nodes=base_nodes + objects + page_nodes)


reranker = FlagEmbeddingReranker(
    top_n=5,
    model="BAAI/bge-reranker-large",
)
recursive_query_engine = recursive_index.as_query_engine(
    similarity_top_k=5, node_postprocessors=[reranker], verbose=True
)


query_engine_tool = QueryEngineTool(
    query_engine=recursive_query_engine,
    metadata=ToolMetadata(
        name="text_only_query_engine",
        description="Tool for querying the reference documents.",
    ),
)
