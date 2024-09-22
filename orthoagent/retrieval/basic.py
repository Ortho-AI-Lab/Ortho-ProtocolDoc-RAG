from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.tools.query_engine import QueryEngineTool
from llama_index.core.tools.types import ToolMetadata


from ..utils.constants import PACKAGE_DIR


reader = SimpleDirectoryReader(
    input_dir=PACKAGE_DIR / "reference_docs"
)
documents = reader.load_data(show_progress=True)
index = VectorStoreIndex.from_documents(
    documents=documents
)

retriever = index.as_retriever()
query_engine = index.as_query_engine(similarity_top_k=3)

query_engine_tool = QueryEngineTool(
    query_engine=query_engine,
    metadata=ToolMetadata(
        name="query_engine",
        description="A query engine that retrieves and "
        "synthesizes information from the reference documents.",
    )
)
