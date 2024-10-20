import nest_asyncio
from llama_parse import LlamaParse
from llama_index.core.node_parser import MarkdownElementNodeParser
from llama_index.core import VectorStoreIndex
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core import StorageContext
from llama_index.core import SimpleDirectoryReader
from pathlib import Path
import qdrant_client


from ..utils.constants import DATA_DIR
from ..utils.find_key import find_key
from ..llm.openai import build_chat_openai


nest_asyncio.apply()


llamaparse_vector_store_path = (
    Path(__file__).resolve().parent / "_vector_store_llamaparse"
)
llamaparse_vector_store_path.mkdir(exist_ok=True)

naive_vector_store_path = Path(__file__).resolve().parent / "_vector_store_naive"
naive_vector_store_path.mkdir(exist_ok=True)


def populate_vector_store_llamaparse() -> VectorStoreIndex:
    """Populate the vector store with LlamaParse and return the index."""

    client = qdrant_client.QdrantClient(path=llamaparse_vector_store_path)
    print(
        "QdrantClient initialized with vector store path:", llamaparse_vector_store_path
    )

    if (llamaparse_vector_store_path / "collection" / "llamaparse_text_store").exists():
        client.delete_collection("llamaparse_text_store")
        print("Deleted existing collection")

    print("Creating new collection and storage context")
    vector_store = QdrantVectorStore(
        collection_name="llamaparse_text_store", client=client
    )
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    print("Begin population of vector store with LlamaParse")
    parser = LlamaParse(
        api_key=find_key("llamacloud"),
        num_workers=4,
        language="en",
        fast_mode=True,
    )

    data_dir_path = DATA_DIR
    assert data_dir_path.exists()

    all_files = sorted(list(data_dir_path.glob("*.pdf")))
    print("Found", len(all_files), "PDF files:")
    for f in all_files:
        print(f"  {f}")
    all_files = [str(f) for f in all_files]

    print(f"Parsing {len(all_files)} files with LlamaParse")
    documents = parser.load_data(file_path=all_files)

    print("Converting parsed document markdown to nodes")
    node_parser = MarkdownElementNodeParser(llm=build_chat_openai(), num_workers=4)
    nodes = node_parser.get_nodes_from_documents(documents=documents)
    index = VectorStoreIndex.from_documents(
        documents=[], storage_context=storage_context
    )

    index.insert_nodes(nodes)

    print("LlamaParse vector store populated")

    return index


def get_vector_store_llamaparse() -> VectorStoreIndex:
    """Returns the LlamaParse vector store index.
    Assumes the vector store has already been populated.
    """
    client = qdrant_client.QdrantClient(path=llamaparse_vector_store_path)
    vector_store = QdrantVectorStore(
        collection_name="llamaparse_text_store", client=client
    )
    index = VectorStoreIndex.from_vector_store(vector_store=vector_store)
    return index


def populate_vector_store_naive() -> VectorStoreIndex:
    """Populate the vector store with naive text and return the index."""

    client = qdrant_client.QdrantClient(path=naive_vector_store_path)
    print("QdrantClient initialized with vector store path:", naive_vector_store_path)

    if (naive_vector_store_path / "collection" / "naive_text_store").exists():
        client.delete_collection("naive_text_store")
        print("Deleted existing collection")

    print("Creating new collection and storage context")
    vector_store = QdrantVectorStore(collection_name="naive_text_store", client=client)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    print("Begin population of vector store with naive text (SimpleDirectoryReader)")
    documents = SimpleDirectoryReader(input_dir=DATA_DIR).load_data()

    index = VectorStoreIndex.from_documents(
        documents=documents, storage_context=storage_context
    )

    print("Naive text vector store populated")

    return index


def get_vector_store_naive() -> VectorStoreIndex:
    """Returns the naive text vector store index.
    Assumes the vector store has already been populated.
    """
    client = qdrant_client.QdrantClient(path=naive_vector_store_path)
    vector_store = QdrantVectorStore(collection_name="naive_text_store", client=client)
    index = VectorStoreIndex.from_vector_store(vector_store=vector_store)
    return index
