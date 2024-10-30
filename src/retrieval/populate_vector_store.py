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


stem_to_company = {
    "DPY_LPS_Limb_Preservation_Sys_Surgtech_0612-36-500r2": "DePuy",
    "Minimally-Invasive-Grower-Custom-Distal-Femoral-Surgical-Technique-Passive-Fixed-Hinge-Tibia": "Onkos",
    "OSSSegmentalDistalFemoralReplacementSurgicalTechnique01462GLBLenREV0316": "Zimmer Biomet",
    "THS_SP_1": "Stryker",
}


company_to_document_title = {
    "DePuy": "Orthogenesis Limb Preservation System Surgical Techniques",
    "Onkos": "Distal Femoral Replacement Surgical Technique: Passive Fixed Hinge Tibia Option",
    "Zimmer Biomet": "Segmental Distal Femoral Replacement Surgical Technique",
    "Stryker": "Triathlon TS Femur and Revision Baseplate",
}


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
        result_type="markdown",
    )

    data_dir_path = DATA_DIR
    assert data_dir_path.exists()

    all_files = sorted(list(data_dir_path.glob("*.pdf")))
    print("Found", len(all_files), "PDF files:")
    for f in all_files:
        print(f"  {f}")
    all_files = [str(f) for f in all_files]

    print(f"Parsing {len(all_files)} files with LlamaParse")

    documents = []

    for file in all_files:
        print(f"  Parsing {file}")
        documents_to_add = parser.load_data(file_path=file)
        for document in documents_to_add:
            company = stem_to_company[Path(file).stem]
            document.metadata["company_name"] = str(company)
            document.metadata["document_title"] = company_to_document_title[company]
        documents.extend(documents_to_add)

    index = VectorStoreIndex.from_documents(
        documents=documents, storage_context=storage_context
    )

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

    data_dir_path = DATA_DIR
    assert data_dir_path.exists()
    all_files = sorted(list(data_dir_path.glob("*.pdf")))

    documents = []

    for file in all_files:
        documents_to_add = SimpleDirectoryReader(input_files=[file]).load_data()
        for document in documents_to_add:
            company = stem_to_company[file.stem]
            document.metadata["company_name"] = str(company)
            document.metadata["document_title"] = company_to_document_title[company]
        documents.extend(documents_to_add)

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
