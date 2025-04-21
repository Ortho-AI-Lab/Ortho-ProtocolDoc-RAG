import nest_asyncio
from llama_parse import LlamaParse
from llama_index.core import VectorStoreIndex, Settings, load_index_from_storage
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core import StorageContext
from llama_index.core import SimpleDirectoryReader
from pathlib import Path
import qdrant_client
import re
from llama_index.core.schema import TextNode
import typing as t
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.core.query_engine import CustomQueryEngine
from llama_index.core.retrievers import BaseRetriever
from llama_index.multi_modal_llms.openai import OpenAIMultiModal
from llama_index.core.schema import NodeWithScore, MetadataMode
from llama_index.core.base.response.schema import Response
from llama_index.core.prompts import PromptTemplate
from llama_index.core.schema import ImageNode


from ..utils.constants import DATA_DIR
from ..utils.find_key import find_key

embed_model = OpenAIEmbedding(model="text-embedding-3-large", dimensions=1536)
# Settings.embed_model = embed_model

llm = OpenAI("gpt-4o-2024-08-06")
Settings.llm = llm


stem_to_company = {
    "Depuy_distal": "DePuy",
    "onko_distal": "Onkos",
    "OSSZimmer_Distal": "Zimmer Biomet",
    "stryker_cut": "Stryker",
}


company_to_document_title = {
    "DePuy": "Orthogenesis Limb Preservation System Surgical Techniques",
    "Onkos": "Distal Femoral Replacement Surgical Technique: Passive Fixed Hinge Tibia Option",
    "Zimmer Biomet": "Segmental Distal Femoral Replacement Surgical Technique",
    "Stryker": "GMRS Distal Femur and Revision Baseplate Surgical Protocol",
}


nest_asyncio.apply()


llamaparse_vector_store_path = (
    Path(__file__).resolve().parent / "_vector_store_llamaparse"
)
llamaparse_vector_store_path.mkdir(exist_ok=True)


llamaparse_vector_store_multiple_path = (
    Path(__file__).resolve().parent / "_vector_store_llamaparse_multiple_docs"
)
llamaparse_vector_store_multiple_path.mkdir(exist_ok=True)

llamaparse_vector_store_multiple_multimodal_path = (
    Path(__file__).resolve().parent / "_store_multiple_multimodal_docs"
)
llamaparse_vector_store_multiple_multimodal_path.mkdir(exist_ok=True)


naive_vector_store_path = Path(__file__).resolve().parent / "_vector_store_naive"
naive_vector_store_path.mkdir(exist_ok=True)


naive_vector_store_multiple_path = (
    Path(__file__).resolve().parent / "_vector_store_naive_multiple_docs"
)
naive_vector_store_multiple_path.mkdir(exist_ok=True)


def populate_vector_store_llamaparse() -> VectorStoreIndex:
    """Populate the vector store with LlamaParse and return the index."""

    client = qdrant_client.QdrantClient(path=llamaparse_vector_store_path)
    print(
        f"QdrantClient initialized with vector store path: {llamaparse_vector_store_path}"
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


def populate_vector_store_llamaparse_multiple_docs(subset=None) -> dict:
    """Populate the vector stores with LlamaParse,
    and return a dictionary of file stem to index."""

    # first, find all the files
    data_dir_path = DATA_DIR
    assert data_dir_path.exists()
    all_files = sorted(list(data_dir_path.glob("*.pdf")))
    print("Found", len(all_files), "PDF files:")
    for f in all_files:
        print(f"  {f}")

    # create a dictionary to store the indexes
    indexes = {}

    for file in all_files:
        # create a new client and vector store for each company
        if file.stem not in subset:
            continue

        client_path = llamaparse_vector_store_multiple_path / file.stem
        client_path.mkdir(exist_ok=True)

        client = qdrant_client.QdrantClient(path=client_path)

        if (client_path / "collection" / "llamaparse_text_store").exists():
            client.delete_collection("llamaparse_text_store")
            print("Deleted existing collection")

        print(f"Creating new collection and storage context for {file.stem}")
        vector_store = QdrantVectorStore(
            collection_name="llamaparse_text_store", client=client
        )

        storage_context = StorageContext.from_defaults(vector_store=vector_store)

        print(f"Begin population of vector store with LlamaParse for {file.stem}")
        parser = LlamaParse(
            api_key=find_key("llamacloud"),
            num_workers=4,
            language="en",
            result_type="markdown",
        )

        print(f"Parsing {file.stem} with LlamaParse")

        documents = parser.load_data(file_path=str(file))

        for document in documents:
            company = stem_to_company[file.stem]
            document.metadata["company_name"] = str(company)
            document.metadata["document_title"] = company_to_document_title[company]

        index = VectorStoreIndex.from_documents(
            documents=documents, storage_context=storage_context
        )

        indexes[file.stem] = index

        print(f"LlamaParse vector store populated for {file.stem}")

    return indexes


def get_vector_store_llamaparse_multiple_docs() -> dict[str, VectorStoreIndex]:
    """Returns a dictionary of file stem to index for the LlamaParse vector stores.
    Assumes the vector stores have already been populated.
    """
    data_dir_path = DATA_DIR
    assert data_dir_path.exists()
    all_files = sorted(list(data_dir_path.glob("*.pdf")))

    indexes = {}

    for file in all_files:
        client_path = llamaparse_vector_store_multiple_path / file.stem
        client = qdrant_client.QdrantClient(path=client_path)
        vector_store = QdrantVectorStore(
            collection_name="llamaparse_text_store", client=client
        )
        index = VectorStoreIndex.from_vector_store(vector_store=vector_store)
        indexes[file.stem] = index

    print(f"LlamaParse vector stores retrieved for {indexes.keys()}.")

    return indexes


def populate_vector_store_llamaparse_multiple_docs_multimodal(subset=None) -> dict:
    """Populate the vector stores with LlamaParse,
    and return a dictionary of file stem to index."""

    # first, find all the files
    data_dir_path = DATA_DIR
    assert data_dir_path.exists()
    all_files = sorted(list(data_dir_path.glob("*.pdf")))
    if subset:
        all_files = [f for f in all_files if f.stem in subset]
    print("Found", len(all_files), "PDF files:")
    for f in all_files:
        print(f"  {f}")

    # create a dictionary to store the indexes
    indexes = {}

    for file in all_files:
        # create a new client and vector store for each company
        stem_path = llamaparse_vector_store_multiple_multimodal_path / file.stem
        stem_path.mkdir(exist_ok=False)

        client_path = stem_path / "qdrant_client"
        client_path.mkdir(exist_ok=False)

        client = qdrant_client.QdrantClient(path=client_path)

        if (client_path / "collection" / "llamaparse_text_store").exists():
            client.delete_collection("llamaparse_text_store")
            print("Deleted existing collection")

        print(f"Creating new collection and storage context for {file.stem}")
        vector_store = QdrantVectorStore(
            collection_name="llamaparse_text_store", client=client
        )

        storage_context = StorageContext.from_defaults(vector_store=vector_store)

        print(f"Begin population of vector store with LlamaParse for {file.stem}")
        parser = LlamaParse(
            api_key=find_key("llamacloud"),
            result_type="markdown",
            parsing_instruction="You are given a medical device surgical technique document.",
            use_vendor_multimodal_model=True,
            vendor_multimodal_model_name="openai-gpt4o",
            show_progress=True,
        )

        print(f"Parsing {file.stem} with LlamaParse")

        md_json_objs = parser.get_json_result(str(file))
        md_json_list = md_json_objs[0]["pages"]

        data_imgs_path = stem_path / "data_images"
        data_imgs_path.mkdir(exist_ok=False)
        parser.get_images(md_json_objs, download_path=str(data_imgs_path))

        def get_page_number(file_name):
            """Gets page number of images using regex on file names"""
            match = re.search(r"-page-(\d+)\.jpg$", str(file_name))
            if match:
                return int(match.group(1))
            return 0

        def get_sorted_image_files(image_dir):
            """Get image files sorted by page."""
            raw_files = [f for f in list(Path(image_dir).iterdir()) if f.is_file()]
            sorted_files = sorted(raw_files, key=get_page_number)
            return sorted_files

        def get_text_nodes(json_dicts, image_dir) -> t.List[TextNode]:
            """Creates nodes from json + images"""

            nodes = []

            docs = [doc["md"] for doc in json_dicts]
            image_files = get_sorted_image_files(image_dir)

            for idx, doc in enumerate(docs):
                node = TextNode(
                    text=doc,
                    metadata={"image_path": str(image_files[idx]), "page_num": idx + 1},
                )
                nodes.append(node)
            return nodes

        text_nodes: list[TextNode] = get_text_nodes(md_json_list, data_imgs_path)
        for node in text_nodes:
            node.to_json()
        index = VectorStoreIndex(
            text_nodes, embed_model=embed_model, storage_context=storage_context
        )
        indexes[file.stem] = index
        print(f"LlamaParse vector store populated for {file.stem}")
    return indexes


def get_retriever_llamaparse_multiple_docs_multimodal(
    subset=None,
) -> dict[str, CustomQueryEngine]:
    """Returns a dictionary of file stem to index for the LlamaParse vector stores.
    Assumes the vector stores have already been populated.
    """
    data_dir_path = DATA_DIR
    assert data_dir_path.exists()
    all_files = sorted(list(data_dir_path.glob("*.pdf")))
    if subset:
        all_files = [f for f in all_files if f.stem in subset]

    retrievers = {}

    for file in all_files:
        stem_path = llamaparse_vector_store_multiple_multimodal_path / file.stem

        client_path = stem_path / "qdrant_client"
        client = qdrant_client.QdrantClient(path=client_path)
        vector_store = QdrantVectorStore(
            collection_name="llamaparse_text_store", client=client
        )
        index = VectorStoreIndex.from_vector_store(vector_store=vector_store)
        text_nodes = index.docstore.docs.values()
        QA_PROMPT_TMPL = """\
        Below we give parsed text from a PDF in two different formats, as well as the image.

        We parse the text in both 'markdown' mode as well as 'raw text' mode. Markdown mode attempts \
        to convert relevant diagrams into tables, whereas raw text tries to maintain the rough spatial \
        layout of the text.

        Use the image information first and foremost. ONLY use the text/markdown information 
        if you can't understand the image.

        ---------------------
        {context_str}
        ---------------------
        Given the context information and not prior knowledge, answer the query. Explain whether you got the answer
        from the parsed markdown or raw text or image, and if there's discrepancies, and your reasoning for the final answer.

        Query: {query_str}
        Answer: """
        QA_PROMPT = PromptTemplate(QA_PROMPT_TMPL)
        gpt_4o_mm = OpenAIMultiModal(
            model="gpt-4o", max_new_tokens=4096, api_key=find_key("openai")
        )

        class MultimodalQueryEngine(CustomQueryEngine):
            qa_prompt: PromptTemplate
            retriever: BaseRetriever
            multi_modal_llm: OpenAIMultiModal

            def __init__(
                self,
                qa_prompt: PromptTemplate,
                retriever: BaseRetriever,
                multi_modal_llm: OpenAIMultiModal,
            ):
                super().__init__(
                    qa_prompt=qa_prompt,
                    retriever=retriever,
                    multi_modal_llm=multi_modal_llm,
                )

            def custom_query(self, query_str: str):
                # retrieve most relevant nodes
                nodes = self.retriever.retrieve(query_str)

                # create image nodes from the image associated with those nodes
                image_nodes = [
                    NodeWithScore(
                        node=ImageNode(image_path=n.node.metadata["image_path"])
                    )
                    for n in nodes
                ]

                # create context string from parsed markdown text
                ctx_str = "\n\n".join(
                    [r.node.get_content(metadata_mode=MetadataMode.LLM) for r in nodes]
                )
                # prompt for the LLM
                fmt_prompt = self.qa_prompt.format(
                    context_str=ctx_str, query_str=query_str
                )

                # use the multimodal LLM to interpret images and generate a response to the prompt
                llm_repsonse = self.multi_modal_llm.complete(
                    prompt=fmt_prompt,
                    image_documents=[image_node.node for image_node in image_nodes],
                )
                return Response(
                    response=str(llm_repsonse),
                    source_nodes=nodes,
                    metadata={"text_nodes": text_nodes, "image_nodes": image_nodes},
                )

        query_engine = MultimodalQueryEngine(
            qa_prompt=QA_PROMPT,
            retriever=index.as_retriever(similarity_top_k=9),
            multi_modal_llm=gpt_4o_mm,
        )
        retrievers[file.stem] = query_engine

    return retrievers


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


def populate_vector_store_naive_multiple_docs() -> dict:
    """Populate the vector stores with naive text,
    and return a dictionary of file stem to index."""

    # first, find all the files
    data_dir_path = DATA_DIR
    assert data_dir_path.exists()
    all_files = sorted(list(data_dir_path.glob("*.pdf")))
    print("Found", len(all_files), "PDF files:")
    for f in all_files:
        print(f"  {f}")

    # create a dictionary to store the indexes
    indexes = {}

    for file in all_files:
        # create a new client and vector store for each company
        client_path = naive_vector_store_multiple_path / file.stem
        client_path.mkdir(exist_ok=True)

        client = qdrant_client.QdrantClient(path=client_path)

        if (client_path / "collection" / "naive_text_store").exists():
            client.delete_collection("naive_text_store")
            print("Deleted existing collection")

        print(f"Creating new collection and storage context for {file.stem}")
        vector_store = QdrantVectorStore(
            collection_name="naive_text_store", client=client
        )

        storage_context = StorageContext.from_defaults(vector_store=vector_store)

        print(f"Begin population of vector store with naive text for {file.stem}")

        documents = SimpleDirectoryReader(input_files=[file]).load_data()

        for document in documents:
            company = stem_to_company[file.stem]
            document.metadata["company_name"] = str(company)
            document.metadata["document_title"] = company_to_document_title[company]

        index = VectorStoreIndex.from_documents(
            documents=documents, storage_context=storage_context
        )

        indexes[file.stem] = index

        print(f"Naive text vector store populated for {file.stem}")

    return indexes


def get_vector_store_naive_multiple_docs() -> dict[str, VectorStoreIndex]:
    """Returns a dictionary of file stem to index for the naive text vector stores.
    Assumes the vector stores have already been populated.
    """
    data_dir_path = DATA_DIR
    assert data_dir_path.exists()
    all_files = sorted(list(data_dir_path.glob("*.pdf")))

    indexes = {}

    for file in all_files:
        client_path = naive_vector_store_multiple_path / file.stem
        client = qdrant_client.QdrantClient(path=client_path)
        vector_store = QdrantVectorStore(
            collection_name="naive_text_store", client=client
        )

        index = VectorStoreIndex.from_vector_store(vector_store=vector_store)
        indexes[file.stem] = index

    print(f"Naive text vector stores retrieved for {indexes.keys()}.")

    return indexes
