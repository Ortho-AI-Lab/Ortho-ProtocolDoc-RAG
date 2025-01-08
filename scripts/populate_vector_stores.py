from pathlib import Path
import sys

dir_path = Path(__file__).resolve().parent.parent
sys.path.append(str(dir_path))


from agents_llamaindex.retrieval.populate_vector_store import (
    populate_vector_store_llamaparse,
    populate_vector_store_naive,
    populate_vector_store_llamaparse_multiple_docs,
    populate_vector_store_naive_multiple_docs,
    populate_vector_store_llamaparse_multiple_docs_multimodal,
    get_vector_store_llamaparse,
    get_vector_store_naive,
    get_vector_store_llamaparse_multiple_docs,
    get_vector_store_naive_multiple_docs,
    get_retriever_llamaparse_multiple_docs_multimodal,
)

from agents_llamaindex.llm.openai import build_chat_openai


if __name__ == "__main__":
    # populate_vector_store_llamaparse()
    # populate_vector_store_naive()
    # populate_vector_store_llamaparse_multiple_docs(subset = ["OSSZimmer_Distal"])
    # populate_vector_store_naive_multiple_docs()

    # populate_vector_store_llamaparse_multiple_docs_multimodal(
    #     subset=[
    #         # "OSSZimmer_Distal",
    #         # "Depuy_distal",
    #         "onko_distal",
    #         "stryker_cut",
    #     ]
    # )
    response = get_retriever_llamaparse_multiple_docs_multimodal()["stryker_cut"].query(
        "What is the maximum stem diameter?"
    )
    print(response)
