from pathlib import Path
import sys

dir_path = Path(__file__).resolve().parent.parent
sys.path.append(str(dir_path))


from src.retrieval.populate_vector_store import (
    populate_vector_store_llamaparse,
    populate_vector_store_naive,
    populate_vector_store_llamaparse_multiple_docs,
    populate_vector_store_naive_multiple_docs,
    get_vector_store_llamaparse,
    get_vector_store_naive,
    get_vector_store_llamaparse_multiple_docs,
    get_vector_store_naive_multiple_docs,
)

from src.llm.openai import build_chat_openai


if __name__ == "__main__":
    # populate_vector_store_llamaparse()
    # populate_vector_store_naive()
    # populate_vector_store_llamaparse_multiple_docs(subset = ["OSSZimmer_Distal"])
    # populate_vector_store_naive_multiple_docs()
    


    idx = get_vector_store_llamaparse_multiple_docs()["OSSZimmer_Distal"]


    model = build_chat_openai()

    chat_engine = idx.as_chat_engine(llm=model)

    retriever = idx.as_retriever()

    print(retriever.retrieve("How long is the distal femoral component?"))
