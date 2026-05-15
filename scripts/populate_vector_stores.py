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
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.core import Settings

# embed_model = OpenAIEmbedding(model="text-embedding-3-large", dimensions=1536)
# Settings.embed_model = embed_model

llm = OpenAI("gpt-4o-2024-08-06")
Settings.llm = llm


if __name__ == "__main__":
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

    response = (
        get_vector_store_llamaparse_multiple_docs()["stryker_cut"]
        .as_query_engine(llm=build_chat_openai())
        .query("What is the maximum stem diameter?")
    )

    # response = get_retriever_llamaparse_multiple_docs_multimodal()["stryker_cut"].query(
    #     "What is the maximum stem diameter?"
    # )
    print(response)
