from pathlib import Path
import sys

dir_path = Path(__file__).resolve().parent.parent
sys.path.append(str(dir_path))


from src.retrieval.populate_vector_store import (
    populate_vector_store_llamaparse,
    populate_vector_store_naive,
)

from src.llm.openai import build_chat_openai


if __name__ == "__main__":
    index_smart = populate_vector_store_llamaparse()
    index_naive = populate_vector_store_naive()

    question = "When should a cement restrictor be used?"

    print(str(index_smart.as_query_engine(llm=build_chat_openai()).query(question)))
    print(str(index_naive.as_query_engine(llm=build_chat_openai()).query(question)))
