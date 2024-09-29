from llama_index.agent.openai import OpenAIAgent

from ..llm.openai import build_chat_openai


from ..retrieval.llamaparse import query_engine_tool as text_only_query_engine_tool


system_prompt = """You are a helpful assistant. 
You will be asked questions about the content of the documents, for which 
you have tools to help you find the answers.

Use the tools to find the answers to the questions.
If you cannot find the answer, you should say so.
"""


simple_agent = OpenAIAgent.from_tools(
    tools=[text_only_query_engine_tool],
    llm=build_chat_openai(),
    verbose=True,
    system_prompt=system_prompt,
)
