from llama_index.agent.openai import OpenAIAgent

from ..llm.openai import build_chat_openai


from ..retrieval.basic import query_engine_tool as basic_query_engine_tool

from ..retrieval.multimodal import query_engine_tool as multimodal_query_engine_tool


system_prompt = """
You are an expert orthopedic surgeon. 

You have access to reference documents that contain information about 
particular implants.

The user will ask you questions about information that can be found in the documents.

It is crucial to provide accurate information to the user. Provide citations 
to the reference documents when answering questions.

If you are unsure about the information, do not provide an answer. Simply 
state that you do not have enough information to answer the question.
"""


simple_agent = OpenAIAgent.from_tools(
    tools=[basic_query_engine_tool],
    llm=build_chat_openai(),
    verbose=True,
    system_prompt=system_prompt,
)


multimodal_agent = OpenAIAgent.from_tools(
    tools=[multimodal_query_engine_tool],
    llm=build_chat_openai(),
    verbose=True,
    system_prompt=system_prompt,

)