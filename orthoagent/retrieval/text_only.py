from llmsherpa.readers import LayoutPDFReader
from llama_index.core import Document
from llama_index.core import VectorStoreIndex
from llama_index.core.tools.query_engine import QueryEngineTool
from llama_index.core.tools.types import ToolMetadata
from ..utils.constants import DATA_DIR


llmsherpa_api_url = "https://readers.llmsherpa.com/api/document/developer/parseDocument?renderFormat=all"
pdf_urls = DATA_DIR.glob("*.pdf")


pdf_reader = LayoutPDFReader(llmsherpa_api_url)


index = VectorStoreIndex([])

for pdf_url in pdf_urls:
    print("chunking", pdf_url.stem)

    doc = pdf_reader.read_pdf(str(pdf_url))

    for i, chunk in enumerate(doc.chunks()):
        print(i)

        index.insert(Document(text=chunk.to_context_text(), extra_info={}))


query_engine = index.as_query_engine()

query_engine_tool = QueryEngineTool(
    query_engine=query_engine,
    metadata=ToolMetadata(
        name="text_only_query_engine",
        description="Tool for querying the reference documents with natural language.",
    ),
)
