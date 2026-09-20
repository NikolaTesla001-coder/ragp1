from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter

text_splitter = CharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=10
)

data=PyPDFLoader('documents_loaders/MT646 - Unit I.pdf')
docs=data.load()
chunks=text_splitter.split_documents(docs)

for chunk in chunks:
    print(chunk.page_content)
    print()