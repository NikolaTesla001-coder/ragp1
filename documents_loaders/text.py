from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter


data=TextLoader('documents_loaders/notes.txt')

text_splitter = CharacterTextSplitter(
    separator='',
    chunk_size=10,
    chunk_overlap=1
)
docs=data.load()

chunks=text_splitter.split_documents(docs)


for i in chunks:
    print(i.page_content)
    print()