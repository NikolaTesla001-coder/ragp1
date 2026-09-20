from langchain_community.document_loaders import WebBaseLoader

url='https://takeuforward.org/dsa/strivers-a2z-sheet-learn-dsa-a-to-z'
data=WebBaseLoader(url)
docs=data.load()
print(docs)