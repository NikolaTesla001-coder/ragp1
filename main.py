from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

embedding_model=HuggingFaceEmbeddings()

vectorstore=Chroma(persist_directory='chroma_db',embedding_function=embedding_model)

retriever=vectorstore.as_retriever(
    search_type='mmr',
    search_kwargs={
         'k' : 4,
         'fetch_k' : 10,#these top 10 will be searched from chunks using similarity,then out of these 10 4 will be selected using mmr
         'lambda_mult':0.5#defines diversity of results,0 means high diversity while 1 will give us almost SAME ans(0 diversity)
    }
)

llm=ChatGroq(model='openai/gpt-oss-120b')

prompt_template= ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a helpful AI assistant.

Use ONLY the provided context to answer the question.

If the answer is not present in the context,
say: "I could not find the answer in the document."
"""
        ),
        (
            "human",
            """Context:
{context}

Question:
{question}
"""
        )
    ]
)

print('rag system created')

print('press 0 to exit')

while True:
    query=input('you:-')
    if query==0:
        break
    docs=retriever.invoke(query)

    context='\n'.join(
        [doc.page_content for doc in docs]
    )

    final_prompt=prompt_template.invoke({
        'context':context,
        'question':query
    })

    response=llm.invoke(final_prompt)

    print(f'\n BOT:{response.content}')