import os

from langchain_community.document_loaders import WebBaseLoader
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_chroma import Chroma
import chromadb
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


def get_embeddings(
    api_key: str,
    endpoint: str,
    deployment: str,
    api_version: str,
) -> AzureOpenAIEmbeddings:
    return AzureOpenAIEmbeddings(
        api_key=api_key,
        azure_endpoint=endpoint,
        azure_deployment=deployment,
        openai_api_version=api_version,
    )


def get_llm(
    api_key: str,
    endpoint: str,
    deployment: str,
    api_version: str,
) -> AzureChatOpenAI:
    return AzureChatOpenAI(
        api_key=api_key,
        azure_endpoint=endpoint,
        azure_deployment=deployment,
        openai_api_version=api_version,
    )


def load_and_split_url(url: str) -> list[Document]:
    loader = WebBaseLoader(web_paths=(url,))
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return splitter.split_documents(docs)


def load_and_split_text(text: str, source: str = "user_input") -> list[Document]:
    doc = Document(page_content=text, metadata={"source": source})
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return splitter.split_documents([doc])


def create_vector_store(
    documents: list[Document],
    embeddings: AzureOpenAIEmbeddings,
    host: str = "localhost",
    port: int = 8000,
    collection_name: str = "rag_collection",
) -> Chroma:
    # サーバーモードのクライアントを作成
    client = chromadb.HttpClient(host=host, port=port)
    
    return Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        client=client,
        collection_name=collection_name,
    )

def add_to_vector_store(
    vector_store: Chroma,
    documents: list[Document],
) -> None:
    vector_store.add_documents(documents=documents)


def query_rag(
    vector_store: Chroma,
    llm: AzureChatOpenAI,
    question: str,
) -> dict:
    retriever = vector_store.as_retriever(search_kwargs={"k": 4})

    prompt = ChatPromptTemplate.from_template(
        "以下のコンテキストに基づいて質問に答えてください。"
        "コンテキストに答えが見つからない場合は、その旨を伝えてください。\n\n"
        "コンテキスト:\n{context}\n\n"
        "質問: {question}"
    )

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    retrieved_docs = retriever.invoke(question)

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    answer = chain.invoke(question)

    return {
        "answer": answer,
        "context": retrieved_docs,
    }
