import os

import streamlit as st

from rag import (
    add_to_vector_store,
    create_vector_store,
    get_embeddings,
    get_llm,
    load_and_split_text,
    load_and_split_url,
    query_rag,
)

st.set_page_config(page_title="RAG Chat", layout="wide")
st.title("RAG Chat")

# --- Session state initialization ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "doc_count" not in st.session_state:
    st.session_state.doc_count = 0
if "chunk_count" not in st.session_state:
    st.session_state.chunk_count = 0

# --- Sidebar: Azure OpenAI settings ---
with st.sidebar:
    st.header("Azure OpenAI 設定")

    api_key = st.text_input(
        "API Key",
        value=os.environ.get("AZURE_OPENAI_API_KEY", ""),
        type="password",
    )
    endpoint = st.text_input(
        "Endpoint",
        value=os.environ.get("AZURE_OPENAI_ENDPOINT", ""),
    )
    chat_deployment = st.text_input(
        "Chat Deployment",
        value=os.environ.get("AZURE_OPENAI_CHAT_DEPLOYMENT", "gpt-4o"),
    )
    embed_deployment = st.text_input(
        "Embedding Deployment",
        value=os.environ.get("AZURE_OPENAI_EMBED_DEPLOYMENT", "text-embedding-3-small"),
    )
    api_version = st.text_input(
        "API Version",
        value=os.environ.get("AZURE_OPENAI_API_VERSION", "2024-10-21"),
    )

    st.divider()
    st.subheader("ドキュメント状態")
    st.metric("ドキュメント数", st.session_state.doc_count)
    st.metric("チャンク数", st.session_state.chunk_count)

# --- Validate settings ---
settings_ok = all([api_key, endpoint, chat_deployment, embed_deployment, api_version])

if not settings_ok:
    st.warning("サイドバーでAzure OpenAIの設定を入力してください。")


# --- Cached resources ---
@st.cache_resource
def cached_embeddings(_api_key, _endpoint, _deployment, _api_version):
    return get_embeddings(_api_key, _endpoint, _deployment, _api_version)


@st.cache_resource
def cached_llm(_api_key, _endpoint, _deployment, _api_version):
    return get_llm(_api_key, _endpoint, _deployment, _api_version)


# --- Document input ---
st.subheader("ドキュメント入力")
tab_url, tab_text = st.tabs(["URL入力", "テキスト貼付"])

with tab_url:
    url_input = st.text_input("URLを入力", placeholder="https://example.com/article")
    url_button = st.button("URLから追加", disabled=not settings_ok)

with tab_text:
    text_input = st.text_area("テキストを貼り付け", height=150)
    text_source = st.text_input("ソース名", value="user_input")
    text_button = st.button("テキストから追加", disabled=not settings_ok)

# --- Process document additions ---
if url_button and url_input:
    with st.spinner("URLからドキュメントを読み込み中..."):
        embeddings = cached_embeddings(api_key, endpoint, embed_deployment, api_version)
        chunks = load_and_split_url(url_input)
        if st.session_state.vector_store is None:
            st.session_state.vector_store = create_vector_store(chunks, embeddings)
        else:
            add_to_vector_store(st.session_state.vector_store, chunks)
        st.session_state.doc_count += 1
        st.session_state.chunk_count += len(chunks)
        st.success(f"{len(chunks)} チャンクを追加しました。")
        st.rerun()

if text_button and text_input:
    with st.spinner("テキストを処理中..."):
        embeddings = cached_embeddings(api_key, endpoint, embed_deployment, api_version)
        chunks = load_and_split_text(text_input, text_source)
        if st.session_state.vector_store is None:
            st.session_state.vector_store = create_vector_store(chunks, embeddings)
        else:
            add_to_vector_store(st.session_state.vector_store, chunks)
        st.session_state.doc_count += 1
        st.session_state.chunk_count += len(chunks)
        st.success(f"{len(chunks)} チャンクを追加しました。")
        st.rerun()

# --- Chat area ---
st.divider()

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "context" in msg:
            with st.expander("取得コンテキスト"):
                for i, doc in enumerate(msg["context"], 1):
                    st.markdown(f"**チャンク {i}** (source: {doc.metadata.get('source', 'N/A')})")
                    st.text(doc.page_content[:300])
                    st.divider()

if prompt := st.chat_input("質問を入力してください", disabled=not settings_ok):
    if st.session_state.vector_store is None:
        st.warning("先にドキュメントを追加してください。")
    else:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("回答を生成中..."):
                llm = cached_llm(api_key, endpoint, chat_deployment, api_version)
                result = query_rag(
                    st.session_state.vector_store, llm, prompt
                )
                st.markdown(result["answer"])
                with st.expander("取得コンテキスト"):
                    for i, doc in enumerate(result["context"], 1):
                        st.markdown(
                            f"**チャンク {i}** (source: {doc.metadata.get('source', 'N/A')})"
                        )
                        st.text(doc.page_content[:300])
                        st.divider()

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": result["answer"],
                "context": result["context"],
            }
        )
