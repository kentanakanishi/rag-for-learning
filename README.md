# RAG Chat

LangChain + Azure OpenAI + Chroma を使った Streamlit ベースの RAG チャットアプリケーション。

URL やテキストからドキュメントを取り込み、その内容に基づいて対話的に質問応答ができます。

## セットアップ

### 前提条件

- Python 3.11+
- [uv](https://docs.astral.sh/uv/)
- Azure OpenAI リソース（Chat モデル + Embedding モデル）

### インストール

```bash
uv sync
```

### 環境変数の設定

`.env.example` をコピーして値を設定します。

```bash
cp .env.example .env
```

```
AZURE_OPENAI_API_KEY=your-api-key-here
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_CHAT_DEPLOYMENT=gpt-4o
AZURE_OPENAI_EMBED_DEPLOYMENT=text-embedding-3-small
AZURE_OPENAI_API_VERSION=2024-10-21
```

環境変数を設定しない場合は、アプリのサイドバーから直接入力できます。

## 起動

```bash
uv run streamlit run app.py
```

## 使い方

1. サイドバーで Azure OpenAI の認証情報を確認・入力
2. 「URL入力」タブまたは「テキスト貼付」タブからドキュメントを追加
3. チャット欄に質問を入力して送信
4. 回答の下にある「取得コンテキスト」を展開すると、参照されたチャンクを確認可能

## ファイル構成

| ファイル | 内容 |
|---|---|
| `app.py` | Streamlit UI（エントリーポイント） |
| `rag.py` | RAG パイプラインロジック |
| `tutorial.ipynb` | LangChain RAG チュートリアルノートブック |
