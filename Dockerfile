FROM python:3.11-slim

# uvのインストール
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# 依存関係ファイルをコピー
COPY pyproject.toml uv.lock ./

# ライブラリインストール
RUN uv sync --frozen --no-dev

# アプリコードをコピー
COPY . .

# uvの仮想環境のPythonを使って起動
CMD ["uv", "run", "streamlit", "run", "app.py", \
     "--server.port=8501", "--server.address=0.0.0.0"]
