"""📥 ツイート収集ページ

対象アカウントのツイートを収集し、ローカルキャッシュに保存する。
API / スクレイピング / CSV・JSONアーカイブの3つの収集方法をサポート。
"""

import streamlit as st
import httpx
import json
import csv
import io
from pathlib import Path

st.set_page_config(page_title="PostLikeMe - 収集", page_icon="📥", layout="wide")

st.title("📥 ツイート収集")
st.markdown("対象アカウントのツイートを収集して、分析の準備をします。")

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
if "api_base_url" not in st.session_state:
    st.session_state.api_base_url = "http://localhost:8000"

st.divider()

# ---------------------------------------------------------------------------
# Collection method tabs
# ---------------------------------------------------------------------------
tab_api, tab_archive = st.tabs(["🔑 API で収集", "📁 ファイルからインポート"])

# ---------------------------------------------------------------------------
# Tab 1: API Collection
# ---------------------------------------------------------------------------
with tab_api:
    st.markdown("### X API / スクレイピングで収集")
    st.caption("X API Bearer Token が必要です。設定で API キーを登録してください。")

    col1, col2 = st.columns([2, 1])
    with col1:
        username = st.text_input(
            "ユーザー名",
            placeholder="elonmusk",
            help="@ なしで入力してください",
            key="api_username",
        )
    with col2:
        collector_type = st.selectbox(
            "収集方法",
            options=["api", "scrape"],
            format_func=lambda x: "公式 API (tweepy)" if x == "api" else "スクレイピング (twscrape)",
        )

    col3, col4 = st.columns(2)
    with col3:
        tweet_count = st.slider("収集ツイート数", min_value=100, max_value=3000, value=500, step=100)
    with col4:
        include_replies = st.checkbox("リプライも含める", value=True)

    if st.button("🚀 収集開始", type="primary", use_container_width=True, key="btn_collect_api"):
        if not username:
            st.error("ユーザー名を入力してください。")
        else:
            with st.spinner(f"@{username} のツイートを収集中... ({tweet_count}件)"):
                try:
                    resp = httpx.post(
                        f"{st.session_state.api_base_url}/api/collect/{username}",
                        json={
                            "username": username,
                            "count": tweet_count,
                            "collector": collector_type,
                            "include_replies": include_replies,
                        },
                        timeout=120.0,
                    )
                    if resp.status_code == 200:
                        data = resp.json()
                        st.success(f"✅ @{username} のツイートを **{data.get('tweet_count', '?')}** 件収集しました！")
                        st.balloons()
                        st.info("次のステップ: 「🔬 分析」ページでプロファイルを構築しましょう。")
                    else:
                        st.error(f"エラー: {resp.status_code} - {resp.text}")
                except httpx.ConnectError:
                    st.error(
                        "API サーバーに接続できません。\n\n"
                        "以下のコマンドでサーバーを起動してください:\n"
                        "```\nuvicorn postlikeme.api.app:app --reload --port 8000\n```"
                    )
                except Exception as e:
                    st.error(f"予期しないエラー: {e}")

# ---------------------------------------------------------------------------
# Tab 2: Archive Import
# ---------------------------------------------------------------------------
with tab_archive:
    st.markdown("### ファイルからインポート")
    st.markdown(
        "Twitter の「データアーカイブ」(JSON) または CSV ファイルからツイートをインポートします。\n\n"
        "**CSV 形式**: `id, text, created_at, is_reply, reply_to_user, like_count, retweet_count, reply_count`"
    )

    archive_username = st.text_input(
        "プロファイル名（ユーザー名）",
        placeholder="myaccount",
        help="インポートしたデータに紐づけるユーザー名",
        key="archive_username",
    )

    uploaded_file = st.file_uploader(
        "ファイルを選択",
        type=["json", "csv"],
        help="Twitter データエクスポート (JSON) または CSV ファイル",
    )

    if uploaded_file is not None:
        file_type = uploaded_file.name.rsplit(".", 1)[-1].lower()

        # Preview
        st.markdown("#### プレビュー")
        content = uploaded_file.read()
        uploaded_file.seek(0)

        if file_type == "json":
            try:
                data = json.loads(content)
                if isinstance(data, list):
                    st.json(data[:3])
                    st.caption(f"合計 {len(data)} 件のツイート")
                elif isinstance(data, dict):
                    st.json({k: v for k, v in list(data.items())[:3]})
                else:
                    st.warning("不明な JSON 構造です。")
            except json.JSONDecodeError as e:
                st.error(f"JSON パースエラー: {e}")

        elif file_type == "csv":
            try:
                text_content = content.decode("utf-8")
                reader = csv.reader(io.StringIO(text_content))
                rows = list(reader)
                if rows:
                    st.dataframe(
                        {"Column": rows[0]} if len(rows) == 1 else
                        [dict(zip(rows[0], row)) for row in rows[1:6]],
                    )
                    st.caption(f"合計 {len(rows) - 1} 行（ヘッダー除く）")
            except Exception as e:
                st.error(f"CSV パースエラー: {e}")

    if st.button(
        "📂 インポート開始",
        type="primary",
        use_container_width=True,
        disabled=uploaded_file is None or not archive_username,
        key="btn_import",
    ):
        if not archive_username:
            st.error("ユーザー名を入力してください。")
        elif uploaded_file is None:
            st.error("ファイルを選択してください。")
        else:
            with st.spinner("ファイルをインポート中..."):
                try:
                    # Save uploaded file temporarily, then call API
                    tmp_path = Path(f"/tmp/postlikeme_upload_{archive_username}.{file_type}")
                    tmp_path.write_bytes(uploaded_file.read())

                    resp = httpx.post(
                        f"{st.session_state.api_base_url}/api/collect/{archive_username}",
                        json={
                            "username": archive_username,
                            "count": 10000,
                            "collector": "archive",
                            "include_replies": True,
                        },
                        timeout=60.0,
                    )
                    if resp.status_code == 200:
                        data = resp.json()
                        st.success(f"✅ インポート完了！ {data.get('tweet_count', '?')} 件")
                    else:
                        st.error(f"エラー: {resp.status_code} - {resp.text}")
                except httpx.ConnectError:
                    st.error("API サーバーに接続できません。サーバーを起動してください。")
                except Exception as e:
                    st.error(f"予期しないエラー: {e}")

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 💡 ヒント")
    st.markdown(
        """
        - **推奨ツイート数**: 500件以上で高品質な分析
        - **最低ツイート数**: 100件（基本的な分析のみ）
        - **最高品質**: 1,000件以上
        - リプライを含めると、リプライスタイルも分析可能
        """
    )

    st.divider()
    st.markdown("### 📊 収集方法の比較")
    st.markdown(
        """
        | 方法 | コスト | 安定性 |
        |------|--------|--------|
        | API | $200/月 | ⭐⭐⭐ |
        | スクレイピング | 無料 | ⭐⭐ |
        | アーカイブ | 無料 | ⭐⭐⭐ |
        """
    )
