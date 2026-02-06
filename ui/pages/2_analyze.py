"""🔬 分析ページ

収集済みツイートからボイスプロファイルを構築する。
分析の進捗をリアルタイム表示し、完了後にサマリーを表示。
"""

import streamlit as st
import httpx

st.set_page_config(page_title="PostLikeMe - 分析", page_icon="🔬", layout="wide")

st.title("🔬 アカウント分析")
st.markdown("収集済みのツイートを分析して、ボイスプロファイルを構築します。")

if "api_base_url" not in st.session_state:
    st.session_state.api_base_url = "http://localhost:8000"

st.divider()

# ---------------------------------------------------------------------------
# Profile selection / username input
# ---------------------------------------------------------------------------
col1, col2 = st.columns([2, 1])

with col1:
    username = st.text_input(
        "分析するユーザー名",
        placeholder="elonmusk",
        help="先に「📥 収集」ページでツイートを収集してください",
    )

with col2:
    rebuild = st.checkbox(
        "プロファイルを再構築",
        value=False,
        help="既存のプロファイルがあっても最初から分析し直す",
    )

# ---------------------------------------------------------------------------
# Analysis pipeline steps (display)
# ---------------------------------------------------------------------------
st.markdown("### 分析パイプライン")

steps = [
    ("1. 前処理", "リツイート除外、テキストクリーニング、絵文字タグ付け"),
    ("2. 構造分析", "語彙豊かさ、文構造、可読性、絵文字・ハッシュタグパターン"),
    ("3. トピック分析", "BERTopic/NMF によるトピック抽出、VADER センチメント"),
    ("4. 性格分析", "Big Five 性格推定、フォーマル度、ユーモアスタイル"),
    ("5. リプライ分析", "リプライパターン、トーン変化、エンゲージメントタイプ"),
    ("6. 例文選択", "Few-shot 用の代表的ツイートを MMR で選択"),
]

# Show steps in expander
with st.expander("分析ステップの詳細を見る", expanded=False):
    for step_name, step_desc in steps:
        st.markdown(f"**{step_name}**: {step_desc}")

# ---------------------------------------------------------------------------
# Run analysis
# ---------------------------------------------------------------------------
if st.button("🔬 分析開始", type="primary", use_container_width=True, disabled=not username):
    progress_bar = st.progress(0, text="分析を準備中...")
    status_text = st.empty()

    step_labels = [s[0] for s in steps]

    try:
        # Update progress for each step (simulated since API is single call)
        for i, (step_name, step_desc) in enumerate(steps):
            progress = int((i / len(steps)) * 80)
            progress_bar.progress(progress, text=f"{step_name}: {step_desc}")

            if i == 0:
                # Actually call the API at the first step
                status_text.info(f"@{username} の分析を開始しました...")

        # Call the analysis API
        resp = httpx.post(
            f"{st.session_state.api_base_url}/api/analyze/{username}",
            json={"username": username, "rebuild": rebuild},
            timeout=300.0,
        )

        progress_bar.progress(100, text="分析完了！")

        if resp.status_code == 200:
            data = resp.json()
            status_text.empty()

            st.success(f"✅ @{username} のボイスプロファイルを構築しました！")
            st.balloons()

            # Show summary
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                st.metric("分析ツイート数", data.get("tweet_count_analyzed", "?"))
            with col_b:
                st.metric("発見トピック数", data.get("topics_found", "?"))
            with col_c:
                st.metric("ステータス", "完了 ✅")

            st.info(
                "次のステップ:\n"
                "- 「👤 プロファイル閲覧」で詳細を確認\n"
                "- 「✍️ ツイート生成」でツイートを生成"
            )

        else:
            status_text.empty()
            error_detail = resp.json().get("detail", resp.text) if resp.headers.get("content-type", "").startswith("application/json") else resp.text
            st.error(f"分析エラー: {error_detail}")

    except httpx.ConnectError:
        progress_bar.empty()
        status_text.empty()
        st.error(
            "API サーバーに接続できません。\n\n"
            "```\nuvicorn postlikeme.api.app:app --reload --port 8000\n```"
        )
    except httpx.ReadTimeout:
        progress_bar.progress(90, text="分析に時間がかかっています...")
        st.warning("分析がタイムアウトしました。大量のツイートの場合は時間がかかります。")
    except Exception as e:
        progress_bar.empty()
        status_text.empty()
        st.error(f"予期しないエラー: {e}")

# ---------------------------------------------------------------------------
# Existing profiles
# ---------------------------------------------------------------------------
st.divider()
st.markdown("### 📁 既存プロファイル")

try:
    resp = httpx.get(f"{st.session_state.api_base_url}/api/profiles", timeout=5.0)
    if resp.status_code == 200:
        profiles = resp.json()
        if profiles:
            for p in profiles:
                with st.container(border=True):
                    cols = st.columns([2, 1, 1, 1])
                    cols[0].markdown(f"**@{p.get('username', '?')}**")
                    cols[1].caption(f"ツイート: {p.get('tweet_count_analyzed', '?')}")
                    cols[2].caption(f"構築日: {p.get('built_at', '?')[:10] if p.get('built_at') else 'N/A'}")
                    if cols[3].button("詳細 →", key=f"view_{p.get('username')}"):
                        st.session_state.selected_profile = p.get("username")
                        st.switch_page("pages/3_profile.py")
        else:
            st.info("まだプロファイルがありません。ツイートを収集して分析を実行してください。")
    else:
        st.warning("プロファイルの取得に失敗しました。")
except httpx.ConnectError:
    st.caption("API サーバーに接続して既存プロファイルを表示します。")
except Exception:
    st.caption("プロファイル情報の取得中...")

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 📋 分析要件")
    st.markdown(
        """
        | ツイート数 | 品質 |
        |-----------|------|
        | < 100 | ⚠️ 低品質 |
        | 200-499 | 🟡 中品質 |
        | 500-999 | 🟢 高品質 |
        | 1000+ | ⭐ 最高品質 |
        """
    )
    st.divider()
    st.markdown("### ⏱️ 所要時間の目安")
    st.markdown(
        """
        - 500 ツイート: ~30秒
        - 1000 ツイート: ~1分
        - BERTopic 使用時は追加で ~30秒
        """
    )
