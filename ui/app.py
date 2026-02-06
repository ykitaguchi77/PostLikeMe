"""PostLikeMe - Streamlit Web UI

メインエントリポイント。サイドバーナビゲーションとグローバル設定を提供。

起動方法:
    streamlit run ui/app.py
"""

import streamlit as st

# ---------------------------------------------------------------------------
# Page config (must be first Streamlit call)
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="PostLikeMe",
    page_icon="🐦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    /* Twitter-like card styling */
    .tweet-card {
        background: #15202b;
        color: #d9d9d9;
        border-radius: 16px;
        padding: 16px 20px;
        margin: 8px 0;
        border: 1px solid #38444d;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
        font-size: 15px;
        line-height: 1.5;
    }
    .tweet-card .username {
        color: #1da1f2;
        font-weight: bold;
        margin-bottom: 4px;
    }
    .tweet-card .score-badge {
        display: inline-block;
        background: #1da1f2;
        color: white;
        border-radius: 12px;
        padding: 2px 10px;
        font-size: 12px;
        margin-top: 8px;
    }
    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
    }
    .metric-card h3 {
        margin: 0;
        font-size: 14px;
        opacity: 0.9;
    }
    .metric-card .value {
        font-size: 32px;
        font-weight: bold;
    }
    /* Header */
    .main-header {
        text-align: center;
        padding: 20px 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Session state initialization
# ---------------------------------------------------------------------------
def init_session_state():
    """Initialize session state variables used across pages."""
    defaults = {
        "api_base_url": "http://localhost:8000",
        "selected_profile": None,
        "profiles_cache": [],
        "last_generated_tweets": [],
        "last_generated_replies": [],
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


init_session_state()

# ---------------------------------------------------------------------------
# Main page content
# ---------------------------------------------------------------------------
st.markdown('<div class="main-header">', unsafe_allow_html=True)
st.title("🐦 PostLikeMe")
st.markdown("**X (Twitter) アカウントを分析して、そっくりのツイートを生成する AI ツール**")
st.markdown("</div>", unsafe_allow_html=True)

st.divider()

# Quick start guide
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("### 1️⃣ 収集")
    st.markdown(
        "対象アカウントのツイートを収集します。"
        "API、スクレイピング、または CSV/JSON アーカイブから取り込めます。"
    )
    if st.button("ツイートを収集する →", key="nav_collect", use_container_width=True):
        st.switch_page("pages/1_collect.py")

with col2:
    st.markdown("### 2️⃣ 分析")
    st.markdown(
        "収集したツイートから文体・性格・トピック傾向を解析し、"
        "ボイスプロファイルを構築します。"
    )
    if st.button("アカウントを分析する →", key="nav_analyze", use_container_width=True):
        st.switch_page("pages/2_analyze.py")

with col3:
    st.markdown("### 3️⃣ 生成")
    st.markdown(
        "ボイスプロファイルを使って、そのアカウントらしい"
        "ツイートやリプライを AI で生成します。"
    )
    if st.button("ツイートを生成する →", key="nav_generate", use_container_width=True):
        st.switch_page("pages/4_generate_tweet.py")

st.divider()

# Status overview
st.subheader("📊 システム状態")

col_status1, col_status2, col_status3 = st.columns(3)

import httpx

try:
    resp = httpx.get(f"{st.session_state.api_base_url}/api/health", timeout=3.0)
    api_ok = resp.status_code == 200
except Exception:
    api_ok = False

with col_status1:
    if api_ok:
        st.success("✅ API サーバー: 稼働中")
    else:
        st.error("❌ API サーバー: 未起動")
        st.caption("起動コマンド: `uvicorn postlikeme.api.app:app --reload`")

with col_status2:
    try:
        resp = httpx.get(f"{st.session_state.api_base_url}/api/profiles", timeout=3.0)
        if resp.status_code == 200:
            profiles = resp.json()
            st.info(f"📁 保存済みプロファイル: **{len(profiles)}** 件")
        else:
            st.info("📁 保存済みプロファイル: 確認中...")
    except Exception:
        st.info("📁 保存済みプロファイル: API接続待ち")

with col_status3:
    st.info("🤖 LLM: 設定ファイルで API キーを確認")

# Sidebar
with st.sidebar:
    st.markdown("## ⚙️ 設定")
    api_url = st.text_input(
        "API サーバー URL",
        value=st.session_state.api_base_url,
        help="FastAPI バックエンドの URL",
    )
    if api_url != st.session_state.api_base_url:
        st.session_state.api_base_url = api_url
        st.rerun()

    st.divider()
    st.markdown("### ページ")
    st.page_link("pages/1_collect.py", label="📥 ツイート収集", icon="1️⃣")
    st.page_link("pages/2_analyze.py", label="🔬 分析", icon="2️⃣")
    st.page_link("pages/3_profile.py", label="👤 プロファイル閲覧", icon="3️⃣")
    st.page_link("pages/4_generate_tweet.py", label="✍️ ツイート生成", icon="4️⃣")
    st.page_link("pages/5_generate_reply.py", label="💬 リプライ生成", icon="5️⃣")
