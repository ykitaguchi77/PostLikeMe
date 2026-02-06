"""💬 リプライ生成ページ

特定のツイートに対して、対象アカウントのスタイルでリプライを生成する。
元ツイートの文脈を理解し、ペルソナに合った返信を候補として提示。
"""

import streamlit as st
import httpx

st.set_page_config(page_title="PostLikeMe - リプライ生成", page_icon="💬", layout="wide")

st.title("💬 リプライ生成")
st.markdown("特定のツイートに対して、その人らしいリプライを生成します。")

if "api_base_url" not in st.session_state:
    st.session_state.api_base_url = "http://localhost:8000"
if "selected_profile" not in st.session_state:
    st.session_state.selected_profile = None

# ---------------------------------------------------------------------------
# Profile selection
# ---------------------------------------------------------------------------
profiles_list = []
try:
    resp = httpx.get(f"{st.session_state.api_base_url}/api/profiles", timeout=5.0)
    if resp.status_code == 200:
        profiles_list = resp.json()
except Exception:
    pass

if not profiles_list:
    st.warning("プロファイルがありません。先に「📥 収集」→「🔬 分析」を実行してください。")
    st.stop()

usernames = [p.get("username", "") for p in profiles_list]
default_idx = 0
if st.session_state.selected_profile in usernames:
    default_idx = usernames.index(st.session_state.selected_profile)

st.divider()

# ---------------------------------------------------------------------------
# Input section
# ---------------------------------------------------------------------------
col1, col2 = st.columns([2, 1])

with col1:
    selected_user = st.selectbox(
        "リプライするアカウント",
        usernames,
        index=default_idx,
        key="gen_reply_user",
    )

with col2:
    candidate_count = st.slider("候補数", min_value=1, max_value=10, value=5, key="reply_count")

# Target tweet input
st.markdown("### 返信先のツイート")
target_tweet = st.text_area(
    "リプライ先のツイート本文",
    placeholder="例: AIが全ての仕事を奪うと思いますか？個人的にはそうは思いませんが、大きな変化は避けられないでしょう。",
    height=120,
    help="リプライしたいツイートのテキストを貼り付けてください",
)

# Preview the target tweet
if target_tweet:
    st.markdown(
        f"""
        <div class="tweet-card" style="opacity: 0.8; border-left: 3px solid #38444d;">
            <div style="font-size: 13px; color: #8899a6;">返信先:</div>
            <div style="font-size: 15px;">{target_tweet}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

temperature = st.slider(
    "創造性 (Temperature)",
    min_value=0.1,
    max_value=1.5,
    value=0.85,
    step=0.05,
    key="reply_temp",
)

# ---------------------------------------------------------------------------
# Generate button
# ---------------------------------------------------------------------------
if st.button(
    "💬 リプライを生成",
    type="primary",
    use_container_width=True,
    disabled=not target_tweet,
):
    with st.spinner(f"@{selected_user} のスタイルでリプライを生成中..."):
        try:
            resp = httpx.post(
                f"{st.session_state.api_base_url}/api/generate/reply",
                json={
                    "username": selected_user,
                    "target_tweet": target_tweet,
                    "count": candidate_count,
                    "temperature": temperature,
                },
                timeout=120.0,
            )

            if resp.status_code == 200:
                data = resp.json()
                candidates = data.get("candidates", [])

                if candidates:
                    st.success(f"✅ {len(candidates)} 件のリプライ候補を生成しました！")

                    st.markdown("### 🏆 生成結果")

                    for i, candidate in enumerate(candidates, 1):
                        text = candidate.get("text", "")
                        score = candidate.get("score", 0.0)
                        pct = int(score * 100)

                        color = "#1da1f2" if pct >= 70 else "#ffad1f" if pct >= 40 else "#e0245e"
                        medal = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else f"#{i}"

                        with st.container(border=True):
                            # Show original + reply together
                            st.markdown(
                                f"""
                                <div class="tweet-card" style="opacity: 0.6; border-left: 3px solid #38444d; padding: 10px 16px;">
                                    <div style="font-size: 12px; color: #8899a6;">元のツイート:</div>
                                    <div style="font-size: 13px;">{target_tweet[:200]}{'...' if len(target_tweet) > 200 else ''}</div>
                                </div>
                                <div class="tweet-card" style="margin-top: -4px; border-left: 3px solid #1da1f2;">
                                    <div class="username">{medal} @{selected_user} のリプライ</div>
                                    <div style="font-size: 16px; margin: 8px 0;">{text}</div>
                                    <span class="score-badge" style="background:{color}">
                                        スタイル一致度: {pct}%
                                    </span>
                                    <span style="color: #8899a6; font-size: 12px; margin-left: 12px;">
                                        {len(text)} 文字
                                    </span>
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )
                            st.code(text, language=None)
                else:
                    st.warning("候補が生成されませんでした。")

            else:
                error = resp.json().get("detail", resp.text) if "json" in resp.headers.get("content-type", "") else resp.text
                st.error(f"生成エラー: {error}")

        except httpx.ConnectError:
            st.error(
                "API サーバーに接続できません。\n\n"
                "```\nuvicorn postlikeme.api.app:app --reload --port 8000\n```"
            )
        except httpx.ReadTimeout:
            st.warning("生成がタイムアウトしました。")
        except Exception as e:
            st.error(f"予期しないエラー: {e}")

# ---------------------------------------------------------------------------
# Sidebar tips
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 💡 リプライ生成のコツ")
    st.markdown(
        """
        **元ツイートの書き方**
        - 実際のツイート本文をそのまま貼り付け
        - 質問形式のツイートだと特に良い返信が生成される
        - 短すぎる元ツイートだと文脈が不足しがち

        **良いリプライの特徴**
        - プロファイルの口調・絵文字パターンと一致
        - 元ツイートの文脈に適切に反応
        - そのアカウントらしい視点・意見を反映
        """
    )

    st.divider()
    st.markdown("### 🔄 使い方の例")
    st.markdown(
        """
        1. 返信したいツイートをコピー
        2. テキストエリアに貼り付け
        3. 「リプライを生成」をクリック
        4. 最適な候補を選んでコピー
        """
    )
