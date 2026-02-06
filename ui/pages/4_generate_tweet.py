"""✍️ ツイート生成ページ

ボイスプロファイルを使って、対象アカウントのスタイルでツイートを生成する。
複数候補を生成し、スタイル一致度スコアで順位付けして表示。
"""

import streamlit as st
import httpx

st.set_page_config(page_title="PostLikeMe - ツイート生成", page_icon="✍️", layout="wide")

st.title("✍️ ツイート生成")
st.markdown("ボイスプロファイルを使って、その人らしいツイートを AI で生成します。")

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
# Generation settings
# ---------------------------------------------------------------------------
col1, col2 = st.columns([2, 1])

with col1:
    selected_user = st.selectbox(
        "スタイルを真似るアカウント",
        usernames,
        index=default_idx,
        key="gen_tweet_user",
    )

with col2:
    candidate_count = st.slider("候補数", min_value=1, max_value=10, value=5)

col3, col4 = st.columns(2)

with col3:
    topic = st.text_input(
        "トピック（任意）",
        placeholder="例: AI の未来, 朝のルーティン, プログラミング",
        help="空欄の場合、その人が普段ツイートしそうなトピックで生成します",
    )

with col4:
    temperature = st.slider(
        "創造性 (Temperature)",
        min_value=0.1,
        max_value=1.5,
        value=0.85,
        step=0.05,
        help="高い = よりクリエイティブ / 低い = よりプロファイルに忠実",
    )

# ---------------------------------------------------------------------------
# Generate button
# ---------------------------------------------------------------------------
if st.button("🚀 ツイートを生成", type="primary", use_container_width=True):
    with st.spinner(f"@{selected_user} のスタイルでツイートを生成中..."):
        try:
            resp = httpx.post(
                f"{st.session_state.api_base_url}/api/generate/tweet",
                json={
                    "username": selected_user,
                    "topic": topic if topic else None,
                    "count": candidate_count,
                    "temperature": temperature,
                },
                timeout=120.0,
            )

            if resp.status_code == 200:
                data = resp.json()
                candidates = data.get("candidates", [])

                if candidates:
                    st.success(f"✅ {len(candidates)} 件のツイート候補を生成しました！")

                    st.markdown("### 🏆 生成結果")
                    st.caption("スタイル一致度が高い順に表示されています。テキストをクリックでコピーできます。")

                    for i, candidate in enumerate(candidates, 1):
                        text = candidate.get("text", "")
                        score = candidate.get("score", 0.0)
                        pct = int(score * 100)

                        # Color based on score
                        color = "#1da1f2" if pct >= 70 else "#ffad1f" if pct >= 40 else "#e0245e"
                        medal = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else f"#{i}"

                        with st.container(border=True):
                            st.markdown(
                                f"""
                                <div class="tweet-card">
                                    <div class="username">{medal} @{selected_user}</div>
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
                            # Copyable text
                            st.code(text, language=None)
                else:
                    st.warning("候補が生成されませんでした。トピックや Temperature を調整してみてください。")

            else:
                error = resp.json().get("detail", resp.text) if "json" in resp.headers.get("content-type", "") else resp.text
                st.error(f"生成エラー: {error}")

        except httpx.ConnectError:
            st.error(
                "API サーバーに接続できません。\n\n"
                "```\nuvicorn postlikeme.api.app:app --reload --port 8000\n```"
            )
        except httpx.ReadTimeout:
            st.warning("生成がタイムアウトしました。LLM API の応答を待っています...")
        except Exception as e:
            st.error(f"予期しないエラー: {e}")

# ---------------------------------------------------------------------------
# Tips sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 💡 生成のコツ")
    st.markdown(
        """
        **トピック指定**
        - 具体的なほど良い結果に
        - 「AI」より「AI が仕事を奪うかどうか」
        - 空欄でもOK（自動選択）

        **Temperature**
        - `0.3-0.5`: 保守的、プロファイルに忠実
        - `0.7-0.9`: バランス良い（推奨）
        - `1.0+`: 創造的だがスタイルがブレやすい

        **候補数**
        - 3-5件がおすすめ
        - 多いほど良い候補が含まれやすい
        """
    )

    st.divider()

    if selected_user:
        st.markdown(f"### 📊 @{selected_user} の特徴")
        matching = [p for p in profiles_list if p.get("username") == selected_user]
        if matching:
            p = matching[0]
            st.caption(f"分析ツイート: {p.get('tweet_count_analyzed', '?')} 件")
