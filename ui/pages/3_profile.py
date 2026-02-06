"""👤 プロファイル閲覧ページ

構築済みのボイスプロファイルをビジュアルに表示する。
性格、トピック、文体、絵文字パターンなどをチャート・表で可視化。
"""

import streamlit as st
import httpx
import json

st.set_page_config(page_title="PostLikeMe - プロファイル", page_icon="👤", layout="wide")

st.title("👤 ボイスプロファイル閲覧")

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
    st.info("まだプロファイルがありません。先に「📥 収集」→「🔬 分析」を実行してください。")
    st.stop()

usernames = [p.get("username", "") for p in profiles_list]
default_idx = 0
if st.session_state.selected_profile in usernames:
    default_idx = usernames.index(st.session_state.selected_profile)

selected = st.selectbox("プロファイルを選択", usernames, index=default_idx)

if not selected:
    st.stop()

# ---------------------------------------------------------------------------
# Load full profile
# ---------------------------------------------------------------------------
profile = None
try:
    resp = httpx.get(
        f"{st.session_state.api_base_url}/api/profiles/{selected}",
        timeout=10.0,
    )
    if resp.status_code == 200:
        profile = resp.json()
except Exception as e:
    st.error(f"プロファイルの読み込みに失敗: {e}")
    st.stop()

if not profile:
    st.error("プロファイルが見つかりません。")
    st.stop()

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.divider()

col_h1, col_h2, col_h3, col_h4 = st.columns(4)
col_h1.metric("ユーザー名", f"@{profile.get('username', '?')}")
col_h2.metric("分析ツイート数", profile.get("tweet_count_analyzed", "?"))
col_h3.metric("構築日", str(profile.get("profile_built_at", ""))[:10])
col_h4.metric("スキーマ版", profile.get("schema_version", "?"))

if profile.get("bio"):
    st.info(f"📝 Bio: {profile['bio']}")

# ---------------------------------------------------------------------------
# Tabs for different profile sections
# ---------------------------------------------------------------------------
tab_personality, tab_topics, tab_style, tab_emoji, tab_reply, tab_raw = st.tabs(
    ["🧠 性格", "📊 トピック", "✏️ 文体", "😀 絵文字", "💬 リプライ", "📄 Raw JSON"]
)

# --- Personality Tab ---
with tab_personality:
    st.markdown("### 🧠 性格プロファイル")

    personality = profile.get("personality_profile", {})

    # Big Five radar chart (using bar chart as Streamlit doesn't have radar natively)
    big5 = personality.get("big_five", {})
    if big5:
        st.markdown("#### Big Five 性格特性")
        big5_data = {
            "開放性 (Openness)": big5.get("openness", 0.5),
            "誠実性 (Conscientiousness)": big5.get("conscientiousness", 0.5),
            "外向性 (Extraversion)": big5.get("extraversion", 0.5),
            "協調性 (Agreeableness)": big5.get("agreeableness", 0.5),
            "神経質性 (Neuroticism)": big5.get("neuroticism", 0.5),
        }
        st.bar_chart(big5_data, horizontal=True)

        # Interpretation
        for trait, score in big5_data.items():
            level = "高い" if score > 0.65 else "中程度" if score > 0.35 else "低い"
            st.caption(f"  {trait}: **{level}** ({score:.2f})")

    col_p1, col_p2, col_p3 = st.columns(3)
    with col_p1:
        formality = personality.get("formality_score", 0.5)
        label = "非常にフォーマル" if formality > 0.8 else "フォーマル" if formality > 0.6 else "カジュアル" if formality > 0.3 else "非常にカジュアル"
        st.metric("フォーマル度", label, f"{formality:.2f}")
    with col_p2:
        st.metric("ユーモアスタイル", personality.get("humor_style", "N/A"))
    with col_p3:
        st.metric("トーン", personality.get("tone", "N/A"))

    if personality.get("catchphrases"):
        st.markdown("#### 口癖・キャッチフレーズ")
        for phrase in personality["catchphrases"]:
            st.markdown(f"- 「{phrase}」")

    assertiveness = personality.get("assertiveness", 0.5)
    st.progress(assertiveness, text=f"主張度: {assertiveness:.0%}")

# --- Topics Tab ---
with tab_topics:
    st.markdown("### 📊 トピック分析")

    topic_profile = profile.get("topic_profile", {})
    topics = topic_profile.get("topics_ranked", [])

    if topics:
        st.markdown("#### トピック重要度ランキング")
        chart_data = {}
        for t in topics[:15]:
            name = t.get("name", "?")
            importance = t.get("importance", 0)
            chart_data[name] = importance

        st.bar_chart(chart_data, horizontal=True)

        # Topic details
        for t in topics[:10]:
            with st.expander(f"📌 {t.get('name', '?')} (重要度: {t.get('importance', 0):.2f})"):
                keywords = t.get("keywords", [])
                if keywords:
                    st.markdown(f"**キーワード**: {', '.join(keywords)}")

    # Opinion map
    opinions = topic_profile.get("opinion_map", [])
    if opinions:
        st.markdown("#### 意見傾向")
        for op in opinions:
            topic_name = op.get("topic", "?")
            label = op.get("label", "neutral")
            mean_s = op.get("sentiment_mean", 0)
            emoji = "🔥" if label == "enthusiastic" else "👍" if "positive" in label else "😐" if label == "neutral" else "🤔" if label == "critical" else "👎"
            st.markdown(f"- {emoji} **{topic_name}**: {label} (平均センチメント: {mean_s:.2f})")

# --- Style Tab ---
with tab_style:
    st.markdown("### ✏️ 文体分析")

    style = profile.get("structural_style", {})

    col_s1, col_s2, col_s3, col_s4 = st.columns(4)
    col_s1.metric("平均ツイート長", f"{style.get('avg_tweet_length', 0):.0f} 文字")
    col_s2.metric("平均文長", f"{style.get('avg_sentence_length', 0):.1f} 語")
    col_s3.metric("断片文比率", f"{style.get('fragment_ratio', 0):.0%}")
    col_s4.metric("疑問文比率", f"{style.get('question_ratio', 0):.0%}")

    # Vocabulary richness
    vocab = style.get("vocabulary_richness", {})
    if vocab:
        st.markdown("#### 語彙豊かさ")
        vcol1, vcol2, vcol3 = st.columns(3)
        vcol1.metric("MTLD", f"{vocab.get('mtld', 0):.1f}")
        vcol2.metric("Yule's K", f"{vocab.get('yules_k', 0):.1f}")
        vcol3.metric("Hapax 比率", f"{vocab.get('hapax_ratio', 0):.2f}")

    # Top vocabulary
    top_vocab = style.get("top_vocabulary", [])
    if top_vocab:
        st.markdown("#### 頻出語彙 (Top 20)")
        vocab_dict = {}
        for item in top_vocab[:20]:
            if isinstance(item, (list, tuple)) and len(item) == 2:
                vocab_dict[item[0]] = item[1]
        if vocab_dict:
            st.bar_chart(vocab_dict)

    # Readability
    readability = style.get("readability_scores", {})
    if readability:
        st.markdown("#### 可読性スコア")
        rcol1, rcol2, rcol3, rcol4 = st.columns(4)
        rcol1.metric("Flesch", f"{readability.get('flesch_reading_ease', 0):.1f}")
        rcol2.metric("Coleman-Liau", f"{readability.get('coleman_liau_index', 0):.1f}")
        rcol3.metric("ARI", f"{readability.get('automated_readability_index', 0):.1f}")
        rcol4.metric("Gunning Fog", f"{readability.get('gunning_fog', 0):.1f}")

    # Tweet length distribution
    dist = style.get("tweet_length_distribution", {})
    hist = dist.get("histogram", {})
    if hist:
        st.markdown("#### ツイート長分布")
        st.bar_chart(hist)

# --- Emoji Tab ---
with tab_emoji:
    st.markdown("### 😀 絵文字プロファイル")

    emoji_data = profile.get("emoji_profile", {})

    col_e1, col_e2, col_e3 = st.columns(3)
    col_e1.metric("ツイートあたり絵文字数", f"{emoji_data.get('usage_rate', 0):.2f}")
    col_e2.metric("絵文字使用ツイート比率", f"{emoji_data.get('emoji_tweet_ratio', 0):.0%}")
    col_e3.metric("絵文字多様性", f"{emoji_data.get('emoji_diversity', 0):.2f}")

    top_emojis = emoji_data.get("top_emojis", [])
    if top_emojis:
        st.markdown("#### よく使う絵文字 Top 10")
        emoji_display = {}
        for item in top_emojis[:10]:
            if isinstance(item, (list, tuple)) and len(item) == 2:
                emoji_display[item[0]] = item[1]
        if emoji_display:
            st.bar_chart(emoji_display)

    # Positional tendency
    pos = emoji_data.get("positional_tendency", {})
    if pos:
        st.markdown("#### 絵文字の配置傾向")
        pos_data = {
            "先頭 (Leading)": pos.get("leading", 0),
            "中間 (Inline)": pos.get("inline", 0),
            "末尾 (Trailing)": pos.get("trailing", 0),
        }
        st.bar_chart(pos_data)

    st.metric("絵文字センチメント", f"{emoji_data.get('emoji_sentiment', 0):.2f}", help="-1.0(ネガティブ)〜+1.0(ポジティブ)")

# --- Reply Tab ---
with tab_reply:
    st.markdown("### 💬 リプライスタイル")

    reply = profile.get("reply_style", {})

    col_r1, col_r2, col_r3 = st.columns(3)
    col_r1.metric("平均リプライ長", f"{reply.get('avg_reply_length', 0):.0f} 文字")
    col_r2.metric("長さ比率（vs オリジナル）", f"{reply.get('length_ratio', 0):.2f}")
    col_r3.metric("エンゲージメントタイプ", reply.get("engagement_type", "N/A"))

    st.metric("トーン変化", reply.get("tone_vs_originals", "N/A"), help="オリジナルツイートと比べたリプライのトーン")

    opening = reply.get("opening_patterns", {})
    if opening:
        st.markdown("#### 冒頭パターン")
        pattern_labels = {
            "greeting": "挨拶型",
            "direct": "直接応答型",
            "quote": "引用応答型",
            "emoji_lead": "絵文字開始型",
            "agreement": "賛同型",
            "disagreement": "反論型",
        }
        display_data = {pattern_labels.get(k, k): v for k, v in opening.items()}
        st.bar_chart(display_data)

# --- Raw JSON Tab ---
with tab_raw:
    st.markdown("### 📄 Raw JSON データ")
    st.json(profile)

    if st.button("📋 JSON をクリップボードにコピー"):
        st.code(json.dumps(profile, ensure_ascii=False, indent=2), language="json")

# ---------------------------------------------------------------------------
# Sidebar actions
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### アクション")
    if st.button("✍️ このプロファイルでツイート生成", use_container_width=True):
        st.session_state.selected_profile = selected
        st.switch_page("pages/4_generate_tweet.py")

    if st.button("💬 このプロファイルでリプライ生成", use_container_width=True):
        st.session_state.selected_profile = selected
        st.switch_page("pages/5_generate_reply.py")

    st.divider()
    if st.button("🗑️ プロファイルを削除", use_container_width=True, type="secondary"):
        if st.session_state.get(f"confirm_delete_{selected}"):
            try:
                resp = httpx.delete(
                    f"{st.session_state.api_base_url}/api/profiles/{selected}",
                    timeout=5.0,
                )
                if resp.status_code == 200:
                    st.success(f"@{selected} のプロファイルを削除しました。")
                    st.rerun()
            except Exception as e:
                st.error(f"削除エラー: {e}")
        else:
            st.session_state[f"confirm_delete_{selected}"] = True
            st.warning("もう一度クリックすると削除されます。")
