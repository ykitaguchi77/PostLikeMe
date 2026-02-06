# PostLikeMe 分析アルゴリズム詳細仕様書

## 概要

本ドキュメントでは、PostLikeMe の分析パイプラインで使用する各アルゴリズムの
**具体的な数式・パラメータ・実装方法**を定義する。

分析パイプラインは以下の順序で実行される:

```
RawTweets → 前処理 → 構造分析 → トピック分析 → 性格分析 → リプライ分析 → 例文選択 → VoiceProfile
```

---

## 1. 前処理 (`preprocessor.py`)

### 1.1 フィルタリング

```
入力: list[RawTweet]
出力: originals: list[CleanTweet], replies: list[CleanTweet]
```

**除外ルール:**
- `is_retweet == True` → 除外（オリジナルコンテンツなし）
- `text` が URL のみ → 除外
- `text` がメンションのみ → 除外
- `language` が対象言語でない → 除外

**分離:**
- `is_reply == False` → `originals`
- `is_reply == True` → `replies`

### 1.2 テキストクリーニング

各ツイートに対して以下を順次適用:

1. **t.co URL の除去**: `re.sub(r'https?://t\.co/\w+', '', text)`
2. **Unicode 正規化**: `unicodedata.normalize('NFC', text)`
3. **絵文字位置のタグ付け**: `emoji.emoji_list(text)` で位置情報を保存後、分析用にはテキストから分離
4. **ユーザーメンション正規化**: `@username` → `@USER`（統計用にオリジナルを保存）
5. **連続空白の正規化**: `re.sub(r'\s+', ' ', text).strip()`

### 1.3 言語検出

- 短いテキスト (<10文字) は言語検出をスキップし、アカウントのデフォルト言語を使用
- 主要言語（最頻出言語）のツイートのみ分析対象とする

---

## 2. 語彙分析 (`structural_analyzer.py` - 語彙セクション)

### 2.1 基本語彙指標

全ツイートを連結したコーパスに対して計算。

| 指標 | 数式 | 解釈 |
|------|------|------|
| **TTR (Type-Token Ratio)** | `V / N` | V=異なり語数, N=総トークン数。テキスト長に依存するため比較用のみ |
| **Root TTR (Guiraud's R)** | `V / √N` | Guiraud (1954)。TTRの長さ補正版 |
| **CTTR** | `V / √(2N)` | Carroll (1964) |

### 2.2 長さ非依存の語彙豊かさ指標（推奨）

| 指標 | 計算方法 | パラメータ |
|------|----------|------------|
| **MTLD** | テキストを順次走査、TTRが閾値を下回るたびにファクターカウンタを増加しリセット。`MTLD = N / factor_count` | 閾値 = 0.720 |
| **MATTR** | 固定幅ウィンドウをスライドし各位置のTTRを平均 | ウィンドウサイズ = 50トークン |
| **Yule's K** | `K = 10⁴ × (M₂ - M₁) / M₁²` | M₁ = N, M₂ = Σ(i² × f(i)) |
| **Yule's I** | `I = M₁² / (M₂ - M₁)` | Kの逆数。大きいほど多様 |
| **vocd-D** | ランダムサブセットでTTRを計算し `TTR = (D/N)[√(1+2N/D) - 1]` にフィット | サブセット範囲: 35-50トークン |

**Yule's K の詳細:**
```
M₁ = N（総トークン数）
M₂ = Σ(i² × f_v(i, N))  ※ f_v(i,N) = ちょうどi回出現する語の種類数
K = 10,000 × (M₂ - M₁) / M₁²
```
- K が大きい → 語彙の繰り返しが多い（語彙が貧弱）
- K が小さい → 語彙が豊か

### 2.3 Hapax 関連指標

| 指標 | 数式 | 説明 |
|------|------|------|
| **Hapax Legomena 比率** | `V₁ / V` | 1回のみ出現する語の割合 |
| **Sichel's S** | `V₂ / V` | 2回出現する語の割合 |
| **Honore's R** | `100 × log(N) / (1 - V₁/V)` | Hapax ベースの豊かさ |
| **Brunet's W** | `N^(V^(-0.172))` | 小さいほど語彙が豊か |

### 2.4 頻出語彙

- spaCy でストップワードを除去
- `collections.Counter` で非ストップワードの頻度カウント
- 上位50語を保存
- レンマ化した上でカウント（`token.lemma_`）

**使用ライブラリ:** `lexicalrichness` (MTLD, MATTR, Yule's K, vocd-D をすべて提供)

---

## 3. 文構造分析 (`structural_analyzer.py` - 文構造セクション)

### 3.1 文長分析

spaCy の `doc.sents` で文分割後:

```python
sentence_lengths = [len([t for t in sent if not t.is_space]) for sent in doc.sents]
```

| 指標 | 計算 |
|------|------|
| 平均文長 | `mean(sentence_lengths)` |
| 文長標準偏差 | `std(sentence_lengths)` |
| 文長中央値 | `median(sentence_lengths)` |
| ツイートあたり文数 | `mean(sentences_per_tweet)` |

### 3.2 断片文（Fragment）検出

spaCy の依存構造解析を使用:

| パターン | 分類 | 検出方法 |
|----------|------|----------|
| ROOT=NOUN, VERB子なし | 名詞句断片 | `root.pos_ == "NOUN" and not any(c.pos_ == "VERB" for c in root.children)` |
| ROOT=ADJ, VERB子なし | 形容詞断片 | `root.pos_ == "ADJ" and not any(c.pos_ == "VERB" for c in root.children)` |
| ROOT=INTJ | 間投詞 | `root.pos_ == "INTJ"` |
| ROOT=VERB, nsubj あり | 完全文 | `any(c.dep_ == "nsubj" for c in root.children)` |
| ROOT=VERB, nsubj なし, 命令形 | 命令文 | `root.morph.get("VerbForm") == ["Imp"]` |
| ROOT=VERB, nsubj なし, 非命令 | 動詞断片 | 上記いずれにも該当しない |

```
fragment_ratio = fragment_count / total_sentence_count
```

### 3.3 疑問文・感嘆文比率

```python
question_ratio = sum(1 for s in sentences if s.text.strip().endswith('?')) / len(sentences)
exclamation_ratio = sum(1 for s in sentences if s.text.strip().endswith('!')) / len(sentences)
statement_ratio = 1.0 - question_ratio - exclamation_ratio
```

### 3.4 品詞（POS）分布

spaCy の Universal POS タグでコーパス全体の分布を計算:

```python
pos_ratios = {
    "noun_ratio":     count(NOUN + PROPN) / total_tokens,
    "verb_ratio":     count(VERB + AUX) / total_tokens,
    "adj_ratio":      count(ADJ) / total_tokens,
    "adv_ratio":      count(ADV) / total_tokens,
    "pronoun_ratio":  count(PRON) / total_tokens,
    "function_word_ratio": count(DET + ADP + CCONJ + SCONJ + PART + PRON + AUX) / total_tokens,
}
```

**注目点:**
- 一人称代名詞（I, me, my）の高比率 → 神経質性と相関
- 機能語比率 → スタイル指紋として最も安定した特徴量

---

## 4. 可読性指標 (`structural_analyzer.py` - 可読性セクション)

### 4.1 使用する指標

個別ツイートではなく**コーパス全体**に対して計算（短文での統計的不安定性を回避）。

| 指標 | 数式 | 特徴 |
|------|------|------|
| **Flesch Reading Ease** | `206.835 - 1.015(W/S) - 84.6(Syl/W)` | 0-100, 高い=易しい |
| **Coleman-Liau Index** | `0.0588L - 0.296S - 15.8` | L=100語あたり文字数, S=100語あたり文数。**音節不要で推奨** |
| **ARI** | `4.71(C/W) + 0.5(W/S) - 21.43` | C=文字数, W=単語数, S=文数。**音節不要で推奨** |
| **Gunning Fog** | `0.4[(W/S) + 100(CW/W)]` | CW=3音節以上の語 |

### 4.2 ツイート向け適応

**問題点:**
- 標準的な可読性公式は100語以上の文章向けに設計
- 個別ツイート (10-40語) では極端なスコア振れが発生
- スラング・略語の音節カウントが不正確

**対策:**
1. 全ツイートを連結してコーパスレベルで計算（各ツイート末尾を文境界として扱う）
2. **Coleman-Liau / ARI を優先**（音節カウント不要、文字数ベースでスラングに強い）
3. 個別ツイートのスコアは分布（平均・標準偏差）としてのみ使用

**使用ライブラリ:** `textstat`

---

## 5. 句読点・記号パターン (`structural_analyzer.py`)

### 5.1 句読点頻度

1000トークンあたりの出現頻度として正規化:

```python
punctuation_per_1k = {
    "period":       count('.') / total_tokens * 1000,
    "comma":        count(',') / total_tokens * 1000,
    "exclamation":  count('!') / total_tokens * 1000,
    "question":     count('?') / total_tokens * 1000,
    "ellipsis":     count('...') / total_tweets,  # ツイートあたり
    "em_dash":      count('—' or '--') / total_tokens * 1000,
    "parentheses":  count('(' or ')') / total_tokens * 1000,
    "quotation":    count('"' or "'") / total_tokens * 1000,
    "semicolon":    count(';') / total_tokens * 1000,
    "colon":        count(':') / total_tokens * 1000,
}
```

### 5.2 大文字パターン

```python
capitalization = {
    "all_caps_ratio":    all_caps_tokens / total_tokens,      # "GREAT" など
    "initial_caps_ratio": initial_cap_sentences / total_sentences,
    "lowercase_start_ratio": lowercase_start_sentences / total_sentences,
    "mid_caps_ratio":    mid_sentence_caps / total_tokens,    # 強調的大文字
}
```

---

## 6. 絵文字分析 (`structural_analyzer.py` - 絵文字セクション)

### 6.1 検出

**ライブラリ:** `emoji`

```python
import emoji

emoji_data = emoji.emoji_list(text)
# 結果: [{"match_start": 5, "match_end": 6, "emoji": "🔥"}, ...]
```

### 6.2 頻度指標

| 指標 | 数式 |
|------|------|
| ツイートあたり絵文字数 | `total_emojis / total_tweets` |
| 絵文字使用ツイート比率 | `tweets_with_emoji / total_tweets` |
| 絵文字多様性 | `unique_emojis / total_emoji_occurrences`（絵文字版TTR）|

### 6.3 位置分析

各絵文字の出現位置を正規化:

```python
position_ratio = emoji_char_index / len(tweet_text)
```

| 位置分類 | 条件 |
|----------|------|
| Leading（先頭） | `position_ratio < 0.2` |
| Inline（中間） | `0.2 ≤ position_ratio ≤ 0.8` |
| Trailing（末尾） | `position_ratio > 0.8` |

全ツイートで集計し、位置傾向の分布を算出。

### 6.4 共起分析

**絵文字ペア共起:**
```python
from collections import Counter
emoji_pairs = Counter()
for tweet in tweets:
    emojis_in_tweet = [e["emoji"] for e in emoji.emoji_list(tweet.text)]
    if len(emojis_in_tweet) >= 2:
        for pair in combinations(sorted(emojis_in_tweet), 2):
            emoji_pairs[pair] += 1
```

**絵文字-単語共起:**
各ツイート内で、絵文字と非ストップワードのペアをカウント。
意味的な関連性を発見（例: "congrats" と 🎉 の共起）。

### 6.5 絵文字センチメント

**ライブラリ:** `emosent-py`（Novak et al., 2015 の研究に基づく751絵文字のセンチメントスコア）

```python
author_emoji_sentiment = mean([sentiment_score(e) for e in all_used_emojis])
# -1.0（ネガティブ）〜 +1.0（ポジティブ）
```

---

## 7. ハッシュタグ分析 (`structural_analyzer.py`)

```python
hashtag_profile = {
    "usage_rate":     total_hashtags / total_tweets,
    "tweets_with_tags": tweets_with_hashtags / total_tweets,
    "top_hashtags":   Counter(all_hashtags).most_common(20),
    "placement_style": classify_placement(hashtag_positions),  # "inline" | "trailing" | "mixed"
    "casing_style":   classify_casing(all_hashtags),           # "CamelCase" | "lowercase" | "UPPERCASE" | "mixed"
}
```

**配置分類:**
- `inline`: 80%以上がツイート中間に出現
- `trailing`: 80%以上がツイート末尾に出現
- `mixed`: それ以外

---

## 8. ツイート長分布 (`structural_analyzer.py`)

```python
lengths = [len(tweet.text) for tweet in tweets]
tweet_length_profile = {
    "mean":     mean(lengths),
    "median":   median(lengths),
    "std":      std(lengths),
    "min":      min(lengths),
    "max":      max(lengths),
    "p25":      percentile(lengths, 25),
    "p75":      percentile(lengths, 75),
    "histogram": histogram(lengths, bins=[0, 50, 100, 140, 200, 280]),
}
```

---

## 9. トピック分析 (`topic_analyzer.py`)

### 9.1 方式選択

| 方式 | ツイートへの適合性 | 前提条件 |
|------|-------------------|----------|
| **BERTopic（推奨）** | 高い。短文に強い。トピック数の事前指定不要 | `sentence-transformers` が必要 |
| **NMF + TF-IDF（代替）** | 中程度。短文への適応が必要 | 軽量、追加依存なし |
| LDA | 低い。短文では性能劣化 | 非推奨 |

### 9.2 BERTopic パイプライン（推奨）

```
ツイート → sentence-transformers 埋め込み → UMAP 次元削減 → HDBSCAN クラスタリング → c-TF-IDF トピック表現
```

**パラメータ設定:**

```python
from umap import UMAP
from hdbscan import HDBSCAN
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

umap_model = UMAP(
    n_neighbors=10,          # 短文向けにデフォルト15から縮小
    n_components=5,          # 中間次元数
    min_dist=0.0,            # クラスタ密度を最大化
    metric='cosine',
    random_state=42
)

hdbscan_model = HDBSCAN(
    min_cluster_size=10,     # 小規模トピックも検出
    metric='euclidean',
    cluster_selection_method='leaf',  # より細かいトピック
    prediction_data=True,
    min_samples=5            # アウトライア削減
)

topic_model = BERTopic(
    embedding_model=embedding_model,
    umap_model=umap_model,
    hdbscan_model=hdbscan_model,
    min_topic_size=10,
    verbose=True
)

topics, probs = topic_model.fit_transform(tweet_texts)
```

**BERTopic を使う理由:**
- トピック数 K の事前指定が不要（HDBSCAN が自動決定）
- アウトライアを自然に処理（トピック -1 に分類）
- `topic_model.topics_over_time()` で時系列トピック変化を追跡可能
- `topic_model.hierarchical_topics()` で階層的トピック構造を取得可能

### 9.3 NMF + TF-IDF パイプライン（代替・軽量版）

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF

vectorizer = TfidfVectorizer(
    max_df=0.85,             # 85%以上のツイートに出現する語を除外
    min_df=3,                # 3件未満のツイートにしか出現しない語を除外
    ngram_range=(1, 2),      # ユニグラム + バイグラム
    sublinear_tf=True,       # 1 + log(tf) で短文の重み補正
    max_features=5000,
)

tfidf_matrix = vectorizer.fit_transform(lemmatized_tweets)

# トピック数 K の決定: コヒーレンススコアで [5, 30] の範囲を探索
best_k = None
best_coherence = -1
for k in range(5, 31):
    nmf = NMF(n_components=k, random_state=42, max_iter=400)
    W = nmf.fit_transform(tfidf_matrix)
    # コヒーレンス計算（top-10語の PMI ベース）
    coherence = compute_coherence(nmf, vectorizer, tweet_texts, top_n=10)
    if coherence > best_coherence:
        best_coherence = coherence
        best_k = k
```

### 9.4 トピック頻度の最新性重み付け

```python
import math
from datetime import datetime

def recency_weight(tweet_date: datetime, now: datetime, half_life_days: int = 90) -> float:
    """指数減衰による最新性重み"""
    lambda_ = math.log(2) / half_life_days  # 半減期パラメータ
    days_ago = (now - tweet_date).days
    return math.exp(-lambda_ * days_ago)

# half_life_days = 90: 90日前のツイートは重み0.5、180日前は0.25
# half_life_days = 30: より最新を重視
```

**トピック重要度:**
```python
topic_importance = {}
for topic_id in unique_topics:
    topic_tweets = [(t, w) for t, w in zip(tweets, weights) if t.topic == topic_id]
    topic_importance[topic_id] = sum(recency_weight(t.created_at, now) for t, w in topic_tweets)

# 正規化して0-1スケールに
total = sum(topic_importance.values())
topic_importance = {k: v/total for k, v in topic_importance.items()}
```

### 9.5 トピック別センチメント/意見マッピング

**ステップ 1: VADER でツイート単位のセンチメント算出**

```python
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

analyzer = SentimentIntensityAnalyzer()

for tweet in tweets:
    scores = analyzer.polarity_scores(tweet.text)
    tweet.sentiment_compound = scores['compound']
    # compound: -1.0（最もネガティブ）〜 +1.0（最もポジティブ）
```

**VADER がツイートに強い理由:**
- ソーシャルメディア向けに設計（スラング、略語、絵文字対応）
- ALL CAPS を強調として認識（"GREAT" > "great"）
- 句読点の繰り返しを強度として認識（"good!!!" > "good"）
- 否定語を検出して極性反転（"not good" → ネガティブ）
- 学習不要、ルールベースで決定的

**ステップ 2: トピック別に集約**

```python
topic_sentiment = {}
for topic_id in unique_topics:
    sentiments = [t.sentiment_compound for t in tweets if t.topic == topic_id]
    mean_s = mean(sentiments)
    topic_sentiment[topic_id] = {
        "mean": mean_s,
        "std": std(sentiments),
        "label": sentiment_label(mean_s),
    }

def sentiment_label(mean_compound: float) -> str:
    if mean_compound > 0.3:   return "enthusiastic"
    if mean_compound > 0.1:   return "generally_positive"
    if mean_compound > -0.1:  return "neutral"
    if mean_compound > -0.3:  return "critical"
    return "negative"
```

---

## 10. 性格・トーン分析 (`personality_analyzer.py`)

### 10.1 Big Five 性格特性のヒューリスティクス推定

研究ベースの言語マーカーを使用し、各特性の近似スコアを算出。

**研究根拠:**
- Yarkoni (2010): 694ブログ、100,000語の分析。LIWC カテゴリと Big Five の相関 (mean |r| = 0.14)
- Pennebaker & King (1999): LIWC と Big Five の基礎的相関
- Moreno et al. (2021): メタ分析、r = 0.26〜0.30

#### Openness（開放性）- 最も検出しやすい特性 (r ≈ 0.30)

| 言語マーカー | 方向 | 相関係数 |
|-------------|------|----------|
| 冠詞 (the, a, an) の使用 | 正 | r = 0.20 |
| 前置詞の使用 | 正 | r = 0.17 |
| 個人代名詞の使用 | 負 | |
| 語彙多様性 (MTLD) | 正 | |
| 平均語長 | 正 | |
| フォーマルな文体 | 正 | |

```python
openness_score = normalize(
    0.25 * z(article_ratio) +
    0.20 * z(preposition_ratio) +
    0.25 * z(mtld) +
    0.15 * z(avg_word_length) +
    0.15 * z(formality_score) +
    -0.20 * z(personal_pronoun_ratio)
)
```

#### Neuroticism（神経質性）

| 言語マーカー | 方向 |
|-------------|------|
| ネガティブ感情語 | 正 |
| 不安/恐怖語 | 正 |
| 一人称単数代名詞 (I, me, my) | 正 |
| 否定語 (not, never, no) | 正 |
| ポジティブ感情語 | 負 |

```python
neuroticism_score = normalize(
    0.30 * z(negative_emotion_ratio) +
    0.20 * z(anxiety_word_ratio) +
    0.20 * z(first_person_singular_ratio) +
    0.15 * z(negation_ratio) +
    -0.15 * z(positive_emotion_ratio)
)
```

#### Extraversion（外向性）

| 言語マーカー | 方向 |
|-------------|------|
| ポジティブ感情語 | 正 |
| 社交語 (meet, party, friends) | 正 |
| ネガティブ感情語 | 負 |
| 総語数 | 正 (話し言葉のみ) |

```python
extraversion_score = normalize(
    0.30 * z(positive_emotion_ratio) +
    0.30 * z(social_word_ratio) +
    0.20 * z(tweet_frequency) +
    -0.20 * z(negative_emotion_ratio)
)
```

#### Agreeableness（協調性）

| 言語マーカー | 方向 | 相関係数 |
|-------------|------|----------|
| 罵倒語 | 負 | r = -0.39（最大の単一相関）|
| ポジティブ感情語 | 正 | |
| 怒り語 | 負 | |
| ネガティブ感情語 | 負 | |

```python
agreeableness_score = normalize(
    -0.35 * z(swear_word_ratio) +
    0.25 * z(positive_emotion_ratio) +
    -0.20 * z(anger_word_ratio) +
    -0.20 * z(negative_emotion_ratio)
)
```

#### Conscientiousness（誠実性）- 最も検出が困難

| 言語マーカー | 方向 |
|-------------|------|
| 達成語 (achieve, complete, goal) | 正 |
| 確信語 (certainly, always, definitely) | 正 |
| フィラー語 | 負 |
| 罵倒語 | 負 |

```python
conscientiousness_score = normalize(
    0.30 * z(achievement_word_ratio) +
    0.25 * z(certainty_word_ratio) +
    -0.25 * z(filler_word_ratio) +
    -0.20 * z(swear_word_ratio)
)
```

**正規化関数:**
```python
def normalize(raw_score: float) -> float:
    """シグモイド関数で [0, 1] にマッピング"""
    return 1.0 / (1.0 + math.exp(-raw_score))
```

**z スコア関数:**
```python
def z(value: float, baseline_mean: float, baseline_std: float) -> float:
    """ベースライン統計に対するzスコア"""
    return (value - baseline_mean) / baseline_std if baseline_std > 0 else 0.0
```

**重要な注意:** 効果量は小〜中程度 (r = 0.14〜0.30)。結果は定性的なバンド (low / moderate / high) として報告し、正確なスコアとは扱わない。

### 10.2 フォーマル度スコア

```python
formality_indicators = {
    "contraction_ratio":    contractions / total_words,       # don't, it's, can't
    "slang_ratio":          slang_words / total_words,        # curated list
    "complete_sentence_ratio": complete_sentences / total_sentences,
    "avg_sentence_length":  mean(sentence_lengths),
    "function_word_ratio":  function_words / total_words,
}

# 0.0 (very casual) 〜 1.0 (very formal)
formality_score = normalize(
    -0.30 * z(contraction_ratio) +
    -0.25 * z(slang_ratio) +
    0.20 * z(complete_sentence_ratio) +
    0.15 * z(avg_sentence_length) +
    0.10 * z(function_word_ratio)
)
```

### 10.3 LLM 支援分析

ヒューリスティクスでは捉えきれないニュアンスを LLM の1回コールで補完:

**入力:** 代表的ツイート 30-50件（エンゲージメント上位から層化抽出）

**LLM に依頼する評価項目:**

```json
{
  "humor_style": "dry | sarcastic | absurdist | wholesome | self_deprecating | none",
  "communication_tone": "casual | professional | academic | provocative | inspirational",
  "rhetorical_devices": ["metaphor", "rhetorical_question", "hyperbole", ...],
  "catchphrases": ["特定の口癖やフレーズ"],
  "disagreement_style": "direct_confrontation | polite_disagreement | avoidance | humor_deflection",
  "thread_usage": "frequent | occasional | rare",
  "big_five_assessment": {
    "openness": "low | moderate | high",
    "conscientiousness": "low | moderate | high",
    "extraversion": "low | moderate | high",
    "agreeableness": "low | moderate | high",
    "neuroticism": "low | moderate | high"
  },
  "overall_persona_summary": "2-3文の人物像要約"
}
```

**統合ルール:**
- ヒューリスティクスと LLM 評価が一致 → 高信頼度として採用
- 不一致の場合 → LLM 評価を優先（文脈理解が優れるため）、ただしヒューリスティクスの数値データもプロファイルに保存

---

## 11. リプライスタイル分析 (`reply_analyzer.py`)

### 11.1 リプライ vs オリジナルの比較

```python
reply_vs_original = {
    "length_ratio":      mean(reply_lengths) / mean(original_lengths),
    "emoji_rate_diff":   reply_emoji_rate - original_emoji_rate,
    "question_rate_diff": reply_question_ratio - original_question_ratio,
    "sentiment_diff":    mean(reply_sentiments) - mean(original_sentiments),
}
```

### 11.2 冒頭パターン分類

リプライの最初のトークン/フレーズをパターンマッチ:

| パターン | 検出方法 | 例 |
|----------|----------|-----|
| 挨拶型 | "hey", "hi", "hello" で開始 | "Hey! Great point..." |
| 直接応答型 | 最初の語が内容語 | "That's exactly right..." |
| 引用応答型 | 引用符または ">" で開始 | "> original text" → 応答 |
| 絵文字開始型 | 絵文字で開始 | "🔥 This is it" |
| 賛同型 | "yes", "agreed", "exactly", "100%" | "Exactly! And also..." |
| 反論型 | "no", "actually", "but", "disagree" | "Actually, I think..." |

```python
opening_patterns = Counter()
for reply in replies:
    pattern = classify_opening(reply.text)
    opening_patterns[pattern] += 1
```

### 11.3 エンゲージメントタイプ分類

リプライ全体の傾向を分類:

```python
def classify_engagement_type(replies: list[CleanTweet]) -> str:
    """
    リプライの主要エンゲージメントタイプを決定
    """
    scores = {
        "supportive":    sum(1 for r in replies if r.sentiment > 0.2 and has_agreement_words(r)),
        "debate":        sum(1 for r in replies if has_disagreement_words(r) or r.sentiment < -0.1),
        "witty":         sum(1 for r in replies if has_humor_markers(r)),
        "informative":   sum(1 for r in replies if avg_length(r) > 100 and has_links_or_data(r)),
    }
    return max(scores, key=scores.get)
```

---

## 12. スタイル指紋 (`structural_analyzer.py` - スタイロメトリー)

### 12.1 機能語頻度ベクトル

最も安定したスタイル指標。著者が無意識に使用するため、トピックに依存しない。

```python
FUNCTION_WORDS = [
    "the", "of", "and", "to", "a", "in", "that", "it", "is", "was",
    "for", "with", "as", "but", "be", "on", "not", "he", "she", "this",
    "from", "by", "at", "or", "an", "which", "have", "had", "were", "been",
    "do", "if", "so", "than", "its", "my", "into", "about", "what", "their",
    "who", "could", "would", "should", "just", "also", "very", "even", "too",
]  # 上位50語

function_word_vector = {
    word: corpus_count(word) / total_tokens
    for word in FUNCTION_WORDS
}
```

### 12.2 文字 N-gram

ツイートの著者推定で 92-98.5% の精度を達成する特徴量。

```python
from sklearn.feature_extraction.text import CountVectorizer

char_vectorizer = CountVectorizer(
    analyzer='char',
    ngram_range=(2, 4),    # 文字バイグラムからクアッドグラム
    max_features=200,
)
char_ngram_matrix = char_vectorizer.fit_transform(tweet_texts)

# 上位100トライグラムとその正規化頻度を保存
top_char_trigrams = get_top_features(char_vectorizer, char_ngram_matrix, n=100)
```

**文字 N-gram が有効な理由:**
- サブワードパターン（接頭辞、接尾辞、スペルの癖）を捉える
- スペルミス、造語、略語を自然に扱う
- 句読点・空白パターンも暗黙的に捕捉
- トークン化の決定に依存しない

### 12.3 短縮形使用パターン

```python
contraction_pairs = {
    ("do", "not"): "don't",
    ("it", "is"): "it's",
    ("I", "am"): "I'm",
    ("can", "not"): "can't",
    ("will", "not"): "won't",
    # ... etc.
}

contraction_preference = {
    pair: contracted_count / (contracted_count + expanded_count)
    for pair in contraction_pairs
}
# 1.0 = 常に短縮形、0.0 = 常に展開形
```

---

## 13. Few-Shot 例文選択 (`example_selector.py`)

### 13.1 エンゲージメントスコア

```python
def engagement_score(tweet: RawTweet) -> float:
    """対数スケールでエンゲージメントをスコアリング"""
    return (
        math.log(1 + tweet.like_count) +
        1.5 * math.log(1 + tweet.retweet_count) +  # リツイートを重み付け
        0.5 * math.log(1 + tweet.reply_count)
    )
```

**リツイート重み 1.5x の理由:** 共有に値するほど印象的 = そのアカウントの「声」をよく表す

### 13.2 MMR (Maximal Marginal Relevance) による多様性確保

```python
def mmr_select(
    candidates: list[Tweet],
    embeddings: np.ndarray,     # sentence-transformers の埋め込み
    target_count: int,
    lambda_param: float = 0.4,  # 0.3-0.5 推奨
) -> list[Tweet]:
    """
    MMR(d) = λ × Sim(d, Q) - (1-λ) × max(Sim(d, d_j) for d_j in S)

    Q = コーパス全体の重心埋め込み
    S = 既に選択されたツイートの集合
    """
    centroid = embeddings.mean(axis=0)  # Q
    selected_indices = []
    candidate_indices = list(range(len(candidates)))

    for _ in range(target_count):
        best_score = -float('inf')
        best_idx = None

        for idx in candidate_indices:
            # コーパス重心への類似度（代表性）
            relevance = cosine_similarity(embeddings[idx], centroid)

            # 既選択ツイートとの最大類似度（冗長性）
            if selected_indices:
                max_sim = max(
                    cosine_similarity(embeddings[idx], embeddings[s])
                    for s in selected_indices
                )
            else:
                max_sim = 0.0

            mmr = lambda_param * relevance - (1 - lambda_param) * max_sim

            if mmr > best_score:
                best_score = mmr
                best_idx = idx

        selected_indices.append(best_idx)
        candidate_indices.remove(best_idx)

    return [candidates[i] for i in selected_indices]
```

**λ = 0.4 の理由:**
- λ = 0.3: 多様性最大化（スタイルの幅を見せる）
- λ = 0.5: 代表性と多様性のバランス
- λ = 0.7: 代表性重視（平均的なスタイルに集中）
- スタイル学習では**幅広い例**が重要なため、0.3-0.4 を推奨

### 13.3 統合選択パイプライン（推奨）

```
1. フィルタ
   - リツイート除外
   - 5語未満のツイート除外
   - URL/メンションのみのツイート除外

2. トピック別層化
   - 各トピックのツイート数に比例してスロットを配分
   - 最低1スロット/トピック（カバレッジ保証）

3. トピック内エンゲージメントランキング
   - engagement_score でソート
   - 上位候補をプール

4. MMR による最終選択
   - λ = 0.4 で多様性を確保
   - 目標: オリジナル 20-30件、リプライ 15-20件

5. 構造的カバレッジ検証
   - 短いツイート / 長いツイート が含まれているか
   - 疑問文 / 感嘆文 が含まれているか
   - 絵文字あり / なし が含まれているか
   - ハッシュタグあり / なし が含まれているか
   - 不足があれば手動で補完
```

### 13.4 リプライ例の選択

追加基準:
- 様々な種類のツイートへの返信を含める（質問、主張、論争的発言、賞賛）
- 短い / 長いリプライの両方を含める
- 異なるトーン（賛同、反論、ユーモア、情報提供）を含める
- 各リプライには**元ツイート（文脈）とセットで**保存

---

## 14. 分析パイプライン統合 (`pipeline.py`)

### 実行順序と依存関係

```
Step 1: preprocessor.preprocess(raw_tweets)
        → originals, replies
        依存: なし

Step 2: structural_analyzer.analyze(originals + replies)
        → structural_style, emoji_profile, hashtag_profile,
          tweet_length_profile, punctuation_patterns, capitalization
        依存: Step 1

Step 3: topic_analyzer.analyze(originals)
        → topics_ranked, opinion_map, semantic_clusters
        依存: Step 1

Step 4: personality_analyzer.analyze(originals, structural_style)
        → personality_profile (heuristics + LLM assessment)
        依存: Step 1, Step 2

Step 5: reply_analyzer.analyze(replies, originals)
        → reply_style
        依存: Step 1

Step 6: example_selector.select(originals, replies, topics)
        → example_tweets
        依存: Step 1, Step 3

Step 7: assemble VoiceProfile
        依存: Step 2-6
```

**並列実行可能:**
- Step 2 と Step 3 は独立して並列実行可能
- Step 4 と Step 5 は Step 2 の後に並列実行可能

### 必要ツイート数のガイドライン

| ツイート数 | 分析品質 | 推奨用途 |
|-----------|----------|----------|
| < 100 | 低（警告表示） | 基本的な構造分析のみ |
| 200-499 | 中 | 構造 + トピック分析 |
| 500-999 | 高 | 全分析パイプライン |
| 1000+ | 最高 | 最高品質プロファイル |

---

## 15. 使用ライブラリまとめ

| ライブラリ | 用途 | 分析ステップ |
|-----------|------|------------|
| `spacy` + `en_core_web_sm` | トークン化、POS、依存構造、NER、文分割 | 2, 3, 10, 11 |
| `lexicalrichness` | TTR, MTLD, MATTR, Yule's K, vocd-D | 2 |
| `textstat` | Flesch, Coleman-Liau, ARI, Gunning Fog | 4 |
| `emoji` | 絵文字検出、位置分析 | 6 |
| `emosent-py` | 絵文字センチメントスコア | 6 |
| `vaderSentiment` | ツイートセンチメント分析 | 9 |
| `sentence-transformers` | ツイート埋め込み (all-MiniLM-L6-v2) | 9, 13 |
| `bertopic` | トピックモデリング | 9 |
| `scikit-learn` | TF-IDF, NMF, 文字N-gram, コサイン類似度 | 9, 12, 13 |
| `numpy` / `scipy` | 統計計算、分布分析 | 全体 |

---

## 参考文献

- Yule, G.U. (1944). *The Statistical Study of Literary Vocabulary*
- Pennebaker, J.W. & King, L.A. (1999). JPSP, 77(6), 1296-1312
- Yarkoni, T. (2010). "Personality in 100,000 Words." J. Research in Personality, 44, 363-373
- Hutto, C.J. & Gilbert, E. (2014). "VADER." ICWSM-14
- Egger, R. & Yu, J. (2022). "A Topic Modeling Comparison..." Frontiers in Sociology
- McCarthy, P.M. & Jarvis, S. (2010). MTLD, vocd-D validation study
- Covington, M.A. & McFall, J.D. (2010). MATTR
- Carbonell, J. & Goldberg, J. (1998). MMR for diversity-based reranking
- Moreno, J.D. et al. (2021). Meta-analysis of text-based personality detection
- Lagutina, K. et al. (2019). "A Survey on Stylometric Text Features." FRUCT 2019
- Novak, P.K. et al. (2015). "Sentiment of Emojis." PLoS ONE, 10(12)
