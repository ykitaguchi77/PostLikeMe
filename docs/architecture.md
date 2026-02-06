# PostLikeMe - アーキテクチャ設計書

## 1. システム概要

PostLikeMe は、X (Twitter) の特定アカウントを分析し、その人のスタイルでツイートやリプライを生成する Python CLI ツールです。

### 2つのフェーズ

**分析フェーズ**: 対象アカウントの公開ツイートを収集し、文体・性格・トピック傾向・口調・ユーモアパターン・絵文字の使い方・ハッシュタグの癖・文構造・意見傾向などを捉えた「ボイスプロファイル」を構築する。

**生成フェーズ**: ボイスプロファイルと LLM を組み合わせ、そのアカウントらしいオリジナルツイートや文脈に応じたリプライを生成する。自動投稿は行わず、人間のレビュー用の候補を生成する。

---

## 2. 技術スタック

| レイヤー | 技術 | 理由 |
|---------|------|------|
| **言語** | Python 3.11+ | NLP・LLM API・async I/O のエコシステムが充実 |
| **CLI フレームワーク** | Typer (Click ベース) | 型ヒント駆動、自動ヘルプ、サブコマンド対応 |
| **ターミナル出力** | Rich | プログレスバー、テーブル、スタイル付きテキスト |
| **データ収集 (主)** | tweepy (X API v2) | 公式・安定・ドキュメント充実 |
| **データ収集 (代替)** | twscrape | 非同期・API キー不要・アカウントプール方式 |
| **データ収集 (手動)** | JSON/CSV インポート | Twitter データエクスポート対応 |
| **LLM (主)** | Anthropic Claude API | スタイル模倣に優れた指示追従性、長コンテキスト |
| **LLM (代替)** | OpenAI GPT-4o / GPT-4.1 | 広く利用可能な代替 |
| **NLP 分析** | spaCy + textstat + カスタムヒューリスティクス | 語彙、可読性、品詞分布、文構造 |
| **埋め込み (任意)** | sentence-transformers | トピッククラスタリング用 |
| **ストレージ** | JSON ファイル (プロファイル) + SQLite (ツイートキャッシュ) | シンプル・ポータブル |
| **設定** | TOML | Python 標準の設定形式 |
| **パッケージング** | pyproject.toml + hatchling | モダン Python パッケージング |
| **テスト** | pytest + pytest-asyncio | 標準テストスタック |
| **リンター** | ruff | 高速オールインワン |

---

## 3. コアコンポーネント詳細設計

### 3.1 データ収集レイヤー (`postlikeme/collectors/`)

ツイート取得を共通インターフェースで抽象化し、残りのシステムをデータソースに依存しない設計にする。

**抽象インターフェース: `BaseCollector`**

```python
class BaseCollector(ABC):
    async def collect_user_tweets(username, count, include_replies) -> list[RawTweet]
    async def collect_tweet_by_id(tweet_id) -> RawTweet
    async def collect_thread(tweet_id) -> list[RawTweet]
    async def get_user_profile(username) -> UserProfile
```

**具象実装:**

1. **`TweepyCollector`** - tweepy で X API v2 Bearer Token 認証。Paginator でページネーション。レートリミット対応 (Basic tier: 900リクエスト/15分)。
2. **`TwscrapeCollector`** - twscrape の非同期 API。アカウントプール方式。API アクセスが不要。
3. **`ArchiveCollector`** - Twitter 公式データエクスポート (JSON) のパース。シンプルな CSV 形式もサポート。

**データモデル: `RawTweet`**

```python
@dataclass
class RawTweet:
    id: str
    text: str
    created_at: datetime
    is_reply: bool
    reply_to_user: str | None
    reply_to_tweet_id: str | None
    is_retweet: bool
    is_quote_tweet: bool
    quoted_text: str | None
    hashtags: list[str]
    mentions: list[str]
    urls: list[str]
    media_types: list[str]       # ["image", "video", "gif"]
    like_count: int
    retweet_count: int
    reply_count: int
    language: str | None
```

**レートリミット戦略:**
- `asyncio.Semaphore` + スライディングウィンドウの `RateLimiter` クラス
- Tweepy: X-Rate-Limit ヘッダーを追跡、429 レスポンスでリセットまでスリープ
- twscrape: 内蔵のアカウントローテーション活用 + 設定可能な遅延 (デフォルト 1.5秒)

**ツイートキャッシュ:**
- SQLite データベース (`~/.postlikeme/cache/tweets.db`)
- スキーマ: `tweets(id PRIMARY KEY, username, raw_json, collected_at)`
- 7日以内のキャッシュがあれば再取得をスキップ

---

### 3.2 ツイート分析・プロファイル構築 (`postlikeme/analysis/`)

生のツイートを LLM への指示に必要な構造化「ボイスプロファイル」に変換するシステムの知的中核。

**分析パイプライン (順次実行):**

#### Step 1: 前処理 (`preprocessor.py`)
- 純粋なリツイートを除外
- オリジナルツイートとリプライを分離
- t.co URL の除去、Unicode 正規化、絵文字位置のタグ付け
- 言語検出
- 結果: クリーンなツイートコーパス (`originals` と `replies`)

#### Step 2: 構造分析 (`structural_analyzer.py`)
spaCy + カスタムヒューリスティクスで定量的スタイル特徴を抽出:

- **語彙指標**: タイプトークン比、語彙豊かさ (Yule's K)、平均単語長、頻出非ストップワード (上位50)
- **文構造**: 平均文長、文長分散、断片 vs 完全文、疑問文頻度、感嘆文頻度
- **句読点パターン**: 省略記号、ダッシュ、括弧、シリアルカンマ
- **大文字パターン**: ALL CAPS 頻度、文頭大文字 vs 小文字
- **絵文字分析**: ツイートあたりの頻度、使用頻度ランキング、位置傾向 (先頭/中/末尾)
- **ハッシュタグパターン**: 頻度、配置 (インライン vs 末尾)、キャメルケース vs 小文字
- **メンションパターン**: 頻度、最頻メンションアカウント
- **ツイート長分布**: ヒストグラム、平均、中央値、標準偏差
- **時間パターン**: 時間帯/曜日別投稿頻度

#### Step 3: トピック・意味分析 (`topic_analyzer.py`)
- **トピック抽出**: spaCy NER + TF-IDF で頻出トピックを特定
- **トピック頻度ランキング**: 最新性で指数減衰重み付け
- **意味クラスター** (任意): 全ツイート埋め込み → HDBSCAN/K-Means クラスタリング
- **意見傾向**: トピックごとのセンチメント分析 (例: `{"AI": "enthusiastic", "政治": "cynical"}`)

#### Step 4: 性格・トーン分析 (`personality_analyzer.py`)
ヒューリスティクスと LLM コール1回の組み合わせ:

**ヒューリスティクス:**
- フォーマル度: 短縮形使用、スラング頻度、文の完全性
- ユーモア指標: "lol"/"lmao"頻度、オチ構造、アイロニーマーカー
- 主張度: 断定文 vs 控えめ表現の比率
- リプライでのエンゲージメントスタイル

**LLM 支援分析:**
- 代表的なツイート30-50件をサンプリングし LLM に構造化分析を依頼
- 評価項目: ユーモアスタイル、コミュニケーショントーン、Big Five 近似、修辞技法、キャッチフレーズ、意見相違の処理方法
- プロファイル構築1回あたり1 API コール

#### Step 5: リプライスタイル分析 (`reply_analyzer.py`)
- リプライ長 vs オリジナルツイート長
- 冒頭パターン (挨拶?直接本題?引用?)
- 賛成/反対パターン
- リプライでの質問使用
- オリジナルからリプライへのトーン変化
- 情報追加 vs 意見 vs ユーモア の傾向

#### 出力: ボイスプロファイル (`VoiceProfile`)

```python
VoiceProfile:
    username: str
    display_name: str
    bio: str
    profile_built_at: datetime
    tweet_count_analyzed: int
    date_range: (earliest, latest)

    structural_style:
        avg_tweet_length, tweet_length_distribution
        avg_sentence_length, vocabulary_richness
        top_vocabulary, punctuation_patterns
        capitalization_patterns, fragment_ratio
        question_ratio, exclamation_ratio

    emoji_profile:
        usage_rate, top_emojis, positional_tendency

    hashtag_profile:
        usage_rate, top_hashtags, placement_style, casing_style

    topic_profile:
        topics_ranked, opinion_map, semantic_clusters

    personality_profile:
        formality_score (0.0-1.0)
        humor_style, tone, big_five_approximation
        rhetorical_devices, catchphrases, assertiveness

    reply_style:
        avg_reply_length, opening_patterns
        tone_vs_originals, engagement_type

    example_tweets:
        originals: list[str]          # 20-30件の代表的オリジナル
        replies: list[(str, str)]     # 15-20件の (文脈, リプライ) ペア

    raw_statistics: dict
```

---

### 3.3 ツイート生成エンジン (`postlikeme/generation/`)

**プロンプトエンジニアリング戦略:**

研究によると、**明示的なスタイル指示付き few-shot プロンプティング**は zero-shot を大幅に上回る (スタイル忠実度 <7% → 最大94.7%)。3層構造のプロンプトを使用:

**Layer 1 - システムプロンプト (キャラクター定義):**
`personality_profile` と `structural_style` から導出されたペルソナ記述。技術仕様ではなくキャラクター描写として記述 (例: 「平均2.3絵文字/ツイート」ではなく「よく🔥や🚀で考えを強調する」)。

**Layer 2 - Few-Shot 例:**
`example_tweets` から選択。高エンゲージメントのツイートを優先（最も「声」を表す）。

**Layer 3 - タスクプロンプト:**
- オリジナルツイート: 「[トピック] についてこの人として書いて」
- リプライ: 「このツイートに対してこの人としてリプライして」

**後処理 (`post_processor.py`):**
- 長さチェック (280文字制限)
- スタイル整合性チェック (絵文字数、ハッシュタグ配置、大文字パターン)
- コンテンツ安全フィルター
- 重複排除 (TF-IDF コサイン類似度)
- 候補ランキング (3-5候補生成 → スタイル準拠度でスコアリング)

---

### 3.4 CLI インターフェース (`postlikeme/cli/`)

```
postlikeme
    collect <username> [--count N] [--include-replies] [--collector api|scrape|archive --file path]
    analyze <username> [--rebuild] [--skip-llm]
    generate tweet <username> [--topic TOPIC] [--count N] [--temperature T]
    generate reply <username> --to-tweet "text or URL" [--count N]
    profile show <username>
    profile list
    profile export <username> [--format json|markdown]
    profile delete <username>
    config init
    config show
    config set <key> <value>
```

---

## 4. データフロー

```
                    [X/Twitter]
                         |
          +--------------+--------------+
          |              |              |
     [Tweepy API]  [twscrape]   [Archive Import]
          |              |              |
          +--------------+--------------+
                         |
                   BaseCollector
                         |
                    [RawTweet]s
                         |
                   +-----+-----+
                   |           |
              [SQLite Cache]   |
                   |           |
                   +-----------+
                         |
                  Analysis Pipeline
                         |
          +--------------+--------------+--------------+
          |              |              |              |
    Preprocessor   Structural    Topic/Semantic   Personality
                    Analyzer      Analyzer         Analyzer
          |              |              |              |
          +--------------+--------------+--------------+
                         |
                  [VoiceProfile JSON]
                         |
              ~/.postlikeme/profiles/<username>.json
                         |
                  Prompt Builder
                         |
          +---------+----+----+---------+
          |         |         |         |
       System    Few-shot    Task    Post-proc
       Prompt    Examples   Prompt   Rules
          |         |         |         |
          +---------+---------+---------+
                         |
                    LLM Client (Claude / GPT)
                         |
                  [Generated Text]
                         |
                   Post-Processor
                   (validate, score, rank)
                         |
                  [Ranked Candidates]
                         |
                    CLI Display (Rich table)
```

---

## 5. プロジェクト構造

```
PostLikeMe/
├── LICENSE
├── README.md
├── pyproject.toml
├── .env.example
├── .gitignore
│
├── src/
│   └── postlikeme/
│       ├── __init__.py
│       ├── __main__.py
│       │
│       ├── cli/
│       │   ├── __init__.py
│       │   ├── app.py               # Typer アプリルート
│       │   ├── collect_cmd.py       # collect コマンド
│       │   ├── analyze_cmd.py       # analyze コマンド
│       │   ├── generate_cmd.py      # generate tweet / reply コマンド
│       │   ├── profile_cmd.py       # profile show/list/export/delete
│       │   └── config_cmd.py        # config init/show/set
│       │
│       ├── collectors/
│       │   ├── __init__.py
│       │   ├── base.py              # BaseCollector ABC, RawTweet
│       │   ├── tweepy_collector.py
│       │   ├── twscrape_collector.py
│       │   ├── archive_collector.py
│       │   └── rate_limiter.py
│       │
│       ├── analysis/
│       │   ├── __init__.py
│       │   ├── pipeline.py          # 分析パイプライン統合
│       │   ├── preprocessor.py
│       │   ├── structural_analyzer.py
│       │   ├── topic_analyzer.py
│       │   ├── personality_analyzer.py
│       │   ├── reply_analyzer.py
│       │   └── example_selector.py
│       │
│       ├── generation/
│       │   ├── __init__.py
│       │   ├── prompt_builder.py
│       │   ├── llm_client.py
│       │   ├── tweet_generator.py
│       │   ├── reply_generator.py
│       │   └── post_processor.py
│       │
│       ├── models/
│       │   ├── __init__.py
│       │   ├── raw_tweet.py
│       │   ├── voice_profile.py
│       │   └── config.py
│       │
│       ├── storage/
│       │   ├── __init__.py
│       │   ├── tweet_cache.py
│       │   ├── profile_store.py
│       │   └── config_store.py
│       │
│       ├── templates/
│       │   ├── system_prompt.j2
│       │   ├── tweet_prompt.j2
│       │   └── reply_prompt.j2
│       │
│       └── utils/
│           ├── __init__.py
│           ├── text.py
│           ├── logging.py
│           └── constants.py
│
├── tests/
│   ├── conftest.py
│   ├── test_collectors/
│   ├── test_analysis/
│   ├── test_generation/
│   ├── test_cli/
│   └── fixtures/
│       ├── sample_tweets.json
│       └── sample_profile.json
│
└── docs/
    ├── architecture.md
    ├── prompt-engineering.md
    └── ethical-guidelines.md
```

**ランタイムストレージ:**

```
~/.postlikeme/
├── config.toml
├── cache/
│   └── tweets.db
├── profiles/
│   ├── elonmusk.json
│   └── ...
└── logs/
    └── postlikeme.log
```

---

## 6. 実装フェーズ

### Phase 1: 基盤構築
**目標**: インストール・起動可能なスケルトンプロジェクト

1. `pyproject.toml` セットアップ (メタデータ、依存関係、CLI エントリポイント)
2. `src/postlikeme/` パッケージ構造作成
3. Typer CLI スケルトン（全コマンドグループのスタブ）
4. TOML 設定管理 (`config init/show/set`)
5. pytest, ruff セットアップ
6. `.gitignore` と `.env.example`

**成果物**: `pip install -e .` 動作、`postlikeme --help` で全コマンド表示

### Phase 2: データ収集
**目標**: X アカウントからツイートを収集・キャッシュ

1. `RawTweet` データクラス定義
2. `BaseCollector` ABC 実装
3. `TweepyCollector` (ページネーション、レートリミット対応)
4. `TwscrapeCollector` (アカウントプール)
5. `ArchiveCollector` (JSON/CSV インポート)
6. SQLite キャッシュ (`tweet_cache.py`)
7. `collect` CLI コマンド (Rich プログレスバー付き)
8. モック API レスポンスでのテスト

**成果物**: `postlikeme collect username --count 500` でツイート取得・キャッシュ

### Phase 3: 分析パイプライン
**目標**: 収集ツイートからボイスプロファイルを構築

1. `preprocessor.py` - フィルタリング、クリーニング、分割
2. `structural_analyzer.py` - spaCy による定量的指標
3. `topic_analyzer.py` - TF-IDF トピック、NER、センチメント
4. `personality_analyzer.py` - ヒューリスティクス + LLM 性格評価
5. `reply_analyzer.py` - リプライ固有の行動パターン
6. `example_selector.py` - 代表的ツイートの階層的選択
7. `VoiceProfile` データクラスと JSON シリアライズ
8. `analysis/pipeline.py` 統合
9. `analyze` と `profile` CLI コマンド

**成果物**: `postlikeme analyze username` で完全なボイスプロファイル生成

### Phase 4: 生成エンジン
**目標**: スタイル一致のツイート・リプライ生成

1. `llm_client.py` (Anthropic / OpenAI バックエンド)
2. Jinja2 プロンプトテンプレート設計・反復
3. `prompt_builder.py` - VoiceProfile → プロンプト組み立て
4. `tweet_generator.py` - N候補生成、リトライ処理
5. `reply_generator.py` - 文脈解析 + 生成
6. `post_processor.py` - 長さ検証、スタイルスコアリング、重複排除
7. `generate tweet` と `generate reply` CLI コマンド

**成果物**: `postlikeme generate tweet username --topic "AI" --count 5` で5候補生成

### Phase 5: 品質向上
**目標**: プロダクション品質の CLI 体験

1. `--verbose` / `--quiet` フラグ
2. 構造化ログ (ローテーティングファイル出力)
3. ユーザーフレンドリーなエラーハンドリング
4. クリップボードコピー対応
5. `--output` ファイル出力フラグ
6. 統合テスト (collect → analyze → generate フルワークフロー)
7. README 作成

### Phase 6: 将来の拡張
- スレッド生成 (マルチツイート)
- プロファイル比較
- スタイル転送 (トピック X をアカウント Y のスタイルで)
- Web UI (FastAPI + HTMX / Streamlit)
- ファインチューニング (LoRA)
- スケジュール生成 (人間承認ループ付き)

---

## 7. 主要技術決定

### 7.1 ツイート収集方法
**決定**: 3つのバックエンドをプラグイン可能なアーキテクチャで対応。

- X API v2 Free tier は読み取り不可（書き込みのみ）→ Basic tier ($200/月) が最低要件
- twscrape は無料だが不安定（X のアンチボット対策）
- アーカイブインポートは API/スクレイピング両方が使えない場合のフォールバック
- **必要データ量**: 最低200ツイート（基本分析）、500+（信頼性の高い分析）、1000+（最高品質）

### 7.2 文体分析方法
**決定**: 決定的 NLP 特徴量 + LLM 1回コールのハイブリッド方式。

- 定量的特徴量（語彙豊かさ、文長、絵文字頻度）は決定的に測定 → 生成制約として使用
- 性格・ユーモア・修辞的洗練度は LLM で評価（30-50ツイートサンプル、1回のAPIコール）

### 7.3 コンテンツ生成方法
**決定**: 明示的スタイル指示付き few-shot プロンプティング。初期実装ではファインチューニングなし。

- 研究結果: few-shot はスタイル忠実度を <7% → 最大94.7% に向上
- VoiceProfile がスタイル指示 (system prompt) と few-shot 例の両方のデータを提供
- Temperature 0.85 で創造性とスタイル準拠のバランス
- 3-5候補生成 → ポストプロセッサでランキング

### 7.4 プロファイルストレージ
**決定**: JSON ファイル (1プロファイル/1ファイル)。

- 人間が読める・編集可能
- 小サイズ (50-200 KB) → DB 不要
- バックアップ・共有・バージョン管理が容易
- `schema_version` フィールドでマイグレーション対応

---

## 8. プライバシー・倫理的考慮

### システムに組み込むガードレール

1. **自動投稿なし**: 生成のみ。`post` / `publish` コマンドは意図的に省略
2. **同意の認識**: `collect` コマンドで X の利用規約・適用法令の遵守は利用者の責任であると表示
3. **データ最小化**: ツイートテキストと基本メタデータのみ保存。DM・非公開ツイート・フォロワーリストは収集しない
4. **なりすまし警告**: 生成コンテンツに「@username のスタイルで AI が生成」と明記
5. **コンテンツフィルター**: ブロックリストで特定トピック・用語・センチメントを除外可能
6. **透明性**: プロファイルは読み取り可能な JSON で保存
7. **平文での認証情報保存なし**: 設定ファイルは制限パーミッション (0600)

### 推奨倫理的使用ポリシー
- 自分のアカウントまたはアカウント所有者の明示的許可がある場合にのみ使用
- AI 生成コンテンツの公開時は常に開示
- なりすまし・欺瞞・ハラスメント・偽情報拡散に使用しない
- X の利用規約と API 使用ポリシーを尊重
- 自動ボットアカウント用の生成コンテンツに使用しない

---

## 9. 依存関係まとめ

**コア (必須):**
- `typer[all]>=0.12` - CLI
- `rich>=13.0` - ターミナルフォーマット
- `tweepy>=4.14` - X API v2
- `anthropic>=0.40` - Claude API
- `spacy>=3.7` - NLP 分析
- `textstat>=0.7` - 可読性指標
- `jinja2>=3.1` - プロンプトテンプレート
- `aiosqlite>=0.20` - 非同期 SQLite
- `pydantic>=2.0` - データバリデーション
- `httpx>=0.27` - 非同期 HTTP

**オプション:**
- `twscrape>=0.12` - スクレイピング
- `openai>=1.50` - OpenAI バックエンド
- `sentence-transformers>=3.0` - 意味クラスタリング
- `keyring>=25.0` - セキュア認証情報保存

**開発:**
- `pytest>=8.0`
- `pytest-asyncio>=0.24`
- `ruff>=0.8`
- `mypy>=1.13`

---

## 10. リスク評価

| リスク | 可能性 | 影響 | 軽減策 |
|--------|--------|------|--------|
| X API 料金のさらなる値上げ | 中 | 高 | マルチバックエンド、アーカイブインポート |
| twscrape がアンチボット対策で動作不能 | 高 | 中 | API へのフォールバック、アーカイブインポート |
| LLM API コスト | 中 | 中 | 分析結果キャッシュ、バッチ生成、安価モデル選択 |
| 生成コンテンツのスタイル不一致 | 中 | 高 | 反復的プロンプト改善、候補ランキング、手動プロファイル調整 |
| 収集時のレートリミット | 高 | 低 | レートリミッター、再開サポート、キャッシュ |
| なりすまし悪用 | 中 | 高 | 自動投稿なし、倫理ガイドライン、開示ラベル |
