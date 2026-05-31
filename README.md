# zenn-content

[Zenn](https://zenn.dev/) と GitHub 連携した記事管理リポジトリ。

## Structure

```
articles/   — Zenn 技術記事（Markdown）
images/     — 記事に埋め込む画像
```

## Setup

```bash
docker compose up -d --build
docker compose exec app bash
```

## Writing

記事ファイルは `articles/<slug>.md` に配置。

```yaml
---
title: "記事タイトル"
emoji: "📝"
type: "tech"
topics: ["topic1", "topic2"]
published: false
---
```

## Tools

- **Claude Code** — AI アシスタント
- **Codex CLI** — コードレビュー
- **textlint** — 日本語文章校正
