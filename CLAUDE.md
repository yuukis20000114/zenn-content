# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Zenn 記事管理リポジトリ。Docker コンテナ上で Claude Code・Codex・MCP を使いながら技術記事を執筆する。

## Common Commands

```bash
# 文章校正（textlint）
npx textlint articles/<slug>.md          # 特定記事を校正
npx textlint articles/*.md               # 全記事を校正
npx textlint --fix articles/<slug>.md    # 自動修正

# Code quality (ruff)
uv run ruff check .           # Run linter
uv run ruff format .          # Format code

# Docker operations (run from host)
docker compose up -d --build  # Build and start container
docker compose exec app bash  # Enter container shell
docker compose down -v        # Stop and remove volumes
```

## Project Structure

- `articles/` — Zenn 技術記事（Markdown）
- `images/` — 記事に埋め込む画像ファイル

## Zenn 記事フォーマット

記事ファイルは `articles/<slug>.md` に配置する。slug は半角英数字・ハイフン・アンダースコアで12〜50文字。

### フロントマター

```yaml
---
title: "記事タイトル"
emoji: "📝"
type: "tech"      # tech: 技術記事 / idea: アイデア記事
topics: ["topic1", "topic2"]  # 最大5つ
published: false  # true で公開
---
```

### 画像の埋め込み

```markdown
![alt text](/images/<filename>)
```

## Configuration Files

- `pyproject.toml` — ruff 設定
- `.claude.json` — MCP server 設定
- `.textlintrc.json` — 日本語文章校正ルール
- `AGENTS.md` — OpenAI Codex レビュワー設定

## MCP Servers Available

| Server | Purpose |
|--------|---------|
| textlint | Japanese text proofreading |
| serena | Semantic code analysis (LSP) |
| context7 | Latest library documentation lookup |
| playwright | Browser automation |
| sequential-thinking | Complex task decomposition |

## Code Style

- Python 3.10-3.11 compatible
- Line length: 88 characters
- Uses ruff for linting and formatting
- Quote style: double quotes
- Indent: 4 spaces

## Codex レビュワー

OpenAI Codex CLI がレビュー専用ツールとしてインストール済み。ユーザーからレビュー依頼があった場合に使用する。

**レビュー実行（推奨）:**
```bash
bash scripts/codex-review.sh              # main との差分をレビュー
bash scripts/codex-review.sh develop       # 任意のブランチとの差分をレビュー
bash scripts/codex-review.sh --file <path> # 特定ファイルをレビュー
```

**ルール:**
- Codex はレビュー専用。ファイル変更は Claude Code が行う
- `--sandbox read-only` を必ず使用すること
- レビュー結果は日本語で要約してユーザーに報告すること

## Permission Policy

Development commands are auto-approved via `Bash(*)`. The following are explicitly DENIED:

| Category | Denied Patterns |
|----------|----------------|
| Filesystem destruction | `rm -rf /`, `dd`, `mkfs` |
| System control | `shutdown`, `reboot`, `halt`, `poweroff` |
| Privilege escalation | `sudo` |
| Destructive git | `push --force`, `push -f`, `reset --hard` |
| Docker destruction | `docker rm`, `docker system prune`, `docker rmi` |
| Network commands | `curl`, `wget`（requires confirmation each time） |

MCP tools (textlint, serena, context7, playwright, sequential-thinking) are also auto-approved.
