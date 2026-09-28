#!/bin/bash
# クラウド(Claude Code on the web)のセッション開始時だけ、依存を入れてテストを走らせる。
# パソコンでは何もしない（uv の環境をそのまま使う）。
set -euo pipefail
[ "${CLAUDE_CODE_REMOTE:-}" = "true" ] || exit 0
cd "$CLAUDE_PROJECT_DIR"
pip install -q -r requirements.txt pytest 2>&1 | grep -v "Running pip as the 'root'" || true
python3 -m pytest tests -q 2>&1 | tail -1
