#!/usr/bin/env bash
# diva 起動スクリプト
#
# Finder でダブルクリックすると、ターミナルが開いてアプリがブラウザで立ち上がります。
#
# 【大事】このターミナルの窓は、アプリを使っているあいだ開けたままにしてください。
#         窓を閉じるとアプリも止まります。終わるときは Control + C。

# 【実行中に編集されても壊れないようにする】
# bash はスクリプトを「何バイト目まで読んだか」で追いながら実行するので、
# 実行中にファイルを書き換えると、ずれた位置から読み直して同じ行を二度
# 実行することがある（実測: 更新中に編集したら取得が二重に走った）。
# 本体を関数にしておけば、bash は定義を丸ごと読んでから実行するので影響を受けない。
__main_body() {
cd "$(dirname "$0")"

echo ""
echo "=================================================="
echo "  🔭 DIVA"
echo "=================================================="
echo ""

die() { echo ""; echo "⚠ $1"; echo ""; read -n 1 -s -r -p "Enter キーでこの窓を閉じます…"; exit 1; }

# ── 環境の用意（初回だけ）──
if [ ! -d ".venv" ]; then
  command -v uv >/dev/null 2>&1 || die "uv が見つかりません。ターミナルで次を実行してください:
    curl -LsSf https://astral.sh/uv/install.sh | sh"
  echo "▶ 初回起動: 環境を作っています（数分かかります）…"
  uv venv --python 3.11 || die "環境の作成に失敗しました"
  uv pip install -r requirements.txt || die "必要なものの取得に失敗しました"
  shasum requirements.txt > .venv/.reqs.sha
elif ! shasum -c .venv/.reqs.sha >/dev/null 2>&1; then
  # 毎回 uv を呼ぶと起動が遅くなるので、requirements.txt が変わったときだけ更新する
  echo "▶ 必要なものを更新しています…"
  uv pip install -r requirements.txt || die "更新に失敗しました"
  shasum requirements.txt > .venv/.reqs.sha
fi

[ -f "data/scout.db" ] || die "データがまだありません。先にターミナルで次を実行してください（約15分）:
    cd \"$(pwd)\"
    .venv/bin/python scripts/weekly_scan.py"

# ── 空いているポートを探す ──
# 8502 が誰かに使われていると Streamlit は起動に失敗する。黙って別の番号に逃がす。
PORT=8502
while lsof -nP -iTCP:$PORT -sTCP:LISTEN >/dev/null 2>&1; do
  PORT=$((PORT+1))
  [ $PORT -gt 8520 ] && die "空いているポートが見つかりませんでした"
done

URL="http://localhost:$PORT"

echo "▶ 起動中… $URL"
echo ""

# ── サーバーを起動し、応答を確認してからブラウザを開く ──
# Streamlit 任せにすると環境によってブラウザが開かないことがあるので、
# ここで自分で開く。
# .venv/bin/streamlit は作ったときのフォルダの場所を中に書き込んでいるので、
# フォルダの名前を変えると動かなくなる（実測: dividend-scout → diva で起動できなくなった）。
# python -m で呼べば場所に依存しない。
.venv/bin/python -m streamlit run app.py \
  --server.port "$PORT" \
  --server.headless true \
  --browser.gatherUsageStats false &
PID=$!
trap 'kill $PID 2>/dev/null' EXIT INT TERM

OPENED=0
for i in $(seq 1 100); do
  if curl -fsS "$URL/healthz" >/dev/null 2>&1; then
    open "$URL"
    OPENED=1
    echo ""
    echo "=================================================="
    echo "  ✅ ブラウザで開きました: $URL"
    echo ""
    echo "  この窓は開けたままにしてください。"
    echo "  終わるときは Control + C。"
    echo "=================================================="
    echo ""
    break
  fi
  kill -0 $PID 2>/dev/null || die "起動に失敗しました。上に出ているメッセージを確認してください"
  sleep 0.3
done

# 30秒待っても応答が無かったとき、黙って待ち続けると「開かない」に見える。
if [ "$OPENED" != "1" ]; then
  echo ""
  echo "⚠ 30秒待ちましたが、まだ応答がありません。"
  echo "  ブラウザで次を開いてみてください: $URL"
  echo "  それでも開かないときは、この窓に出ているメッセージを確認してください。"
  echo ""
fi

wait $PID

}

__main_body "$@"
