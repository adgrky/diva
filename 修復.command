#!/usr/bin/env bash
# 検算が消してしまった保有を元に戻す（2026-09-28 の修正ぶん。1回やれば十分）
#
# Finder でダブルクリックするだけ。まず何が起きたかを見せて、
# 「戻しますか？」と聞いてから戻します。
__main_body() {
cd "$(dirname "$0")"
PY=".venv/bin/python"
[ -x "$PY" ] || PY="python3"

echo ""
echo "=================================================="
echo "  🩹 消えた保有の確認"
echo "=================================================="
echo ""
$PY scripts/repair_audit_leak.py
echo ""
if $PY scripts/repair_audit_leak.py | grep -q "表示だけです"; then
  read -r -p "上の内容で元に戻しますか？ (y を押して Enter / やめるならそのまま Enter) " ans
  if [ "$ans" = "y" ]; then
    $PY scripts/repair_audit_leak.py --apply
  else
    echo "何も変えずに終わります。"
  fi
fi
echo ""
read -n 1 -s -r -p "Enter キーでこの窓を閉じます…"
}
__main_body "$@"
