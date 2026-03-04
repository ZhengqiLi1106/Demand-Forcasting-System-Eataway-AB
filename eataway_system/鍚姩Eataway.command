#!/bin/bash
# ============================================================
# Eataway 启动脚本 — 双击即可运行
# 放到桌面或 eataway 文件夹，双击启动
# ============================================================

# 脚本所在目录
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# 激活 Python 环境（如果你用 conda 或 venv，取消下面对应注释）
# source ~/anaconda3/bin/activate eataway
# source "$DIR/venv/bin/activate"

# 检查 Flask 是否安装
if ! python3 -c "import flask" 2>/dev/null; then
    echo "正在安装 Flask..."
    pip3 install flask --quiet
fi

echo ""
echo "  ╔══════════════════════════════╗"
echo "  ║  EATAWAY 预测系统启动中...  ║"
echo "  ║  http://localhost:5000       ║"
echo "  ╚══════════════════════════════╝"
echo ""

# 启动 Flask（会自动打开浏览器）
python3 "$DIR/app.py"
