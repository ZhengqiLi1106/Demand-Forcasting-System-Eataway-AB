# Eataway 预测系统

## 文件结构
```
eataway_system/
├── 启动Eataway.command   ← 双击这个启动
├── app.py                ← Flask 服务器
└── templates/
    └── index.html        ← 网页界面
```

## 第一次使用

1. 把 `eataway_system/` 文件夹放到桌面 Eataway 文件夹里
2. 打开终端，运行：
   ```bash
   chmod +x ~/Desktop/eataway/eataway_system/启动Eataway.command
   ```
   （只需要做一次）
3. 以后每次双击 `启动Eataway.command` 即可

## 每周操作流程

1. 双击 `启动Eataway.command`
2. 浏览器自动打开 `http://localhost:5000`
3. 点击顶部 **「Kör Pipeline」** 标签页
4. 点击 **「Kör Veckopipeline」** 按钮
5. 等待 5~10 分钟，页面自动刷新
6. 切到 **「Chaufförsvy」** 或 **「Köksvy」** 查看结果
7. 点击 **「↓ Ladda ner CSV」** 下载当周数据

## 修改路径

如果你的文件路径不是 `/Users/zihaoyang/Desktop/eataway`，
打开 `app.py`，修改开头的：

```python
BASE_DIR      = Path("/Users/zihaoyang/Desktop/eataway")
OUTPUT_DIR    = BASE_DIR / "output_v7"
FEATURE_SCRIPT = BASE_DIR / "feature_v3.py"
TRAIN_SCRIPT   = BASE_DIR / "eataway_train_v7.py"
```

## 安装依赖（第一次）

```bash
pip3 install flask pandas lightgbm scikit-learn
```
