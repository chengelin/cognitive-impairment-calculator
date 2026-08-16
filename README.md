# 双模式经颅超声认知障碍风险计算器

## 运行方法

双击 `start.bat`。首次运行会创建 Python 环境并安装依赖，完成后浏览器会自动打开。

也可以在命令行中运行：

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe -m streamlit run app.py
```

## 模型口径

- 输入为 8 个临床原始值，不是标准化后的 z 分数。
- 每个样本分别通过 5 个外层交叉验证折的标准化器和 XGBoost 模型。
- 页面显示 5 个模型预测概率的算术平均值。
- 分类阈值固定为论文分析使用的 0.50。
- 页面中的个体贡献为未校准 XGBoost 决策空间的方向性解释；最终风险概率来自校准后的五折集成。

本工具目前仅完成内部验证，只用于科研展示和辅助筛查研究，不能替代 MoCA-B、神经心理评估或临床诊断。

