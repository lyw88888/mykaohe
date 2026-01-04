# train_model.py
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import pickle

# ✅ 修正1：使用原始字符串读取 CSV 文件（注意文件名！）
df = pd.read_csv(r"D:\streamlit_env\student_data_adjusted_rounded.csv")

# 打印数据形状确认加载成功
print("数据加载成功，形状:", df.shape)

feature_cols = ["每周学习时长（小时）", "上课出勤率", "期中考试分数", "作业完成率"]
target_col = "期末考试分数"

X = df[feature_cols]
y = df[target_col]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"✅ 模型训练完成！MSE: {mse:.2f}, R²: {r2:.2f}")

# ✅ 修正2：保存到当前目录或指定路径
with open(r"D:\streamlit_env\model.pkl", "wb") as f:
    pickle.dump(model, f)

with open(r"D:\streamlit_env\feature_columns.pkl", "wb") as f:
    pickle.dump(feature_cols, f)

print("✅ 模型已保存！")
