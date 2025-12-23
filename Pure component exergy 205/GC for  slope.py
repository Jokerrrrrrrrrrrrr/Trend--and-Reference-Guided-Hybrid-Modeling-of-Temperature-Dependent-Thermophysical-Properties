# linner model
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# 1. 读取数据
file_path = "Pure component exergy 205.xlsx"  # 替换为你实际的文件路径
df = pd.read_excel(file_path, sheet_name="Sheet1")

# 2. 获取温度和目标值
temp_cols = df.columns[31:41]  # 10个温度点列（AF到AO）
target_cols = df.columns[41:51]  # 10个目标值列（AP到AY）

# 3. 计算每个物质的9个斜率，并取其中位数
slope_medians = []
for i, row in df.iterrows():
    temps = row[temp_cols].values
    targets = row[target_cols].values

    slopes = [(targets[t+1] - targets[t]) / (temps[t+1] - temps[t]) for t in range(len(temps)-1)]
    slope_medians.append(np.median(slopes))

target_slopes = np.array(slope_medians)

# 4. 获取基团浓度（M到AE列）
group_cols = df.columns[12:31]  # M到AE是基团浓度列
X = df[group_cols].values
y = target_slopes

# 5. 创建并训练回归模型
model = LinearRegression()
model.fit(X, y)

# 6. 预测
predicted_slopes = model.predict(X)

# 7. 计算评价指标
mse = mean_squared_error(y, predicted_slopes)
r2 = r2_score(y, predicted_slopes)

print(f"MSE = {mse:.4f}")
print(f"R² = {r2:.4f}")

# 8. 创建输出 DataFrame，仅包含序号、实际斜率和预测斜率
output_df = pd.DataFrame({
    "序号": df.index + 1,
    "实际斜率": target_slopes,
    "预测斜率": predicted_slopes
})

# 9. 保存预测结果
output_df.to_excel("predictions_with_slopes_only.xlsx", index=False)
print("预测结果已保存为: predictions_with_slopes_only.xlsx")

#
# #ml model
# import pandas as pd
# import numpy as np
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.metrics import mean_squared_error, r2_score
# from sklearn.model_selection import train_test_split
#
# # 1. 读取数据
# file_path = "Pure component exergy 205.xlsx"  # 替换为你实际的文件路径
# df = pd.read_excel(file_path, sheet_name="Sheet1")
#
# # 2. 获取温度和目标值
# temp_cols = df.columns[31:41]  # AF-AO
# target_cols = df.columns[41:51]  # AP-AY
#
# # 3. 计算每个物质的9个斜率，并取中位数
# slope_medians = []
# for i, row in df.iterrows():
#     temps = row[temp_cols].values
#     targets = row[target_cols].values
#     slopes = [(targets[t+1] - targets[t]) / (temps[t+1] - temps[t]) for t in range(len(temps)-1)]
#     slope_medians.append(np.median(slopes))
#
# target_slopes = np.array(slope_medians)
#
# # 4. 获取基团浓度（M-AE列）
# group_cols = df.columns[12:31]
# X = df[group_cols].values
# y = target_slopes
#
# # 5. 数据划分（训练集/测试集 8:2）
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#
# # 6. 创建随机森林模型
# rf_model = RandomForestRegressor(n_estimators=200, random_state=42)
# rf_model.fit(X_train, y_train)
#
# # 7. 预测
# y_train_pred = rf_model.predict(X_train)
# y_test_pred = rf_model.predict(X_test)
# y_all_pred = rf_model.predict(X)  # 对所有数据预测，用于保存到Excel
#
# # 8. 计算评价指标
# mse_train = mean_squared_error(y_train, y_train_pred)
# r2_train = r2_score(y_train, y_train_pred)
#
# mse_test = mean_squared_error(y_test, y_test_pred)
# r2_test = r2_score(y_test, y_test_pred)
#
# print(f"训练集 MSE = {mse_train:.4f}, R² = {r2_train:.4f}")
# print(f"测试集 MSE = {mse_test:.4f}, R² = {r2_test:.4f}")
#
# # 9. 保存预测结果
# output_df = pd.DataFrame({
#     "序号": df.index + 1,
#     "实际斜率": target_slopes,
#     "预测斜率": y_all_pred
# })
# output_df.to_excel("rf_predictions_with_slopes.xlsx", index=False)
# print("预测结果已保存为: rf_predictions_with_slopes.xlsx")


#devide train and test
# import pandas as pd
# import numpy as np
# from sklearn.linear_model import LinearRegression
# from sklearn.metrics import mean_squared_error, r2_score
# from sklearn.model_selection import train_test_split  # 导入数据划分工具
#
# # 1. 读取数据
# file_path = "Pure component exergy 205.xlsx"  # 替换为你实际的文件路径
# df = pd.read_excel(file_path, sheet_name="Sheet1")
#
# # 2. 获取温度和目标值
# # 温度数据（AF到AO列）
# temp_cols = df.columns[31:41]  # 10个温度点列（AF到AO）
#
# # 目标值（AP到AY列）
# target_cols = df.columns[41:51]  # 10个目标值列（AP到AY）
#
# # 3. 计算每个物质的9个斜率，并取其中位数
# slope_medians = []
#
# for i, row in df.iterrows():
#     # 获取温度数据和目标值
#     temps = row[temp_cols].values
#     targets = row[target_cols].values
#
#     # 计算相邻温度点之间的斜率
#     slopes = []
#     for t in range(len(temps) - 1):
#         slope = (targets[t + 1] - targets[t]) / (temps[t + 1] - temps[t])
#         slopes.append(slope)
#
#     # 计算斜率的中位数，作为目标预测值
#     median_slope = np.median(slopes)
#     slope_medians.append(median_slope)
#
# # 将中位数斜率作为目标值
# target_slopes = np.array(slope_medians)
#
# # 4. 获取基团浓度（M到AE列）
# group_cols = df.columns[12:31]  # M到AE是基团浓度列
#
# # 5. 使用基团浓度预测斜率
# X = df[group_cols].values  # 基团浓度数据
# y = target_slopes  # 目标斜率（中位数）
#
# # 6. 数据划分（8:2）
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#
# # 创建回归模型
# model = LinearRegression()
#
# # 训练模型
# model.fit(X_train, y_train)
#
# # 预测斜率
# predicted_train_slopes = model.predict(X_train)
# predicted_test_slopes = model.predict(X_test)
#
# # 7. 评估模型的性能（训练集和测试集）
# mse_train = mean_squared_error(y_train, predicted_train_slopes)
# r2_train = r2_score(y_train, predicted_train_slopes)
#
# mse_test = mean_squared_error(y_test, predicted_test_slopes)
# r2_test = r2_score(y_test, predicted_test_slopes)
#
# print(f"训练集 MSE = {mse_train:.4f}")
# print(f"训练集 R² = {r2_train:.4f}")
#
# print(f"测试集 MSE = {mse_test:.4f}")
# print(f"测试集 R² = {r2_test:.4f}")
#
# # 8. 创建输出 DataFrame，仅包含序号、实际斜率和预测斜率
# output_df = pd.DataFrame({
#     "序号": df.index + 1,  # 物质的序号，从1开始
#     "实际斜率": target_slopes,  # 实际斜率（中位数）
#     "预测斜率": model.predict(X)  # 预测斜率（使用完整数据集进行预测）
# })
#
# # 9. 保存预测结果
# output_df.to_excel("predictions_with_slopes_only_8to2_split.xlsx", index=False)  # 保存为新的Excel文件
# print("预测结果已保存为: predictions_with_slopes_only_8to2_split.xlsx")
