import pandas as pd
import numpy as np
from sklearn.linear_model import HuberRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers

# ========= 1. 读取数据 =========
file_path = "heat capacity 207.xlsx"
df = pd.read_excel(file_path, sheet_name="Sheet1")
df = df.dropna(subset=[df.columns[0]])  # 删除缺失值
df[df.columns[0]] = df[df.columns[0]].astype(int)

# ========= 2. 列定义 =========
group_cols = df.columns[11:30]  # 19个基团列
temp_cols = df.columns[30:40]  # 10个温度点
cp_cols = df.columns[40:50]  # 10个Cp值列
target_column_T1 = 'ASPEN Half Critical T'

# ========= 3. 子模型训练 =========
X_groups = df[group_cols]
valid_mask = ~df[target_column_T1].isna()

poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X_groups[valid_mask])

# GradientBoostingRegressor 预测 T1
y_T1 = df.loc[valid_mask, target_column_T1].values
T1_model = GradientBoostingRegressor(n_estimators=300, learning_rate=0.05, max_depth=4, random_state=0).fit(X_poly,
                                                                                                            y_T1)

# Cp1, Cp2 使用 HuberRegressor
Cp1_model = HuberRegressor(max_iter=9000).fit(X_groups, df.iloc[:, 9])
Cp2_model = HuberRegressor(max_iter=9000).fit(X_groups, df.iloc[:, 50])

# ========= 4. 构建训练数据 =========
X_total, y_total, material_ids, temperatures = [], [], [], []
X_poly_all = poly.transform(X_groups)

for i, row in df.iterrows():
    material_id = row.iloc[0]
    Nk = row[group_cols].values  # 19 个基团
    temps = row[temp_cols].values  # 10 个温度点
    cps = row[cp_cols].values  # 10 个Cp值

    Nk_df = pd.DataFrame([Nk], columns=group_cols)
    Nk_poly = X_poly_all[i:i + 1]

    try:
        # 新模型：直接预测 T1（无需 log 和 exp）
        T1 = T1_model.predict(Nk_poly)[0]
        if T1 <= 0 or np.isnan(T1):
            continue
        T2 = T1 * 1.5
        Cp1 = Cp1_model.predict(Nk_df)[0]
        Cp2 = Cp2_model.predict(Nk_df)[0]
        slope = (Cp2 - Cp1) / (T2 - T1)
    except:
        continue

    # 生成交互项：基团特征与温度点交互
    # 每个基团特征与所有 10 个温度点的交互项，生成 (19, 10) 的矩阵
    Nk_expanded = np.tile(Nk, (10, 1))  # 重复 10 次，构建每个基团与温度的交互矩阵
    Nk_times_temps = Nk_expanded * temps.reshape(-1, 1)  # 进行逐元素相乘，得到 (19, 10) 的交互项矩阵

    # 把每个物质的特征向量展平，最终形成一个向量
    features = np.concatenate([
        Nk_times_temps.flatten(),  # 19*10 交互项展平
        (slope * temps).flatten()  # 斜率与每个温度点的交互项
    ])

    X_total.append(features)  # 添加特征数据
    y_total.append(cps)  # 添加对应的Cp值（10个温度点的Cp值）
    material_ids.append(material_id)
    temperatures.append(temps)

# 确保 X_total 和 y_total 形状一致
X_total = np.array(X_total, dtype=np.float64)  # 转换为 NumPy 数组，确保为 float64 数组
y_total = np.array(y_total, dtype=np.float64)  # 转换为 NumPy 数组，确保为 float64 数组

# 提取已知的温度点（如 T1 和 T2）的 Cp 值
y_known = df[[cp_cols[0], cp_cols[1]]].values  # 假设已知的温度点是 T1 和 T2

y_known = np.array(y_known, dtype=np.float64)  # 转换为 NumPy 数组，确保为 float64 数组

# 打印形状，确保数据一致性
print(f"X_total shape: {X_total.shape}")
print(f"y_total shape: {y_total.shape}")
print(f"y_known shape: {y_known.shape}")

# ========= 6. 构建多任务学习模型 =========
# 模型架构设计：共享底层特征提取层
input_layer = Input(shape=(X_total.shape[1],))  # 输入层

# 底层共享层
shared_layer = Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.01))(input_layer)
shared_layer = Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.01))(shared_layer)

# 主任务分支：预测所有温度点的Cp值
main_output = Dense(10, name='main_output')(shared_layer)  # 输出10个目标值（每个温度点的Cp）

# 辅助任务分支：预测已知温度点的Cp值
auxiliary_output = Dense(y_known.shape[1], name='auxiliary_output')(shared_layer)  # 输出2个目标值（T1和T2的Cp）

# 构建模型
model = Model(inputs=input_layer, outputs=[main_output, auxiliary_output])


# ========= 7. 编译模型 =========
def composite_loss(y_true_main, y_pred_main, y_true_aux, y_pred_aux, alpha=0.5):
    main_loss = np.mean((y_true_main - y_pred_main) ** 2)  # MSE
    aux_loss = np.mean((y_true_aux - y_pred_aux) ** 2)  # MSE for auxiliary task
    return main_loss + alpha * aux_loss  # Alpha is a weight factor for the auxiliary task


# 为每个输出指定指标
model.compile(optimizer=Adam(),
              loss={'main_output': 'mean_squared_error', 'auxiliary_output': 'mean_squared_error'},
              metrics={'main_output': ['mae'], 'auxiliary_output': ['mae']})  # 为每个输出指定mae

# ========= 8. 训练模型 =========
history = model.fit(X_total, {'main_output': y_total, 'auxiliary_output': y_known},
                    epochs=50, batch_size=32)

# ========= 9. 模型评估 =========
# 评估模型
y_pred = model.predict(X_total)
y_pred_main = y_pred[0]  # 主任务预测
y_pred_aux = y_pred[1]  # 辅助任务预测

# 计算R²、MSE和ARD
mse_main = mean_squared_error(y_total, y_pred_main)
r2_main = r2_score(y_total, y_pred_main)

mse_aux = mean_squared_error(y_known, y_pred_aux)
r2_aux = r2_score(y_known, y_pred_aux)

print(f"主任务 R²: {r2_main:.4f}, MSE: {mse_main:.2f}")
print(f"辅助任务 R²: {r2_aux:.4f}, MSE: {mse_aux:.2f}")

# ========= 10. 输出预测结果 =========
# 扁平化 y_total, y_pred_main, y_pred_aux
y_total_flat = y_total.flatten()  # 扁平化为一维数组
y_pred_main_flat = y_pred_main.flatten()  # 扁平化为一维数组
y_pred_aux_flat = y_pred_aux.flatten()  # 扁平化为一维数组

# 创建 DataFrame
results = pd.DataFrame({
    "Material_ID": np.repeat(material_ids, 10),  # 每个物质重复10次，对应10个温度点
    "Temperature (K)": np.tile(temperatures, 10),  # 对应的10个温度点
    "Cp_measured": y_total_flat,  # 扁平化后的Cp值
    "Cp_predicted_main": y_pred_main_flat,  # 扁平化后的主任务预测Cp值
    "Cp_predicted_auxiliary": np.tile(y_pred_aux_flat, 5)  # 扁平化后的辅助任务预测Cp值（每个物质2个温度点的Cp预测）
})

# 保存预测结果
results.to_excel("multi_task_learning_results.xlsx", index=False)
print("✅ 已保存预测结果为: multi_task_learning_results.xlsx")
