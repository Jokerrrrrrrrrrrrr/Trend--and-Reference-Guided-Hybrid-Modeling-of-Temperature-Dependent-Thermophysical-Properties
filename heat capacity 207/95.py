import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import warnings

warnings.filterwarnings('ignore')  # 忽略特征名称警告

# 1. 读取数据（没有列名）
file_path = "heat capacity 207.xlsx"
df = pd.read_excel(file_path, sheet_name=0)  # 读取第一个工作表

# 提取基团向量（L列到AD列，即从第12列到第30列）
group_vectors = df.iloc[:, 11:30]  # 这里提取基团向量（去掉前11列）

# SMILES列（假设SMILES列是第2列）
smiles = df.iloc[:, 1]  # SMILES在第二列

# T1热容列（J列，即第10列）
T1_Cp = df.iloc[:, 9].values  # T1热容（第10列）

# 取log1p
group_vectors_log = np.log1p(group_vectors)


# 2. 计算MSC相似度（优化版本）
def compute_msc(target_vector, reference_vector, alpha=np.e):
    """
    计算目标向量与参考向量之间的MSC相似度，检查是否有除以零的情况
    """
    target_vector = np.array(target_vector)
    reference_vector = np.array(reference_vector)

    # 计算每个位置的最小值和最大值
    min_vals = np.minimum(target_vector, reference_vector)
    max_vals = np.maximum(target_vector, reference_vector)

    sum_min = np.sum(min_vals)
    sum_max = np.sum(max_vals)

    # 检查sum_max是否为零
    if sum_max < 1e-6:  # 如果 sum_max 非常小，可能导致数值溢出
        print("警告：sum_max 太小，目标分子与参考分子的相似度计算返回 NaN")
        return np.nan  # 返回 NaN，避免除以零错误

    msc = (alpha * sum_min - 1) / (alpha * sum_max - 1)

    # 检查计算结果是否非常大
    if np.abs(msc) > 1e10:
        print("警告：计算出的 MSC 值过大，目标分子与参考分子的相似度返回 NaN")
        return np.nan

    return msc


# 3. 检查第 95 个物质的基团向量
target_idx = 94  # 由于Python的索引从0开始，第95个物质的索引是94
target_vector = group_vectors_log.iloc[target_idx].values
print(f"第 95 个物质的基团向量：{target_vector}")

# 4. 计算第 95 个物质与其他物质的相似度
similarities = []
for i in range(group_vectors_log.shape[0]):
    if i == target_idx:
        continue  # 排除目标分子与自己比较
    ref_vector = group_vectors_log.iloc[i].values
    similarity = compute_msc(target_vector, ref_vector, alpha=np.e)
    similarities.append(similarity)

similarities = np.array(similarities)

# 打印第 95 个物质与其他物质的相似度
print(f"第 95 个物质与其他物质的相似度：{similarities}")
print(f"最大相似度：{np.max(similarities)}")
print(f"最小相似度：{np.min(similarities)}")

# 5. 筛选出相似度大于 0.9 的参考分子
R_threshold = 0.7
mask = (similarities > R_threshold) & (np.arange(len(similarities)) != target_idx)
selected_indices = np.where(mask)[0]

# 打印筛选的相似分子
print(f"第 95 个物质与相似度大于 {R_threshold} 的分子的数量：{len(selected_indices)}")
print(f"选择的参考分子的索引：{selected_indices}")

# 6. 检查训练集是否满足要求
if len(selected_indices) == 0:
    print(f"警告：目标分子 {target_idx} 没有找到相似分子，使用备用预测...")

    # 使用均值进行预测
    target_prediction = np.mean(T1_Cp)  # 使用均值进行预测
    selected_indices = [target_idx]  # 备用方案：使用目标分子本身

# 7. 使用相似分子进行训练
top_sim_vectors = group_vectors_log.iloc[selected_indices]
top_sim_T1_Cp = T1_Cp[selected_indices]

model = LinearRegression()
model.fit(top_sim_vectors, top_sim_T1_Cp)

# 8. 预测第 95 个物质的 T1 热容
target_df = pd.DataFrame([target_vector], columns=group_vectors_log.columns)
target_prediction = model.predict(target_df)[0]

print(f"第 95 个物质的预测值：{target_prediction}")
print(f"第 95 个物质的实际值：{T1_Cp[target_idx]}")

# 9. 查看回归系数，确认是否有异常
print(f"回归系数：{model.coef_}")
