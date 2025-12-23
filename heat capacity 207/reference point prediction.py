# import pandas as pd
# import numpy as np
# from sklearn.linear_model import LinearRegression
# from sklearn.metrics import r2_score  # 导入R²计算方法
# import warnings
#
# warnings.filterwarnings('ignore')  # 忽略特征名称警告
#
# # 1. 读取数据（没有列名）
# file_path = "heat capacity 207.xlsx"
# df = pd.read_excel(file_path, sheet_name=0)  # 读取第一个工作表
#
# # 提取基团向量（L列到AD列，即从第12列到第30列）
# group_vectors = df.iloc[:, 11:30]  # 这里提取基团向量（去掉前11列）
#
# # SMILES列（假设SMILES列是第2列）
# smiles = df.iloc[:, 1]  # SMILES在第二列
#
# # T1热容列（J列，即第10列）
# T1_Cp = df.iloc[:, 9].values  # T1热容（第10列）
#
# # T2热容列（AY列，即第50列）
# T2_Cp = df.iloc[:, 50].values  # T2热容（第50列）
#
# # 取log1p
# group_vectors_log = np.log1p(group_vectors)
#
# # 2. 计算MSC相似度（优化版本）
# def compute_msc(target_vector, reference_vector, alpha=np.e):
#     """
#     计算目标向量与参考向量之间的MSC相似度，检查是否有除以零的情况
#     """
#     target_vector = np.array(target_vector)
#     reference_vector = np.array(reference_vector)
#
#     # 计算每个位置的最小值和最大值
#     min_vals = np.minimum(target_vector, reference_vector)
#     max_vals = np.maximum(target_vector, reference_vector)
#
#     sum_min = np.sum(min_vals)
#     sum_max = np.sum(max_vals)
#
#     if sum_max == 0:
#         print("警告：分母为0，目标分子与参考分子的相似度计算返回 NaN")
#         return np.nan
#
#     msc = (alpha * sum_min - 1) / (alpha * sum_max - 1)
#     return msc
#
# # 3. 循环每个目标分子预测
# predictions = []
# reliability_indices = []
# actual_values = []  # 添加实际值用于比较
#
# print("开始预测，共", len(group_vectors_log), "个分子...")
#
# # 对每个目标分子进行预测
# for target_idx, target_vector in enumerate(group_vectors_log.values):
#     if target_idx % 10 == 0:  # 每10个分子打印一次进度
#         print(f"处理第 {target_idx}/{len(group_vectors_log)} 个分子...")
#
#     try:
#         similarities = []  # 存储所有的相似度
#         for i in range(group_vectors_log.shape[0]):  # 遍历所有参考分子
#             if i == target_idx:  # 排除目标分子与自己比较
#                 continue
#             ref_vector = group_vectors_log.iloc[i].values  # 获取参考分子的基团频数
#             similarity = compute_msc(target_vector, ref_vector, alpha=np.e)  # 计算相似度
#             similarities.append(similarity)  # 将相似度添加到列表
#
#         similarities = np.array(similarities)  # 转换为 numpy 数组
#
#         # 选择 R > 0.9 的训练集（排除目标分子自身）
#         R_threshold = 0.7
#         mask = (similarities > R_threshold) & (np.arange(len(similarities)) != target_idx)
#         selected_indices = np.where(mask)[0]
#
#         # 如果没有分子满足阈值，选择相似度最高的一个（排除自身）
#         if len(selected_indices) == 0:
#             all_indices = np.arange(len(similarities))
#             exclude_self = all_indices[all_indices != target_idx]
#             if len(exclude_self) > 0:
#                 max_sim_idx = exclude_self[np.argmax(similarities[exclude_self])]
#                 selected_indices = [max_sim_idx]
#             else:
#                 selected_indices = [target_idx]
#
#         # 选择训练集对应的基团向量和T1热容
#         top_sim_vectors = group_vectors_log.iloc[selected_indices]
#         top_sim_T1_Cp = T1_Cp[selected_indices]
#
#         # 检查训练样本数量是否足够
#         if len(selected_indices) < 2:
#             if len(selected_indices) == 1:
#                 target_prediction = top_sim_T1_Cp[0]
#             else:
#                 target_prediction = np.mean(T1_Cp)  # 备用方案
#         else:
#             # 训练模型（线性回归）
#             model = LinearRegression()
#             model.fit(top_sim_vectors, top_sim_T1_Cp)
#
#             # 预测目标分子（转换为DataFrame以保持特征名称一致）
#             target_df = pd.DataFrame([target_vector], columns=group_vectors_log.columns)
#             target_prediction = model.predict(target_df)[0]
#
#         # 最大相似度作为可靠性指数
#         max_similarity = np.max(similarities[selected_indices])
#
#         predictions.append(target_prediction)
#         reliability_indices.append(max_similarity)
#         actual_values.append(T1_Cp[target_idx])  # 记录实际值
#
#     except Exception as e:
#         print(f"处理第 {target_idx} 个分子时出错: {e}")
#         predictions.append(np.mean(T1_Cp))
#         reliability_indices.append(0.5)
#         actual_values.append(T1_Cp[target_idx])
#
# # 4. 输出结果
# results = pd.DataFrame({
#     'SMILES': smiles,
#     'Actual_T1_Cp': actual_values,
#     'Predicted_T1_Cp': predictions,
#     'Reliability_Index': reliability_indices,
#     'Absolute_Error': np.abs(np.array(actual_values) - np.array(predictions))
# })
#
# # 计算整体性能指标
# mae = np.mean(results['Absolute_Error'])
# r2 = r2_score(actual_values, predictions)  # 计算 R² 值
#
# print(f"预测完成！平均绝对误差(MAE): {mae:.4f}")
# print(f"可靠性指数平均值: {np.mean(reliability_indices):.4f}")
# print(f"R² 值: {r2:.4f}")  # 打印 R² 值
#
# results.to_excel("Predictions_with_MSC_similarity_LinearRegression.xlsx", index=False)
# print("✅ 已保存预测结果为: Predictions_with_MSC_similarity_LinearRegression.xlsx")
#
# from sklearn.linear_model import Ridge
# from sklearn.metrics import r2_score  # 导入R²计算方法
# import pandas as pd
# import numpy as np
# import warnings
#
# warnings.filterwarnings('ignore')  # 忽略特征名称警告
#
# # 1. 读取数据（没有列名）
# file_path = "heat capacity 207.xlsx"
# df = pd.read_excel(file_path, sheet_name=0)  # 读取第一个工作表
#
# # 提取基团向量（L列到AD列，即从第12列到第30列）
# group_vectors = df.iloc[:, 11:30]  # 这里提取基团向量（去掉前11列）
#
# # SMILES列（假设SMILES列是第2列）
# smiles = df.iloc[:, 1]  # SMILES在第二列
#
# # T1热容列（J列，即第10列）
# T1_Cp = df.iloc[:, 9].values  # T1热容（第10列）
#
# # 取log1p（对数转换） - 仅在计算相似度时使用
# group_vectors_log = np.log1p(group_vectors)
#
# # 2. 计算MSC相似度（优化版本）
# def compute_msc(target_vector, reference_vector, alpha=np.e):
#     """
#     计算目标向量与参考向量之间的MSC相似度，检查是否有除以零的情况
#     """
#     target_vector = np.array(target_vector)
#     reference_vector = np.array(reference_vector)
#
#     # 计算每个位置的最小值和最大值
#     min_vals = np.minimum(target_vector, reference_vector)
#     max_vals = np.maximum(target_vector, reference_vector)
#
#     sum_min = np.sum(min_vals)
#     sum_max = np.sum(max_vals)
#
#     if sum_max == 0:
#         print("警告：分母为0，目标分子与参考分子的相似度计算返回 NaN")
#         return np.nan
#
#     msc = (alpha * sum_min - 1) / (alpha * sum_max - 1)
#     return msc
#
# # 3. 循环每个目标分子预测
# predictions = []
# reliability_indices = []
# actual_values = []  # 添加实际值用于比较
#
# print("开始预测，共", len(group_vectors_log), "个分子...")
#
# # 对每个目标分子进行预测
# for target_idx, target_vector in enumerate(group_vectors_log.values):
#     if target_idx % 10 == 0:  # 每10个分子打印一次进度
#         print(f"处理第 {target_idx}/{len(group_vectors_log)} 个分子...")
#
#     try:
#         similarities = []  # 存储所有的相似度
#         for i in range(group_vectors_log.shape[0]):  # 遍历所有参考分子
#             if i == target_idx:  # 排除目标分子与自己比较
#                 continue
#             ref_vector = group_vectors_log.iloc[i].values  # 获取参考分子的基团频数
#             similarity = compute_msc(target_vector, ref_vector, alpha=np.e)  # 计算相似度
#             similarities.append(similarity)  # 将相似度添加到列表
#
#         similarities = np.array(similarities)  # 转换为 numpy 数组
#
#         # 选择 R > 0.9 的训练集（排除目标分子自身）
#         R_threshold = 0.9
#         mask = (similarities > R_threshold) & (np.arange(len(similarities)) != target_idx)
#         selected_indices = np.where(mask)[0]
#
#         # 如果没有分子满足阈值，选择相似度最高的一个（排除自身）
#         if len(selected_indices) == 0:
#             all_indices = np.arange(len(similarities))
#             exclude_self = all_indices[all_indices != target_idx]
#             if len(exclude_self) > 0:
#                 max_sim_idx = exclude_self[np.argmax(similarities[exclude_self])]
#                 selected_indices = [max_sim_idx]
#             else:
#                 selected_indices = [target_idx]
#
#         # 选择训练集对应的基团向量和T1热容（回归使用原始基团向量）
#         top_sim_vectors = group_vectors.iloc[selected_indices]  # 使用原始基团向量
#         top_sim_T1_Cp = T1_Cp[selected_indices]
#
#         # 检查训练样本数量是否足够
#         if len(selected_indices) < 2:
#             if len(selected_indices) == 1:
#                 target_prediction = top_sim_T1_Cp[0]
#             else:
#                 target_prediction = np.mean(T1_Cp)  # 备用方案
#         else:
#             # 使用岭回归模型（Ridge Regression）
#             model = Ridge(alpha=1.0)  # alpha 控制正则化强度
#             model.fit(top_sim_vectors, top_sim_T1_Cp)
#
#             # 预测目标分子
#             target_df = pd.DataFrame([target_vector], columns=group_vectors.columns)
#             target_prediction = model.predict(target_df)[0]
#
#         # 最大相似度作为可靠性指数
#         max_similarity = np.max(similarities[selected_indices])
#
#         predictions.append(target_prediction)
#         reliability_indices.append(max_similarity)
#         actual_values.append(T1_Cp[target_idx])  # 记录实际值
#
#     except Exception as e:
#         print(f"处理第 {target_idx} 个分子时出错: {e}")
#         predictions.append(np.mean(T1_Cp))
#         reliability_indices.append(0.5)
#         actual_values.append(T1_Cp[target_idx])
#
# # 4. 输出结果
# results = pd.DataFrame({
#     'SMILES': smiles,
#     'Actual_T1_Cp': actual_values,
#     'Predicted_T1_Cp': predictions,
#     'Reliability_Index': reliability_indices,
#     'Absolute_Error': np.abs(np.array(actual_values) - np.array(predictions))
# })
#
# # 计算整体性能指标
# mae = np.mean(results['Absolute_Error'])
# r2 = r2_score(actual_values, predictions)  # 计算 R² 值
#
# print(f"预测完成！平均绝对误差(MAE): {mae:.4f}")
# print(f"可靠性指数平均值: {np.mean(reliability_indices):.4f}")
# print(f"R² 值: {r2:.4f}")  # 打印 R² 值
#
# results.to_excel("Predictions_with_MSC_similarity_Ridge_Regression.xlsx", index=False)
# print("✅ 已保存预测结果为: Predictions_with_MSC_similarity_Ridge_Regression.xlsx")

from sklearn.linear_model import HuberRegressor
from sklearn.metrics import r2_score  # 导入R²计算方法
import pandas as pd
import numpy as np
import warnings
from sklearn.linear_model import LinearRegression

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

# 取log1p（对数转换） - 仅在计算相似度时使用
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

    msc = (alpha ** sum_min - 1) / (alpha ** sum_max - 1)
    return msc

# 3. 循环每个目标分子预测
predictions = []
reliability_indices = []
actual_values = []  # 添加实际值用于比较

print("开始预测，共", len(group_vectors_log), "个分子...")

# 对每个目标分子进行预测
for target_idx, target_vector in enumerate(group_vectors_log.values):
    if target_idx % 10 == 0:  # 每10个分子打印一次进度
        print(f"处理第 {target_idx}/{len(group_vectors_log)} 个分子...")

    try:
        similarities = []  # 存储所有的相似度
        for i in range(group_vectors_log.shape[0]):  # 遍历所有参考分子
            if i == target_idx:  # 排除目标分子与自己比较
                continue
            ref_vector = group_vectors_log.iloc[i].values  # 获取参考分子的基团频数
            similarity = compute_msc(target_vector, ref_vector, alpha=np.e)  # 计算相似度
            similarities.append(similarity)  # 将相似度添加到列表

        similarities = np.array(similarities)  # 转换为 numpy 数组

        # 选择 R > 0.9 的训练集（排除目标分子自身）
        R_threshold = 0.5
        mask = (similarities > R_threshold)
        selected_indices = np.where(mask)[0]

        # 如果没有分子满足阈值，选择相似度最高的一个（排除自身）
        if len(selected_indices) == 0:
            all_indices = np.arange(len(similarities))
            exclude_self = all_indices[all_indices != target_idx]
            if len(exclude_self) > 0:
                max_sim_idx = exclude_self[np.argmax(similarities[exclude_self])]
                selected_indices = [max_sim_idx]
            else:
                selected_indices = [target_idx]

        # 选择训练集对应的基团向量和T1热容（回归使用原始基团向量）
        top_sim_vectors = group_vectors.iloc[selected_indices]  # 使用原始基团向量
        top_sim_T1_Cp = T1_Cp[selected_indices]

        # 检查训练样本数量是否足够
        if len(selected_indices) < 2:
            if len(selected_indices) == 1:
                target_prediction = top_sim_T1_Cp[0]
            else:
                target_prediction = np.mean(T1_Cp)  # 备用方案
        else:
            # 使用 Huber 回归模型（Huber Regressor）
            model = HuberRegressor(max_iter=9000,epsilon=1.35)  # epsilon 控制模型对异常值的鲁棒性
            model.fit(top_sim_vectors, top_sim_T1_Cp)

            # 预测目标分子
            target_prediction = model.predict([group_vectors.iloc[target_idx]])[0]
        # else:
        #     # 使用线性回归模型
        #     model = LinearRegression()
        #     model.fit(top_sim_vectors, top_sim_T1_Cp)
        #
        #     # 预测目标分子
        #     target_prediction = model.predict([group_vectors.iloc[target_idx]])[0]
        # 最大相似度作为可靠性指数
        max_similarity = np.max(similarities[selected_indices])

        predictions.append(target_prediction)
        reliability_indices.append(max_similarity)
        actual_values.append(T1_Cp[target_idx])  # 记录实际值

    except Exception as e:
        print(f"处理第 {target_idx} 个分子时出错: {e}")
        predictions.append(np.mean(T1_Cp))
        reliability_indices.append(0.5)
        actual_values.append(T1_Cp[target_idx])

# 4. 输出结果
results = pd.DataFrame({
    'SMILES': smiles,
    'Actual_T1_Cp': actual_values,
    'Predicted_T1_Cp': predictions,
    'Reliability_Index': reliability_indices,
    'Absolute_Error': np.abs(np.array(actual_values) - np.array(predictions))
})

# 计算整体性能指标
mae = np.mean(results['Absolute_Error'])
r2 = r2_score(actual_values, predictions)  # 计算 R² 值

print(f"预测完成！平均绝对误差(MAE): {mae:.4f}")
print(f"可靠性指数平均值: {np.mean(reliability_indices):.4f}")
print(f"R² 值: {r2:.4f}")  # 打印 R² 值

results.to_excel("Predictions_with_MSC_similarity_Huber_Regression.xlsx", index=False)
print("✅ 已保存预测结果为: Predictions_with_MSC_similarity_Huber_Regression.xlsx")
