import pandas as pd

# 1. 读取原始 Excel 文件
input_path = "selected_25_liquid_density_descriptors.xlsx"
df = pd.read_excel(input_path)

# 2. 列定义
temp_cols = df.columns[-20:-10]        # 后20列中前10列是温度
vp_cols = df.columns[-10:]             # 后10列是蒸汽压
descriptor_cols = df.columns[:-20]     # 前面所有列是描述符

# 3. 展开每个样本的10个温度和蒸汽压点
rows = []
for _, row in df.iterrows():
    for i in range(10):
        entry = row[descriptor_cols].to_dict()
        entry["Temperature"] = row[temp_cols[i]]
        entry["Density"] = row[vp_cols[i]]
        rows.append(entry)

# 4. 构造新的 DataFrame
df_transformed = pd.DataFrame(rows)

# 5. 保存为新的 Excel 文件
output_path = "Transformed_density_Dataset.xlsx"
df_transformed.to_excel(output_path, index=False)

print(f"✅ 数据转换完成，已保存为：{output_path}")
