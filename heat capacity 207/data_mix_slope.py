import pandas as pd

# === 1. 读取原始数据与 slope 文件 ===
train_df = pd.read_excel("Transformed_hp_Dataset.xlsx")
slope_df = pd.read_csv("slope_values.csv")

# === 2. 扩展 slope 为每个物质对应 10 行 ===
slope_expanded = pd.DataFrame({'slope': slope_df['slope'].repeat(10).values})

# === 3. 合并 slope 列到原始数据中 ===
train_df_with_slope = pd.concat([train_df.reset_index(drop=True), slope_expanded.reset_index(drop=True)], axis=1)

# === 4. 保存为新的 EXCEL文件 ===
output_path = "Transformed_hp_with_slope.xlsx"
train_df_with_slope.to_excel(output_path, index=False)

print(f"✅ 已成功保存为: {output_path}")
