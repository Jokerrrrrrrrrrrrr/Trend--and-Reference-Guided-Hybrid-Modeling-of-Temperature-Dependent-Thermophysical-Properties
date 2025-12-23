import pandas as pd
from sklearn.feature_selection import VarianceThreshold

# 读取数据
df = pd.read_csv("descriptors_mordred_208.csv")

# 1. 删除缺失值比例大于 30% 的列
df = df.loc[:, df.isnull().mean() < 0.3]

# 2. 删除包含非数字（字符串错误提示）的列
def is_float_column(series):
    try:
        pd.to_numeric(series.dropna())
        return True
    except:
        return False

numeric_cols = [col for col in df.columns if is_float_column(df[col])]
df = df[numeric_cols]

# 3. 删除所有值相同的特征（方差为0）
selector = VarianceThreshold(threshold=0.0)
features = df.select_dtypes(include=['float64', 'int64'])
features_selected = selector.fit_transform(features)
selected_cols = features.columns[selector.get_support()]
df_clean = df[selected_cols]

# 4. 删除含缺失值的行
df_clean = df_clean.dropna()

# 保存结果
df_clean.to_csv("describe word_cleaned_208.csv", index=False)


