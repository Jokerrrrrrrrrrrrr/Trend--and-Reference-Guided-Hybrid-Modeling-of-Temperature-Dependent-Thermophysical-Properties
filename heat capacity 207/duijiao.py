import openpyxl

# 加载Excel文件
workbook = openpyxl.load_workbook("heat capacity 207.xlsx")
sheet = workbook.worksheets[8]  # 获取第4个工作表（索引从0开始）

# 提取主对角线
main_diagonal = [sheet.cell(row=i, column=i).value for i in range(1, 208)]

print("主对角线数据：", main_diagonal)
