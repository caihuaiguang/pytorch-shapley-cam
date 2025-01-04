def parse_results(file_path, has_road=False):
    results = {}
    with open(file_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            if "Model:" in line:
                model = line.split(":")[1].split(",")[0].strip()
                method = line.split(":")[2].split(",")[0].strip()
                key = (model, method)
                results[key] = {}
            elif "Average ADCC:" in line:
                results[key]['adcc'] = float(line.split(":")[1].strip())
            elif "Average AvgDrop:" in line:
                results[key]['avg_drop'] = float(line.split(":")[1].strip())
            elif "Average Coherency:" in line:
                results[key]['coherency'] = float(line.split(":")[1].strip())
            elif "Average Complexity:" in line:
                results[key]['complexity'] = float(line.split(":")[1].strip())
            elif has_road and "Average ROAD:" in line:
                results[key]['road'] = float(line.split(":")[1].strip())
    return results

def format_value(value):
    """将值乘以100，并以2.2f格式返回字符串"""
    if isinstance(value, (int, float)):
        return f"{value * 100:.2f}"
    return value
def generate_latex_table(results):
    table = "\\begin{table*}[htbp]\n"
    table += "\\begin{tabular}{|c|c|c|c|c|c|c|}\n"
    table += "\\hline\n"
    table += "Model & CAM Method & Average ADCC & Average AvgDrop & Average Coherency & Average Complexity & Average ROAD \\\\\n"
    table += "\\hline\n"

    for (model, method), metrics in results.items():
        adcc = format_value(metrics.get('adcc', 'N/A'))
        avg_drop = format_value(metrics.get('avg_drop', 'N/A'))
        coherency = format_value(metrics.get('coherency', 'N/A'))
        complexity = format_value(metrics.get('complexity', 'N/A'))
        road = format_value(metrics.get('road', 'N/A'))
        table += f"{model} & {method} & {adcc} & {avg_drop} & {coherency} & {complexity} & {road} \\\\\n"
        table += "\\hline\n"

    table += "\\end{tabular}\n"
    table += "\\end{table*}"
    return table

# 读取txt文件并合并数据
file_path1 = 'output_ADCC.txt'  # 第一个文件路径
file_path2 = 'output_ROAD.txt' # 第二个文件路径

results1 = parse_results(file_path1)
results2 = parse_results(file_path2, has_road=True)

# 合并两个文件的结果
for key in results2:
    if key in results1:
        results1[key].update(results2[key])
    else:
        results1[key] = results2[key]

# 生成LaTeX表格
latex_table = generate_latex_table(results1)

# 输出或保存LaTeX表格
print(latex_table)
# 或者将表格保存到文件中
with open('output_table.tex', 'w') as f:
    f.write(latex_table)
