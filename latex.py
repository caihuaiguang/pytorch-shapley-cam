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

def format_value(value, best=False, second_best=False):
    """Multiply the value by 100 and return it as a string in 2.2f format. 'best' indicates if it is the best, and 'second_best' indicates if it is the second-best."""
    if isinstance(value, (int, float)):
        formatted_value = f"{value * 100:.2f}"
        if best:
            formatted_value = f"\\textbf{{{formatted_value}}}"
        elif second_best:
            formatted_value = f"\\underline{{{formatted_value}}}"
        return formatted_value
    return value

def generate_latex_table(results):
    table = "\\begin{table*}[htbp]\n"
    table += "\\small\n"
    table += "\\setlength{\\tabcolsep}{.25em}\n"
    table += "\\centering\n"
    table += "\\caption{Evaluation of different CAM-based approaches with existing and proposed metrics on six different backbones.}\n"
    table += "\\label{tab:results}\n"
    table += "\\begin{tabular}{l cccc c cccc c}\n"
    table += "\\hline\n"
    
    backbone_pairs = [
        ("resnet18", "ResNet-18", "resnet50", "ResNet-50"),
        ( "resnet101", "ResNet-101", "resnext50", "ResNeXt-50"),
        ("swint_t", "Swin-T", "swint_s", "Swin-S"),
        ("swint_b", "Swin-B", "vgg16", "VGG-16"),
        ("efficientnetb0", "EfficientNet-B0", "mobilenetv2", "MobileNetV2"),
    ]
    
    for backbone1, display_name1, backbone2, display_name2 in backbone_pairs:
        table += f"& \\multicolumn{{4}}{{c}}{{\\textbf{{{display_name1}}}}} & & \\multicolumn{{4}}{{c}}{{\\textbf{{{display_name2}}}}} \\\\\n"
        table += "\\cline{2-5} \\cline{7-10}\n"
        table += "\\textbf{Method} & Avg Drop $\\downarrow$ & Coherency $\\uparrow$ & Complexity $\\downarrow$ & \\textbf{ADCC} $\\uparrow$ & & Avg Drop $\\downarrow$ & Coherency $\\uparrow$ & Complexity $\\downarrow$ & \\textbf{ADCC} $\\uparrow$ \\\\\n"
        table += "\\hline\n"
        
        # methods = ["randomcam", "hirescam", "gradcamplusplus", "xgradcam", "layercam", "gradcam", "gradcamelementwise", "shapleycam"]
        method_pairs = [("randomcam", "RandomCAM"),("gradcam","GradCAM"), ("hirescam","HiResCAM"), ("gradcamplusplus","GradCAM++"), ("xgradcam","XGradCAM"), ("layercam","LayerCAM"),  ("gradcamelementwise","$\\text{GradCAM}_{E}$"), ("shapleycam","ShapleyCAM")]
        
        # Find the best and second-best values for each metric
        metrics = ['avg_drop', 'coherency', 'complexity', 'adcc']
        
        def get_ranks(results, backbone, metric, reverse=False):
            values = [(results.get((backbone, method), {}).get(metric, float('-inf') if reverse else float('inf')), method) for method,_ in method_pairs]
            values.sort(reverse=reverse)
            best = values[0][1] if values else None
            second_best = values[1][1] if len(values) > 1 else None
            return best, second_best
        
        for method, method_displayname in method_pairs:
            row1 = results.get((backbone1, method), {})
            row2 = results.get((backbone2, method), {})
            
            # Determine which methods are the best and second-best for each metric
            best_avg_drop1, second_best_avg_drop1 = get_ranks(results, backbone1, 'avg_drop')
            best_coherency1, second_best_coherency1 = get_ranks(results, backbone1, 'coherency', reverse=True)
            best_complexity1, second_best_complexity1 = get_ranks(results, backbone1, 'complexity')
            best_adcc1, second_best_adcc1 = get_ranks(results, backbone1, 'adcc', reverse=True)
            
            best_avg_drop2, second_best_avg_drop2 = get_ranks(results, backbone2, 'avg_drop')
            best_coherency2, second_best_coherency2 = get_ranks(results, backbone2, 'coherency', reverse=True)
            best_complexity2, second_best_complexity2 = get_ranks(results, backbone2, 'complexity')
            best_adcc2, second_best_adcc2 = get_ranks(results, backbone2, 'adcc', reverse=True)
            
            avg_drop1 = format_value(row1.get('avg_drop', '-'), best=(method == best_avg_drop1), second_best=(method == second_best_avg_drop1))
            coherency1 = format_value(row1.get('coherency', '-'), best=(method == best_coherency1), second_best=(method == second_best_coherency1))
            complexity1 = format_value(row1.get('complexity', '-'), best=(method == best_complexity1), second_best=(method == second_best_complexity1))
            adcc1 = format_value(row1.get('adcc', '-'), best=(method == best_adcc1), second_best=(method == second_best_adcc1))
            
            avg_drop2 = format_value(row2.get('avg_drop', '-'), best=(method == best_avg_drop2), second_best=(method == second_best_avg_drop2))
            coherency2 = format_value(row2.get('coherency', '-'), best=(method == best_coherency2), second_best=(method == second_best_coherency2))
            complexity2 = format_value(row2.get('complexity', '-'), best=(method == best_complexity2), second_best=(method == second_best_complexity2))
            adcc2 = format_value(row2.get('adcc', '-'), best=(method == best_adcc2), second_best=(method == second_best_adcc2))
            
            table += f"{method_displayname} & {avg_drop1} & {coherency1} & {complexity1} & {adcc1} & & {avg_drop2} & {coherency2} & {complexity2} & {adcc2} \\\\\n"
        
        table += "\\hline\n"
    
    table += "\\end{tabular}\n"
    table += "\\end{table*}"
    
    return table

# file_path1 = 'output_ADCC_softmax_test.txt'  
file_path1 = 'output_ADCC_pre_test.txt'  
results1 = parse_results(file_path1)
latex_table = generate_latex_table(results1)
print(latex_table)
with open('output_table_pre_test.tex', 'w') as f:
    f.write(latex_table)
