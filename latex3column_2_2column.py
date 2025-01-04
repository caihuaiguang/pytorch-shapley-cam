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
            elif "Average AvgDrop:" in line or "Average Drop:" in line:
                results[key]['ad'] = float(line.split(":")[1].strip())
            elif "Average Coherency:" in line:
                results[key]['coherency'] = float(line.split(":")[1].strip())
            elif "Average Complexity:" in line:
                results[key]['complexity'] = float(line.split(":")[1].strip())
            elif "Average Increase:" in line:
                results[key]['increase'] = float(line.split(":")[1].strip())
            elif "Average Drop in Deletion:" in line:
                results[key]['add'] = float(line.split(":")[1].strip())
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

def generate_latex_table_original(results):


    method_pairs = [
    ("randomcam", "RandomCAM"), 
    ("gradcamplusplus", "GradCAM++"), 
    ("xgradcam", "XGradCAM"), 
    ("layercam", "LayerCAM"),
    ("gradcamelementwise", "GradCAM-E"), 
    ("shapleycam", "ShapleyCAM-E"), 
    ("hirescam", "HiResCAM"),  
    ("shapleycam_hires", "ShapleyCAM-H"),
    ("gradcam", "GradCAM"), 
    ("shapleycam_mean", "ShapleyCAM"), 
    ]
    table = "\\begin{table*}[htbp]\n"
    table += "\\setlength{\\tabcolsep}{.25em}\n"
    table += "\\renewcommand{\\arraystretch}{1.05}\n"
    table += "\\centering\n"
    table += "\\caption{Evaluation of different CAM methods with six metrics on twelve different backbones with the last convolutional layer \\\\ or the first normalization layer of the last transformer block as the target layer.}\n"
    table += "\\label{tab:results}\n"
    table += "\\begin{tabular}{l cccccc cc cccccc cc cccccc}\n"
    table += "\\hline\n"
    
    backbone_pairs = [
        ("resnet18", "ResNet-18 (69.76\\%)", "resnet50", "ResNet-50 (76.13\\%)", "resnet101", "ResNet-101 (77.38\\%)"),
        ("resnet152", "ResNet-152 (78.32\\%)","resnext50", "ResNeXt-50 (77.62\\%)", "vgg16", "VGG-16 (71.59\\%)"),
        ("efficientnetb0", "EfficientNet-B0 (77.69\\%)", "mobilenetv2", "MobileNet-V2 (71.88\\%)", "swint_t", "Swin-T (80.91\\%)"),
        ("swint_s", "Swin-S (83.05\\%)","swint_b", "Swin-B (84.71\\%)", "swint_l", "Swin-L (85.83\\%)")
    ]
    
    for backbone1, display_name1, backbone2, display_name2, backbone3, display_name3 in backbone_pairs:
        table += f"& \\multicolumn{{6}}{{c}}{{\\textbf{{{display_name1}}}}} & & \\multicolumn{{6}}{{c}}{{\\textbf{{{display_name2}}}}} & & \\multicolumn{{6}}{{c}}{{\\textbf{{{display_name3}}}}} \\\\\n"
        table += "\\cline{2-7} \\cline{9-14} \\cline{16-21}\n"
        table += "\\textbf{Method} & AD $\\downarrow$ & Coh $\\uparrow$ & Com $\\downarrow$ & \\textbf{ADCC} $\\uparrow$ & \\textbf{IC} $\\uparrow$ & \\textbf{ADD} $\\uparrow$ & & AD $\\downarrow$ & Coh $\\uparrow$ & Com $\\downarrow$ & \\textbf{ADCC} $\\uparrow$ & \\textbf{IC} $\\uparrow$ & \\textbf{ADD} $\\uparrow$ & & AD $\\downarrow$ & Coh $\\uparrow$ & Com $\\downarrow$ & \\textbf{ADCC} $\\uparrow$ & \\textbf{IC} $\\uparrow$ & \\textbf{ADD} $\\uparrow$ \\\\\n"
        table += "\\hline\n"

        metrics = ['ad', 'coherency', 'complexity', 'adcc', 'increase', 'add']
        
        def get_ranks(results, backbone, metric, reverse=False):
            values = [(results.get((backbone, method), {}).get(metric, float('-inf') if reverse else float('inf')), method) for method, _ in method_pairs if method != "randomcam"]
            values.sort(reverse=reverse)
            best = values[0][1] if values else None
            second_best = values[1][1] if len(values) > 1 else None
            return best, second_best
        
        for method, method_displayname in method_pairs:
            row1 = results.get((backbone1, method), {})
            row2 = results.get((backbone2, method), {})
            row3 = results.get((backbone3, method), {})
            
            best_avg_drop1, second_best_avg_drop1 = get_ranks(results, backbone1, 'ad')
            best_coherency1, second_best_coherency1 = get_ranks(results, backbone1, 'coherency', reverse=True)
            best_complexity1, second_best_complexity1 = get_ranks(results, backbone1, 'complexity')
            best_adcc1, second_best_adcc1 = get_ranks(results, backbone1, 'adcc', reverse=True)
            best_increase1, second_best_increase1 = get_ranks(results, backbone1, 'increase', reverse=True)
            best_add1, second_best_add1 = get_ranks(results, backbone1, 'add', reverse=True)
            
            best_avg_drop2, second_best_avg_drop2 = get_ranks(results, backbone2, 'ad')
            best_coherency2, second_best_coherency2 = get_ranks(results, backbone2, 'coherency', reverse=True)
            best_complexity2, second_best_complexity2 = get_ranks(results, backbone2, 'complexity')
            best_adcc2, second_best_adcc2 = get_ranks(results, backbone2, 'adcc', reverse=True)
            best_increase2, second_best_increase2 = get_ranks(results, backbone2, 'increase', reverse=True)
            best_add2, second_best_add2 = get_ranks(results, backbone2, 'add', reverse=True)
            
            best_avg_drop3, second_best_avg_drop3 = get_ranks(results, backbone3, 'ad')
            best_coherency3, second_best_coherency3 = get_ranks(results, backbone3, 'coherency', reverse=True)
            best_complexity3, second_best_complexity3 = get_ranks(results, backbone3, 'complexity')
            best_adcc3, second_best_adcc3 = get_ranks(results, backbone3, 'adcc', reverse=True)
            best_increase3, second_best_increase3 = get_ranks(results, backbone3, 'increase', reverse=True)
            best_add3, second_best_add3 = get_ranks(results, backbone3, 'add', reverse=True)
            
            avg_drop1 = format_value(row1.get('ad', '-'), best=(method == best_avg_drop1), second_best=(method == second_best_avg_drop1))
            coherency1 = format_value(row1.get('coherency', '-'), best=(method == best_coherency1), second_best=(method == second_best_coherency1))
            complexity1 = format_value(row1.get('complexity', '-'), best=(method == best_complexity1), second_best=(method == second_best_complexity1))
            adcc1 = format_value(row1.get('adcc', '-'), best=(method == best_adcc1), second_best=(method == second_best_adcc1))
            increase1 = format_value(row1.get('increase', '-'), best=(method == best_increase1), second_best=(method == second_best_increase1))
            add1 = format_value(row1.get('add', '-'), best=(method == best_add1), second_best=(method == second_best_add1))
            
            avg_drop2 = format_value(row2.get('ad', '-'), best=(method == best_avg_drop2), second_best=(method == second_best_avg_drop2))
            coherency2 = format_value(row2.get('coherency', '-'), best=(method == best_coherency2), second_best=(method == second_best_coherency2))
            complexity2 = format_value(row2.get('complexity', '-'), best=(method == best_complexity2), second_best=(method == second_best_complexity2))
            adcc2 = format_value(row2.get('adcc', '-'), best=(method == best_adcc2), second_best=(method == second_best_adcc2))
            increase2 = format_value(row2.get('increase', '-'), best=(method == best_increase2), second_best=(method == second_best_increase2))
            add2 = format_value(row2.get('add', '-'), best=(method == best_add2), second_best=(method == second_best_add2))
            
            avg_drop3 = format_value(row3.get('ad', '-'), best=(method == best_avg_drop3), second_best=(method == second_best_avg_drop3))
            coherency3 = format_value(row3.get('coherency', '-'), best=(method == best_coherency3), second_best=(method == second_best_coherency3))
            complexity3 = format_value(row3.get('complexity', '-'), best=(method == best_complexity3), second_best=(method == second_best_complexity3))
            adcc3 = format_value(row3.get('adcc', '-'), best=(method == best_adcc3), second_best=(method == second_best_adcc3))
            increase3 = format_value(row3.get('increase', '-'), best=(method == best_increase3), second_best=(method == second_best_increase3))
            add3 = format_value(row3.get('add', '-'), best=(method == best_add3), second_best=(method == second_best_add3))
            
            table += f"{method_displayname} & {avg_drop1} & {coherency1} & {complexity1} & {adcc1} & {increase1} & {add1} & & {avg_drop2} & {coherency2} & {complexity2} & {adcc2} & {increase2} & {add2} & & {avg_drop3} & {coherency3} & {complexity3} & {adcc3} & {increase3} & {add3} \\\\\n"
        
    
            # 添加虚线分隔符在 HiResCAM 和 GradCAM 之间
            if method == "layercam":
                table += "\\cdashline{2-21}\n"
            elif method == "shapleycam_hires":
                table += "\\cdashline{2-21}\n"
            elif method == "shapleycam":
                table += "\\cdashline{2-21}\n"
        table += "\\hline\n"
    table += "\\end{tabular}\n"
    table += "\\end{table*}\n"
    return table

def generate_latex_table(results):
    method_pairs = [
    ("randomcam", "RandomCAM"), 
    ("gradcamplusplus", "GradCAM++"), 
    ("xgradcam", "XGradCAM"), 
    ("layercam", "LayerCAM"),
    ("gradcamelementwise", "GradCAM-E"), 
    ("shapleycam", "ShapleyCAM-E"), 
    ("hirescam", "HiResCAM"),  
    ("shapleycam_hires", "ShapleyCAM-H"),
    ("gradcam", "GradCAM"), 
    ("shapleycam_mean", "ShapleyCAM"), 
    ]

    
    table = "\\begin{table*}[htbp]\n"
    # table += "\\small\n"
    table += "\\setlength{\\tabcolsep}{.25em}\n"
    table += "\\renewcommand{\\arraystretch}{1.05}\n"
    table += "\\centering\n"
    table += "\\caption{Evaluation of different CAM methods with six metrics on eight different backbones.}\n"
    table += "\\label{tab:results}\n"
    table += "\\begin{tabular}{l cccccc cc cccccc}\n"
    table += "\\hline\n"
    
    backbone_pairs = [
        ("resnet18", "ResNet-18 (69.76\\%)", "resnet50", "ResNet-50 (76.13\\%)"),
        ("resnet101", "ResNet-101 (77.38\\%)", "resnet152", "ResNet-152 (78.32\\%)"),
        ("resnext50", "ResNeXt-50 (77.62\\%)", "mobilenetv2", "MobileNet-V2 (71.88\\%)"),
        ("vgg16", "VGG-16 (71.59\\%)",  "efficientnetb0", "EfficientNet-B0 (77.69\\%)"),
    ]
    
    for backbone1, display_name1, backbone2, display_name2 in backbone_pairs:
        table += f"& \\multicolumn{{6}}{{c}}{{\\textbf{{{display_name1}}}}} & & \\multicolumn{{6}}{{c}}{{\\textbf{{{display_name2}}}}} \\\\\n"
        table += "\\cline{2-7} \\cline{9-14}\n"
        table += "\\textbf{Method} & AD $\\downarrow$ & Coh $\\uparrow$ & Com $\\downarrow$ & \\textbf{ADCC} $\\uparrow$ & \\textbf{IC} $\\uparrow$ & \\textbf{ADD} $\\uparrow$ & & AD $\\downarrow$ & Coh $\\uparrow$ & Com $\\downarrow$ & \\textbf{ADCC} $\\uparrow$ & \quad\\textbf{IC} $\\uparrow$ & \\textbf{ADD} $\\uparrow$ \\\\\n"
        table += "\\hline\n"
 
        # Find the best and second-best values for each metric
        metrics = ['ad', 'coherency', 'complexity', 'adcc', 'increase', 'add']
        
        def get_ranks(results, backbone, metric, reverse=False):
            values = [(results.get((backbone, method), {}).get(metric, float('-inf') if reverse else float('inf')), method) for method, _ in method_pairs if method != "randomcam"]
            values.sort(reverse=reverse)
            best = values[0][1] if values else None
            second_best = values[1][1] if len(values) > 1 else None
            return best, second_best
        
        for method, method_displayname in method_pairs:
            row1 = results.get((backbone1, method), {})
            row2 = results.get((backbone2, method), {})
            
            # Determine which methods are the best and second-best for each metric
            best_avg_drop1, second_best_avg_drop1 = get_ranks(results, backbone1, 'ad')
            best_coherency1, second_best_coherency1 = get_ranks(results, backbone1, 'coherency', reverse=True)
            best_complexity1, second_best_complexity1 = get_ranks(results, backbone1, 'complexity')
            best_adcc1, second_best_adcc1 = get_ranks(results, backbone1, 'adcc', reverse=True)
            best_increase1, second_best_increase1 = get_ranks(results, backbone1, 'increase', reverse=True)
            best_add1, second_best_add1 = get_ranks(results, backbone1, 'add', reverse=True)
            
            best_avg_drop2, second_best_avg_drop2 = get_ranks(results, backbone2, 'ad')
            best_coherency2, second_best_coherency2 = get_ranks(results, backbone2, 'coherency', reverse=True)
            best_complexity2, second_best_complexity2 = get_ranks(results, backbone2, 'complexity')
            best_adcc2, second_best_adcc2 = get_ranks(results, backbone2, 'adcc', reverse=True)
            best_increase2, second_best_increase2 = get_ranks(results, backbone2, 'increase', reverse=True)
            best_add2, second_best_add2 = get_ranks(results, backbone2, 'add', reverse=True)
            
            avg_drop1 = format_value(row1.get('ad', '-'), best=(method == best_avg_drop1), second_best=(method == second_best_avg_drop1))
            coherency1 = format_value(row1.get('coherency', '-'), best=(method == best_coherency1), second_best=(method == second_best_coherency1))
            complexity1 = format_value(row1.get('complexity', '-'), best=(method == best_complexity1), second_best=(method == second_best_complexity1))
            adcc1 = format_value(row1.get('adcc', '-'), best=(method == best_adcc1), second_best=(method == second_best_adcc1))
            increase1 = format_value(row1.get('increase', '-'), best=(method == best_increase1), second_best=(method == second_best_increase1))
            add1 = format_value(row1.get('add', '-'), best=(method == best_add1), second_best=(method == second_best_add1))
            
            avg_drop2 = format_value(row2.get('ad', '-'), best=(method == best_avg_drop2), second_best=(method == second_best_avg_drop2))
            coherency2 = format_value(row2.get('coherency', '-'), best=(method == best_coherency2), second_best=(method == second_best_coherency2))
            complexity2 = format_value(row2.get('complexity', '-'), best=(method == best_complexity2), second_best=(method == second_best_complexity2))
            adcc2 = format_value(row2.get('adcc', '-'), best=(method == best_adcc2), second_best=(method == second_best_adcc2))
            increase2 = format_value(row2.get('increase', '-'), best=(method == best_increase2), second_best=(method == second_best_increase2))
            add2 = format_value(row2.get('add', '-'), best=(method == best_add2), second_best=(method == second_best_add2))
            
            table += f"{method_displayname} & {avg_drop1} & {coherency1} & {complexity1} & {adcc1} & {increase1} & {add1} & & {avg_drop2} & {coherency2} & {complexity2} & {adcc2} & {increase2} & {add2} \\\\\n"
        
            # 添加虚线分隔符在 HiResCAM 和 GradCAM 之间
            if method == "layercam":
                table += "\\cdashline{2-14}\n"
            elif method == "shapleycam_hires":
                table += "\\cdashline{2-14}\n"
            elif method == "shapleycam":
                table += "\\cdashline{2-14}\n"
        table += "\\hline\n"
    table += "\\end{tabular}\n"
    table += "\\end{table*}"
    
    return table
def generate_latex_table_swin(results):
    method_pairs = [
    ("randomcam", "RandomCAM"), 
    ("gradcamplusplus", "GradCAM++"), 
    ("xgradcam", "XGradCAM"), 
    ("layercam", "LayerCAM"),
    ("gradcamelementwise", "GradCAM-E"), 
    ("shapleycam", "ShapleyCAM-E"), 
    ("hirescam", "HiResCAM"),  
    ("shapleycam_hires", "ShapleyCAM-H"),
    ("gradcam", "GradCAM"), 
    ("shapleycam_mean", "ShapleyCAM"), 
    ]

    table = "\\begin{table*}[htbp]\n"
    # table += "\\small\n"
    table += "\\setlength{\\tabcolsep}{.25em}\n"
    table += "\\renewcommand{\\arraystretch}{1.05}\n"
    table += "\\centering\n"
    table += "\\caption{Evaluation of different CAM methods with six metrics on Swin Transformer.}\n"
    table += "\\label{tab:results}\n"
    table += "\\begin{tabular}{l cccccc cc cccccc}\n"
    table += "\\hline\n"

    backbone_pairs = [
        ("swint_t", "Swin-T (80.91\\%)", "swint_s", "Swin-S (83.05\\%)"),
        ("swint_b", "Swin-B (84.71\\%)", "swint_l", "Swin-L (85.83\\%)"),
    ]
    
    for backbone1, display_name1, backbone2, display_name2 in backbone_pairs:
        table += f"& \\multicolumn{{6}}{{c}}{{\\textbf{{{display_name1}}}}} & & \\multicolumn{{6}}{{c}}{{\\textbf{{{display_name2}}}}} \\\\\n"
        table += "\\cline{2-7} \\cline{9-14}\n"
        table += "\\textbf{Method} & AD $\\downarrow$ & Coh $\\uparrow$ & Com $\\downarrow$ & \\textbf{ADCC} $\\uparrow$ & \\textbf{IC} $\\uparrow$ & \\textbf{ADD} $\\uparrow$ & & AD $\\downarrow$ & Coh $\\uparrow$ & Com $\\downarrow$ & \\textbf{ADCC} $\\uparrow$ & \quad\\textbf{IC} $\\uparrow$ & \\textbf{ADD} $\\uparrow$ \\\\\n"
        table += "\\hline\n"
        

        # Find the best and second-best values for each metric
        metrics = ['ad', 'coherency', 'complexity', 'adcc', 'increase', 'add']
        
        def get_ranks(results, backbone, metric, reverse=False):
            values = [(results.get((backbone, method), {}).get(metric, float('-inf') if reverse else float('inf')), method) for method, _ in method_pairs if method != "randomcam"]
            values.sort(reverse=reverse)
            best = values[0][1] if values else None
            second_best = values[1][1] if len(values) > 1 else None
            return best, second_best
        
        for method, method_displayname in method_pairs:
            row1 = results.get((backbone1, method), {})
            row2 = results.get((backbone2, method), {})
            
            # Determine which methods are the best and second-best for each metric
            best_avg_drop1, second_best_avg_drop1 = get_ranks(results, backbone1, 'ad')
            best_coherency1, second_best_coherency1 = get_ranks(results, backbone1, 'coherency', reverse=True)
            best_complexity1, second_best_complexity1 = get_ranks(results, backbone1, 'complexity')
            best_adcc1, second_best_adcc1 = get_ranks(results, backbone1, 'adcc', reverse=True)
            best_increase1, second_best_increase1 = get_ranks(results, backbone1, 'increase', reverse=True)
            best_add1, second_best_add1 = get_ranks(results, backbone1, 'add', reverse=True)
            
            best_avg_drop2, second_best_avg_drop2 = get_ranks(results, backbone2, 'ad')
            best_coherency2, second_best_coherency2 = get_ranks(results, backbone2, 'coherency', reverse=True)
            best_complexity2, second_best_complexity2 = get_ranks(results, backbone2, 'complexity')
            best_adcc2, second_best_adcc2 = get_ranks(results, backbone2, 'adcc', reverse=True)
            best_increase2, second_best_increase2 = get_ranks(results, backbone2, 'increase', reverse=True)
            best_add2, second_best_add2 = get_ranks(results, backbone2, 'add', reverse=True)
            
            avg_drop1 = format_value(row1.get('ad', '-'), best=(method == best_avg_drop1), second_best=(method == second_best_avg_drop1))
            coherency1 = format_value(row1.get('coherency', '-'), best=(method == best_coherency1), second_best=(method == second_best_coherency1))
            complexity1 = format_value(row1.get('complexity', '-'), best=(method == best_complexity1), second_best=(method == second_best_complexity1))
            adcc1 = format_value(row1.get('adcc', '-'), best=(method == best_adcc1), second_best=(method == second_best_adcc1))
            increase1 = format_value(row1.get('increase', '-'), best=(method == best_increase1), second_best=(method == second_best_increase1))
            add1 = format_value(row1.get('add', '-'), best=(method == best_add1), second_best=(method == second_best_add1))
            
            avg_drop2 = format_value(row2.get('ad', '-'), best=(method == best_avg_drop2), second_best=(method == second_best_avg_drop2))
            coherency2 = format_value(row2.get('coherency', '-'), best=(method == best_coherency2), second_best=(method == second_best_coherency2))
            complexity2 = format_value(row2.get('complexity', '-'), best=(method == best_complexity2), second_best=(method == second_best_complexity2))
            adcc2 = format_value(row2.get('adcc', '-'), best=(method == best_adcc2), second_best=(method == second_best_adcc2))
            increase2 = format_value(row2.get('increase', '-'), best=(method == best_increase2), second_best=(method == second_best_increase2))
            add2 = format_value(row2.get('add', '-'), best=(method == best_add2), second_best=(method == second_best_add2))
            
            table += f"{method_displayname} & {avg_drop1} & {coherency1} & {complexity1} & {adcc1} & {increase1} & {add1} & & {avg_drop2} & {coherency2} & {complexity2} & {adcc2} & {increase2} & {add2} \\\\\n"
        
            if method == "layercam":
                table += "\\cdashline{2-14}\n"
            elif method == "shapleycam_hires":
                table += "\\cdashline{2-14}\n"
            elif method == "shapleycam":
                table += "\\cdashline{2-14}\n"
        table += "\\hline\n"
    table += "\\end{tabular}\n"
    table += "\\end{table*}\n\n"
    
    return table
# 调整 method_pairs 顺序


# method_pairs = [
#     ("randomcam", "RandomCAM"), 
#     ("gradcamplusplus", "GradCAM++"), 
#     ("layercam", "LayerCAM"),
#     ("gradcamelementwise", "GradCAM-E"), 
#     # ("shapleycam", "ShapleyCAM-E"), 
#     # same
#     ("gradcam", "GradCAM"), 
#     # ("shapleycam_mean", "ShapleyCAM"), 
#     # ("xgradcam", "XGradCAM"), 
#     # ("hirescam", "HiResCAM"),  
#     # ("shapleycam_hires", "ShapleyCAM-H"),
# ]


file_path1 = 'gist2_conv.txt'  


results1 = parse_results(file_path1)
latex_table = generate_latex_table(results1)
latex_table_swin = generate_latex_table_swin(results1)
# print(latex_table)
with open('output_table_3_column_2_2column.tex', 'w') as f:
    f.write(latex_table)
    f.write(latex_table_swin)