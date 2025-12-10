import json
import os
import glob
import csv
import re

def calculate_metrics(ground_truth_file, results_file):
    try:
        with open(ground_truth_file, 'r') as f:
            ground_truth_data = json.load(f)
    except FileNotFoundError:
        print(f"Ground truth file not found: {ground_truth_file}")
        return None

    try:
        with open(results_file, 'r') as f:
            results_data = json.load(f)
    except FileNotFoundError:
        print(f"Results file not found: {results_file}")
        return None

    # Process ground truth
    gt_lookup = {}
    for item in ground_truth_data:
        scene_id = item['scene_id']
        question = item['conversations'][0]['value'].replace("\n<image>", "").strip()
        answer = item['ground_truth_answer'].lower()
        gt_lookup[(scene_id, question)] = answer

    # Process results
    res_lookup = {}
    raw_predictions = []
    if isinstance(results_data, dict):
        for key, value in results_data.items():
            if isinstance(value, list):
                raw_predictions.extend(value)
    elif isinstance(results_data, list):
        raw_predictions = results_data
    
    for pred in raw_predictions:
        scene_id = pred['scene_id']
        question = pred['instruction'].strip()
        response = pred['response'].lower().strip().rstrip('.')
        
        key = (scene_id, question)
        if key not in res_lookup:
            res_lookup[key] = []
        res_lookup[key].append(response)

    tp = 0
    tn = 0
    fp = 0
    fn = 0
    
    matched_gt_count = 0
    yes_responses = 0
    
    # Iterate over Ground Truth to find matches in Results
    for (scene_id, question), gt_answer in gt_lookup.items():
        if (scene_id, question) in res_lookup:
            matched_gt_count += 1
            responses = res_lookup[(scene_id, question)]
            response = responses[-1] # Take the last one
            
            if response == 'yes':
                yes_responses += 1

            if gt_answer == 'yes':
                if response == 'yes':
                    tp += 1
                else:
                    fn += 1
            else: # gt_answer == 'no'
                if response == 'yes':
                    fp += 1
                else:
                    tn += 1

    total_predictions = matched_gt_count

    if total_predictions == 0:
        return {
            "Accuracy": 0, "Precision": 0, "Recall": 0, "F1 Score": 0, "Yes (%)": 0
        }

    accuracy = (tp + tn) / total_predictions
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    yes_percentage = (yes_responses / total_predictions) * 100
    
    return {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "Yes (%)": yes_percentage
    }

def main():
    results_dir = "eval_results"
    output_csv = "validation_summary.csv"
    
    configs = [
        {
            "dataset": "scannet200",
            "gt_dir": "scannet_scannet200",
            "pattern": os.path.join(results_dir, "leo-sft_noact_scannet200_*.json"),
            "regex": r"leo-sft_noact_scannet200_(.+?)_(template_\d+)\.json",
            "is_nested": False
        },
        {
            "dataset": "nyu40",
            "gt_dir": "scannet_nyu40",
            "pattern": os.path.join(results_dir, "leo-sft_noact_nyu40_*.json"),
            "regex": r"leo-sft_noact_nyu40_(.+?)_(template_\d+)\.json",
            "is_nested": False
        }
    ]
    
    summary_data = []
    
    for config in configs:
        print(f"Processing dataset: {config['dataset']}")
        files = glob.glob(config['pattern'])
        files.sort()
        
        print(f"Found {len(files)} result files for {config['dataset']}.")
        
        for result_file in files:
            # For nested files, we match against the relative path from results_dir
            rel_path = os.path.relpath(result_file, results_dir)
            
            match = re.search(config['regex'], rel_path)
            
            # Special check for nyu40 to avoid matching scannet200 files if regex is too broad
            # But here scannet200 files are flat, nyu40 are nested, so pattern handles it.
            # However, regex for nyu40 might match scannet200 if not careful?
            # nyu40 regex expects folder structure.
            
            if match:
                q_type = match.group(1)
                template = match.group(2)
                
                # Skip if q_type contains "scannet200" (just in case of overlap, though pattern should prevent it)
                if "scannet200" in q_type:
                    continue

                gt_filename = f"{q_type}_{template}.json"
                gt_path = os.path.join(config['gt_dir'], gt_filename)
                
                print(f"Processing {rel_path}...")
                print(f"  GT: {gt_path}")
                
                metrics = calculate_metrics(gt_path, result_file)
                
                if metrics:
                    row = {
                        "Dataset": config['dataset'],
                        "File": rel_path,
                        "Type": q_type,
                        "Template": template,
                        **metrics
                    }
                    summary_data.append(row)
                else:
                    print("  Failed to calculate metrics.")
            else:
                # print(f"Skipping file {rel_path} (does not match regex)")
                pass

    # Calculate averages per dataset and type
    # Key: (dataset, type)
    group_metrics = {}
    
    for row in summary_data:
        key = (row['Dataset'], row['Type'])
        if key not in group_metrics:
            group_metrics[key] = {
                "Accuracy": [], "Precision": [], "Recall": [], "F1 Score": [], "Yes (%)": []
            }
        
        group_metrics[key]["Accuracy"].append(row["Accuracy"])
        group_metrics[key]["Precision"].append(row["Precision"])
        group_metrics[key]["Recall"].append(row["Recall"])
        group_metrics[key]["F1 Score"].append(row["F1 Score"])
        group_metrics[key]["Yes (%)"].append(row["Yes (%)"])

    # Add average rows
    for (dataset, q_type), metrics in group_metrics.items():
        avg_row = {
            "Dataset": dataset,
            "File": f"AVERAGE_{dataset}_{q_type}",
            "Type": q_type,
            "Template": "AVERAGE",
            "Accuracy": sum(metrics["Accuracy"]) / len(metrics["Accuracy"]),
            "Precision": sum(metrics["Precision"]) / len(metrics["Precision"]),
            "Recall": sum(metrics["Recall"]) / len(metrics["Recall"]),
            "F1 Score": sum(metrics["F1 Score"]) / len(metrics["F1 Score"]),
            "Yes (%)": sum(metrics["Yes (%)"]) / len(metrics["Yes (%)"])
        }
        summary_data.append(avg_row)

    # Calculate global averages per type (across datasets)
    global_type_metrics = {}
    
    for row in summary_data:
        # Skip existing average rows
        if row['Template'] == 'AVERAGE':
            continue
            
        q_type = row['Type']
        if q_type not in global_type_metrics:
            global_type_metrics[q_type] = {
                "Accuracy": [], "Precision": [], "Recall": [], "F1 Score": [], "Yes (%)": []
            }
        
        global_type_metrics[q_type]["Accuracy"].append(row["Accuracy"])
        global_type_metrics[q_type]["Precision"].append(row["Precision"])
        global_type_metrics[q_type]["Recall"].append(row["Recall"])
        global_type_metrics[q_type]["F1 Score"].append(row["F1 Score"])
        global_type_metrics[q_type]["Yes (%)"].append(row["Yes (%)"])

    # Add global average rows
    for q_type, metrics in global_type_metrics.items():
        avg_row = {
            "Dataset": "ALL",
            "File": f"AVERAGE_ALL_{q_type}",
            "Type": q_type,
            "Template": "AVERAGE",
            "Accuracy": sum(metrics["Accuracy"]) / len(metrics["Accuracy"]),
            "Precision": sum(metrics["Precision"]) / len(metrics["Precision"]),
            "Recall": sum(metrics["Recall"]) / len(metrics["Recall"]),
            "F1 Score": sum(metrics["F1 Score"]) / len(metrics["F1 Score"]),
            "Yes (%)": sum(metrics["Yes (%)"]) / len(metrics["Yes (%)"])
        }
        summary_data.append(avg_row)

    # Sort data
    type_order = {'random': 0, 'popular': 1, 'adversarial': 2}
    dataset_order = {'scannet200': 0, 'nyu40': 1, 'ALL': 2}
    
    def get_sort_key(row):
        d_dataset = row['Dataset']
        t_type = row['Type']
        t_template = row['Template']
        
        dataset_rank = dataset_order.get(d_dataset, 99)
        type_rank = type_order.get(t_type, 99)
        
        if t_template == 'AVERAGE':
            template_rank = 99
        else:
            # Extract number from template_X
            match = re.search(r'(\d+)', t_template)
            if match:
                template_rank = int(match.group(1))
            else:
                template_rank = 0
        
        return (dataset_rank, type_rank, template_rank)

    summary_data.sort(key=get_sort_key)

    if summary_data:
        # Reorder columns
        cols = ["Dataset", "File", "Type", "Template", "Precision", "Recall", "F1 Score", "Accuracy", "Yes (%)"]
        
        with open(output_csv, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=cols)
            writer.writeheader()
            for row in summary_data:
                writer.writerow(row)
                
        print(f"\nSummary saved to {output_csv}")
        # Print a simple table
        header = f"{'Dataset':<12} {'Type':<15} {'Template':<15} {'Prec':<10} {'Rec':<10} {'F1':<10} {'Acc':<10} {'Yes%':<10}"
        print(header)
        print("-" * len(header))
        for row in summary_data:
            print(f"{row['Dataset']:<12} {row['Type']:<15} {row['Template']:<15} {row['Precision']:.4f}     {row['Recall']:.4f}     {row['F1 Score']:.4f}     {row['Accuracy']:.4f}     {row['Yes (%)']:.2f}")
    else:
        print("No data to summarize.")

if __name__ == "__main__":
    main()
