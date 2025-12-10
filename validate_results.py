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
    gt_base_dir = "scannet_scannet200"
    output_csv = "validation_summary.csv"
    
    # Pattern to match the result files
    # leo-sft_noact_scannet200_{type}_{template}.json
    pattern = os.path.join(results_dir, "leo-sft_noact_scannet200_*.json")
    files = glob.glob(pattern)
    files.sort()
    
    summary_data = []
    
    print(f"Found {len(files)} result files.")
    
    for result_file in files:
        filename = os.path.basename(result_file)
        
        # Extract type and template using regex
        # Expected format: leo-sft_noact_scannet200_(adversarial|popular|random)_(template_\d+).json
        match = re.search(r"leo-sft_noact_scannet200_(.+?)_(template_\d+)\.json", filename)
        
        if match:
            q_type = match.group(1)
            template = match.group(2)
            
            gt_filename = f"{q_type}_{template}.json"
            gt_path = os.path.join(gt_base_dir, gt_filename)
            
            print(f"Processing {filename}...")
            print(f"  GT: {gt_path}")
            
            metrics = calculate_metrics(gt_path, result_file)
            
            if metrics:
                row = {
                    "File": filename,
                    "Type": q_type,
                    "Template": template,
                    **metrics
                }
                summary_data.append(row)
            else:
                print("  Failed to calculate metrics.")
        else:
            print(f"Skipping file {filename} (does not match pattern)")

    # Calculate averages per type
    type_metrics = {}
    
    for row in summary_data:
        q_type = row['Type']
        if q_type not in type_metrics:
            type_metrics[q_type] = {
                "Accuracy": [], "Precision": [], "Recall": [], "F1 Score": [], "Yes (%)": []
            }
        
        type_metrics[q_type]["Accuracy"].append(row["Accuracy"])
        type_metrics[q_type]["Precision"].append(row["Precision"])
        type_metrics[q_type]["Recall"].append(row["Recall"])
        type_metrics[q_type]["F1 Score"].append(row["F1 Score"])
        type_metrics[q_type]["Yes (%)"].append(row["Yes (%)"])

    # Add average rows
    for q_type, metrics in type_metrics.items():
        avg_row = {
            "File": f"AVERAGE_{q_type}",
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
    
    def get_sort_key(row):
        t_type = row['Type']
        t_template = row['Template']
        
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
        
        return (type_rank, template_rank)

    summary_data.sort(key=get_sort_key)

    if summary_data:
        # Reorder columns
        cols = ["File", "Type", "Template", "Precision", "Recall", "F1 Score", "Accuracy", "Yes (%)"]
        
        with open(output_csv, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=cols)
            writer.writeheader()
            for row in summary_data:
                writer.writerow(row)
                
        print(f"\nSummary saved to {output_csv}")
        # Print a simple table
        header = f"{'File':<55} {'Type':<15} {'Template':<15} {'Prec':<10} {'Rec':<10} {'F1':<10} {'Acc':<10} {'Yes%':<10}"
        print(header)
        print("-" * len(header))
        for row in summary_data:
            print(f"{row['File']:<55} {row['Type']:<15} {row['Template']:<15} {row['Precision']:.4f}     {row['Recall']:.4f}     {row['F1 Score']:.4f}     {row['Accuracy']:.4f}     {row['Yes (%)']:.2f}")
    else:
        print("No data to summarize.")

if __name__ == "__main__":
    main()
