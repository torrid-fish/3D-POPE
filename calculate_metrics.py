import json
import argparse
import os

def calculate_metrics(ground_truth_file, results_file):
    print(f"Loading ground truth from {ground_truth_file}")
    with open(ground_truth_file, 'r') as f:
        ground_truth_data = json.load(f)

    print(f"Loading results from {results_file}")
    with open(results_file, 'r') as f:
        results_data = json.load(f)

    # Process ground truth
    # Key: (scene_id, question) -> Value: answer
    gt_lookup = {}
    for item in ground_truth_data:
        scene_id = item['scene_id']
        question = item['conversations'][0]['value'].replace("\n<image>", "").strip()
        answer = item['ground_truth_answer'].lower()
        gt_lookup[(scene_id, question)] = answer

    # Process results
    # results_data is a dict, we need to iterate over its values (lists of predictions)
    # We will build a lookup for results: (scene_id, cleaned_question) -> list of responses
    res_lookup = {}
    
    raw_predictions = []
    if isinstance(results_data, dict):
        for key, value in results_data.items():
            if isinstance(value, list):
                raw_predictions.extend(value)
    elif isinstance(results_data, list):
        raw_predictions = results_data
    else:
        print("Error: Unknown format for results file.")
        return

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
    missing_res_count = 0
    yes_responses = 0
    
    # Iterate over Ground Truth to find matches in Results
    for (scene_id, question), gt_answer in gt_lookup.items():
        if (scene_id, question) in res_lookup:
            matched_gt_count += 1
            # If there are multiple results, we take the last one (assuming it's the most recent run)
            # or we could warn. Let's take the last one.
            responses = res_lookup[(scene_id, question)]
            if len(responses) > 1:
                # print(f"Warning: Multiple results found for {scene_id}, {question}. Using the last one.")
                pass
            
            response = responses[-1]
            
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
        else:
            missing_res_count += 1
            if missing_res_count <= 5:
                 print(f"Missing result for GT: Scene '{scene_id}', Question '{question}'")

    total_predictions = matched_gt_count
    print(f"Ground Truth items: {len(gt_lookup)}")
    print(f"Matched items: {matched_gt_count}")
    print(f"GT items missing in Results: {missing_res_count}")

    if total_predictions == 0:
        print("No predictions matched with ground truth.")
        return

    accuracy = (tp + tn) / total_predictions
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    yes_percentage = (yes_responses / total_predictions) * 100
    
    # Calculate yes percentage for ALL results (just for info)
    yes_responses_all = 0
    for r_list in res_lookup.values():
        for r in r_list:
            if r == 'yes':
                yes_responses_all += 1
    total_res_count = sum(len(v) for v in res_lookup.values())
    yes_percentage_all = (yes_responses_all / total_res_count) * 100 if total_res_count > 0 else 0

    print("-" * 30)
    print(f"Metrics (on matched data):")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Yes (%): {yes_percentage:.2f}%")
    print(f"Yes (%) (All results in file): {yes_percentage_all:.2f}%")
    print("-" * 30)
    print(f"TP: {tp}, TN: {tn}, FP: {fp}, FN: {fn}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate metrics from ground truth and results files.")
    
    # Positional arguments
    parser.add_argument("results_file", nargs='?', help="Path to the generated results JSON file")
    parser.add_argument("gt_file", nargs='?', help="Path to the dataset ground truth JSON file")
    
    # Optional flags (for backward compatibility or explicit naming)
    parser.add_argument("--results", help="Path to results JSON file")
    parser.add_argument("--gt", help="Path to ground truth JSON file")
    
    args = parser.parse_args()
    
    # Determine paths (flags take precedence, then positional, then defaults)
    results_path = args.results if args.results else args.results_file
    gt_path = args.gt if args.gt else args.gt_file
    
    # Default values if nothing provided
    if not results_path:
        results_path = "eval_results/leo-sft_noact_adversarial_template_1/probe/results.json"
    if not gt_path:
        gt_path = "scannet_scannet200/adversarial_template_1.json"
    
    print(f"Using Results File: {results_path}")
    print(f"Using Ground Truth File: {gt_path}")
    
    if not os.path.exists(gt_path):
        print(f"Error: Ground truth file not found at {gt_path}")
    elif not os.path.exists(results_path):
        print(f"Error: Results file not found at {results_path}")
    else:
        calculate_metrics(gt_path, results_path)
