import json
import argparse
import subprocess
import os
import sys
import math

def main():
    parser = argparse.ArgumentParser(description="Run POPE evaluation from JSON file.")
    parser.add_argument("json_file", help="Path to the input JSON file")
    args = parser.parse_args()

    json_path = os.path.abspath(args.json_file)
    
    if not os.path.exists(json_path):
        print(f"Error: File {json_path} not found.")
        sys.exit(1)

    print(f"Reading data from {json_path}...")
    with open(json_path, 'r') as f:
        data = json.load(f)

    scene_ids = []
    instructions = []

    for item in data:
        if 'scene_id' in item and 'conversations' in item:
            scene_ids.append(item['scene_id'])
            # TODO: Extract instruction and remove <image> tag
            instruction = item['conversations'][0]['value'].replace('\n<image>', '').replace('<image>', '').strip()
            instructions.append(instruction)

    if not scene_ids:
        print("No valid data found in JSON.")
        sys.exit(1)

    # Parameters from eval_leo.sh
    # We use absolute path for trained_model to be safe when changing directory
    # sft_noact is inside embodied-generalist
    workspace_root = os.getcwd()
    model_name = "sft_noact"
    trained_model_abs = os.path.join(workspace_root, "embodied-generalist", model_name)
    eval_note = args.json_file.split('/')[-1].replace('.json', '')
    
    # We need to run this from embodied-generalist directory
    target_cwd = os.path.join(workspace_root, "embodied-generalist")
    
    if not os.path.exists(target_cwd):
        print(f"Error: Directory {target_cwd} not found.")
        sys.exit(1)

    total_items = len(scene_ids)
    print(f"Found {total_items} items. Processing all at once...")

    # Set environment variables
    env = os.environ.copy()
    env["HYDRA_FULL_ERROR"] = "1"

    # Format lists for Hydra
    def format_list_for_hydra(items):
        escaped_items = []
        for item in items:
            # Escape double quotes
            esc = item.replace('"', '\\"')
            # Wrap in quotes
            escaped_items.append(f'"{esc}"')
        return '[' + ','.join(escaped_items) + ']'

    scene_ids_str = format_list_for_hydra(scene_ids)
    instructions_str = format_list_for_hydra(instructions)
    
    cmd = [
        sys.executable, "launch.py",
        "--mode", "python",
        "--run_file", "inference.py",
        "--config", "configs/default.yaml",
        f"name=leo-{model_name}",
        f"note={eval_note}",
        f"pretrained_ckpt_path={trained_model_abs}",
        "data.scan_family_base=../",
        "base_dir=../eval_results/",
        "llm.cfg_path=lmsys/vicuna-7b-v1.1",
        "vision3d.backbone.path=./pointnetpp_vil3dref.pth",
        'probe.sources=["scannet"]',
        f"probe.scene_ids={scene_ids_str}",
        f"probe.instructions={instructions_str}",
        "probe.save_obj_tokens=false",
        "dataloader.eval.batchsize=1"
    ]
    
    # Run the command
    try:
        subprocess.run(cmd, cwd=target_cwd, env=env, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error during execution: {e}")
        sys.exit(1)

    print("Evaluation completed.")

if __name__ == "__main__":
    main()
