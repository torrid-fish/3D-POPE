#!/bin/bash

TRAINED_MODEL="sft_noact"
SCENE_IDS=("scene0011_00" "scene0011_00")
INSTRUCTIONS=("Describe this scene." "Is there a chair in this scene?")
EVAL_NOTE="test"
# Join SCENE_IDS with commas and escaped quotes
printf -v SCENE_IDS_STR '\\"%s\\",' "${SCENE_IDS[@]}"
SCENE_IDS_STR="${SCENE_IDS_STR%,}"

# Escape spaces in INSTRUCTIONS and join with commas and escaped quotes
ESCAPED_INSTRUCTIONS=()
for inst in "${INSTRUCTIONS[@]}"; do
    ESCAPED_INSTRUCTIONS+=("${inst// /\\ }")
done
printf -v INSTRUCTIONS_STR '\\"%s\\",' "${ESCAPED_INSTRUCTIONS[@]}"
INSTRUCTIONS_STR="${INSTRUCTIONS_STR%,}"

# Run the evaluation script under original embodied-generalist directory
cd embodied-generalist

HYDRA_FULL_ERROR=1 python launch.py \
    --mode python \
    --run_file inference.py \
    --config configs/default.yaml \
    name="LEO-${TRAINED_MODEL}"\
    note=${EVAL_NOTE} \
    pretrained_ckpt_path=${TRAINED_MODEL} \
    data.scan_family_base="../" \
    base_dir="../eval_results/" \
    llm.cfg_path="lmsys/vicuna-7b-v1.1" \
    vision3d.backbone.path="./pointnetpp_vil3dref.pth" \
    probe.sources=[\"scannet\"] \
    probe.scene_ids=[${SCENE_IDS_STR}] \
    probe.instructions=[${INSTRUCTIONS_STR}] \
    probe.save_obj_tokens=false \
    dataloader.eval.batchsize=1

cd ..