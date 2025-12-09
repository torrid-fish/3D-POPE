#!/bin/bash

########### Scannet200 Evaluation ###############

# 1 has been evaluated already
for template_id in 2 3 4; do
    python eval_leo.py \
        scannet_scannet200/adversarial_template_${template_id}.json
done

for template_id in 1 2 3 4; do
    python eval_leo.py \
        scannet_scannet200/popular_template_${template_id}.json
done

for template_id in 1 2 3 4; do
    python eval_leo.py \
        scannet_scannet200/random_template_${template_id}.json
done

############## Nyu40 Evaluation #################

for template_id in 1 2 3 4; do
    python eval_leo.py \
        scannet_nyu40/adversarial_template_${template_id}.json
done

for template_id in 1 2 3 4; do
    python eval_leo.py \
        scannet_nyu40/popular_template_${template_id}.json
done

for template_id in 1 2 3 4; do
    python eval_leo.py \
        scannet_nyu40/random_template_${template_id}.json
done
