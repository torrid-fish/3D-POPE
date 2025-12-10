#!/bin/bash

########### Scannet200 Evaluation ###############

for template_id in 1 2 3 4; do
    python eval_leo.py \
        scannet_scannet200/adversarial_template_${template_id}.json
    git add eval_results/
    git commit -m "chore: scannet200_adversarial_template_${template_id} complete"
done

for template_id in 1 2 3 4; do
    python eval_leo.py \
        scannet_scannet200/popular_template_${template_id}.json
    git add eval_results/
    git commit -m "chore: scannet200_popular_template_${template_id}"
done

for template_id in 1 2 3 4; do
    python eval_leo.py \
        scannet_scannet200/random_template_${template_id}.json
    git add eval_results/
    git commit -m "chore: scannet200_random_template_${template_id} complete"
done

############## Nyu40 Evaluation #################

for template_id in 1 2 3 4; do
    python eval_leo.py \
        scannet_nyu40/adversarial_template_${template_id}.json
    git add eval_results/
    git commit -m "chore: nyu40_adversarial_template_${template_id} complete"
done

for template_id in 1 2 3 4; do
    python eval_leo.py \
        scannet_nyu40/popular_template_${template_id}.json
    git add eval_results/
    git commit -m "chore: nyu40_popular_template_${template_id} complete"
done

for template_id in 1 2 3 4; do
    python eval_leo.py \
        scannet_nyu40/random_template_${template_id}.json
    git add eval_results/
    git commit -m "chore: nyu40_random_template_${template_id} complete"
done
