# Reproducing classifier experiments

Modify the paths in classifier_src/global_paths.py to point to the correct dataset folders and run the mimic_generate_df.py file before running any experiments.

We used the following command to train our best model with the MAPLEZ annotations:
`torchrun src/train_pytorch.py --experiment=<experiment name> --backend tensor -j 8 --print-freq=100 --model v2_m --batch-size 16 --lr-warmup-epochs 4 --epochs=40 --output-dir=./runs/ --augmix-severity 0 --ra-magnitude 0 --cutmix-alpha 0 --mixup-alpha 0 --use_old_aug false --lr 0.5 --pad false --epochs_only_last_layer 10 --lr-step-size=15 --lr-scheduler steplr --use_hard_labels false --ignore_comparison_uncertainty true --label_smoothing 0 --labeler  llm --auto-augment ta_wide --wd 5e-5 --n_hidden_neurons_in_heads 1024 --severity_loss_multiplier 0 --location_loss_multiplier 0.01 --include_ap true --share_first_classifier_layer false`

To run each of the columns from Tab. 5, modify the following flags:
- CheXpert: `--labeler chexpert --use_hard_labels true --ignore_comparison_uncertainty false --auto-augment old`
- VQA: `--labeler vqa --use_hard_labels true --ignore_comparison_uncertainty false --auto-augment old --share_first_classifier_layer true --location_loss_multiplier 0.001`
- $\lambda_{loc}=0$: `--location_loss_multiplier 0.`
- Cat. Labels: `--use_hard_labels true`
- Use "Stable": `--ignore_comparison_uncertainty false`
- MAPLEZ-G: `--labeler llm_generic`
- All Changes: `--labeler llm_generic --ignore_comparison_uncertainty false --use_hard_labels true --location_loss_multiplier 0.`

Add these flags to test a trained model: `--test-only true --split test --resume <experiment folder>/model_best_epoch.pth`


## Requirements

It was tested with

- Python 3.11.5
- h5py                      3.9.0
- imageio                   2.31.3
- matplotlib                3.8.0
- numpy                     1.25.2
- pandas                    2.1.0
- pillow                    9.4.0
- pytorch                   2.0.1
- scikit-image              0.21.0
- scikit-learn              1.3.0
- scipy                     1.11.2
- torchvision               0.15.2
- torchxrayvision           1.2.0
- tqdm                      4.66.1