<div align="center">
  <h1>P2A-VLA</h1>
  <p><strong>Progressive Perception-to-Action Alignment for Vision-Language-Action Models</strong></p>
  <p>
    <a href="https://55dupup.github.io/P2A-VLA-offical/">Project Website</a> ·
    <a href="#installation">Installation</a> ·
    <a href="#training">Training</a> ·
    <a href="#evaluation">Evaluation</a> ·
    <a href="LICENSE">MIT License</a>
  </p>
</div>

P2A-VLA studies progressive alignment between perception and continuous robot actions. This repository contains the research code. The method overview, results, and demonstrations are available on the [project website](https://55dupup.github.io/P2A-VLA-offical/), whose source is maintained in a [separate website repository](https://github.com/55Dupup/P2A-VLA-offical).

## Installation

Use Linux, Python 3.10, and a CUDA-capable NVIDIA GPU. Run the commands below from the repository root unless another directory is specified.

```bash
git clone https://github.com/55Dupup/P2A-VLA.git
cd P2A-VLA
conda create -n p2a-vla python=3.10 -y
conda activate p2a-vla
```

Install a CUDA-compatible PyTorch and torchvision build using the [PyTorch installation instructions](https://pytorch.org/get-started/locally/), then install the project dependencies:

```bash
pip install -e .
pip install "transformers==4.40.1" packaging ninja
pip install flash-attn --no-build-isolation
```

Choose a FlashAttention build compatible with your PyTorch, CUDA, and GPU. The repository's `environment.yml` provides reference package versions; the commands above create a separate environment. Choose CUDA packages compatible with your system.

### Benchmark Environments

Install only the environments needed for your experiment. Keep benchmark checkouts outside the P2A-VLA repository and replace the example paths with your local directories.

**LIBERO**

```bash
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git /path/to/LIBERO
pip install -e /path/to/LIBERO
pip install -r experiments/robot/libero/libero_requirements.txt
```

**CALVIN**

```bash
git clone --recurse-submodules https://github.com/mees/calvin.git /path/to/calvin
cd /path/to/calvin
bash install.sh
cd /path/to/P2A-VLA
```

**MetaWorld**

```bash
git clone https://github.com/Farama-Foundation/Metaworld.git /path/to/Metaworld
pip install -e /path/to/Metaworld
```

For headless simulation, configure the rendering backend required by your machine, for example `MUJOCO_GL=egl`.

## Model and Data Preparation

### VLM Backbone

Download the [Prismatic Qwen2.5-0.5B backbone](https://huggingface.co/Stanford-ILIAD/prism-qwen25-extra-dinosiglip-224px-0_5b) into the location used by the current training implementation:

```bash
python -c 'from huggingface_hub import snapshot_download; snapshot_download(repo_id="Stanford-ILIAD/prism-qwen25-extra-dinosiglip-224px-0_5b", local_dir="pretrained_models/prism-qwen25-extra-dinosiglip-224px-0_5b")'
```

Keep the checked-in `pretrained_models/configs/` directory. The current training code uses these configuration files when constructing the VLA model. The downloaded backbone initializes training; it is not a P2A-VLA policy checkpoint.

### Training Data

Fine-tuning expects TFDS/RLDS datasets. Available data sources include [LIBERO RLDS](https://huggingface.co/datasets/openvla/modified_libero_rlds) and [CALVIN ABC RLDS](https://huggingface.co/datasets/zhouhongyi/calvin_abc_rlds). Prepare MetaWorld demonstrations as an RLDS builder registered as `metaworld_mt50`.

Point `--data_root_dir` to the parent of the dataset builder directory:

```text
/path/to/rlds/
├── libero_spatial_no_noops/1.0.0/
├── libero_object_no_noops/1.0.0/
├── libero_goal_no_noops/1.0.0/
├── libero_10_no_noops/1.0.0/
├── calvin_abc_rlds/1.0.0/
└── metaworld_mt50/1.0.0/
```

CALVIN evaluation additionally needs the original simulation dataset and environment checkout; training RLDS files alone are insufficient for rollout evaluation.

## Training

The common entrypoint is `vla-scripts/finetune.py`. The example below enables progressive alignment and local structural regularization for LIBERO Spatial:

```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nnodes 1 --nproc_per_node 2 \
  vla-scripts/finetune.py \
  --vlm_path ./pretrained_models/prism-qwen25-extra-dinosiglip-224px-0_5b \
  --config_file_path ./pretrained_models/configs \
  --data_root_dir /path/to/rlds \
  --dataset_name libero_spatial_no_noops \
  --run_root_dir outputs/libero \
  --use_minivlm True --use_l1_regression True --use_film False \
  --num_images_in_input 2 --use_proprio True \
  --use_lora True --lora_rank 64 --use_fz False \
  --use_pro_version True --use_gate True --use_latent_outputs True \
  --use_contrastive_loss True --contrastive_weight 0.1 \
  --action_alignment_strategy improved \
  --action_use_manifold_regularization True \
  --action_manifold_topk 8 --action_manifold_temperature 0.2 \
  --action_learnable_alignment_weights True \
  --batch_size 8 --grad_accumulation_steps 8 --learning_rate 2e-4 \
  --max_steps 150005 --num_steps_before_decay 100000 --save_freq 5000 \
  --image_aug True --save_latest_checkpoint_only False \
  --merge_lora_during_training True \
  --wandb_project P2A-VLA --run_id_note libero-spatial-pra-lsc
```

For the other benchmarks, replace the dataset, observation settings, output directory, and run note:

| Benchmark | `--dataset_name` | `--num_images_in_input` | `--use_proprio` |
| --- | --- | ---: | --- |
| LIBERO Spatial | `libero_spatial_no_noops` | 2 | `True` |
| LIBERO Object | `libero_object_no_noops` | 2 | `True` |
| LIBERO Goal | `libero_goal_no_noops` | 2 | `True` |
| LIBERO Long | `libero_10_no_noops` | 2 | `True` |
| CALVIN ABC | `calvin_abc_rlds` | 2 | `True` |
| MetaWorld MT50 | `metaworld_mt50` | 1 | `False` |

Set `--nproc_per_node` to the number of selected GPUs. `--batch_size` is per GPU; effective batch size is `batch_size × GPU count × grad_accumulation_steps`. The command is a launch template; select the training schedule appropriate to the experiment to match your dataset and compute budget.

The current implementation names the alignment switches `use_contrastive_loss`, `action_alignment_strategy`, and `action_use_manifold_regularization`. Keep these settings and the gated/latent action-head architecture consistent with the experiment. The legacy LIBERO and CALVIN shell launchers contain example paths and experiment defaults; the direct command above exposes the required settings explicitly.

### Checkpoints and Logs

Training writes checkpoints under `--run_root_dir`. With `--merge_lora_during_training True`, saved checkpoints include merged model weights for evaluation. Retain the processor/configuration files, `dataset_statistics.json`, action-head weights, and the proprio projector when proprioception is enabled. The exact weight suffix depends on the checkpoint-saving mode.

The current training entrypoint initializes W&B in offline mode. Logs are written locally and can be synchronized later if needed.

## Evaluation

Use a checkpoint from your own P2A-VLA training run. Match the action-head, observation, and proprioception settings to those used in training; the backbone download is not an evaluation checkpoint.

### LIBERO

```bash
CUDA_VISIBLE_DEVICES=0 python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint /path/to/merged-checkpoint \
  --task_suite_name libero_spatial \
  --num_images_in_input 2 --use_proprio True --use_film False \
  --use_pro_version True --use_gate True --use_latent_outputs True \
  --num_trials_per_task 50 --save_video True \
  --local_log_dir outputs/eval/libero/logs \
  --local_video_dir outputs/eval/libero/videos
```

Select `libero_spatial`, `libero_object`, `libero_goal`, or `libero_10` for the corresponding trained suite.

### CALVIN ABC → D

Before running the current evaluator, update `CALVIN_ROOT` and the dataset path passed to `make_env(...)` in `vla-scripts/evaluate_calvin.py`. They contain placeholder paths that must be configured locally. Set `CALVIN_ROOT` to your CALVIN checkout and the dataset path to the directory containing the intended D evaluation split's `validation/` folder. The current `--calvin_path` option does not replace both hard-coded locations.

```bash
CUDA_VISIBLE_DEVICES=0 python vla-scripts/evaluate_calvin.py \
  --pretrained_checkpoint /path/to/merged-checkpoint \
  --num_images_in_input 2 --use_proprio True --use_film False \
  --use_pro_version True --use_gate True --use_latent_outputs True \
  --local_log_dir outputs/eval/calvin \
  --run_id_note calvin-abc-to-d --save_video True
```

### MetaWorld MT50

The MetaWorld evaluation wrapper accepts environment variables and forwards additional command-line options to the Python evaluator:

```bash
CUDA_VISIBLE_DEVICES=0 \
PRETRAINED_CHECKPOINT=/path/to/merged-checkpoint \
BENCHMARK=MT50 \
OUTPUT_DIR=outputs/eval/metaworld \
NUM_TRIALS_PER_TASK=10 \
SAVE_VIDEO=y \
bash vla-scripts/evaluate_metaworld.sh
```

Its default observation configuration is one image without proprioception. Review the task coverage and skipped-environment information in the evaluation output when reporting MT50 results.

## Repository Layout

```text
experiments/robot/libero/       LIBERO evaluation
experiments/robot/metaworld/    MetaWorld evaluation
vla-scripts/finetune.py         Common training entrypoint
vla-scripts/evaluate_calvin.py CALVIN evaluation
prismatic/                     VLM, action heads, and RLDS loading
pretrained_models/configs/     Model configuration files
pyproject.toml                 Package dependencies
LICENSE                        MIT license and attribution
```

## License

The code is distributed under the [MIT License](LICENSE). The original copyright and license notice are retained in `LICENSE`. Third-party models, datasets, and dependencies retain their respective licenses.

## Acknowledgements

P2A-VLA builds on [VLA-Adapter](https://github.com/OpenHelix-Team/VLA-Adapter), [Prismatic VLMs](https://github.com/TRI-ML/prismatic-vlms), [OpenVLA](https://github.com/openvla/openvla), [OpenVLA-OFT](https://github.com/moojink/openvla-oft), and [MiniVLA](https://github.com/Stanford-ILIAD/openvla-mini). We thank their authors and the teams behind [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO), [CALVIN](https://github.com/mees/calvin), and [MetaWorld](https://github.com/Farama-Foundation/Metaworld) for their open-source contributions.
