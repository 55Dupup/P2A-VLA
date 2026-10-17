from pathlib import Path
import os
import subprocess
import tempfile
import unittest


REPO_ROOT = Path(__file__).resolve().parents[1]


class RoboCasaCosmosScriptsTest(unittest.TestCase):
    def test_training_script_uses_cosmos_dataset_7d_three_view_proprio_defaults(self):
        script = (REPO_ROOT / "vla-scripts" / "finetune_robocasa_cosmos.sh").read_text()

        self.assertIn('data_root_dir="${DATA_ROOT_DIR:-/path/to/datasets/robocasa_cosmos_success_only_rlds}"', script)
        self.assertIn('data_name="${DATASET_NAME:-robocasa_cosmos}"', script)
        self.assertIn('run_root_dir="${RUN_ROOT_DIR:-${ROOT_DIR}/outputs/robocasa-cosmos}"', script)
        self.assertIn('num_steps_before_decay="${NUM_STEPS_BEFORE_DECAY:-30000}"', script)
        self.assertIn('max_steps="${MAX_STEPS:-45005}"', script)
        self.assertIn("--num_images_in_input 3", script)
        self.assertIn("--use_proprio True", script)
        self.assertIn("--image_aug True", script)
        self.assertIn("--use_gate True", script)
        self.assertIn("--use_latent_outputs True", script)

    def test_single_eval_script_uses_official_cosmos_eval_alignment_defaults(self):
        script = (REPO_ROOT / "vla-scripts" / "evaluate_robocasa_cosmos.sh").read_text()

        self.assertIn('DATASET_NAME=${DATASET_NAME:-robocasa_cosmos_success_only}', script)
        self.assertIn('UNNORM_KEY=${UNNORM_KEY:-${DATASET_NAME}}', script)
        self.assertIn('TASK_NAME="${TASK_NAME:-TurnOffMicrowave}"', script)
        self.assertIn('N_TRIALS="${N_TRIALS:-50}"', script)
        self.assertIn('LOG_DIR=${LOG_DIR:-experiments/robocasa/tmp/robocasa_cosmos_single}', script)
        self.assertIn('IMAGE_OBS_KEY=${IMAGE_OBS_KEY:-video.left_image}', script)
        self.assertIn('EXTRA_IMAGE_OBS_KEYS=${EXTRA_IMAGE_OBS_KEYS:-video.right_image,video.wrist_image}', script)
        self.assertIn('STATE_OBS_KEYS=${STATE_OBS_KEYS:-state}', script)
        self.assertIn('FLIP_IMAGES=${FLIP_IMAGES:-True}', script)
        self.assertIn('USE_PROPRIO=${USE_PROPRIO:-True}', script)
        self.assertIn('PROPRIO_DIM=${PROPRIO_DIM:-9}', script)
        self.assertIn('N_ACTION_STEPS=${N_ACTION_STEPS:-16}', script)
        self.assertIn('APPEND_FIXED_MOBILE_BASE_ACTION=${APPEND_FIXED_MOBILE_BASE_ACTION:-True}', script)
        self.assertIn('ROBOTS=${ROBOTS:-PandaMobile}', script)
        self.assertIn('OBJ_INSTANCE_SPLIT=${OBJ_INSTANCE_SPLIT:-B}', script)
        self.assertIn('LAYOUT_AND_STYLE_IDS=${LAYOUT_AND_STYLE_IDS:-((1,1),(2,2),(4,4),(6,9),(7,10))}', script)
        self.assertIn('"${python_bin}" experiments/robot/robocasa/run_robocasa_cosmos_eval.py', script)

    def test_single_eval_allows_omitting_checkpoint_when_env_default_is_set(self):
        with tempfile.TemporaryDirectory() as log_dir:
            env = os.environ.copy()
            env.update({
                "PYTHON_BIN": "/bin/echo",
                "DEFAULT_CHECKPOINT_PATH": "outputs/robocasa-cosmos/default_chkpt",
                "LOG_DIR": log_dir,
                "CUDA_VISIBLE_DEVICES": "",
            })
            result = subprocess.run(
                ["bash", "vla-scripts/evaluate_robocasa_cosmos.sh", "TurnOffMicrowave", "3"],
                cwd=REPO_ROOT,
                env=env,
                text=True,
                capture_output=True,
                check=True,
            )

        self.assertIn("--pretrained_checkpoint outputs/robocasa-cosmos/default_chkpt", result.stdout)
        self.assertIn("--task_name TurnOffMicrowave", result.stdout)
        self.assertIn("--num_trials_per_task 3", result.stdout)
        self.assertIn("--append_fixed_mobile_base_action True", result.stdout)


if __name__ == "__main__":
    unittest.main()
