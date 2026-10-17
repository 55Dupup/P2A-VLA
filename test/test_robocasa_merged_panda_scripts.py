from pathlib import Path
import os
import re
import subprocess
import tempfile
import unittest


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT = "/path/to/checkpoint"


class RoboCasaMergedPandaScriptsTest(unittest.TestCase):
    def test_legacy_robocasa_launchers_are_removed(self):
        for rel_path in (
            "vla-scripts/finetune_robocasa.sh",
            "vla-scripts/evaluate_robocasa.sh",
            "vla-scripts/evaluate_robocasa_batch.sh",
        ):
            with self.subTest(path=rel_path):
                self.assertFalse((REPO_ROOT / rel_path).exists())

    def test_training_script_defaults_to_converted_100demos_dataset(self):
        script = (REPO_ROOT / "vla-scripts" / "finetune_robocasa_merged_panda.sh").read_text()

        self.assertIn('data_root_dir="${DATA_ROOT_DIR:-/path/to/datasets/robocasa_merged_24_tasks_100demos_v1_rlds}"', script)
        self.assertIn('data_name="${DATASET_NAME:-robocasa_merged_panda_100demos}"', script)
        self.assertIn("--num_images_in_input 3", script)
        self.assertIn("--use_proprio False", script)
        self.assertIn("--use_gate True", script)
        self.assertIn("--use_latent_outputs True", script)
        self.assertIn('--use_contrastive_loss "${use_contrastive_loss}"', script)
        self.assertIn('--contrastive_weight "${contrastive_weight}"', script)
        self.assertIn('--action_alignment_strategy "${action_alignment_strategy}"', script)
        self.assertIn('--action_use_manifold_regularization "${action_use_manifold_regularization}"', script)
        self.assertIn('--action_manifold_topk "${action_manifold_topk}"', script)
        self.assertIn('--action_manifold_temperature "${action_manifold_temperature}"', script)
        self.assertIn('--action_learnable_alignment_weights "${action_learnable_alignment_weights}"', script)
        self.assertIn('--wandb_entity "${wandb_entity}"', script)
        self.assertIn('--wandb_project "${wandb_project}"', script)

    def test_training_script_uses_metaworld_style_launcher_defaults(self):
        script = (REPO_ROOT / "vla-scripts" / "finetune_robocasa_merged_panda.sh").read_text()

        self.assertIn('SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"', script)
        self.assertIn('ROOT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"', script)
        self.assertIn('python_bin="${PYTHON_BIN:-python}"', script)
        self.assertIn('nproc_per_node="${NPROC_PER_NODE:-}"', script)
        self.assertIn('IFS=\',\' read -r -a gpu_ids <<< "${cuda_visible_devices}"', script)
        self.assertIn('"${python_bin}" -m torch.distributed.run', script)

    def test_single_eval_defaults_match_official_protocol_and_training(self):
        script = (REPO_ROOT / "vla-scripts" / "evaluate_robocasa_merged_panda.sh").read_text()

        self.assertIn(f'DEFAULT_CHECKPOINT_PATH="${{DEFAULT_CHECKPOINT_PATH:-{DEFAULT_CHECKPOINT}}}"', script)
        self.assertIn('python_bin="${PYTHON_BIN:-python}"', script)
        self.assertIn('CHECKPOINT_PATH="${CHECKPOINT_PATH:-${DEFAULT_CHECKPOINT_PATH}}"', script)
        self.assertIn('first_arg="$1"', script)
        self.assertIn('ENV_NAME="${first_arg}"', script)
        self.assertIn('N_EPISODES="${3:-${N_EPISODES}}"', script)
        self.assertIn('n_envs="${N_ENVS:-1}"', script)
        self.assertIn('use_proprio="${USE_PROPRIO:-False}"', script)
        self.assertIn('action_space_mode="${ACTION_SPACE_MODE:-auto}"', script)
        self.assertIn('DATASET_NAME=${DATASET_NAME:-robocasa_merged_24_tasks_100demos_v1}', script)
        self.assertIn('ENV_SPLIT=${ENV_SPLIT:-target}', script)
        self.assertIn('OBJ_REGISTRIES=${OBJ_REGISTRIES:-lightwheel}', script)
        self.assertIn('--env_split "${ENV_SPLIT}"', script)
        self.assertIn('--obj_registries "${OBJ_REGISTRIES}"', script)
        self.assertIn('UNNORM_KEY=${UNNORM_KEY:-${DATASET_NAME}}', script)
        self.assertIn('--num_images_in_input 3', script)
        self.assertIn('--use_proprio "${use_proprio}"', script)
        self.assertIn('--use_pro_version True', script)
        self.assertIn('--use_gate True', script)
        self.assertIn('--use_latent_outputs True', script)
        self.assertIn('DICT_ACTION_KEYS=${DICT_ACTION_KEYS:-action.end_effector_position,action.end_effector_rotation,action.gripper_close,action.base_motion,action.control_mode}', script)
        self.assertIn('DICT_ACTION_DIMS=${DICT_ACTION_DIMS:-3,3,1,4,1}', script)
        self.assertIn('--dict_action_keys "${DICT_ACTION_KEYS}"', script)
        self.assertIn('--dict_action_dims "${DICT_ACTION_DIMS}"', script)
        self.assertIn('"${python_bin}" experiments/robot/robocasa/simulation_env.py', script)

    def _run_single_eval_with_echo_python(self, *args):
        with tempfile.TemporaryDirectory() as log_dir:
            env = os.environ.copy()
            env.update({
                "PYTHON_BIN": "/bin/echo",
                "LOG_DIR": log_dir,
                "CUDA_VISIBLE_DEVICES": "",
            })
            result = subprocess.run(
                ["bash", "vla-scripts/evaluate_robocasa_merged_panda.sh", *args],
                cwd=REPO_ROOT,
                env=env,
                text=True,
                capture_output=True,
                check=True,
            )
        return result.stdout

    def test_single_eval_allows_omitting_checkpoint_positional_arg(self):
        stdout = self._run_single_eval_with_echo_python(
            "robocasa/TurnOffSinkFaucet",
            "turn off the sink faucet",
            "7",
        )

        self.assertIn(f"--pretrained_checkpoint {DEFAULT_CHECKPOINT}", stdout)
        self.assertIn("--env_name robocasa/TurnOffSinkFaucet", stdout)
        self.assertIn("--task_description turn off the sink faucet", stdout)
        self.assertIn("--n_episodes 7", stdout)

    def test_single_eval_accepts_explicit_checkpoint_positional_arg(self):
        checkpoint = "outputs/custom_checkpoint_chkpt"
        stdout = self._run_single_eval_with_echo_python(
            checkpoint,
            "robocasa/PressMicrowaveStopButton",
            "press the stop button on the microwave",
            "9",
        )

        self.assertIn(f"--pretrained_checkpoint {checkpoint}", stdout)
        self.assertIn("--env_name robocasa/PressMicrowaveStopButton", stdout)
        self.assertIn("--task_description press the stop button on the microwave", stdout)
        self.assertIn("--n_episodes 9", stdout)

    def test_batch_eval_defaults_to_official_episode_count_and_default_env_list(self):
        batch_eval = (REPO_ROOT / "vla-scripts" / "evaluate_robocasa_merged_panda_batch.sh").read_text()

        self.assertIn('SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"', batch_eval)
        self.assertIn('ROOT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"', batch_eval)
        self.assertIn(f'DEFAULT_CHECKPOINT_PATH="${{DEFAULT_CHECKPOINT_PATH:-{DEFAULT_CHECKPOINT}}}"', batch_eval)
        self.assertIn('CHECKPOINT_PATH="${CHECKPOINT_PATH:-${DEFAULT_CHECKPOINT_PATH}}"', batch_eval)
        self.assertIn('ENV_LIST_FILE="${ENV_LIST_FILE:-experiments/robot/robocasa/robocasa_merged_panda_envs.txt}"', batch_eval)
        self.assertIn('first_arg="$1"', batch_eval)
        self.assertIn('N_EPISODES="${2:-${N_EPISODES}}"', batch_eval)
        match = re.search(r"ENV_LIST_FILE=\"\$\{ENV_LIST_FILE:-(?P<path>[^}]+)\}\"", batch_eval)
        self.assertIsNotNone(match)
        env_list_path = REPO_ROOT / match.group("path")
        self.assertTrue(env_list_path.exists())
        self.assertIn("RAN_ANY=false", batch_eval)
        self.assertIn("No evaluation tasks were found in ENV_LIST_FILE", batch_eval)

    def test_batch_eval_allows_omitting_checkpoint_positional_arg(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            env_list = tmp_path / "envs.txt"
            log_dir = tmp_path / "logs"
            env_list.write_text(
                "robocasa/TurnOffSinkFaucet|turn off the sink faucet\n",
                encoding="utf-8",
            )
            env = os.environ.copy()
            env.update({
                "PYTHON_BIN": "/bin/echo",
                "LOG_DIR": str(log_dir),
                "CUDA_VISIBLE_DEVICES": "",
            })
            subprocess.run(
                ["bash", "vla-scripts/evaluate_robocasa_merged_panda_batch.sh", str(env_list), "6"],
                cwd=REPO_ROOT,
                env=env,
                text=True,
                capture_output=True,
                check=True,
            )
            task_log = log_dir / "robocasa_TurnOffSinkFaucet.log"
            log_text = task_log.read_text(encoding="utf-8")

        self.assertIn(f"--pretrained_checkpoint {DEFAULT_CHECKPOINT}", log_text)
        self.assertIn("--env_name robocasa/TurnOffSinkFaucet", log_text)
        self.assertIn("--task_description turn off the sink faucet", log_text)
        self.assertIn("--n_episodes 6", log_text)


if __name__ == "__main__":
    unittest.main()
