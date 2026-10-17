import tempfile
import unittest
from pathlib import Path

import numpy as np
from gymnasium import Env, spaces

from experiments.robot.robocasa.simulation_env import RoboCasaObservationAdapterWrapper
from experiments.robot.robocasa.wrappers.video_recording_wrapper import VideoRecordingWrapper


class DummyOfficialRoboCasaEnv(Env):
    def __init__(self):
        image_space = spaces.Box(low=0, high=255, shape=(4, 4, 3), dtype=np.uint8)
        self.observation_space = spaces.Dict(
            {
                "observation.images.robot0_agentview_left": image_space,
                "observation.images.robot0_eye_in_hand": image_space,
                "observation.images.robot0_agentview_right": image_space,
                "observation.state": spaces.Box(low=-1, high=1, shape=(16,), dtype=np.float32),
            }
        )
        self.action_space = spaces.Box(low=-1, high=1, shape=(12,), dtype=np.float32)

    def reset(self, seed=None, options=None):
        return self._obs(), {}

    def step(self, action):
        return self._obs(), 0.0, True, False, {"success": True}

    def _obs(self):
        return {
            "observation.images.robot0_agentview_left": np.full((4, 4, 3), 20, dtype=np.uint8),
            "observation.images.robot0_eye_in_hand": np.full((4, 4, 3), 30, dtype=np.uint8),
            "observation.images.robot0_agentview_right": np.full((4, 4, 3), 10, dtype=np.uint8),
            "observation.state": np.arange(16, dtype=np.float32),
        }


class FakeVideoRecorder:
    def __init__(self):
        self._ready = False
        self.file_path = None

    def is_ready(self):
        return self._ready

    def start(self, file_path):
        self.file_path = Path(file_path)
        self.file_path.write_bytes(b"video")
        self._ready = True

    def write_frame(self, img):
        self.file_path.write_bytes(b"video-frame")

    def stop(self):
        self._ready = False


class DummyRenderableEnv(DummyOfficialRoboCasaEnv):
    def render(self):
        return np.zeros((4, 4, 3), dtype=np.uint8)


class RoboCasaEvalRuntimeTest(unittest.TestCase):
    def test_official_robocasa_observation_keys_are_normalized_to_training_schema_without_proprio(self):
        wrapped = RoboCasaObservationAdapterWrapper(
            DummyOfficialRoboCasaEnv(),
            task_description="turn off the sink faucet",
            use_proprio=False,
        )

        obs, _ = wrapped.reset()

        self.assertEqual(set(obs), {
            "video.right_image",
            "video.left_image",
            "video.wrist_image",
            "annotation.human.coarse_action",
        })
        self.assertEqual(obs["video.right_image"][0, 0, 0], 10)
        self.assertEqual(obs["video.left_image"][0, 0, 0], 20)
        self.assertEqual(obs["video.wrist_image"][0, 0, 0], 30)
        self.assertEqual(obs["annotation.human.coarse_action"], "turn off the sink faucet")
        self.assertNotIn("state", obs)

    def test_video_recording_wrapper_finalizes_last_episode_on_close(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            wrapped = VideoRecordingWrapper(
                DummyRenderableEnv(),
                FakeVideoRecorder(),
                video_dir=Path(tmp_dir),
                steps_per_render=1,
            )
            wrapped.reset()
            wrapped.step(np.zeros(12, dtype=np.float32))
            wrapped.close()

            finalized_videos = list(Path(tmp_dir).glob("*_success1.mp4"))

        self.assertEqual(len(finalized_videos), 1)


if __name__ == "__main__":
    unittest.main()
