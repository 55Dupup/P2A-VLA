import os
import uuid
from pathlib import Path

import gymnasium as gym
import imageio
import numpy as np


class VideoRecorder:
    def __init__(self, fps, codec="libx264", quality=7):
        self.fps = fps
        self.codec = codec
        self.quality = quality
        self._writer = None
        self._file_path = None

    @classmethod
    def create_h264(cls, fps, codec="libx264", **_):
        return cls(fps=fps, codec=codec)

    def is_ready(self):
        return self._writer is not None

    def start(self, file_path):
        self.stop()
        self._file_path = str(file_path)
        self._writer = imageio.get_writer(self._file_path, fps=self.fps, codec=self.codec, quality=self.quality)

    def write_frame(self, img: np.ndarray):
        if not self.is_ready():
            raise RuntimeError("Must call start() before writing frames.")
        self._writer.append_data(img)

    def stop(self):
        if self._writer is not None:
            self._writer.close()
            self._writer = None
            self._file_path = None

    def __del__(self):
        self.stop()


class VideoRecordingWrapper(gym.Wrapper):
    def __init__(
        self,
        env,
        video_recorder: VideoRecorder,
        mode="rgb_array",
        video_dir: Path | None = None,
        steps_per_render=1,
        **kwargs,
    ):
        super().__init__(env)
        if video_dir is not None:
            video_dir.mkdir(parents=True, exist_ok=True)

        self.mode = mode
        self.render_kwargs = kwargs
        self.steps_per_render = steps_per_render
        self.video_dir = video_dir
        self.video_recorder = video_recorder
        self.file_path = None
        self.step_count = 0
        self.is_success = False

    def _finalize_video_file(self):
        self.video_recorder.stop()
        if self.video_dir is None or self.file_path is None or not os.path.exists(self.file_path):
            return

        original_filestem = Path(self.file_path).stem
        if original_filestem.endswith(f"_success{int(self.is_success)}"):
            return

        new_filestem = f"{original_filestem}_success{int(self.is_success)}"
        new_file_path = self.video_dir / f"{new_filestem}.mp4"
        os.rename(self.file_path, new_file_path)
        self.file_path = new_file_path

    def reset(self, **kwargs):
        result = super().reset(**kwargs)
        self.step_count = 1
        self._finalize_video_file()

        self.is_success = False
        self.file_path = None
        if self.video_dir is not None:
            self.file_path = self.video_dir / f"{uuid.uuid4()}.mp4"
        return result

    def step(self, action):
        result = super().step(action)
        self.step_count += 1

        if self.file_path is not None and (self.step_count % self.steps_per_render) == 0:
            if not self.video_recorder.is_ready():
                self.video_recorder.start(self.file_path)

            frame = self.env.render()
            if frame is not None:
                frame = np.asarray(frame, dtype=np.uint8)
                self.video_recorder.write_frame(frame)

            info = result[-1]
            if isinstance(info, dict) and "success" in info:
                success_value = np.asarray(info["success"]).reshape(-1)
                self.is_success = bool(success_value[-1]) if success_value.size > 0 else False
        return result

    def render(self, mode="rgb_array", **kwargs):
        if self.video_recorder.is_ready():
            self.video_recorder.stop()
        return self.file_path

    def close(self):
        self._finalize_video_file()
        return super().close()
