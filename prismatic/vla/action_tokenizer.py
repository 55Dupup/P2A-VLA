"""
action_tokenizer.py

Extension class; wraps base LLM/VLM tokenizer with logic to discretize and tokenize continuous robot actions.
"""

import json
from functools import partial
from pathlib import Path
from typing import List, Union

import numpy as np
import torch
from transformers import PreTrainedTokenizerBase
from transformers.models.qwen2.tokenization_qwen2_fast import Qwen2TokenizerFast

from prismatic.overwatch.overwatch import initialize_overwatch

overwatch = initialize_overwatch(__name__)


class ActionTokenizer:
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        bins: int = 256,
        min_action: int = -1,
        max_action: int = 1,
        use_extra: bool = False,
    ) -> None:
        """
        Discretizes continuous robot actions into N bins per dimension and maps to the least used tokens.

        NOTE =>> by default, assumes a BPE-style tokenizer akin to the LlamaTokenizer, where *the least used tokens*
                 appear at the end of the vocabulary!

        :param tokenizer: Base LLM/VLM tokenizer to extend.
        :param bins: Number of bins for each continuous value; we'll adopt a uniform binning strategy.
        :param min_action: Minimum action value (for clipping, setting lower bound on bin interval).
        :param max_action: Maximum action value (for clipping, setting upper bound on bin interval).
        :param use_extra: Use the extra tokens (not just the last ones), only implemented for Qwen2
        """
        self.tokenizer, self.n_bins, self.min_action, self.max_action = tokenizer, bins, min_action, max_action

        # Create Uniform Bins + Compute Bin Centers
        self.bins = np.linspace(min_action, max_action, self.n_bins)
        self.bin_centers = (self.bins[:-1] + self.bins[1:]) / 2.0

        self.tokenizer_len = self.tokenizer.vocab_size
        if isinstance(tokenizer, Qwen2TokenizerFast) and use_extra:
            self.tokenizer_len = len(self.tokenizer)
        elif use_extra:
            raise NotImplementedError("Cannot use extra tokens for this tokenizer!")

        # [Contract] Set "action_token_begin_idx" based on `self.tokenizer.vocab_size - (self.n_bins + 1)`
        #   =>> Assumes we're always overwriting the final `n_bins` tokens of the vocabulary!
        self.action_token_begin_idx: int = int(self.tokenizer_len - (self.n_bins + 1))
        self.action_token_end_idx: int = int(self.tokenizer_len)

    def __call__(self, action: np.ndarray, use_minivlm) -> Union[str, List[str]]:
        """Clip & bin actions to *the last `n_bins` tokens* of the vocabulary (e.g., tokenizer.vocab[-256:])."""
        action = np.clip(action, a_min=float(self.min_action), a_max=float(self.max_action))
        discretized_action = np.digitize(action, self.bins)

        # import pdb; pdb.set_trace()
        if use_minivlm:
            return (self.tokenizer_len - discretized_action).tolist()

        else:
            # Handle single element vs. batch
            if len(discretized_action.shape) <= 1:
                return self.tokenizer.decode(list(self.tokenizer_len - discretized_action))
            else:
                return self.tokenizer.batch_decode((self.tokenizer_len - discretized_action).tolist())

    def decode_token_ids_to_actions(self, action_token_ids: np.ndarray) -> np.ndarray:
        """
        将模型输出的动作 token ID 反向映射回连续的动作值（使用 bin 中心近似）。

        注意 np.digitize 的特性：
            - 返回值范围是 [1, n_bins+1]
            - 当值正好等于 max_action 或超出时，会返回 n_bins+1
            - 但我们只有 n_bins 个 bin 中心，因此需要特殊处理最后一个索引
        """
        # 先恢复成离散 bin 索引：vocab_len - token_id
        discretized_actions = self.tokenizer_len - action_token_ids

        # 关键处理：
        # 1. 所有索引减 1 → 变为 [0, n_bins]
        # 2. 如果出现 n_bins（即原 digitize 返回 n_bins+1），会越界
        # 3. 因此 clip 到 [0, n_bins-1]，让 n_bins 强制映射到最后一个有效 bin 中心
        discretized_actions = np.clip(discretized_actions - 1, a_min=0, a_max=self.bin_centers.shape[0] - 1)

        # 查表得到对应的 bin 中心值，即近似的连续动作
        return self.bin_centers[discretized_actions]

    @property
    def vocab_size(self) -> int:
        """返回动作专用的“词汇表”大小，即 bin 的数量"""
        return self.n_bins

    @property
    def required_future_horizon(self) -> int:
        """当前动作 tokenizer 不需要预测未来的动作步，因此返回 0（某些高级版本可能需要）"""
        return 0


class VQActionTokenizer(ActionTokenizer):
    """Loads a torch model (VqVaE) that turns"""

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        vq_vae_path="",
        device="cpu",
        use_extra: bool = False,
    ):
        self.tokenizer = tokenizer
        self.device = device

        ### VQ VAE loading ###

        # NOTE: if this errors, you need to install vqvae, source: https://github.com/jayLEE0301/vq_bet_official
        from vqvae.vqvae import VqVae

        self.vq_path = Path(vq_vae_path)
        assert self.vq_path.exists(), f"Missing VQ VAE path: {self.vq_path}"
        vq_model_path = self.vq_path / "checkpoints" / "model.pt"
        vq_config_path = self.vq_path / "config.json"
        assert vq_model_path.exists(), f"Missing VQ checkpoint path: {vq_model_path}"
        assert vq_config_path.exists(), f"Missing VQ config path: {vq_config_path}"
        with open(vq_config_path, "r") as f:
            vq_config = dict(json.load(f))
        # set the load checkpoint
        vq_config["load_dir"] = vq_model_path
        vq_config["eval"] = True
        vq_config["device"] = self.device
        overwatch.info(f"Loading VQ VAE for Action Tokenization from {vq_config_path}...")
        # instantiate the vqvae and load
        self.vq_vae = VqVae(**vq_config)
        overwatch.info(f"Found VQ VAE parameters: \n{self.vq_vae}")
        ### TOKENIZATION arguments ###
        # number of bins to assign for each "action" dimension
        self.n_bins = self.vq_vae.vqvae_n_embed

        self.tokenizer_len = self.tokenizer.vocab_size
        if isinstance(tokenizer, Qwen2TokenizerFast) and use_extra:
            self.tokenizer_len = len(self.tokenizer)
        elif use_extra:
            raise NotImplementedError("Cannot use extra tokens for this tokenizer!")

        # [Contract] Set "action_token_begin_idx" based on `self.tokenizer.vocab_size - (self.n_bins + 1)`
        #   =>> Assumes we're always overwriting the final `n_bins` tokens of the vocabulary!
        self.action_token_begin_idx: int = int(self.tokenizer_len - (self.n_bins + 1))
        self.action_token_end_idx: int = int(self.tokenizer_len)

    def __call__(self, action: np.ndarray) -> Union[str, List[str]]:
        # make sure shape matches (1 x T x A)
        action = torch.from_numpy(action).to(self.device).reshape((1, self.vq_vae.input_dim_h, self.vq_vae.input_dim_w))
        # action is (1 x T x A), codes will be (1 x GROUPS) each between 0 and BINS-1
        _, vq_code = self.vq_vae.get_code(action)
        assert torch.all(vq_code >= 0) and torch.all(vq_code < self.n_bins)

        # vq_codes will be between [0, n_bins-1], so we subtract them from vocab_size - 1
        # for example, code 0 maps to vocab_size - 1
        return self.tokenizer.decode(list(self.tokenizer_len - 1 - vq_code[0].numpy()))

    def decode_token_ids_to_actions(self, action_token_ids: np.ndarray) -> np.ndarray:
        # first convert from tokens to bins (inverse of what happens in __call__)
        action_token_ids = self.tokenizer_len - 1 - action_token_ids
        initial_shape = action_token_ids.shape
        # these directly correspond to the bins
        action_token_ids = np.clip(action_token_ids, 0, self.n_bins - 1)
        action_token_ids = torch.from_numpy(action_token_ids).to(self.device).reshape(-1, self.vq_vae.vqvae_groups)
        assert torch.all(action_token_ids >= 0) and torch.all(action_token_ids < self.n_bins)
        # (1 x G) --> (1 x Z_DIM)
        latent = self.vq_vae.draw_code_forward(action_token_ids)
        # --> (1 x A) --> (A,)
        ret_action = self.vq_vae.get_action_from_latent(latent)

        # reshape to be a flat array if the input was a single action
        if action_token_ids.shape[0] == 1 and len(initial_shape) == 1:
            return ret_action[0, 0]

        # get the first horizon element of the returned actions (VQ might return an action horizon)
        # TODO parameterize this
        return ret_action[:, 0]

    @property
    def required_future_horizon(self) -> int:
        # the number of future action horizon elements
        return self.vq_vae.input_dim_h - 1


ACTION_TOKENIZERS = {
    "action_tokenizer": ActionTokenizer,
    "extra_action_tokenizer": partial(ActionTokenizer, use_extra=True),
    # libero
    "libero_vq_action_tokenizer": partial(
        VQActionTokenizer, vq_vae_path="vq/pretrain_vq+mx-libero_90+fach-7+ng-7+nemb-128+nlatent-512"
    ),
    "libero_vq_extra_action_tokenizer": partial(
        VQActionTokenizer, vq_vae_path="vq/pretrain_vq+mx-libero_90+fach-7+ng-7+nemb-128+nlatent-512", use_extra=True
    ),
    "libero_vq_h0_extra_action_tokenizer": partial(
        VQActionTokenizer, vq_vae_path="vq/pretrain_vq+mx-libero_90+fach-0+ng-7+nemb-128+nlatent-512", use_extra=True
    ),
    # bridge
    "bridge_vq_extra_action_tokenizer": partial(
        VQActionTokenizer,
        vq_vae_path="vq/pretrain_modvq+mx-bridge_dataset+fach-7+ng-7+nemb-256+nlatent-512",
        use_extra=True,
    ),
}
