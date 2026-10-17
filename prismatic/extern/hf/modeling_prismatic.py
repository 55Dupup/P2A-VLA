"""
modeling_prismatic.py

Core HuggingFace-style PrismaticPreTrainedModel and PrismaticForConditionalGeneration class definitions.
Inherits from the default `transformers.PretrainedModel`. Meant to be standalone and self-contained,
but exactly replicate the logic in `prismatic.models.vlms.prismatic.py`.
"""

import logging
from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, ClassVar, Dict, List, Optional, Tuple, Union
import numpy as np
import timm
import tokenizers
import torch
import torch.nn as nn
import transformers
from timm.models.vision_transformer import LayerScale
from transformers import AutoModelForCausalLM, PretrainedConfig, PreTrainedModel
from transformers.modeling_outputs import ModelOutput

from prismatic.training.train_utils import (
    get_current_action_mask,
    get_next_actions_mask,
)
from prismatic.vla.constants import (
    ACTION_DIM,
    ACTION_PROPRIO_NORMALIZATION_TYPE,
    ACTION_TOKEN_BEGIN_IDX,
    IGNORE_INDEX,
    NUM_ACTIONS_CHUNK,
    STOP_INDEX,
    NormalizationType,
    NUM_TOKENS
)
from .configuration_prismatic import OpenVLAConfig, PrismaticConfig



# Set up logger
logger = logging.getLogger(__name__)


# === Utility Functions for Monkey-Patching ===
def unpack_tuple(fn: Callable[[Any], Tuple[Any]]) -> Callable[[Any], Any]:
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        result = fn(*args, **kwargs)
        return result[0] if isinstance(result, tuple) else result

    return wrapper



# HF Transformers overwrites parameters with names containing `gamma`; we're going to patch VisionBackbone.LayerScale.
#   =>> TIMM :: https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/vision_transformer.py#L109
#   =>> Transformers :: https://github.com/huggingface/transformers/blob/main/src/transformers/modeling_utils.py#L3960
def _ls_new_forward(self, x: torch.Tensor) -> torch.Tensor:
    return x.mul_(self.scale_factor) if self.inplace else x * self.scale_factor



def ls_apply_patch(ls_module: LayerScale):
    ls_module.scale_factor = nn.Parameter(ls_module.gamma.clone())
    ls_module.forward = _ls_new_forward.__get__(ls_module, LayerScale)
    del ls_module.gamma



# === Prismatic Vision Backbone (nn.Module) Definitions (w/ Fused Backbone Support) ===
class PrismaticVisionBackbone(nn.Module):
    """
    Vision backbone for Prismatic models that handles image feature extraction.

    Supports both single backbone (e.g., SigLIP) and fused backbone (e.g., SigLIP + DINOv2) configurations.
    For fused backbones, features from both models are concatenated along the feature dimension.
    """

    def __init__(
        self,
        use_fused_vision_backbone: bool,
        image_sizes: List[int],
        timm_model_ids: List[str],
        timm_override_act_layers: List[Optional[str]],
    ) -> None:
        """
        Initialize the vision backbone.

        Args:
            use_fused_vision_backbone: Whether to use two backbones and fuse their features
            image_sizes: List of image sizes for each backbone
            timm_model_ids: List of TIMM model IDs to use for each backbone
            timm_override_act_layers: List of activation layer overrides for each backbone
        """
        super().__init__()
        self.use_fused_vision_backbone = use_fused_vision_backbone
        self.num_images_in_input = 1  # Default value, can be overridden later

        # Validate number of (fused) vision backbones
        if len(timm_model_ids) > 2:
            raise ValueError("Prismatic models only support up to 2 (fused) vision backbones!")

        # Create primary featurizer
        self.featurizer = self._create_featurizer(
            model_id=timm_model_ids[0], img_size=image_sizes[0], act_layer=timm_override_act_layers[0]
        )
        self.embed_dim = self.featurizer.embed_dim

        # Create secondary featurizer if using fused backbone
        if self.use_fused_vision_backbone:
            self.fused_featurizer = self._create_featurizer(
                model_id=timm_model_ids[1], img_size=image_sizes[1], act_layer=timm_override_act_layers[1]
            )
            self.embed_dim += self.fused_featurizer.embed_dim

        # Patch LayerScale modules for HF compatibility
        self._patch_layer_scales()


    def _create_featurizer(self, model_id: str, img_size: int, act_layer: Optional[str]) -> nn.Module:
        """
        Create a TIMM-based featurizer model with appropriate configurations.

        Args:
            model_id: The TIMM model ID to load
            img_size: Input image size for the model
            act_layer: Override for the activation layer type

        Returns:
            A configured featurizer model
        """
        featurizer = timm.create_model(
            model_id,
            pretrained=False,
            num_classes=0,
            img_size=img_size,
            act_layer=act_layer,
        )

        # Monkey-patch the forward function to extract the second-to-last layer features
        num_blocks = len(featurizer.blocks)
        featurizer.forward = unpack_tuple(partial(featurizer.get_intermediate_layers, n={num_blocks - 2}))

        return featurizer


    def _patch_layer_scales(self) -> None:
        """
        Patch all LayerScale modules to be compatible with HF's parameter naming.

        HF Transformers overwrites parameters with names containing 'gamma',
        so we need to rename and modify the forward method.
        """
        # Patch primary featurizer
        for module in self.featurizer.modules():
            if isinstance(module, LayerScale):
                ls_apply_patch(module)

        # Patch secondary featurizer if it exists
        if self.use_fused_vision_backbone:
            for module in self.fused_featurizer.modules():
                if isinstance(module, LayerScale):
                    ls_apply_patch(module)


    def get_num_patches(self) -> int:
        """
        Returns the number of vision patches output by the vision backbone.

        Returns:
            Number of patches per image
        """
        return self.featurizer.patch_embed.num_patches


    def get_num_images_in_input(self) -> int:
        """
        Returns the number of input images for the vision backbone.

        Returns:
            Number of images expected in the input
        """
        return self.num_images_in_input


    def set_num_images_in_input(self, num_images_in_input: int) -> None:
        """
        Sets the number of input images for the vision backbone.

        Args:
            num_images_in_input: Number of images to expect in the input
        """
        self.num_images_in_input = num_images_in_input


    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Implements the forward pass for the vision backbone.

        If `self.use_fused_vision_backbone == True`, uses both SigLIP and DINOv2 transformers to extract visual features
        (otherwise uses SigLIP only). Allows multi-image inputs (but only for fused vision backbone).

        Args:
            pixel_values (torch.Tensor): Pixels for input image(s), (B, C, H, W).
        """
        if self.num_images_in_input == 1:
            if not self.use_fused_vision_backbone:
                return self.featurizer(pixel_values)

            # Split `pixel_values :: [bsz, 2 * 3, resolution, resolution]` =>> featurize =>> channel stack
            img, img_fused = torch.split(pixel_values, [3, 3], dim=1)
            patches, patches_fused = self.featurizer(img), self.fused_featurizer(img_fused)

            return torch.cat([patches, patches_fused], dim=2)

        else:
            assert self.use_fused_vision_backbone, "Multi-image inputs require using fused backbone!"

            # Split `pixel_values` into individual images (each with 6 channels: 3 for SigLIP + 3 for DINOv2)
            images = torch.split(pixel_values, [6] * self.num_images_in_input, dim=1)

            # Process each image and collect patches
            all_patches = []
            for img in images:
                # Split each image further into two stacks of channels (each with 3 channels)
                img_regular, img_fused = torch.split(img, [3, 3], dim=1)

                # Get patches from both SigLIP and DINOv2 vision transformers
                patches = self.featurizer(img_regular)
                patches_fused = self.fused_featurizer(img_fused)

                # Concatenate SigLIP and DINOv2 patches along the hidden dimension
                combined_patches = torch.cat([patches, patches_fused], dim=2)
                all_patches.append(combined_patches)

            # Concatenate all patches along the patch dimension
            return torch.cat(all_patches, dim=1)



# === Prismatic Projector (nn.Module) Definitions ===
class PrismaticProjector(nn.Module):
    def __init__(self, use_fused_vision_backbone: bool, vision_dim: int, llm_dim: int) -> None:
        super().__init__()
        self.use_fused_vision_backbone = use_fused_vision_backbone
        self.vision_dim, self.llm_dim = vision_dim, llm_dim

        # Switch on `use_fused_vision_backbone` =>> use slightly different MLPs and projection factors!
        if not self.use_fused_vision_backbone:
            self.fc1 = nn.Linear(self.vision_dim, self.llm_dim, bias=True)
            self.fc2 = nn.Linear(self.llm_dim, self.llm_dim, bias=True)
            self.act_fn1 = nn.GELU()
        else:
            initial_projection_dim = 4 * vision_dim
            self.fc1 = nn.Linear(self.vision_dim, initial_projection_dim, bias=True)
            self.fc2 = nn.Linear(initial_projection_dim, self.llm_dim, bias=True)
            self.fc3 = nn.Linear(self.llm_dim, self.llm_dim, bias=True)
            self.act_fn1 = nn.GELU()
            self.act_fn2 = nn.GELU()

    def forward(self, img_patches: torch.Tensor) -> torch.Tensor:
        if not self.use_fused_vision_backbone:
            projected_features = self.fc1(img_patches)
            projected_features = self.act_fn1(projected_features)
            projected_features = self.fc2(projected_features)
        else:
            projected_features = self.fc1(img_patches)
            projected_features = self.act_fn1(projected_features)
            projected_features = self.fc2(projected_features)
            projected_features = self.act_fn2(projected_features)
            projected_features = self.fc3(projected_features)

        return projected_features



# === Main HF Class Definitions ===
@dataclass
class PrismaticCausalLMOutputWithPast(ModelOutput):
    """Base class for Prismatic casual (visually-conditioned) language model outputs; also exposes visual features."""

    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None

    # Additions for VLMs
    projector_features: Optional[torch.FloatTensor] = None

    language_embeddings: Optional[torch.FloatTensor] = None
    layered_visual_embeddings: Optional[torch.FloatTensor] = None      # 新增：多层 visual list
    layered_text_embeddings: Optional[torch.FloatTensor] = None          # 新增：多层 text list
    layered_action_embeddings: Optional[torch.FloatTensor] = None      # 新增：多层 action list 



class PrismaticPreTrainedModel(PreTrainedModel):
    config_class: PretrainedConfig = PrismaticConfig
    base_model_prefix: str = "model"
    supports_gradient_checkpointing: bool = True

    _no_split_modules: ClassVar[List[str]] = ["PrismaticProjector"]
    _skip_keys_device_placement: str = "past_key_values"
    _supports_flash_attn_2: bool = True

    def _init_weights(self, module: nn.Module) -> None:
        # Important :: this HF ported version is *not* meant for training from scratch; only inference and fine-tuning!
        #   => As such, this init_weights code is not correct; if training VLMs from scratch, use the main codebase at
        #      https://github.com/TRI-ML/prismatic-vlms
        std = (
            self.config.initializer_range
            if hasattr(self.config, "initializer_range")
            else self.config.text_config.initializer_range
        )

        if hasattr(module, "class_embedding"):
            module.class_embedding.data.normal_(mean=0.0, std=std)

        if isinstance(module, (nn.Linear, nn.Conv2d)):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    @property
    def _supports_sdpa(self) -> bool:
        """Check LLM supports SDPA Attention"""
        return self.language_model._supports_sdpa



class PrismaticForConditionalGeneration(PrismaticPreTrainedModel):
    """
    OPENVLA通用 VLA 模型核心，继承自 Hugging Face 的 PreTrainedModel，负责所有标准的训练逻辑（loss 计算、gradient checkpointing、generation with cache 等）。
    负责模型的通用前向、损失计算和自回归生成。
    """
    def __init__(self, config: PrismaticConfig) -> None:
        super().__init__(config)

        # [Validation] Lightweight Validate on `config` Fields + Dependency Versions
        if config.use_fused_vision_backbone is None:
            raise ValueError("Missing config field `use_fused_vision_backbone`")

        if timm.__version__ not in {"0.9.10", "0.9.11", "0.9.12", "0.9.16"}:
            raise NotImplementedError(
                "TIMM Version must be >= 0.9.10 and < 1.0.0 (breaking); please raise a GitHub Issue "
                "if you urgently need support for latest TIMM versions."
            )

        if (transformers.__version__ != "4.40.1") or (tokenizers.__version__ != "0.19.1"):
            logger.warning(
                f"Expected `transformers==4.40.1` and `tokenizers==0.19.1` but got "
                f"`transformers=={transformers.__version__}` and `tokenizers=={tokenizers.__version__}`; "
                f"there might be inference-time regressions due to dependency changes. If in doubt, please"
                f"use the above versions."
            )
        
        # Instantiate PrismaticVisionBackbone (w/ Potential Fused Backbone)
        self.vision_backbone = PrismaticVisionBackbone( #视觉编码器（基于 DINOv2 + SigLIP 的融合 ViT），提取图像 patch 特征
            config.use_fused_vision_backbone, config.image_sizes, config.timm_model_ids, config.timm_override_act_layers
        )

        # Create Multimodal Projector
        self.projector = PrismaticProjector( #一个小型 MLP，将视觉特征投影到 LLM 的嵌入空间
            config.use_fused_vision_backbone,
            vision_dim=self.vision_backbone.embed_dim,
            llm_dim=config.text_config.hidden_size,
        )

        # Instantiate LLM Backbone
        self.language_model = AutoModelForCausalLM.from_config( #LLM 主干
            config.text_config, attn_implementation=config._attn_implementation
        )

        self.vocab_size = config.text_config.vocab_size
        self.pad_token_id = config.pad_token_id
        self.llm_dim = config.text_config.hidden_size
        
        #Action query token
        self.action_queries = nn.Embedding(NUM_TOKENS, self.llm_dim)# 一个可学习的嵌入（nn.Embedding(NUM_TOKENS, llm_dim)），初始化为零，用于代表动作 token 的查询（query）。
        self.action_queries.weight.data.zero_()

        # HF Boilerplate =>> initializes weights via `_init_weights()` and sets gradient checkpointing
        self.post_init()

    # === `PreTrainedModel` Boilerplate ===
    def get_input_embeddings(self) -> nn.Module:
        return self.language_model.get_input_embeddings()
    def set_version(self, version: str):
        self.version = version
        return self.version


    def set_input_embeddings(self, value: nn.Module) -> None:
        self.language_model.set_input_embeddings(value)

    def get_output_embeddings(self) -> nn.Module:
        return self.language_model.get_output_embeddings()

    def set_output_embeddings(self, new_embeddings: nn.Module) -> None:
        self.language_model.set_output_embeddings(new_embeddings)

    def get_decoder(self) -> nn.Module:
        return self.language_model.get_decoder()

    def set_decoder(self, decoder: nn.Module) -> None:
        self.language_model.set_decoder(decoder)

    def tie_weights(self) -> None:
        self.language_model.tie_weights()  # Note: `Llama-2` and `Mistral` don't tie weights (no-op)

    def resize_token_embeddings(
        self, new_num_tokens: Optional[int] = None, pad_to_multiple_of: Optional[int] = None
    ) -> nn.Embedding:
        updated_embeddings = self.language_model.resize_token_embeddings(new_num_tokens, pad_to_multiple_of)

        # Update config/instance variables
        self.config.text_config.vocab_size = updated_embeddings.num_embeddings
        self.vocab_size = updated_embeddings.num_embeddings

        return updated_embeddings

    def _replace_input_embeddings(self, input_embeddings, all_actions_mask, noisy_action_features):
        """
        input_embeddings 中动作 token 对应的位置用 noisy_action_features 覆盖
        Replace embeddings in input_embeddings at positions where all_actions_mask is True
        with embeddings from noisy_action_features, using vectorized operations.

        Args:
            input_embeddings: Tensor of shape (B, S, D)
            all_actions_mask: Boolean tensor of shape (B, S)
            noisy_action_features: Tensor of shape (B, K, D) where K is the number of True values in mask per sample

        Returns:
            Modified input_embeddings tensor
        """
        # Clone input to avoid modifying the original tensor
        new_input_embeddings = input_embeddings.clone()

        # Create a tensor with the same shape of input_embeddings to hold the noisy action features
        repositioned_noisy_action_features = torch.zeros_like(input_embeddings)

        # Create batch indices for splicing
        batch_indices = torch.arange(input_embeddings.shape[0], device=input_embeddings.device)
        batch_indices = batch_indices.unsqueeze(1).expand(-1, noisy_action_features.shape[1])

        # Get indices where mask is True for each sample
        masked_indices = torch.stack([torch.where(mask)[0] for mask in all_actions_mask])

        # Move the noisy action features into their correct positions
        # print(noisy_action_features.size())
        
        repositioned_noisy_action_features[batch_indices, masked_indices] = noisy_action_features

        # Combine original input embeddings and noisy action embeddings using the mask
        new_input_embeddings = torch.where(
            all_actions_mask.unsqueeze(-1), repositioned_noisy_action_features, new_input_embeddings
        )

        return new_input_embeddings

    def _process_action_masks(self, labels): #生成动作掩码
        """Helper to get action masks from labels"""
        current_action_mask = get_current_action_mask(labels)
        next_actions_mask = get_next_actions_mask(labels)
        all_actions_mask = current_action_mask | next_actions_mask  # (B, seq_len)
        return all_actions_mask

    def _process_vision_features(self, pixel_values, language_embeddings=None, use_film=False):
        """Process vision features with optional FiLM conditioning"""
        if use_film:
            # FiLM: Infuse language inputs into visual features
            patch_features = self.vision_backbone(pixel_values, language_embeddings)  # (bsz, 256 * num_images, D)
        else:
            patch_features = self.vision_backbone(pixel_values)  # (bsz, 256 * num_images, D)

        # Project patch embeddings into language embedding space
        return self.projector(patch_features)

    def _process_proprio_features(self, projected_patch_embeddings, proprio, proprio_projector):
        """Process proprioceptive features and append to vision features"""
        if proprio_projector is not None and proprio is not None:
            # projected_patch_embeddings: (bsz, num_patches * num_images, llm_dim)
            # proprio: (bsz, proprio_dim) or (propro_dim,)
            proprio = proprio.reshape(projected_patch_embeddings.shape[0], -1)  # (bsz, proprio_dim)
            proprio_features = proprio_projector(proprio)  # (bsz, llm_dim)
            proprio_features = proprio_features.unsqueeze(dim=1)  # (bsz, 1, llm_dim)
            # For simplicity, just append proprio token to the end of projected vision patch tokens
            return torch.cat((projected_patch_embeddings, proprio_features), dim=1)
        return projected_patch_embeddings

    def _build_multimodal_attention(self, input_embeddings, projected_patch_embeddings, attention_mask):
        """Build multimodal embeddings and attention mask"""
        # Update attention mask
        
        projected_patch_attention_mask = None
        if attention_mask is not None:
            projected_patch_attention_mask = torch.full(
                (projected_patch_embeddings.shape[0], projected_patch_embeddings.shape[1]),
                fill_value=True,
                dtype=attention_mask.dtype,
                device=attention_mask.device,
            )

        # Build multimodal embeddings & attention mask; insert embeddings after <BOS> token (1:)
        # 插入视觉 patch 嵌入到文本序列的 BOS token 之后 顺序：BOS + 视觉 patches + 剩下的input_embeddings
        # [BOS, vision patches..., prompt token, action token, eos]
        multimodal_embeddings = torch.cat(
            [input_embeddings[:, :1, :], projected_patch_embeddings, input_embeddings[:, 1:, :]], dim=1
        )

        multimodal_attention_mask = None
        if attention_mask is not None:
            multimodal_attention_mask = torch.cat(
                [attention_mask[:, :1], projected_patch_attention_mask, attention_mask[:, 1:]], dim=1
            )

        return multimodal_embeddings, multimodal_attention_mask

    def _build_multimodal_labels(self, labels, projected_patch_embeddings):
        """Build multimodal labels with IGNORE_INDEX for patch embeddings"""
        if labels is not None:
            projected_patch_labels = torch.full(
                (projected_patch_embeddings.shape[0], projected_patch_embeddings.shape[1]),
                fill_value=IGNORE_INDEX,
                dtype=labels.dtype,
                device=labels.device,
            )
            return torch.cat([labels[:, :1], projected_patch_labels, labels[:, 1:]], dim=1)
        return None

    # === Core Prismatic VLM `forward()` Logic ===
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_projector_features: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        proprio=None,
        proprio_projector=None,
        noisy_actions=None,
        noisy_action_projector=None,
        diffusion_timestep_embeddings=None,
        use_film: bool = False,
    ) -> Union[Tuple, PrismaticCausalLMOutputWithPast]:
        """Run a forward pass through the VLM, returning a PrismaticCausalLMOutputWithPast instance."""
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        output_projector_features = output_projector_features if output_projector_features is not None else False
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Respect `use_cache` only if not training (even if `gradient_checkpointing` is off)
        use_cache = use_cache and not self.training

        # Instantiate Placeholder for Projector Features
        projected_patch_embeddings = None

        # === Handle Generation with Cache (`input_ids.shape[1] == 1`) =>> requires `past_keys_values` ===
        if input_ids.shape[1] == 1: #即单步生成，只输入下一个 token
            assert input_ids.shape[0] == 1, "Generation is only currently supported for batch size of 1!"
            assert past_key_values is not None, "You must provide `past_key_values` during cached generation!"
            assert labels is None, "Unexpected key `labels` provided during cached generation!"

            language_model_output = self.language_model(
                input_ids=input_ids,
                attention_mask=None,
                position_ids=None,
                past_key_values=past_key_values,
                inputs_embeds=None,
                labels=None,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

        # === Handle Unimodal Forward ===
        elif pixel_values is None: #纯语言模式（Unimodal / Language-only）
            assert (input_ids is not None) and (inputs_embeds is None), "Missing `input_ids` in language-only forward!"
            assert past_key_values is None, "Unexpected key `past_key_values` provided during language-only forward!"

            language_model_output = self.language_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=None,
                past_key_values=None,
                inputs_embeds=None,
                labels=labels,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

        # === Handle Multimodal Forward ===
        elif (input_ids.shape[0] == pixel_values.shape[0]) or (inputs_embeds.shape[0] == pixel_values.shape[0]):
            assert past_key_values is None, "Unexpected key `past_key_values` provided during multimodal forward!"

            #Step1 从 LLM 的词嵌入层得到文本 token 的嵌入 (B, seq_len, D)
            input_embeddings = self.get_input_embeddings()(input_ids)

            
            #Step2 从 labels 中提取动作位置掩码 all_actions_mask（当前动作 + 下一个动作的位置）。
            #在这些位置，用可学习的 action_queries 替换原始文本嵌入（通过 _replace_input_embeddings）
            all_actions_mask = self._process_action_masks(labels)

            # Extract the language portion of the input embeddings (i.e. remove the action tokens portion)
            
            # print(input_embeddings[~all_actions_mask].size())
            # TODO 从inputs_embeds中提取语言 token 的嵌入（即非动作 token 的嵌入）
            language_embeddings = input_embeddings[~all_actions_mask].reshape(
                input_embeddings.shape[0], -1, input_embeddings.shape[2]
            )  # (B, lang_seq_len, llm_dim)

            # Get visual features 处理视觉特征 用语言嵌入调制视觉特征（use_film=True 时
            projected_patch_embeddings = self._process_vision_features(pixel_values, language_embeddings, use_film)

            # Process action embeddings 将input_embeddings中对应的action token的位置的embedding替换为 noisy action features（如果提供了 noisy_actions）
            if noisy_actions is not None:
                action_queries = self.action_queries.weight  # (1, h)
                action_queries = action_queries.view(1, action_queries.shape[0], action_queries.shape[1]).repeat(input_embeddings.shape[0], 1, 1)  # (b, chunk_size, h)
                all_actions_mask = self._process_action_masks(labels)
                input_embeddings = self._replace_input_embeddings(
                    input_embeddings, all_actions_mask, action_queries)
            else:
                action_queries = self.action_queries.weight  # (1, h)
                action_queries = action_queries.view(1, action_queries.shape[0], action_queries.shape[1]).repeat(input_embeddings.shape[0], 1, 1)  # (b, chunk_size, h)
                all_actions_mask = self._process_action_masks(labels)
                input_embeddings = self._replace_input_embeddings(
                    input_embeddings, all_actions_mask, action_queries)

            # Build multimodal embeddings & attention mask
            #将投影后的视觉 patch 嵌入 插入到文本序列的 BOS token 之后（即：[BOS] + [视觉 patch tokens] + [剩余文本 tokens + action queries]）。
            #相应扩展 attention_mask（视觉部分全可见）。
            multimodal_embeddings, multimodal_attention_mask = self._build_multimodal_attention(
                input_embeddings, projected_patch_embeddings, attention_mask
            )
            
            # Build labels for multimodal sequence if needed 
            # 视觉 patch 位置的 labels 设为 IGNORE_INDEX（不计算损失，只计算文本/动作部分损失）。
            multimodal_labels = self._build_multimodal_labels(labels, projected_patch_embeddings)
            # print(f"multimodal_labels:{multimodal_labels}")
            # Dispatch to language model
            # 用构建好的 multimodal_embeddings 和 multimodal_attention_mask 调用 language_model。
            language_model_output = self.language_model( #prismatic
                input_ids=None,
                attention_mask=multimodal_attention_mask,
                position_ids=None,
                past_key_values=None,
                inputs_embeds=multimodal_embeddings,
                labels=None,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                ) 
            
            # # TODO DEBUG 拆分LLM输出的visual embedding/text embedding/action_embeddings
            # # begin 计算分割索引
            # B = multimodal_embeddings.shape[0]
            # bos_start = 0
            # visual_start = 1
            # visual_end = visual_start + projected_patch_embeddings.shape[1]  # T_v
            # text_action_start = visual_end
            # text_action_end = multimodal_embeddings.shape[1]  # T_t + T_a

            # # 调整 mask 为 text_action 部分的子集（原始 all_actions_mask 去掉 BOS）
            # adjusted_actions_mask = all_actions_mask[:, 1:]  # (B, T_t + T_a)

            # # 分割多层 hidden_states
            # layered_visual = []
            # layered_text = []
            # layered_action = []
            # for layer_hidden in language_model_output.hidden_states:  # 循环每个层 (B, total_seq_len, D)
            #     # Visual 部分
            #     processed_visual = layer_hidden[:, visual_start:visual_end, :]  # (B, T_v, D)

            #     # Text + Action 混合部分
            #     processed_text_action = layer_hidden[:, text_action_start:text_action_end, :]  # (B, T_t + T_a, D)

            #     # 纯 Text 部分（过滤动作位置）
            #     processed_text = processed_text_action[~adjusted_actions_mask].reshape(B, -1, layer_hidden.shape[-1])  # (B, T_t, D)

            #     # 纯 Action 部分
            #     processed_action = processed_text_action[adjusted_actions_mask].reshape(B, -1, layer_hidden.shape[-1])  # (B, T_a, D)

            #     layered_visual.append(processed_visual)
            #     layered_text.append(processed_text)
            #     layered_action.append(processed_action)

            # # end

        # === Otherwise =>> Assume Invalid! ===
        elif (input_ids.shape[0] != pixel_values.shape[0]) or (inputs_embeds.shape[0] != pixel_values.shape[0]):
            raise ValueError("Non-homogenous batch of (text, image) input -- forward() does not support mixed batches!")

        else:
            raise ValueError(
                "Invalid PrismaticForConditionalGeneration `forward()` call with provided arguments:\n"
                f"=> `input_ids` = {input_ids is not None}\n"
                f"=> `attention_mask` = {attention_mask is not None}\n"
                f"=> `pixel_values` = {pixel_values is not None}\n"
                f"=> `labels` = {labels is not None}\n"
                f"=> `input_embeds` = {inputs_embeds is not None}\n"
                f"=> `past_key_values` = {past_key_values is not None}\n"
                f"=> `use_cache` = {use_cache}"
            )

        # Unpack `language_model_output` and return PrismaticCausalLMOutputWithPast (or tuple if not `return_dict`)
        if not return_dict:
            if output_projector_features and (projected_patch_embeddings is not None):
                return *language_model_output, projected_patch_embeddings

            return language_model_output

        return PrismaticCausalLMOutputWithPast(
            loss=language_model_output.loss,
            past_key_values=language_model_output.past_key_values,
            hidden_states=language_model_output.hidden_states,
            attentions=language_model_output.attentions,
            projector_features=projected_patch_embeddings, # vision backbone 输出的视觉特征
            # language_embeddings=language_embeddings,      # 新增：原始输入的语言嵌入
            # layered_visual_embeddings=layered_visual,      # 新增：多层 visual list
            # layered_text_embeddings=layered_text,          # 新增：多层 text list
            # layered_action_embeddings=layered_action,      # 新增：多层 action list
            )

    # === GenerationMixin Methods ===
    def prepare_inputs_for_generation(
        self,
        input_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs: str,
    ) -> Dict[str, torch.Tensor]:
        """Borrowed from `LlamaForCausalLM` and simplified for batch size = 1; mirrors original PrismaticVLM logic."""
        if ((input_ids is not None) and (input_ids.shape[0] > 1)) or (
            (inputs_embeds is not None) and (inputs_embeds.shape[0] > 1)
        ):
            raise ValueError("Generation with batch size > 1 is not currently supported!")

        # Handle `past_key_values` (cache) =>> assume `input_ids` just has unprocessed tokens
        if past_key_values is not None:
            input_ids = input_ids[:, -1:]

        # If `input_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"input_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        # Make sure `pixel_values` are preserved in `model_inputs`
        model_inputs.update(
            {
                "attention_mask": attention_mask,
                "pixel_values": pixel_values,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
            }
        )

        return model_inputs

    # Defer to Language Model (all handle this differently, with different return types)
    def _reorder_cache(self, *args, **kwargs) -> Any:
        return self.language_model._reorder_cache(*args, **kwargs)



class OpenVLAForActionPrediction(PrismaticForConditionalGeneration):
    """
    OPENVLA 动作预测模型核心，为机器人策略（policy）推理设计，隐藏了复杂的序列构造细节。
    训练时：通常使用父类 PrismaticForConditionalGeneration 的 forward() 进行自回归训练（预测离散 action tokens）。
    推理时：使用这个类（或其实例）的 predict_action() 方法，直接输入图像 + 文本指令，输出连续的、去归一化的动作值（形状如 (num_actions_chunk, action_dim)，默认 8×7）。
    它不进行自回归逐步生成，而是一次性预测一个动作 chunk（通常 8 个动作），这在实际机器人控制中更高效。
    """
    config_class: PretrainedConfig = OpenVLAConfig

    def __init__(self, config: OpenVLAConfig) -> None:
        super().__init__(config)
        self.norm_stats = config.norm_stats
        

        # Compute action bins
        self.bins = np.linspace(-1, 1, config.n_action_bins)
        self.bin_centers = (self.bins[:-1] + self.bins[1:]) / 2.0

        # Compute vocab size for de-tokenization -- revert added "multiple of"
        self.vocab_size = self.config.text_config.vocab_size - self.config.pad_to_multiple_of

    def _prepare_input_for_action_prediction(self, input_ids, attention_mask):
        """Prepares input for action prediction by adding necessary tokens
            在原始 prompt 后追加 64 个 placeholder action token（NUM_TOKENS = 通常 64 = 8 actions × 8 bins per dim? 实际是 ACTION_DIM × NUM_ACTIONS_CHUNK）
            再追加一个 STOP token
            扩展 attention_mask（新位置全可见）
        """
        # 添加action tokens 占位符
        placeholder_action_token_ids = ( 
            torch.ones((input_ids.shape[0], NUM_TOKENS)).to(input_ids.device).to(input_ids.dtype)
        )
        # 跟原始的文本token拼接构成action_mask
        input_ids = torch.cat([input_ids, placeholder_action_token_ids], dim=-1)

        # 添加 stop token到序列末尾，保持跟训练时一致
        stop_token_id = torch.ones((input_ids.shape[0], 1)).to(input_ids.device).to(input_ids.dtype) * STOP_INDEX
        input_ids = torch.cat([input_ids, stop_token_id], dim=-1) # [TEXT,ACTION,STOP] 48+64+1

        # Extend the attention mask to fit the new shape of input
        # Note: Only batch size == 1 supported right now
        mask_extension = ( # 扩展 attention mask 以适应新的输入形状
            torch.ones((attention_mask.shape[0], input_ids.shape[-1] - attention_mask.shape[-1]))
            .to(attention_mask.device)
            .to(attention_mask.dtype)
        )
        attention_mask = torch.cat([attention_mask, mask_extension], dim=-1)

        return input_ids, attention_mask

    def _prepare_labels_for_action_prediction(self, labels, input_ids):
        """Creates labels tensor for action prediction if not provided"""
        # Extend labels tensor with fake action labels
        ARBITRARY_ACTION_TOKEN_IDX = ACTION_TOKEN_BEGIN_IDX + 1
        labels_extension = (
            torch.ones((labels.shape[0], input_ids.shape[-1] - labels.shape[-1])).to(labels.device).to(labels.dtype)
            * ARBITRARY_ACTION_TOKEN_IDX
        )
        labels = torch.cat([labels, labels_extension], dim=-1)

        # Replace last label token with stop token
        # 同样将最后一个标签位置设为 STOP token
        labels[:, -1] = STOP_INDEX

        return labels

    def _unnormalize_actions(self, normalized_actions, unnorm_key=None):
        """Unnormalize actions using dataset statistics"""
        action_norm_stats = self.get_action_stats(unnorm_key)

        if ACTION_PROPRIO_NORMALIZATION_TYPE == NormalizationType.BOUNDS:
            mask = action_norm_stats.get("mask", np.ones_like(action_norm_stats["min"], dtype=bool))
            action_high, action_low = np.array(action_norm_stats["max"]), np.array(action_norm_stats["min"])
        elif ACTION_PROPRIO_NORMALIZATION_TYPE == NormalizationType.BOUNDS_Q99:
            mask = action_norm_stats.get("mask", np.ones_like(action_norm_stats["q01"], dtype=bool))
            action_high, action_low = np.array(action_norm_stats["q99"]), np.array(action_norm_stats["q01"])
        else:
            raise ValueError("Unsupported action/proprio normalization type detected!")

        actions = np.where(
            mask,
            0.5 * (normalized_actions + 1) * (action_high - action_low + 1e-8) + action_low,
            normalized_actions,
        )

        return actions


    def _regression_or_discrete_prediction(
        self,
        input_embeddings, #[原始input_ids=[bos,prompt], action tokens, eos]
        all_actions_mask,
        projected_patch_embeddings,
        attention_mask,
        labels,
        NUM_PATCHES,
        NUM_PROMPT_TOKENS,
        action_head=None,
        proprio=None,
        proprio_projector=None,
    ):
        """Run L1 regression-based continuous action prediction or discrete action tokens prediction."""

        action_queries = self.action_queries.weight  # (1, h)
        action_queries = action_queries.view(1, action_queries.shape[0], action_queries.shape[1]).repeat(input_embeddings.shape[0], 1, 1)  # (b, chunk_size, h)
        # Replace action token embeddings with noisy action embeddings
        input_embeddings = self._replace_input_embeddings(input_embeddings.clone(), all_actions_mask, action_queries)

        # Build multimodal embeddings & attention mask
        # TODO note: multimodal_embeddings:[Bos, vision patches..., prompt token, action token, eos]
        multimodal_embeddings, multimodal_attention_mask = self._build_multimodal_attention(
            input_embeddings, projected_patch_embeddings, attention_mask
        )

        # Forward pass through language model
        language_model_output = self.language_model(
            input_ids=None,
            attention_mask=multimodal_attention_mask,
            position_ids=None,
            past_key_values=None,
            inputs_embeds=multimodal_embeddings,
            labels=None,
            use_cache=None,
            output_attentions=False,
            output_hidden_states=True,
            return_dict=True,
        )

# 初始化用于存储各部分 Hidden States 的列表
        multi_layer_hidden_states = [] # 存放 Visual + Action 的混合特征 (VLA-Adapter 核心输入)
        multi_layer_visual_states = [] # 存放纯 Visual 特征
        multi_layer_prompt_states = [] # 存放纯 Text Prompt 特征
        multi_layer_action_states = [] # 存放纯 Action 特征

        # [逻辑] 假设输入序列的 Layout 结构如下：
        # Index:  [0]   | [1 ... 512] | [513 ... 560]  | [561 ... 624]
        # Token:  BOS   | Visual(512) | Prompt(48)     | Action(64)
        
        # 遍历每一层 hidden_states
        # language_model_output.hidden_states 是一个 tuple，包含每一层的输出 (B, Seq_Len, D)
        for item in language_model_output.hidden_states:
            
            # [修改 26-1-9] 动态获取 batch_size
            # 原始代码硬编码为 1，导致无法进行 Batch 推理或验证。
            # 修改后：item.shape[0] 自动适配 batch_size=1 或 batch_size>1。
            batch_size = item.shape[0]
            hidden_dim = item.shape[-1]

            # === 1. 定义切片索引 (增强可读性) ===
            # 相比直接写数字，定义 start/end 变量更不容易出错
            vis_start = 1 # 跳过 BOS token (Index 0)
            vis_end   = vis_start + NUM_PATCHES # 1 + 512
            
            prompt_start = vis_end
            prompt_end   = prompt_start + NUM_PROMPT_TOKENS # 513 + 48
            
            act_start = prompt_end
            act_end   = act_start + NUM_TOKENS # 561 + 64

            # === 2. 提取并 Reshape (核心修改) ===
            # [修改] 使用 batch_size 变量，而非硬编码 '1'
            # [逻辑] 确保 .to(torch.bfloat16) 类型匹配
            
            # Extract Action: (B, 1, 64, D)
            actions_hidden_states = item[:, act_start:act_end, :].reshape(batch_size, 1, NUM_TOKENS, -1).to(torch.bfloat16)
            
            # Extract Prompt: (B, 1, 48, D)
            prompt_hidden_states = item[:, prompt_start:prompt_end, :].reshape(batch_size, 1, NUM_PROMPT_TOKENS, -1).to(torch.bfloat16)
            
            # Extract Visual: (B, 1, 512, D)
            # [逻辑] 原始代码从 0 开始取是错的，因为 0 是 BOS，这里修正为从 1 开始
            visual_hidden_states = item[:, vis_start:vis_end, :].reshape(batch_size, 1, NUM_PATCHES, -1).to(torch.bfloat16)

            # === 3. 组合混合特征 ===
            # [注意] 这里将 Visual 和 Action 拼接，丢弃了 Prompt。
            # 这是为了符合下游 Action Head 的特定输入需求。
            # Shape: (B, 1, 512+64, D)
            mixed_states = torch.cat((visual_hidden_states, actions_hidden_states), dim=2)

            # === 4. 存入列表 ===
            multi_layer_hidden_states.append(mixed_states)
            multi_layer_visual_states.append(visual_hidden_states)
            multi_layer_action_states.append(actions_hidden_states)
            multi_layer_prompt_states.append(prompt_hidden_states)

        # === 5. 堆叠层维度 ===
        # 将 List[Tensor] 堆叠为 Tensor
        # 结果 Shape: [Batch, Layers(25), Seq_Len, Dim]
        multi_layer_hidden_states = torch.cat(multi_layer_hidden_states, dim=1) 
        multi_layer_prompt_states = torch.cat(multi_layer_prompt_states, dim=1) 
        multi_layer_visual_states = torch.cat(multi_layer_visual_states, dim=1) 
        multi_layer_action_states = torch.cat(multi_layer_action_states, dim=1) 

        # 获取最后一层的输出，用于某些特定的投影层输入
        last_hidden_states = language_model_output.hidden_states[-1]

        # === 6. 动作预测 ===
        if action_head is not None:
            # L1 regression prediction
            # [修改] 参数传递修正
            # 之前的代码可能错误地将 'mixed_states' 传给了 'multi_layer_action_states'。
            # 这里显式地传入拆分好的纯净特征。
            normalized_actions = action_head.predict_action(
                multi_layer_hidden_states, # 主输入：Visual + Action 混合特征
                proprio=proprio,
                proprio_projector=proprio_projector,
                latent_outputs=last_hidden_states,          # VLM 最后一层
                multi_layer_prompt_states=multi_layer_prompt_states, # 纯文本层
                multi_layer_visual_states=multi_layer_visual_states, # 纯视觉层
                multi_layer_action_states=multi_layer_action_states  # [关键] 纯动作层 (不再传混合特征)
            )
            normalized_actions = normalized_actions.reshape(NUM_ACTIONS_CHUNK, ACTION_DIM)
            normalized_actions = normalized_actions.float().cpu().detach().numpy()
        else:
            # Discrete token-based prediction
            predicted_action_token_ids = (
                language_model_output.logits[
                    :,
                    NUM_PATCHES + NUM_PROMPT_TOKENS : NUM_PATCHES + NUM_PROMPT_TOKENS + ACTION_DIM * NUM_ACTIONS_CHUNK,
                ]
                .argmax(dim=2)
                .cpu()
                .numpy()
            )
            discretized_actions = self.vocab_size - predicted_action_token_ids
            discretized_actions = np.clip(discretized_actions - 1, a_min=0, a_max=self.bin_centers.shape[0] - 1)
            normalized_actions = self.bin_centers[discretized_actions]
            normalized_actions = normalized_actions.reshape(NUM_ACTIONS_CHUNK, ACTION_DIM)

        return normalized_actions, actions_hidden_states


    def predict_action(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        unnorm_key: Optional[str] = None,
        proprio=None,
        proprio_projector=None,
        action_head=None, #可选，用于连续预测
        noisy_action_projector=None,
        use_film: bool = False,
        **kwargs: str,
    ) -> np.ndarray:
        """Predict actions from input sequence, with options for different prediction methods.

        Args:
            input_ids: Input token ids
            unnorm_key: Key for unnormalization statistics
            proprio: Proprioceptive features
            proprio_projector: Projector for proprioceptive features
            action_head: Optional head for L1 regression or diffusion-based prediction
            noisy_action_projector: Projector for noisy actions in diffusion-based prediction
            use_film: Whether to use FiLM conditioning
            **kwargs: Additional arguments including pixel_values and attention_mask

        Returns:
            Tuple of (unnormalized_actions, action_hidden_states)
        """

        pixel_values = kwargs["pixel_values"] # [1, 12, 224, 224]
        attention_mask = kwargs["attention_mask"] #[1, 48]

        # Create fake labels tensor (needed for action mask)
        # 真实推理时没有 labels，但需要 all_actions_mask 来知道动作位置在哪里替换为 action_queries
        # 所以这里先把input task prompt的长度确定，尾巴再加上动作 token 的长度
        labels = input_ids.clone()#[1, 48]
        labels[:] = IGNORE_INDEX # 将输入的text的token标签全部设为 IGNORE_INDEX

        # FIXME Get number of tokens in prompt (excluding the start token)
        NUM_PROMPT_TOKENS = input_ids.shape[-1] - 1  # Subtract action tokens and stop token

        # Prepare inputs by adding action tokens and stop token
        # [原始input_ids=[bos,prompt], placeholder action tokens, stop token]
        input_ids, attention_mask = self._prepare_input_for_action_prediction(input_ids, attention_mask) #[1, 113, 896] 48+64+1?

        # 添加 fake labels for action prediction；末尾加了一个stop token
        labels = self._prepare_labels_for_action_prediction(labels, input_ids) #[1,113]

        # 查表，Token转化到 Embedding :[[bos,prompt], placeholder action tokens, stop token]
        input_embeddings = self.get_input_embeddings()(input_ids) #[1, 113, 896]
        # 提取动作掩码的位置
        all_actions_mask = self._process_action_masks(labels)

        # TODO 提取action mask=false的位置 Extract language embeddings [1, 49, 896] [bos + prompt +eos]?
        language_embeddings = input_embeddings[~all_actions_mask].reshape(
            input_embeddings.shape[0], -1, input_embeddings.shape[2]
        )

        # Process vision features [1, 512, 896]
        projected_patch_embeddings = self._process_vision_features(pixel_values, language_embeddings, use_film)

        # Add proprioceptive features if provided
        use_proprio = proprio_projector is not None and proprio is not None
        if use_proprio:
            proprio = torch.Tensor(proprio).to(projected_patch_embeddings.device, dtype=projected_patch_embeddings.dtype) #[8]

        # Calculate number of patches (including proprio token and/or diffusion timestep embedding if present)
        NUM_PATCHES = self.vision_backbone.get_num_patches() * self.vision_backbone.get_num_images_in_input() #NUM_PATCHES

        # Run regression or discrete token-based prediction
        normalized_actions, actions_hidden_states = self._regression_or_discrete_prediction(
            input_embeddings,
            all_actions_mask,
            projected_patch_embeddings,
            attention_mask,
            labels,
            NUM_PATCHES,
            NUM_PROMPT_TOKENS,
            action_head=action_head,
            proprio=proprio, # [8]
            proprio_projector=proprio_projector,
            )#normalized_actions.shape (8, 7);actions_hidden_states.shape [1, 1, 64, 896]
           
        # Unnormalize predicted actions
        actions = self._unnormalize_actions(normalized_actions, unnorm_key)

        return actions, actions_hidden_states



    @staticmethod
    def _check_unnorm_key(norm_stats: Dict[str, Dict[str, Any]], unnorm_key: Optional[str]) -> str:
        """Validate and resolve the unnormalization key for action statistics"""
        if unnorm_key is None:
            assert len(norm_stats) == 1, (
                f"Your model was trained on more than one dataset, "
                f"please pass a `unnorm_key` from the following options to choose the statistics "
                f"used for un-normalizing actions: {norm_stats.keys()}"
            )
            unnorm_key = next(iter(norm_stats.keys()))

        assert unnorm_key in norm_stats, (
            f"The `unnorm_key` you chose is not in the set of available dataset statistics, "
            f"please choose from: {norm_stats.keys()}"
        )
        return unnorm_key

    def get_action_dim(self, unnorm_key: Optional[str] = None) -> int:
        """Get the dimensionality of the policy's action space."""
        unnorm_key = self._check_unnorm_key(self.norm_stats, unnorm_key)
        return len(self.norm_stats[unnorm_key]["action"]["min"])

    def get_action_stats(self, unnorm_key: Optional[str] = None) -> Dict[str, Any]:
        """Get all the logged statistics for the given dataset."""
        unnorm_key = self._check_unnorm_key(self.norm_stats, unnorm_key)
        return self.norm_stats[unnorm_key]["action"]
