"""
action_heads.py

Implementations of various action heads, which serve as alternatives to VLM sequential token prediction.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from prismatic.vla.constants import ACTION_DIM, ACTION_TOKEN_BEGIN_IDX, IGNORE_INDEX, NUM_ACTIONS_CHUNK, PROPRIO_DIM, STOP_INDEX, NUM_TOKENS
import numpy as np

def learnable_random_perturbations(seq_len, dim, device, dtype):
    random_perturbations = nn.Parameter(torch.zeros(seq_len, dim, device=device, dtype=dtype))
    nn.init.normal_(random_perturbations, mean=0.0, std=0.02)
    return random_perturbations
class AttentionPool1D(nn.Module):
    """Lightweight token attention pooling for the improved alignment path."""

    def __init__(self, hidden_dim):
        super().__init__()
        self.score = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        attn_logits = self.score(x).squeeze(dim=-1)
        attn_weights = torch.softmax(attn_logits, dim=1)
        return torch.sum(attn_weights.unsqueeze(dim=-1) * x, dim=1)


class L1RegressionActionHead(nn.Module):
    """Simple MLP-based action head that generates continuous actions via L1 regression."""
    def __init__(
        self,
        input_dim=4096,# LLM 的 hidden dimension（如 Qwen2-7B 是 4096）
        hidden_dim=4096,
        action_dim=7,
        num_task_tokens=512, #视觉 patch 数量（这里叫 task tokens），常见值 256/512
        use_pro_version=False,
        use_gate=False,
        use_latent_outputs=False,
        use_contrastive_loss=False, # TODO 是否使用对比学习来对齐表征
        alignment_strategy="original",
        use_manifold_regularization=False,
        manifold_topk=8,
        manifold_temperature=0.2,
        learnable_alignment_weights=False,
    ):
        super().__init__()
        self.num_task_tokens = num_task_tokens
        self.action_dim = action_dim 
        self.hidden_dim = hidden_dim
        self.model = MLPResNet(
            num_blocks=24, 
            input_dim=input_dim*ACTION_DIM, # actions_chunk 的输入维度
            hidden_dim=hidden_dim, 
            output_dim=action_dim, 
            use_pro_version=use_pro_version,
            use_gate=use_gate, #是否使用门控策略
            use_contrastive_loss=use_contrastive_loss, # TODO 是否启用对比学习的方式在hidden层做表征对齐
            alignment_strategy=alignment_strategy,
            use_manifold_regularization=use_manifold_regularization,
            manifold_topk=manifold_topk,
            manifold_temperature=manifold_temperature,
            learnable_alignment_weights=learnable_alignment_weights,
            )
        self.use_latent_outputs = use_latent_outputs
        if self.use_latent_outputs: #使用最后一层的VLM latent outputs 作为初始化的x 替代原有的 dummy action tokens的做法
            # 1. 加一个 LayerNorm，先把 VLM 的输出稳住（非常重要！）
            self.latent_norm = nn.LayerNorm(hidden_dim)
            # 2. Projector
            self.latent_projector = nn.Linear(hidden_dim, hidden_dim) 
            # 3. Positional Bias
            self.pos_bias = nn.Parameter(torch.zeros(1, self.action_dim*NUM_ACTIONS_CHUNK, hidden_dim))
            nn.init.normal_(self.pos_bias, mean=0.0, std=0.02)
            # ================= [关键修改] =================
            # 4. 零初始化策略 (Zero-Init Strategy)
            # 将投影层的权重和偏置全部设为 0
            nn.init.constant_(self.latent_projector.weight, 0.0)
            nn.init.constant_(self.latent_projector.bias, 0.0)
            # =============================================
        
    def predict_action(
            self, 
            actions_hidden_states,  # multi-layer hidden states for action tokens, [B, K层, patchs+action tokens, hidden_dim]
            proprio=None, # 本体状态（如关节角度），[B, proprio_dim]
            proprio_projector=None, # 将 proprio 投影到 LLM hidden_dim
            phase="Inference",
            latent_outputs=None, # (B, seq_len, hidden_dim)
            multi_layer_prompt_states=None,# [B, K, len, D]
            multi_layer_visual_states=None,
            multi_layer_action_states=None
            ):
        batch_size = actions_hidden_states.shape[0]
        device = actions_hidden_states.device

        proprio_features = None
        if proprio is not None:
            if proprio_projector is None:
                raise ValueError("proprio_projector must be provided when proprio is not None")
            proprio = proprio.reshape(batch_size, -1).to(torch.bfloat16)  # (bsz, proprio_dim)
            proprio_features = proprio_projector(proprio)  # (bsz, llm_dim) 本体状态表征
            proprio_features = proprio_features.unsqueeze(dim=1)  # (bsz, 1, llm_dim)

        task_hidden_states = actions_hidden_states[:, :, :self.num_task_tokens, :] # vision token
        actions_hidden_states = actions_hidden_states[:, :, self.num_task_tokens:, :] # action queries

        # FIXME
        if self.use_latent_outputs: #使用最后一层的VLM latent outputs 作为初始化的x 替代原有的 dummy action tokens的做法
            # 1. Mean Pooling: [B, seq_len, D] -> [B, 1, D]
            latent_summary = latent_outputs[:, -1, :].unsqueeze(dim=1)  # 取最后一个 有效 token 的 hidden state 作为 summary (B, 1, D)
            # 2. [新增] 先 Norm，防止 VLM 输出数值过大
            latent_summary = self.latent_norm(latent_summary)
            # 3. Projection
            projected_latent = self.latent_projector(latent_summary)
            # 4. Add Bias
            cond_actions_hidden_states = projected_latent + self.pos_bias.to(actions_hidden_states.dtype)
        else: #原始做法：用全0的dummy tokens作为输入
            cond_actions_hidden_states = torch.zeros(
                (batch_size, self.action_dim * NUM_ACTIONS_CHUNK, self.hidden_dim),
                device=device, dtype=actions_hidden_states.dtype
            ).detach()  #QUERY TOKENS: dummy tokens to predict actions_chunks 

        rearranged_actions_hidden_states = cond_actions_hidden_states.reshape(
            batch_size, NUM_ACTIONS_CHUNK, -1
        )  # (batch, action_chunk_len, action_dim * hidden_dim)

        if phase == "Training": #FIXME 训练时添加可学习的随机扰动（learnable noise）
            batch_size, seq_len, dim = rearranged_actions_hidden_states.shape #[B, chunk_len, dim]
            random_perturbations = learnable_random_perturbations(seq_len, dim, device=rearranged_actions_hidden_states.device, dtype=rearranged_actions_hidden_states.dtype) #[1, seq_len, dim]
            rearranged_actions_hidden_states = (rearranged_actions_hidden_states + random_perturbations) # 对chunk的每个位置添加可学习扰动
        # import ipdb; ipdb.set_trace()
        action = self.model( # MLPResNet
            rearranged_actions_hidden_states, #扰动后的 chunk hidden states (batch, chunk_len, action_dim * hidden_dim)
            h_a=actions_hidden_states,#(batch, k, action tokens=NUM_TOKENS, hidden_dim)
            h_t=task_hidden_states,#(batch, k, visual tokens, hidden_dim)
            p=proprio_features, #()(batch, 1, hidden_dim)
            phase=phase,
            multi_layer_prompt_states=multi_layer_prompt_states,
            multi_layer_visual_states=multi_layer_visual_states,
            multi_layer_action_states=multi_layer_action_states
            )

        return action
    

class MLPResNet(nn.Module):
    """MLP with residual connection blocks."""
    def __init__(
            self, 
            num_blocks, 
            input_dim, 
            hidden_dim, 
            output_dim,
            use_pro_version=False,
            use_gate=False,
            use_contrastive_loss=False, # TODO 是否使用对比学习来对齐表征
            alignment_strategy="original",
            use_manifold_regularization=False,
            manifold_topk=8,
            manifold_temperature=0.2,
            learnable_alignment_weights=False,
            ):
        
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(input_dim)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.mlp_resnet_blocks = nn.ModuleList()
        self.use_contrastive_loss = use_contrastive_loss
        if alignment_strategy not in {"original", "improved"}:
            raise ValueError(f"Unsupported alignment_strategy: {alignment_strategy}")
        self.alignment_strategy = alignment_strategy
        self.use_manifold_regularization = use_manifold_regularization and alignment_strategy == "improved"
        self.manifold_topk = manifold_topk
        self.manifold_temperature = manifold_temperature
        self.learnable_alignment_weights = learnable_alignment_weights and alignment_strategy == "improved"
        self.last_alignment_stats = {}
        if self.use_contrastive_loss:
            # ================== SigLIP 核心参数 ==================
            # 初始 logit_scale 设为 log(10)，bias 设为 -10 是 SigLIP 官方推荐的最优起点
            self.logit_scale = nn.Parameter(torch.tensor(np.log(10.0)))
            self.logit_bias = nn.Parameter(torch.tensor(-10.0))

            # TODO 需要定义 projection layer 来将 MLPResNet 中间层的 visual/prompt/action states 投影到对比学习的空间来计算对比损失
            self.c_proj_prompt = nn.Linear(hidden_dim, hidden_dim) # prompt 投影层 
            self.c_proj_visual = nn.Linear(hidden_dim, hidden_dim) # 
            self.c_proj_action = nn.Linear(hidden_dim, hidden_dim) # 
            self.c_proj_x = nn.Linear(hidden_dim, hidden_dim) # MLPResNet 中间层的 x 的投影层
            if self.alignment_strategy == "improved":
                self.x_pool = AttentionPool1D(hidden_dim)
                self.prompt_pool = AttentionPool1D(hidden_dim)
                self.visual_pool = AttentionPool1D(hidden_dim)
                self.action_pool = AttentionPool1D(hidden_dim)
                self.prompt_direction_proj = nn.Linear(hidden_dim, hidden_dim)
                self.fusion_gate = nn.Linear(hidden_dim * 2, 1)
                self.fusion_proj = nn.Linear(hidden_dim * 3, hidden_dim)
                num_alignment_terms = 4 if self.use_manifold_regularization else 3
                if self.learnable_alignment_weights:
                    self.alignment_weight_logits = nn.Parameter(torch.zeros(num_alignment_terms))
                else:
                    fixed_weights = [1.0, 0.1, 1.0]
                    if self.use_manifold_regularization:
                        fixed_weights.append(0.05)
                    self.register_buffer(
                        "alignment_fixed_weights",
                        torch.tensor(fixed_weights, dtype=torch.float32),
                        persistent=False,
                    )
        for _ in range(num_blocks):
            if use_pro_version:
                if use_gate:
                    self.mlp_resnet_blocks.append(MLPResNetBlock_Pro_Gated(dim=hidden_dim))
                else:
                    self.mlp_resnet_blocks.append(MLPResNetBlock_Pro(dim=hidden_dim))
            else:
                self.mlp_resnet_blocks.append(MLPResNetBlock(dim=hidden_dim))
                
        self.layer_norm2 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, h_a=None, h_t=None, p=None, phase="Inference", 
                    multi_layer_prompt_states=None, multi_layer_visual_states=None, multi_layer_action_states=None):
            x = self.layer_norm1(x) 
            x = self.fc1(x) 
            x = self.relu(x) 
            contrastive_loss = 0.0 
            alignment_stats_accumulator = {}
            for i, block in enumerate(self.mlp_resnet_blocks):
                # ================== 1. 记录上一层的 X (用于计算残差) ==================
                if phase == "Training" and self.use_contrastive_loss:
                    x_old_pooled = self._pool_tokens(x, pooler=getattr(self, "x_pool", None))
                    x_old_representation = self.c_proj_x(x_old_pooled)
                # ================== 2. 前向传播 (所有阶段都必须执行!) ==================
                x = block(x, h_t=h_t[:, i+1, :], h_a=h_a[:, i+1, :], p=p) 

                # ================== 3. 计算多目标对比损失 ==================
                if phase == "Training" and self.use_contrastive_loss:
                    # A. 提取 VLM 各个模态的表征
                    layer_prompt_tokens = multi_layer_prompt_states[:, i, :, :]
                    layer_prompt_state = self._pool_tokens(layer_prompt_tokens, pooler=getattr(self, "prompt_pool", None))
                    prompt_representation = self.c_proj_prompt(layer_prompt_state) 
                    
                    layer_visual_tokens = multi_layer_visual_states[:, i, :, :]
                    layer_visual_state = self._pool_tokens(layer_visual_tokens, pooler=getattr(self, "visual_pool", None)) #spatial foring 就是约束 vlm layer i 中的视觉表征接近 用预训练VGGT获得的表征
                    visual_representation = self.c_proj_visual(layer_visual_state)
                    
                    layer_action_tokens = multi_layer_action_states[:, i, :, :]
                    layer_action_state = self._pool_tokens(layer_action_tokens, pooler=getattr(self, "action_pool", None))
                    action_representation = self.c_proj_action(layer_action_state)
                    
                    # B. 计算当前 X 的表征 以及 它的残差
                    x_new_pooled = self._pool_tokens(x, pooler=getattr(self, "x_pool", None))
                    x_new_representation = self.c_proj_x(x_new_pooled)
                    
                    # 计算残差 (当前层新增的信息) 实现：Iterative Semantic Refinement 每次对于动作表征的更新都在prompt的表征引导下进行。
                    x_residual_representation = x_new_representation - x_old_representation
                    
                    # C. 调用对齐函数计算 step_loss
                    if self.alignment_strategy == "improved":
                        step_loss, step_stats = self.compute_improved_alignment_loss(
                            x_new_representation,
                            x_residual_representation,
                            prompt_representation,
                            visual_representation,
                            action_representation,
                        )
                    else:
                        step_loss, step_stats = self.compute_contrastive_loss(
                            x_new_representation, 
                            x_residual_representation, 
                            prompt_representation, 
                            visual_representation, 
                            action_representation
                        ) 
                    
                    # 累加损失 (求均值避免层数过多导致 loss 爆炸)
                    contrastive_loss += step_loss / len(self.mlp_resnet_blocks) 
                    for key, value in step_stats.items():
                        alignment_stats_accumulator[key] = alignment_stats_accumulator.get(key, 0.0) + (
                            value / len(self.mlp_resnet_blocks)
                        )
                    
            x = self.layer_norm2(x) 
            x = self.fc2(x) 
            self.last_alignment_stats = alignment_stats_accumulator if (
                phase == "Training" and self.use_contrastive_loss
            ) else {}
            
            if self.use_contrastive_loss and phase == "Training":
                return x, contrastive_loss 
            return x

    def _pool_tokens(self, hidden_states, pooler=None):
        if self.alignment_strategy == "improved" and pooler is not None:
            return pooler(hidden_states)
        return torch.max(hidden_states, dim=1)[0]

    def _fuse_prompt_visual(self, prompt_rep, visual_rep):
        if self.alignment_strategy != "improved":
            return prompt_rep + visual_rep

        fusion_gate = torch.sigmoid(self.fusion_gate(torch.cat([prompt_rep, visual_rep], dim=-1)))
        gated_mix = fusion_gate * visual_rep + (1.0 - fusion_gate) * prompt_rep
        interaction = prompt_rep * visual_rep
        return self.fusion_proj(torch.cat([gated_mix, prompt_rep, interaction], dim=-1))

    def _combine_alignment_terms(self, losses):
        stats = {}
        if self.alignment_strategy != "improved":
            lambda_visual = 1.0
            lambda_prompt = 0.1
            lambda_action = 1.0
            total_loss = (lambda_visual * losses[0]) + (lambda_prompt * losses[1]) + (lambda_action * losses[2])
            if len(losses) > 3:
                total_loss = total_loss + 0.05 * losses[3]
            stats["alignment_weight_visual"] = losses[0].new_tensor(lambda_visual)
            stats["alignment_weight_prompt"] = losses[0].new_tensor(lambda_prompt)
            stats["alignment_weight_action"] = losses[0].new_tensor(lambda_action)
            if len(losses) > 3:
                stats["alignment_weight_manifold"] = losses[0].new_tensor(0.05)
            return total_loss, stats

        if self.learnable_alignment_weights:
            raw_weights = F.softplus(self.alignment_weight_logits[: len(losses)]) + 1e-4
            normalized_weights = (len(losses) * raw_weights) / raw_weights.sum()
            total_loss = 0.0
            for idx, loss in enumerate(losses):
                total_loss = total_loss + normalized_weights[idx] * loss
            stats["alignment_weight_visual"] = normalized_weights[0]
            stats["alignment_weight_prompt"] = normalized_weights[1]
            stats["alignment_weight_action"] = normalized_weights[2]
            if len(losses) > 3:
                stats["alignment_weight_manifold"] = normalized_weights[3]
            return total_loss, stats

        fixed_weights = self.alignment_fixed_weights[: len(losses)].to(losses[0].dtype)
        total_loss = 0.0
        for idx, loss in enumerate(losses):
            total_loss = total_loss + fixed_weights[idx] * loss
        stats["alignment_weight_visual"] = fixed_weights[0]
        stats["alignment_weight_prompt"] = fixed_weights[1]
        stats["alignment_weight_action"] = fixed_weights[2]
        if len(losses) > 3:
            stats["alignment_weight_manifold"] = fixed_weights[3]
        return total_loss, stats

    def _graph_manifold_loss(self, latent_representation, teacher_representation):
        batch_size = latent_representation.shape[0]
        if batch_size <= 1:
            return latent_representation.new_zeros(())

        # `torch.cdist` is not implemented for bfloat16 on CUDA, so compute the
        # local geometry regularizer in float32 for numerical stability.
        latent_representation = F.normalize(latent_representation.float(), dim=-1)
        with torch.no_grad():
            teacher_representation = F.normalize(teacher_representation.float(), dim=-1)
            affinity = torch.matmul(teacher_representation, teacher_representation.T)
            affinity.fill_diagonal_(-1e4)

            if self.manifold_topk is not None and self.manifold_topk > 0 and batch_size > 2:
                k = min(self.manifold_topk, batch_size - 1)
                topk_idx = torch.topk(affinity, k=k, dim=-1).indices
                mask = torch.zeros_like(affinity, dtype=torch.bool)
                mask.scatter_(1, topk_idx, True)
                affinity = affinity.masked_fill(~mask, -1e4)

            weights = torch.softmax(affinity / self.manifold_temperature, dim=-1)
            weights = 0.5 * (weights + weights.T)
            weights.fill_diagonal_(0.0)

        pairwise_distance = torch.cdist(latent_representation, latent_representation, p=2).pow(2)
        return (weights * pairwise_distance).sum() / (weights.sum() + 1e-6)

    def compute_improved_alignment_loss(self, x_new, x_residual, prompt_rep, visual_rep, action_rep):
        loss_visual_x = self._siglip_loss(visual_rep, x_new)

        prompt_direction = self.prompt_direction_proj(prompt_rep)
        normalized_residual = F.normalize(x_residual.float(), dim=-1)
        normalized_prompt_direction = F.normalize(prompt_direction.float(), dim=-1)
        loss_prompt_xres = 1.0 - F.cosine_similarity(normalized_residual, normalized_prompt_direction, dim=-1).mean()

        fused_context_rep = self._fuse_prompt_visual(prompt_rep, visual_rep)
        loss_pv_action = self._siglip_loss(fused_context_rep, action_rep)

        losses = [loss_visual_x, loss_prompt_xres, loss_pv_action]
        if self.use_manifold_regularization:
            teacher_representation = 0.5 * (fused_context_rep.detach() + action_rep.detach())
            manifold_loss = self._graph_manifold_loss(x_new, teacher_representation)
            losses.append(manifold_loss)

        total_loss, weight_stats = self._combine_alignment_terms(losses)
        stats = {
            "alignment_loss_visual_raw": loss_visual_x.detach(),
            "alignment_loss_prompt_raw": loss_prompt_xres.detach(),
            "alignment_loss_action_raw": loss_pv_action.detach(),
        }
        if self.use_manifold_regularization:
            stats["alignment_loss_manifold_raw"] = losses[3].detach()
        stats.update({k: v.detach() for k, v in weight_stats.items()})
        stats["alignment_loss_raw_mean"] = torch.stack([loss.detach() for loss in losses]).mean()
        stats["alignment_loss_weighted"] = total_loss.detach()
        return total_loss, stats

    def compute_contrastive_loss(self, x_new, x_residual, prompt_rep, visual_rep, action_rep, temperature=0.1):
        """
        计算多维度的特征对齐对比损失
        """
        # 1. Visual 与 X_new 的对齐
        # loss_visual_x = self._info_nce(visual_rep, x_new, temperature)
        loss_visual_x = self._siglip_loss(visual_rep, x_new)
        # 2. Prompt 与 X_residual 的对齐 ？TODO 应该是visual condition下的X_residual 与 prompt 的对齐，来强化 prompt 对于动作更新的引导作用
        # 迫使当前层学到的 "新增/增量信息" 对应文本的语义引导
        # loss_prompt_xres = self._info_nce(prompt_rep, x_residual, temperature)
        loss_prompt_xres = self._siglip_loss(prompt_rep, x_residual)

        # 3. (Prompt + Visual) 联合表征 与 Action 的对齐
        # 简单的加法融合是在同维度隐空间中最常用的多模态融合方式
        combined_pv_rep = self._fuse_prompt_visual(prompt_rep, visual_rep)
        # loss_pv_action = self._info_nce(combined_pv_rep, action_rep, temperature)
        loss_pv_action = self._siglip_loss(combined_pv_rep, action_rep)

        total_loss, weight_stats = self._combine_alignment_terms([loss_visual_x, loss_prompt_xres, loss_pv_action])
        stats = {
            "alignment_loss_visual_raw": loss_visual_x.detach(),
            "alignment_loss_prompt_raw": loss_prompt_xres.detach(),
            "alignment_loss_action_raw": loss_pv_action.detach(),
            "alignment_loss_raw_mean": torch.stack(
                [loss_visual_x.detach(), loss_prompt_xres.detach(), loss_pv_action.detach()]
            ).mean(),
            "alignment_loss_weighted": total_loss.detach(),
        }
        stats.update({k: v.detach() for k, v in weight_stats.items()})
        return total_loss, stats

    def _info_nce(self, feat1, feat2, temperature=0.1):
        """
        标准的 InfoNCE (Info Noise-Contrastive Estimation) 损失
        feat1, feat2 形状皆为 [Batch_Size, Hidden_Dim]
        """
        # L2 归一化，把特征映射到单位超球面上
        feat1 = torch.nn.functional.normalize(feat1, dim=-1)
        feat2 = torch.nn.functional.normalize(feat2, dim=-1)

        # 计算余弦相似度矩阵 [B, B]
        # 矩阵对角线元素是正样本对，非对角线元素是负样本对
        logits = torch.matmul(feat1, feat2.T) / temperature

        # 构造目标标签: 0, 1, 2, ..., B-1 (即对角线)
        batch_size = logits.shape[0]
        labels = torch.arange(batch_size, device=logits.device)

        # 双向 CrossEntropy (类似 CLIP: Image2Text 和 Text2Image 都要算)
        loss_1to2 = torch.nn.functional.cross_entropy(logits, labels)
        loss_2to1 = torch.nn.functional.cross_entropy(logits.T, labels)
        
        return (loss_1to2 + loss_2to1) / 2.0

    def _siglip_loss(self, feat1, feat2):
        """
        SigLIP 对比损失：将全局多分类降维为独立的二分类
        输入形状皆为 [Batch_Size, Hidden_Dim]
        """
        # 1. L2 归一化特征
        feat1 = F.normalize(feat1, dim=-1)
        feat2 = F.normalize(feat2, dim=-1)

        # 2. 限制 scale 上限防止梯度爆炸 (最大值为 e^4.605 ≈ 100)
        logit_scale = torch.clamp(self.logit_scale.exp(), max=100.0)
        
        # 3. 计算余弦相似度并应用缩放与偏置 [B, B]
        logits = torch.matmul(feat1, feat2.T) * logit_scale + self.logit_bias

        # 4. 构造二分类标签 (对角线正样本=1，非对角线负样本=-1)
        batch_size = logits.shape[0]
        labels = 2.0 * torch.eye(batch_size, device=logits.device) - 1.0

        # 5. 计算逐像素的 LogSigmoid 损失并求均值
        # F.logsigmoid 相比手动计算 -log(1/(1+e^-x)) 在数值上更稳定，防止 nan
        loss = -F.logsigmoid(labels * logits)
        
        return loss.mean()
def apply_rope(q, k, cos, sin):
    """
    RoPE:
    q, k: (B, H, T, D)   # D must be an even number
    cos/sin: (T, D)
    """
    cos = cos.unsqueeze(0).unsqueeze(0)  # (1, 1, T, D)
    sin = sin.unsqueeze(0).unsqueeze(0) # 预计算的位置旋转角度，形状 (T, D)（seq_len, head_dim）


    def rotate_half(x):
        # Swap even and odd dimensions and flip the signs
        x1 = x[..., ::2]   # Even subdimension
        x2 = x[..., 1::2]  # odd subdimension

        return torch.stack((-x2, x1), dim=-1).reshape_as(x) 


    q_rot = (q * cos) + (rotate_half(q) * sin)
    k_rot = (k * cos) + (rotate_half(k) * sin)
    #x'_{2i}   = x_{2i} * cos θ - x_{2i+1} * sin θ
    #x'_{2i+1} = x_{2i} * sin θ + x_{2i+1} * cos θ
    return q_rot, k_rot



class RotaryPositionEmbedding(nn.Module):
    def __init__(self, dim, base=10000):
        """
        dim = head_dim
        """
        super().__init__()
        assert dim % 2 == 0, "RoPE head_dim must be an even number"
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, seq_len, device, dtype):
        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype) ## [0, 1, 2, ..., T-1]
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)  # (T, dim/2)
        # t[0] * inv_freq[0], t[0] * inv_freq[1], ...
        # t[1] * inv_freq[0], t[1] * inv_freq[1], ...
        emb = torch.cat([freqs, freqs], dim=-1)  # (T, dim) 复制一份补全奇数维度
        return emb.cos().to(dtype), emb.sin().to(dtype)



class MLPResNetBlock(nn.Module):
    """
    One residual MLP block with cross-attention conditioning.

    This block applies multi-head attention over:
      - token features (self-attention),
      - task-related hidden states (h_t),
      - action/proprioception-related hidden states (h_a, p).
    The outputs are combined via a gating mechanism, projected back to the
    hidden dimension, and passed through a small feedforward sub-network with
    residual connection.

    Args:
        dim (int): Dimensionality of the hidden features. Must be divisible by num_heads.

    Inputs:
        x (torch.Tensor): Input tensor of shape (batch_size, seq_len, hidden_dim).
        h_t (torch.Tensor, optional): Task-related hidden states of shape
                                      (batch_size, K, hidden_dim).
        h_a (torch.Tensor, optional): Action-related hidden states of shape
                                      (batch_size, 1, hidden_dim).
        p (torch.Tensor, optional): Additional conditioning features
                                    (e.g., proprioception), shape (batch_size, 1, hidden_dim).

    Returns:
        torch.Tensor: Output tensor of shape (batch_size, seq_len, hidden_dim).
    """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        
        # Main feedforward network
        self.ffn = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.ReLU(),
        )

        self.num_heads = 8
        self.head_dim = dim // self.num_heads

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.o_proj = nn.Linear(dim, dim)

        self.gating_factor = nn.Parameter(torch.zeros(1))



    def forward(self, x, h_t=None, h_a=None, p=None):
        """
        x: (batch_size, seq_len, hidden_dim)
        h, t, p: (batch_size, 1, hidden_dim) or None
        """

        g = self.gating_factor
        ratio_g = nn.Tanh()(g)

        conditions = []
        if h_a is not None:
            conditions.append(h_a)
        if p is not None:
            conditions.append(p)

        h = torch.cat(conditions, dim=1)  # (batch_size, cond_len, hidden_dim)

        B = x.size(0)
        T = x.size(1)
        C = x.size(2)
        K_t = h.size(1)
        K = h_t.size(1)

        task_k = h
        task_v = h

        adapter_k = h_t
        adapter_v = h_t

        q_1 = self.q_proj(x) # (B, T, C)
        k_tokens = self.k_proj(x)             # (B, T, C)
        v_tokens = self.v_proj(x)             # (B, T, C)
        k_task = self.k_proj(task_k)    # (B, K, C)
        v_task = self.v_proj(task_v)    # (B, K, C)

        k_adapter = self.k_proj(adapter_k)    # (B, K, C)
        v_adapter = self.v_proj(adapter_v)    # (B, K, C)

        # (B, seq_len, C) -> (B, num_heads, seq_len, head_dim)
        q_1 = q_1.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        
        k_tokens = k_tokens.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v_tokens = v_tokens.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k_task = k_task.view(B, K_t, self.num_heads, self.head_dim).transpose(1, 2)
        v_task = v_task.view(B, K_t, self.num_heads, self.head_dim).transpose(1, 2)

        k_adapter = k_adapter.view(B, K, self.num_heads, self.head_dim).transpose(1, 2)
        v_adapter = v_adapter.view(B, K, self.num_heads, self.head_dim).transpose(1, 2)

        attn_scores_tokens = torch.matmul(q_1, k_tokens.transpose(-2, -1)) # (B, H, T, T)
        attn_scores_task = torch.matmul(q_1, k_task.transpose(-2, -1)) * 1 # (B, H, T, K)
        attn_scores_adapter = torch.matmul(q_1, k_adapter.transpose(-2, -1)) * ratio_g # (B, H, T, K)

        attn_scores = torch.cat([attn_scores_tokens, attn_scores_task, attn_scores_adapter], dim=-1) # (B, H, T, T+K)
        attn_scores = attn_scores / math.sqrt(self.head_dim)
        attn_weights = torch.softmax(attn_scores, dim=-1) # (B, H, T, T+K)

        v_combined = torch.cat([v_tokens, v_task, v_adapter], dim=2) # (B, H, T+K, head_dim)
        output = torch.matmul(attn_weights, v_combined) # (B, H, T, head_dim)

        output = output.transpose(1, 2).contiguous().view(B, T, C)
        output = self.o_proj(output)

        x = self.ffn(output + x) 

        return x



class MLPResNetBlock_Pro(nn.Module):
    """One MLP ResNet block with separate projections for self, adapter, task + RoPE, now with FiLM modulation."""

    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.ffn = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.ReLU(),
            )

        # Q (from x only)
        self.q_proj = nn.Linear(dim, dim)

        # Self-Attention: K, V
        self.k_self = nn.Linear(dim, dim)
        self.v_self = nn.Linear(dim, dim)

        # Adapter cross-attention: K, V
        self.k_adapter = nn.Linear(dim, dim)
        self.v_adapter = nn.Linear(dim, dim)

        # Task cross-attention: K, V
        self.k_task = nn.Linear(dim, dim)
        self.v_task = nn.Linear(dim, dim)

        self.o_proj = nn.Linear(dim, dim)

        # gating
        self.gating_factor = nn.Parameter(torch.zeros(1))

        # RoPE
        self.rope = RotaryPositionEmbedding(self.head_dim)

        # ---- FiLM ----
        # FiLM is useless; to avoid conflict with chkpt, it can be kept as is for now.
        self.film_gen = nn.Sequential(
            nn.Linear(dim, dim * 2),  # output γ and β
            )


    def apply_film(self, x, gamma, beta):
        """FiLM: per-channel modulation"""
        return gamma.unsqueeze(1) * x + beta.unsqueeze(1)


    def forward(self, x, h_a=None, h_t=None, p=None):
        """
        h_a: layer i adapter tokens
        h_t: layer i task tokens 
        p:   possible conditioning vector (for FiLM)
        """
        # import ipdb; ipdb.set_trace()
        g = self.gating_factor
        ratio_g = torch.tanh(g)

        # concat available adapter tokens; no-proprio runs should still keep h_a
        adapter_inputs = []
        if h_a is not None:
            adapter_inputs.append(h_a)
        if p is not None:
            adapter_inputs.append(p)
        h_adapter = torch.cat(adapter_inputs, dim=1) if len(adapter_inputs) > 1 else (adapter_inputs[0] if adapter_inputs else None)

        h_task = h_t # 也是第k层的task tokens [B,task_tokens,dim]
        B, T, C = x.shape # [1, 8, 896]
        K_a = h_adapter.size(1) if h_adapter is not None else 0 #NUM_TOKENS +1: 64+1
        K_t = h_task.size(1) if h_task is not None else 0 #num_task_tokens: 512

        # Q
        q_1 = self.q_proj(x) # (B, T, C) → 只从 x 投影 Q

        # self tokens
        k_tokens = self.k_self(x) # self-attn 的 K (B, T, C) [1, 8, 896]
        v_tokens = self.v_self(x) # self-attn 的 V (B, T, C) [1, 8, 896]

        # adapter tokens
        k_adapter = self.k_adapter(h_adapter) # adapter cross 的 K (B, K_a, C) [1, 65, 896]
        v_adapter = self.v_adapter(h_adapter)# adapter cross 的 V (B, K_a, C) [1, 65, 896]

        # task tokens
        k_task = self.k_task(h_task)# task cross 的 K (B, K_t, C) [1, 512, 896]
        v_task = self.v_task(h_task)# task cross 的 V (B, K_t, C) [1, 512, 896]


        # reshape -> multi-head
        def reshape_heads(t, B, L): #t shape:[B,L,C] 实际上就是将C拆开
            return t.view(B, L, self.num_heads, self.head_dim).transpose(1, 2) # (B, L, C) → (B, num_heads=8, L, head_dim=512)


        q_1 = reshape_heads(q_1, B, T) #(B, T, C)：[1, 8, 896]-> [B, num_heads=8, T=8, head_dim=112]
        k_tokens, v_tokens = reshape_heads(k_tokens, B, T), reshape_heads(v_tokens, B, T) #(B, T, C) #torch.Size([1, 8, 8, 112])
        k_adapter, v_adapter = reshape_heads(k_adapter, B, K_a), reshape_heads(v_adapter, B, K_a) #torch.Size([1, 8, 65, 112])
        k_task, v_task = reshape_heads(k_task, B, K_t), reshape_heads(v_task, B, K_t) # [1, 8, 512, 112]

        # RoPE
        cos_main, sin_main = self.rope(seq_len=T, device=x.device, dtype=x.dtype) # [65, 112]
        q_1, k_tokens = apply_rope(q_1, k_tokens, cos_main, sin_main)
        cos_a, sin_a = self.rope(seq_len=K_a, device=x.device, dtype=x.dtype)
        _, k_adapter = apply_rope(k_adapter, k_adapter, cos_a, sin_a)     
        cos_t, sin_t = self.rope(seq_len=K_t, device=x.device, dtype=x.dtype)
        _, k_task = apply_rope(k_task, k_task, cos_t, sin_t)

        # attention scores
        attn_scores = [torch.matmul(q_1, k_tokens.transpose(-2, -1))] #torch.matmul(q_1, k_tokens.transpose(-2, -1)).shape:torch.Size([1, 8, 8, 8])
        attn_scores.append(torch.matmul(q_1, k_adapter.transpose(-2, -1))) #append torch.Size([1, 8, 8, 65])
        attn_scores.append(torch.matmul(q_1, k_task.transpose(-2, -1)) * ratio_g)# append torch.Size([1, 8, 8, 512])
        attn_scores = torch.cat(attn_scores, dim=-1) / math.sqrt(self.head_dim) # (B, H, T, T+K_a+K_t) torch.Size([1, 8, 8, 585])
        #TODO 原始做法是三个分支的attn一起做softmax 类似于MOE的选择性关注不同的分支
        attn_weights = torch.softmax(attn_scores, dim=-1) #torch.Size([1, 8, 8, 585]

        # combine V
        v_list = [v_tokens,v_adapter,v_task]# torch.Size([1, 8, 8, 112]) + torch.Size([1, 8, 65, 112])+ torch.Size([1, 8, 512, 112])
        v_combined = torch.cat(v_list, dim=2) # (B, H, T+K_a+K_t, dim)
        #等价于Y_original = (weights_self @ V_self) + (weights_adapter @ V_adapter) + (weights_task @ V_task)
        output = torch.matmul(attn_weights, v_combined)
        output = output.transpose(1, 2).contiguous().view(B, T, C)
        output = self.o_proj(output)

        # # ---- FiLM ---- 
        # gamma_beta = self.film_gen(p)  # [B, 2C]
        # gamma, beta = gamma_beta.chunk(2, dim=-1)  # [B, C], [B, C]
        # output = self.apply_film(output, gamma, beta)

        # residual + FFN
        x = self.ffn(output + x) #torch.Size([1, 8, 896])
        return x


class MLPResNetBlock_Pro_Gated(nn.Module):
    """
    改进版 MLP ResNet Block：
    - 支持三种注意力来源：self-attention、adapter cross-attention、task cross-attention
    - 每个注意力分支独立计算注意力输出
    - 为每个分支添加独立的 Gated Attention（参考 Qwen Gated Attention 论文）
      即在注意力输出 Y 后乘以 query-dependent 的 sigmoid 门控：Y' = Y ⊙ σ(x @ W_g)
    - 结构与原始版本高度对应，便于对比和迁移
    """

    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.dim = dim                  # 模型隐藏维度
        self.num_heads = num_heads      # 多头注意力头数
        self.head_dim = dim // num_heads  # 每个头的维度

        # --------------------- FFN 部分 ---------------------
        self.ffn = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.ReLU(),
        )

        # --------------------- Q/K/V 投影 ---------------------
        # 共享的 Query，只来自主序列 x
        self.q_proj = nn.Linear(dim, dim)

        # Self-attention 的 K、V（来自主序列 x）
        self.k_self = nn.Linear(dim, dim)
        self.v_self = nn.Linear(dim, dim)

        # Adapter cross-attention 的 K、V（来自 adapter tokens + proprio）
        self.k_adapter = nn.Linear(dim, dim)
        self.v_adapter = nn.Linear(dim, dim)

        # Task cross-attention 的 K、V（来自 task tokens）
        self.k_task = nn.Linear(dim, dim)
        self.v_task = nn.Linear(dim, dim)

        # 注意力多头拼接后的最终输出投影
        self.o_proj = nn.Linear(dim, dim)

        # --------------------- 原始 gating factor（用于 task 分支的 attn_score 缩放） ---------------------
        self.gating_factor = nn.Parameter(torch.zeros(1))  # 可学习标量，控制 task 分支强度

        # --------------------- RoPE 位置编码 ---------------------
        self.rope = RotaryPositionEmbedding(self.head_dim)  # 每个头独立的旋转位置编码

        # --------------------- FiLM（当前未使用，可保留兼容性） ---------------------
        self.film_gen = nn.Sequential(
            nn.Linear(dim, dim * 2),  # 输出 gamma (scale) 和 beta (bias)
        )

        # --------------------- 新增：三个独立的分支门控投影（核心改动） ---------------------
        # 每个分支都有独立的 query-dependent 门控线性层
        # 门控值 = sigmoid(x @ w_g_xxx)，形状 (B, T, dim)，用于逐元素乘注意力输出
        self.w_g_self    = nn.Linear(dim, dim)   # self-attention 分支门控
        self.w_g_adapter = nn.Linear(dim, dim)   # adapter cross-attention 分支门控
        self.w_g_task    = nn.Linear(dim, dim)   # task cross-attention 分支门控

    def apply_film(self, x, gamma, beta):
        """FiLM 条件调制（当前未启用，仅保留接口）"""
        return gamma.unsqueeze(1) * x + beta.unsqueeze(1)

    def forward(self, x, h_a=None, h_t=None, p=None):
        """
        前向传播
        Args:
            x:   主序列输入 [B, T, dim]，作为 Query 和 self-attention 的 KV 来源
            h_a: adapter tokens（第 i 层）[B, num_adapter_tokens, dim]
            h_t: task tokens（第 i 层）[B, num_task_tokens, dim]
            p:   proprioception 或其他条件向量 [B, 1, dim]，会与 h_a 拼接
        """
        # 保存残差
        residual = x

        # # gating_factor 用于控制 task 分支的注意力分数强度
        # ratio_g = torch.tanh(self.gating_factor)

        # --------------------- 准备 adapter 和 task 输入 ---------------------
        # 将可用的 adapter tokens 拼接；no-proprio 时仍保留 h_a 分支
        adapter_inputs = []
        if h_a is not None:
            adapter_inputs.append(h_a)
        if p is not None:
            adapter_inputs.append(p)
        h_adapter = torch.cat(adapter_inputs, dim=1) if len(adapter_inputs) > 1 else (adapter_inputs[0] if adapter_inputs else None)
        h_task = h_t

        B, T, C = x.shape                     # 主序列形状
        K_a = h_adapter.size(1) if h_adapter is not None else 0   # adapter 序列长度
        K_t = h_task.size(1) if h_task is not None else 0         # task 序列长度

        # --------------------- 共享 Query 投影与 reshape ---------------------
        q = self.q_proj(x)                                            # [B, T, dim]
        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)  # [B, num_heads, T, head_dim]

        # ==================================== Self-Attention 分支 ====================================
        # K、V 来自主序列 x
        k_self = self.k_self(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v_self = self.v_self(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        # 应用 RoPE（self-attention 需要对 Q 和 K 都旋转）
        cos_main, sin_main = self.rope(seq_len=T, device=x.device, dtype=x.dtype)
        q_self_rot, k_self_rot = apply_rope(q, k_self, cos_main, sin_main)

        # 计算注意力
        attn_score_self = torch.matmul(q_self_rot, k_self_rot.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn_weight_self = torch.softmax(attn_score_self, dim=-1)
        y_self = torch.matmul(attn_weight_self, v_self)   # [B, num_heads, T, head_dim]

        # 应用 self 分支门控（Gated Attention 核心）
        gate_self = torch.sigmoid(self.w_g_self(x))       # [B, T, dim]，query-dependent
        y_self_gated = y_self.transpose(1, 2).contiguous().view(B, T, C) * gate_self

        # ==================================== Adapter Cross-Attention 分支 ====================================
        if h_adapter is not None:
            k_adapter = self.k_adapter(h_adapter).view(B, K_a, self.num_heads, self.head_dim).transpose(1, 2)
            v_adapter = self.v_adapter(h_adapter).view(B, K_a, self.num_heads, self.head_dim).transpose(1, 2)

            # Cross-attention 只对 K 施加 RoPE（Q 已经旋转过）
            cos_a, sin_a = self.rope(seq_len=K_a, device=x.device, dtype=x.dtype)
            _, k_adapter_rot = apply_rope(k_adapter, k_adapter, cos_a, sin_a)  # q 不需要再次旋转

            attn_score_adapter = torch.matmul(q_self_rot, k_adapter_rot.transpose(-2, -1)) / math.sqrt(self.head_dim)
            attn_weight_adapter = torch.softmax(attn_score_adapter, dim=-1)
            y_adapter = torch.matmul(attn_weight_adapter, v_adapter)

            # 应用 adapter 分支门控
            gate_adapter = torch.sigmoid(self.w_g_adapter(x))
            y_adapter_gated = y_adapter.transpose(1, 2).contiguous().view(B, T, C) * gate_adapter
        else:
            y_adapter_gated = torch.zeros(B, T, C, device=x.device, dtype=x.dtype)

        # ==================================== Task Cross-Attention 分支 ====================================
        if h_task is not None:
            k_task = self.k_task(h_task).view(B, K_t, self.num_heads, self.head_dim).transpose(1, 2)
            v_task = self.v_task(h_task).view(B, K_t, self.num_heads, self.head_dim).transpose(1, 2)

            cos_t, sin_t = self.rope(seq_len=K_t, device=x.device, dtype=x.dtype)
            _, k_task_rot = apply_rope(k_task, k_task, cos_t, sin_t)

            # 注意：这里保留了原始的 ratio_g 缩放（控制 task 分支强度）
            attn_score_task = (torch.matmul(q_self_rot, k_task_rot.transpose(-2, -1))) / math.sqrt(self.head_dim)
            attn_weight_task = torch.softmax(attn_score_task, dim=-1)
            y_task = torch.matmul(attn_weight_task, v_task)

            # 应用 task 分支门控
            gate_task = torch.sigmoid(self.w_g_task(x))
            y_task_gated = y_task.transpose(1, 2).contiguous().view(B, T, C) * gate_task
        else:
            y_task_gated = torch.zeros(B, T, C, device=x.device, dtype=x.dtype)

        # ==================================== 合并三个分支输出 ====================================
        # output = y_self_gated + y_adapter_gated + y_task_gated * ratio_g  # [B, T, dim]
        # 去除 ratio_g，全部由门控控制
        output = y_self_gated + y_adapter_gated + y_task_gated  # [B, T, dim]
        output = self.o_proj(output)                             # 最终线性投影

        # --------------------- 残差连接 + FFN ---------------------
        x = self.ffn(output + residual)   # 最终输出

        return x
