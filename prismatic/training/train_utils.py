"""Utils for training/fine-tuning scripts."""

import torch

from prismatic.vla.constants import ACTION_DIM, ACTION_TOKEN_BEGIN_IDX, IGNORE_INDEX


def get_current_action_mask(token_ids):
    """
    token_ids: shape [B, seq_len]
       example:
            token_ids:   [-100, -100, ..., , act1, act2, ..., act7, act8, ..., act56, EOS]
            有效位置:   [False, ..., , True (act1), True (act2), ..., True (act56), True (EOS)]
            cumsum:      [0, ..., , 1 (act1), 2 (act2), ..., 7 (act7), 8 (act8), ...]
    Outputs：返回一个布尔 mask，True 的位置对应当前一步的 7 个动作 token
    """
    # import pdb; pdb.set_trace()
    # Create a tensor marking positions of IGNORE_INDEX
    newline_positions = token_ids != IGNORE_INDEX #步骤1：找出非 padding 位置（有效 token）

    # Calculate cumulative sum to identify regions between newlines
    cumsum = torch.cumsum(newline_positions, dim=1) #步骤2：在序列维度上累积计数（从左到右数有效 token 个数）

    # Create the mask
    mask = (1 <= cumsum) & (cumsum <= ACTION_DIM) #步骤3：有效 token 计数在 [1, 7] 的位置 → 当前动作位置

    # Extract the action part only
    action_tokens_only_mask = token_ids > ACTION_TOKEN_BEGIN_IDX # 步骤4：进一步确保这些位置是动作 token（ID > 起始阈值）
    mask = action_tokens_only_mask * mask # 步骤5：交集，确保既是当前动作位置又是真正的动作 token

    return mask


def get_next_actions_mask(token_ids):
    # import pdb; pdb.set_trace()
    # Create a tensor marking positions of IGNORE_INDEX
    newline_positions = token_ids != IGNORE_INDEX

    # Calculate cumulative sum to identify regions between newlines
    cumsum = torch.cumsum(newline_positions, dim=1)

    # Create the mask
    mask = cumsum > ACTION_DIM #未来动作位置：有效 token 计数大于 7 的位置

    # Extract the action part only
    action_tokens_only_mask = token_ids > ACTION_TOKEN_BEGIN_IDX
    mask = action_tokens_only_mask * mask

    return mask


def compute_token_accuracy(predicted_token_ids, ground_truth_token_ids, mask):
    correct_preds = (predicted_token_ids == ground_truth_token_ids) & mask
    accuracy = correct_preds.sum().float() / mask.sum().float()
    return accuracy


def compute_actions_l1_loss(action_tokenizer, predicted_token_ids, ground_truth_token_ids, mask):
    pred_continuous_actions = torch.tensor(
        action_tokenizer.decode_token_ids_to_actions(predicted_token_ids[mask].cpu().numpy())
    )
    true_continuous_actions = torch.tensor(
        action_tokenizer.decode_token_ids_to_actions(ground_truth_token_ids[mask].cpu().numpy())
    )
    l1_loss = torch.nn.functional.l1_loss(pred_continuous_actions, true_continuous_actions)
    return l1_loss
