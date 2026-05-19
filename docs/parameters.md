## Parameter table

| Parameter | Where set | Current | After retrain (state_t=6) | Constraint |
|---|---|---|---|---|
| `obs.workspace_rgb.horizon` | [`policy_io/lerobot.yaml`](../mimic-video/model/cosmos_predict2/configs/dataloading/policy_io/lerobot.yaml) | 5 | 5 | obs window stays — 1 s of past context regardless |
| `action.workspace_rgb.horizon` | [`policy_io/lerobot.yaml`](../mimic-video/model/cosmos_predict2/configs/dataloading/policy_io/lerobot.yaml) | **56** | **16** | ⓐ must satisfy `obs + action ≡ 1 + 4n` for clean VAE compression |
| `obs.joint_pos_lowdim.horizon` | [`policy_io/lerobot.yaml`](../mimic-video/model/cosmos_predict2/configs/dataloading/policy_io/lerobot.yaml) | 1 | 1 | independent (just current joint state) |
| `action.joint_action_lowdim.horizon` | [`policy_io/lerobot.yaml`](../mimic-video/model/cosmos_predict2/configs/dataloading/policy_io/lerobot.yaml) | 15 | 15 | independent (output spec — number of actions predicted) |
| `num_frames` (V2W finetune clip) | [`data_video.py`](../mimic-video/model/cosmos_predict2/configs/defaults/data_video.py) | **61** | **21** | ⓐ must equal `obs.workspace_rgb.horizon + action.workspace_rgb.horizon` |
| `state_t` (DiT positional embeddings) | DiT model config (saved in checkpoint) | **16** | **6** | ⓑ must equal VAE latent count from `num_frames` |
| VAE latent count | derived from `num_frames` via VAE math | `16 = 1 + (61-1)/4` | `6 = 1 + (21-1)/4` | ⓑ derived; matches `state_t` |
| Action decoder cross-attn KV length | derived from DiT output | 16 timesteps | 6 timesteps | ⓒ inherits from DiT — no separate setting |
| `obs.workspace_rgb.target_frequency` | [`policy_io/lerobot.yaml`](../mimic-video/model/cosmos_predict2/configs/dataloading/policy_io/lerobot.yaml) | 5 | 5 | ⓓ must match V2W training rate |
| `action.workspace_rgb.target_frequency` | [`policy_io/lerobot.yaml`](../mimic-video/model/cosmos_predict2/configs/dataloading/policy_io/lerobot.yaml) | 5 | 5 | ⓓ must match V2W training rate |
| V2W effective sampling rate | [`dataset_video.py`](../mimic-video/model/cosmos_predict2/data/dataset_video.py) (`step = data_fps/5`) | 5 Hz (hardcoded) | 5 Hz | ⓓ hardcoded in dataloader |
| `obs.joint_pos_lowdim.target_frequency` | [`policy_io/lerobot.yaml`](../mimic-video/model/cosmos_predict2/configs/dataloading/policy_io/lerobot.yaml) | 5 | 5 | independent (controls obs interpolation) |
| `action.joint_action_lowdim.target_frequency` | [`policy_io/lerobot.yaml`](../mimic-video/model/cosmos_predict2/configs/dataloading/policy_io/lerobot.yaml) | 5 | 5 | independent (controls action prediction rate) |

## Coupling constraints

### ⓐ Clip-shape chain

`num_frames` in V2W finetune **must equal** `obs.workspace_rgb.horizon + action.workspace_rgb.horizon` in action decoder training. Both must hit a clean VAE shape (`1 + 4n`).

- Current: both = 61 (5 + 56)
- After retrain: both = 21 (5 + 16)

### ⓑ Latent-shape chain

VAE latent count `1 + (num_frames - 1) / 4` **must equal** the DiT's `state_t`. The DiT bakes `state_t` into its positional embeddings, so this is set by V2W training and consumed downstream.

- 61 frames → 16 latents → `state_t=16`
- 21 frames → 6 latents → `state_t=6`

### ⓒ Cross-attn KV length

The action decoder has **no separate setting** for cross-attn KV length. It cross-attends to whatever the DiT produces. Shrinking `state_t` automatically shrinks the action decoder's input — no changes to action decoder architecture needed.

### ⓓ Rate chain

`workspace_rgb.target_frequency` (obs and action sides) **must equal** the rate the V2W DiT was trained at. That rate is **hardcoded to 5 Hz** in [`dataset_video.py`](../mimic-video/model/cosmos_predict2/data/dataset_video.py) via `step = data_fps / 5.0` — you can't change this without modifying that line and retraining the V2W DiT.

## Independent parameters (no coupling)

These can be tuned in isolation without breaking the video pipeline:

- `obs.joint_pos_lowdim.horizon` and `target_frequency` — joint-state obs is read directly by the action decoder, no V2W involvement.
- `action.joint_action_lowdim.horizon` — your prediction count. Currently 15; could be 30 (longer chunks), 10 (shorter), etc. Only affects the action decoder's output head dims.
- `action.joint_action_lowdim.target_frequency` — your prediction rate. Currently 5; could be 10 (Jonas's smoothness suggestion) at the cost of cache invalidation and retraining.

## Workflow for shrinking `state_t` (e.g., 16 → 6)

1. **V2W side** (Jonas's pipeline):
   - Change `num_frames=21` in [`data_video.py`](../mimic-video/model/cosmos_predict2/configs/defaults/data_video.py).
   - Re-finetune V2W DiT with new `state_t=6` (architectural change to positional embeddings).
   - Publish new `iter_*_fused.pt` checkpoint.

2. **Action decoder side** (your pipeline):
   - Change `action.workspace_rgb.horizon: 56 → 16` in [`policy_io/lerobot.yaml`](../mimic-video/model/cosmos_predict2/configs/dataloading/policy_io/lerobot.yaml).
   - Relax the VAE shim allowlist at [`precompute_video_latents.py:55`](../mimic-video/model/scripts/precompute_video_latents.py) (`{1, 5, 61}` → add the new T).
   - Update `LATENT_SHAPE` at [`precompute_video_latents.py:67`](../mimic-video/model/scripts/precompute_video_latents.py) to match new latent timesteps.
   - Point `VIDEO_DIT_PATH` at the new checkpoint.
   - Re-run [`scripts/precompute_video_latents.sh`](../scripts/precompute_video_latents.sh) (stats_id changes).
   - Re-train the action decoder.
