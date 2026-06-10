# Video and Action Model Parameters

The video and action models share several temporal parameters. Keep these
values aligned when preparing data, finetuning the Video2World backbone, and
training the action decoder.

## Current Configuration

| Parameter | Where set | Value | Purpose |
|---|---|---:|---|
| Video clip length | [`data_video.py`](../mimic-video/model/cosmos_predict2/configs/defaults/data_video.py) | 21 frames | Input length used to finetune the SO-101 Video2World datasets |
| Observation image horizon | [`policy_io/lerobot.yaml`](../mimic-video/model/cosmos_predict2/configs/dataloading/policy_io/lerobot.yaml) | 5 frames | Past visual context provided to the model |
| Future image horizon | [`policy_io/lerobot.yaml`](../mimic-video/model/cosmos_predict2/configs/dataloading/policy_io/lerobot.yaml) | 16 frames | Future visual sequence represented during action training |
| Video latent timesteps (`state_t`) | [`config_video2world.py`](../mimic-video/model/cosmos_predict2/configs/config_video2world.py) | 6 | Temporal size of the VAE latent and DiT positional embeddings |
| Image sampling frequency | [`policy_io/lerobot.yaml`](../mimic-video/model/cosmos_predict2/configs/dataloading/policy_io/lerobot.yaml) | 5 Hz | Sampling rate for observation and future image sequences |
| Joint observation horizon | [`policy_io/lerobot.yaml`](../mimic-video/model/cosmos_predict2/configs/dataloading/policy_io/lerobot.yaml) | 1 | Current robot joint state |
| Predicted action horizon | [`policy_io/lerobot.yaml`](../mimic-video/model/cosmos_predict2/configs/dataloading/policy_io/lerobot.yaml) | 15 actions | Number of low-dimensional actions predicted per chunk |
| Predicted action frequency | [`policy_io/lerobot.yaml`](../mimic-video/model/cosmos_predict2/configs/dataloading/policy_io/lerobot.yaml) | 10 Hz | Execution rate of the predicted action sequence |

The active temporal configuration is:

```text
5 observation frames + 16 future frames = 21 video frames
21 video frames -> 6 VAE latent timesteps -> state_t = 6
```

The 61-frame entries in `data_video.py` belong to upstream Bridge and LIBERO
datasets. The SO-101 datasets (`ex1_all_v4`, `ex2_all_v4`, and `ex3_all`) use
21 frames.

## Required Constraints

### Video Clip Length

The Video2World clip length must match the total image horizon used by the
action model:

```text
num_frames = obs.workspace_rgb.horizon + action.workspace_rgb.horizon
```

For the current configuration:

```text
21 = 5 + 16
```

Cosmos VAE clips must have a length of the form `1 + 4n`. The corresponding
latent length is:

```text
state_t = 1 + (num_frames - 1) / 4
```

For a 21-frame clip, this produces six latent timesteps.

### Video Checkpoint Compatibility

`state_t` is part of the Video2World model configuration and positional
embedding shape. An action-model run must therefore use a video checkpoint
trained with the same clip length and `state_t`.

The action decoder does not configure its cross-attention sequence length
separately. It consumes the temporal features produced by the video DiT.

### Image Sampling Rate

The observation and future image frequencies in `policy_io/lerobot.yaml` must
match the effective sampling rate used during Video2World finetuning. The
SO-101 configuration currently samples image sequences at 5 Hz.

Changing this rate requires regenerating the training data and retraining the
video backbone and action model.

## Independently Tunable Parameters

The following parameters are not tied to the Video2World latent length:

- `obs.joint_pos_lowdim.horizon` controls how much joint-state history is read.
- `obs.joint_pos_lowdim.target_frequency` controls joint-state interpolation.
- `action.joint_action_lowdim.horizon` controls the number of predicted actions.
- `action.joint_action_lowdim.target_frequency` controls the action execution
  rate.

Changing an action output horizon or frequency changes the action-model target
and requires rebuilding its caches and retraining the action decoder, but does
not require changing the video DiT architecture.

## Changing the Video Clip Length

To use a different clip length:

1. Choose a frame count of the form `1 + 4n`.
2. Set `num_frames` for the SO-101 datasets in
   [`data_video.py`](../mimic-video/model/cosmos_predict2/configs/defaults/data_video.py).
3. Set `state_t = 1 + (num_frames - 1) / 4` in
   [`config_video2world.py`](../mimic-video/model/cosmos_predict2/configs/config_video2world.py).
4. Finetune a new Video2World checkpoint with that configuration.
5. Update the image horizons in
   [`policy_io/lerobot.yaml`](../mimic-video/model/cosmos_predict2/configs/dataloading/policy_io/lerobot.yaml)
   so their sum equals `num_frames`.
6. Point `VIDEO_DIT_PATH` at the compatible Video2World checkpoint.
7. Re-run [`precompute_video_latents.sh`](../scripts/precompute_video_latents.sh)
   to rebuild the latent cache.
8. Retrain the action decoder.

The latent precomputation script derives its output shape from the active
Video2World pipeline configuration, so no separate hard-coded latent shape
needs to be updated.
