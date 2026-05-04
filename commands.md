# Sample commands for lerobot

Set `repo_id` as absolute paths unless you want to save it to your huggingface cache.

## Train

```bash
lerobot-train \
  --dataset.repo_id=/home/arc_user/Sorting-piper-robot-server/data/2026-04-11_10-47-28_video \
  --policy.type=act \
  --output_dir=outputs/train/test_single_arm \
  --job_name=act_single_arm \
  --policy.device=cuda \
  --wandb.enable=true \
  --policy.repo_id=${HF_USER}/act_policy
```

## Merge

```bash
lerobot-edit-dataset \
  --new_repo_id /home/dell/workspaces/robot_learning_project/data/20260428/record-test-merged \
  --repo_id /home/dell/workspaces/robot_learning_project/data/20260428/record-test-merged \
  --operation.type merge \
  --operation.repo_ids "['/home/dell/workspaces/robot_learning_project/data/20260428/record-test-2', '/home/dell/workspaces/robot_learning_project/data/20260428/record-test-3', '/home/dell/workspaces/robot_learning_project/data/20260428/record-test-4']"
```

## Delete Episodes

```bash
lerobot-edit-dataset \
  --repo_id /home/arc_user/Sorting-piper-robot-server/data/2026-04-08_21-33-57 \
  --operation.type delete_episodes \
  --operation.episode_indices "[4]"
```

## Evaluate

```bash
lerobot-record \
  --robot.type=so101_follower \
  --robot.port=/dev/ttyACM0 \
  --robot.id=my_awesome_follower_arm \
  --robot.cameras="{ front: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}}" \
  --display_data=true \
  --dataset.repo_id=${HF_USER}/eval_act_your_dataset \
  --dataset.root="${HOME}/workspaces/robot_learning_project/data/eval_act_your_dataset_1" \
  --dataset.fps=30 \
  --dataset.num_episodes=10 \
  --dataset.reset_time_s=0 \
  --dataset.single_task="Your task description" \
  --dataset.video=true \
  --dataset.streaming_encoding=true \
  --dataset.encoder_threads=2 \
  --dataset.vcodec=auto \
  --dataset.push_to_hub=false \
  --policy.path=${HOME}/workspaces/robot_learning_project/act_policy/checkpoints/last/pretrained_model
```
