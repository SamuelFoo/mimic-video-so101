hf download robot-learning/Ex1_attempt_1 \
  --repo-type dataset \
  --local-dir /cluster/scratch/samfoo/robot_learning/data/ex1_attempt_1

hf download robot-learning/Ex1_attempt_2 \
  --repo-type dataset \
  --local-dir /cluster/scratch/samfoo/robot_learning/data/ex1_attempt_2

conda activate lerobot
pip install 'zarr<3' 'numcodecs<0.16'

python mimic-video/data_preprocessing/action/process_lerobot.py \
  --repo-id ex1_merged \
  --root /cluster/scratch/samfoo/robot_learning/data/ex1_merged \
  --output-dir /cluster/scratch/samfoo/robot_learning/data/ex1_merged-zarr \
  --overwrite
