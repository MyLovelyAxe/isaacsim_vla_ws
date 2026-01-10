from safety_estimator.data.dataloader import TrajectoryDataLoader
from pathlib import Path

dataloader = TrajectoryDataLoader(
    dataset_path=Path("~/isaacsim_vla_ws/record").expanduser(),
    history_len=10,
    future_len=10,
    train_set_ratio=0.8,
    valid_set_ratio=0.1,
    test_set_ratio=0.1,
    batch_size=32,
    examine_mode=False,
    verbose=False,
)

# training

train_set = dataloader.train_set
for batch_idx in range(train_set.batch_num):
    input, gt = train_set.get_batch(batch_idx)
    print(f"batch {batch_idx}:")
    print(f"input shape: {input.shape}")
    print(f"ground truth shape: {gt.shape}")