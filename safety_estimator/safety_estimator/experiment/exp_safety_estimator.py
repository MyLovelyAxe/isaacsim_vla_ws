import os
import torch
from pathlib import Path
from dataclasses import dataclass
from safety_estimator.data.dataloader import (
    TrajectoryDataLoader,
    TrajectoryDataSet,
    DEFAULT_DATASET_PATH,
)
from safety_estimator.model.safety_estimator_network import (
    SafetyEstimatorNetwork,
)

DEFAULT_OUTPUT_PATH = Path(__file__).parent.parent.resolve() / "checkpoints"

@dataclass
class SafetyEsimatiorExp:
    """Manage the complete training & validation & testing for safety estimator."""


    dataset_path: Path = DEFAULT_DATASET_PATH
    """The folder that contains the recorded trajectories."""
    history_len: int = 10
    """The window length of history of joint states Q and executed actions A, i.e. t-N, ..., t."""
    future_len: int = 10
    """The horizon length of future of joint states Q and executed actions A, i.e. t, ..., t+M."""
    encoded_dim: int = 16
    """the dimension of encoded input for both history and proposed action."""
    train_set_ratio: float = 0.8
    """Ratio of training set in the whole dataset"""
    valid_set_ratio: float = 0.1
    """Ratio of validation set in the whole dataset"""
    test_set_ratio: float = 0.1
    """Ratio of testing set in the whole dataset"""
    batch_size: int = 32
    """Number of samples for one single batch."""
    warmup_len: int = 30 # TODO: this is a temporary compromise, remove it later
    """The number of timestamp at beginning of each trajectory to warm-up, difference beween Q and A is large but still considered as safe."""
    motion_stuck_risk_thr: float = 0.8
    """Threshold for determining risk label based on motion stuck."""
    train_epoch: int = 50
    """Number of epochs to train the network."""
    device_choice: str = "cuda:0"
    """The device choice for training or inference, i.e. cpu or gpu."""
    learning_rate: float = 3e-4
    """Learning rate for optimizer to update gradients."""
    output_path: Path = DEFAULT_OUTPUT_PATH
    """The path to store the trained checkpoints."""
    log_id: str = ""
    """Unique ID for one single experiment."""
    examine_mode: bool = False
    """In examine mode, the returned samples are not yet normalized."""
    verbose: bool = False
    """If True: log the process of creating dataloader."""


    @property
    def train_set(self) -> TrajectoryDataSet:
        """Dataset for training."""
        return self.dataloader.train_set
    
    @property
    def valid_set(self) -> TrajectoryDataSet:
        """Dataset for validation."""
        return self.dataloader.valid_set

    @property
    def test_set(self) -> TrajectoryDataSet:
        """Dataset for testing."""
        return self.dataloader.test_set


    def __post_init__(self):

        os.makedirs(self.output_path, exist_ok=True)
        self.device = self.select_device()
        self.dataloader = self.create_dataloader()
        self.model = self.create_model().to(self.device)
        self.criterion = self.create_loss().to(self.device)
        self.optimizer = self.create_optimizer()
        self.lr_schedulor = self.create_lr_schedulor()


    def select_device(self):
        """Check availability of GPU and choose device."""

        if self.device_choice == "cuda:0":
            if torch.cuda.is_available():
                print(f"GPU available, choose {self.device_choice}.")
                return torch.device(self.device_choice)
            else:
                print("No cuda GPU available, force to use CPU.")
                return torch.device("cpu")
        elif self.device_choice == "cpu":
            print("Choose to use CPU.")
            return torch.device("cpu")


    def create_dataloader(self) -> TrajectoryDataLoader:
        """The whole data including training, validation and testing sets."""

        return TrajectoryDataLoader(
            dataset_path=self.dataset_path,
            history_len=self.history_len,
            future_len=self.future_len,
            train_set_ratio=self.train_set_ratio,
            valid_set_ratio=self.valid_set_ratio,
            test_set_ratio=self.test_set_ratio,
            batch_size=self.batch_size,
            warmup_len=self.warmup_len,
            motion_stuck_risk_thr=self.motion_stuck_risk_thr,
            examine_mode=self.examine_mode,
            verbose=self.verbose,
        )


    def create_model(self) -> SafetyEstimatorNetwork:
        """The network to estimate risk score."""

        return SafetyEstimatorNetwork(
            history_len=self.history_len,
            encoded_dim=self.encoded_dim,
        )


    def create_loss(self):
        """Loss funtion."""

        return torch.nn.BCEWithLogitsLoss()
        
    
    def create_lr_schedulor(self):
        """The schedular to update learning rate for each epoch."""

        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer=self.optimizer,
            T_max=self.train_epoch,
            eta_min=0.0,
        )


    def create_optimizer(self):
        """Optimizer to update gradients."""

        return torch.optim.Adam(
            params=self.model.parameters(), 
            lr=self.learning_rate, 
            betas=(0.9, 0.999), 
            weight_decay=1e-5,
        )


    def train(self):

        self.model.train()

        for epoch in range(self.train_epoch):

            epoch_loss = 0.0
            for param_group in self.optimizer.param_groups:
                current_lr = param_group['lr']

            for batch_idx in range(self.train_set.batch_num):

                history, prop_act, risk_label = self.train_set.get_batch(batch_idx)
                history = history.to(self.device)
                prop_act = prop_act.to(self.device)
                risk_label = risk_label.to(self.device)

                if self.verbose:
                    print(f"batch {batch_idx}:")
                    print(f"history shape: {history.shape}")
                    print(f"prop_act shape: {prop_act.shape}")
                    print(f"risk_label shape: {risk_label.shape}")
                
                risk_score = self.model(history, prop_act)
                loss = self.criterion(risk_score, risk_label)

                # TODO: remove
                # get a parameter to check if it is actually updated
                # p0 = next(self.model.parameters())
                # before = p0.data.flatten()[0].item()


                self.model.zero_grad()
                loss.backward()
                epoch_loss += loss.item()
                _ = self.optimizer.step()


                # TODO: remove
                # after = p0.data.flatten()[0].item()
                # print("param delta:", after - before)

            epoch_loss = epoch_loss / self.train_set.batch_num
            self.lr_schedulor.step()

            print('training: ')
            print(f'End of epoch {epoch}: current lr {current_lr:5.4f} | epoch_loss {epoch_loss:5.4f} ')


    def validate(self):

        # TODO:
        self.model.eval()


    def test(self):

        # TODO:
        self.model.eval()

if __name__ == "__main__":

    exp = SafetyEsimatiorExp(
        dataset_path=Path("~/isaacsim_vla_ws/record").expanduser(),
        train_epoch=10,
        verbose=False,
    )
    exp.train()
