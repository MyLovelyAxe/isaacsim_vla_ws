import os
import torch
from typing import Optional
from pathlib import Path
from dataclasses import dataclass
from safety_estimator.data.dataloader import (
    TrajectoryDataLoader,
    TrajectoryDataSet,
    DEFAULT_DATASET_PATH,
)
from safety_estimator.utils.metrics import compute_metrics
from safety_estimator.model.safety_estimator_network import (
    SafetyEstimatorNetwork,
)
from safety_estimator.utils.tools import (
    save_model,
    load_model,
    log_epoch,
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
    """The dimension of encoded input for both history and proposed action."""
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
    load_model_pt: Optional[Path] = None
    """The path of .pt of a pretrained model for testing."""
    log_process: bool = True
    """Whether save the log process of training and validation into .json."""
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
        best_auc = 0.0
        for epoch in range(self.train_epoch):

            train_loss = 0.0
            for param_group in self.optimizer.param_groups:
                current_lr = param_group['lr']

            for batch_idx in range(self.train_set.batch_num):

                self.model.zero_grad()

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
                loss.backward()
                train_loss += loss.item()
                _ = self.optimizer.step()

            train_loss = train_loss / self.train_set.batch_num
            self.lr_schedulor.step()

            epoch_info = {"epoch": epoch, "lr": current_lr, "Train loss": train_loss, }
            print(f"End of epoch {epoch}: ")
            print(f"Train loss {train_loss:5.3f} | Current lr {current_lr:5.4f} |")

            valid_metrics = self.validate()
            print(
                "Valid loss {valid_loss:5.3f} | "
                "Valid F1: {f1:5.3f} | Valid Accuracy: {accuracy:5.3f} | "
                "Valid Recall: {recall:5.3f} | Valid AUC: {auc:5.3f} |"
                .format(**valid_metrics),flush=True,
            )
            epoch_info.update(valid_metrics)

            if valid_metrics["auc"] > best_auc:
                best_auc = valid_metrics["auc"]
                save_model(
                    model=self.model, 
                    epoch=epoch,
                    best_auc=best_auc,
                    mean_q=self.train_set.mean_q,
                    std_q=self.train_set.std_q,
                    mean_a=self.train_set.mean_a,
                    std_a=self.train_set.std_a,
                    eps=self.train_set.eps,
                    history_len=self.history_len,
                    encoded_dim=self.encoded_dim,
                    output_path=self.output_path, 
                    log_id=self.log_id,
                )

            if self.log_process:
                log_epoch(
                    epoch_info=epoch_info, 
                    log_path=self.output_path / f"{self.log_id}.json",
                )


    def validate(self):

        self.model.eval()
        valid_loss = 0.0
        risk_score_lst = list()
        risk_label_lst = list()

        with torch.no_grad():

            for batch_idx in range(self.valid_set.batch_num):

                history, prop_act, risk_label = self.valid_set.get_batch(batch_idx)
                history = history.to(self.device)
                prop_act = prop_act.to(self.device)
                risk_label = risk_label.to(self.device) # shape (batch_size, 1)
                
                risk_score = self.model(history, prop_act) # shape (batch_size, 1)
                loss = self.criterion(risk_score, risk_label)

                valid_loss += loss.item()
                risk_score_lst.append(risk_score)
                risk_label_lst.append(risk_label)

        valid_loss = valid_loss / self.valid_set.batch_num
        all_risk_scores = torch.cat(risk_score_lst, dim=0)
        all_risk_labels = torch.cat(risk_label_lst, dim=0)

        valid_metrics = dict(valid_loss=valid_loss)
        valid_metrics.update(compute_metrics(
            risk_score=all_risk_scores.detach().cpu().numpy(), 
            risk_label=all_risk_labels.detach().cpu().numpy(),
        ))

        return valid_metrics


    def test(self):
        """Test a pretrained weights on testing set."""

        self.model = load_model(
            model=self.model,  
            device=self.device, 
            load_model_pt=self.load_model_pt,
        )
        test_loss = 0.0
        risk_score_lst = list()
        risk_label_lst = list()

        with torch.no_grad():

            for batch_idx in range(self.test_set.batch_num):

                history, prop_act, risk_label = self.test_set.get_batch(batch_idx)
                history = history.to(self.device)
                prop_act = prop_act.to(self.device)
                risk_label = risk_label.to(self.device) # shape (batch_size, 1)
                
                risk_score = self.model(history, prop_act) # shape (batch_size, 1)
                loss = self.criterion(risk_score, risk_label)

                test_loss += loss.item()
                risk_score_lst.append(risk_score)
                risk_label_lst.append(risk_label)

        test_loss = test_loss / self.test_set.batch_num
        all_risk_scores = torch.cat(risk_score_lst, dim=0)
        all_risk_labels = torch.cat(risk_label_lst, dim=0)

        test_metrics = dict(test_loss=test_loss)
        test_metrics.update(compute_metrics(
            risk_score=all_risk_scores.detach().cpu().numpy(), 
            risk_label=all_risk_labels.detach().cpu().numpy(),
        ))

        print(
            "Test loss {test_loss:5.3f} | "
            "Test F1: {f1:5.3f} | Test Accuracy: {accuracy:5.3f} | "
            "Test Recall: {recall:5.3f} | Test AUC: {auc:5.3f} |"
            .format(**test_metrics),flush=True,
        )

        if self.log_process:
            loaded_model_log_id = self.load_model_pt.name.replace(".pt", "")
            log_epoch(
                epoch_info=test_metrics, 
                # store in the same json with training and validation log
                log_path=self.output_path / f"{loaded_model_log_id}.json",
            )

if __name__ == "__main__":

    exp = SafetyEsimatiorExp(
        dataset_path=Path("~/isaacsim_vla_ws/record").expanduser(),
        train_epoch=10,
        verbose=False,
    )
    exp.train()
