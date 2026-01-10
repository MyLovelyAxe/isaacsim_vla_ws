
import torch
import torch.nn as nn
from typing import Tuple
from safety_estimator.utils.robot_config import SO100_JOINTS_LIMITS_RADIAN


class SafetyEstimatorNetwork(nn.Module):
    """The network estimating risk score based on history trajectory.
    
    Input of network:

        - history: 
            - definition: concatenated joint states and executed actions in the past timestamps
            - shape: (N, 12), (history_len, 2 x joints_len)
            - remark: joint states and executed actions have the same shape (N, 6), but concatenated
        - proposed next action:
            - definition: the proposed action from VLA model for next timestamp
            - shape: (6,), (joints_len,)
            - remark: the risk score is for how riskty this is
            
    Output of network:

        - risk score: 
            - definition: how risky a proposed next action is based on history
            - shape: (1,)
            - remark: the closer to 1, the more risky; vise versa
    """

    def __init__(
        self,
        history_len: int = 10,
        encoded_dim: int = 16,
    ):
        super().__init__()
        # the window length of history of joint states Q and executed actions A, i.e. t-N, ..., t
        self.history_len = history_len
        # the dimension of encoded input (both history and proposed action)
        self.encoded_dim = encoded_dim

        # create layers
        self.history_encoder = nn.Linear(
            in_features=self.history_feat_len,
            out_features=self.encoded_dim,
        )
        self.prop_act_encoder = nn.Linear(
            in_features=self.prop_act_feat_len,
            out_features=self.encoded_dim,
        )
        self.estimator = nn.Sequential(
            nn.Linear(
                in_features=self.N*self.encoded_dim+self.encoded_dim,
                out_features=128,
            ),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=256),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=16),
            nn.ReLU(),
            nn.Linear(in_features=16, out_features=1),
        )


    @property
    def N(self) -> int:
        """The window length of history along timestamp dimension."""
        return self.history_len

    @property
    def joints_len(self) -> int:
        """The number of joints of robot arm SO100, i.e. fixed 6"""
        return len(SO100_JOINTS_LIMITS_RADIAN)

    @property
    def history_feat_len(self) -> int:
        """The length of feature for history.
        
        A history sample is concatenated by:
            - joint states: feature length 6
            - executed actions: feature length 6
        i.e. fixed 12
        """
        return 2 * self.joints_len
    
    @property
    def history_sample_shape(self) -> Tuple[int]:
        """One single history sample should have this shape."""
        return (self.N, self.history_feat_len)
    
    @property
    def prop_act_feat_len(self) -> int:
        """The length of feature for proposed next action.
        
        i.e. the same with executed action feature length.
        """
        return self.joints_len
    
    @property
    def prop_act_sample_shape(self) -> Tuple[int]:
        """One single proposed next action sample should have this shape."""
        return (self.prop_act_feat_len)
    

    def forward(self, history, prop_act):
        # history shape: (batch_size, N, 12)
        # prop_act shape: (batch_size, 6)

        # input: (batch_size, N, 12)
        # output: (batch_size, N, encoded_dim), type: torch.float32
        encoded_history = self.history_encoder(history)

        # input: (batch_size, 6)
        # output: (batch_size, encoded_dim), type: torch.float32
        encoded_prop_act = self.prop_act_encoder(prop_act)

        # (batch_size, N, encoded_dim) -> (batch_size, N x encoded_dim), type: torch.float32
        flatten_encoded_history = encoded_history.reshape(-1, self.N * self.encoded_dim)

        # encoded_input: (batch_size, N x encoded_dim + encoded_dim), type: torch.float32
        encoded_input = torch.concat([
            flatten_encoded_history,
            encoded_prop_act,
        ], dim=1)

        # risk_score: shape: (batch_size, 1), type: torch.float32
        risk_score = self.estimator(encoded_input)

        return risk_score


if __name__ == "__main__":

    network = SafetyEstimatorNetwork(
        history_len=10,
    )
    print(f"Correct shape of proposed next action: {network.prop_act_sample_shape}")
    
    dummy_history = torch.rand([32, 10, 12])
    dummy_prop_act = torch.rand([32, 6])
    dummy_risk_score = network(dummy_history, dummy_prop_act)
    print(f"dummy_risk_score: shape({dummy_risk_score.shape}, type: {dummy_risk_score.dtype})")