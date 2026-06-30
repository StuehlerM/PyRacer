# JEPA (Joint Embedding Predictive Architecture) package
#
# A self-supervised alternative to reward-based RL.
# Learns a world model and plans toward goals without rewards.

from .agent import JEPAAgent
from .model import StateEncoder, Predictor, create_target_encoder, vicreg_loss
from .memory import TransitionBuffer, GoalBuffer
