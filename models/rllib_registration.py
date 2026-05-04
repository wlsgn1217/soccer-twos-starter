"""
Ray RLlib 1.4: ModelCatalog.register_custom_model checks tf.keras.Model even when
using PyTorch only; if TensorFlow is not installed, tf is None and registration
crashes. We register via tune's global registry (same backend as ModelCatalog)
without that check.
"""
from ray.tune.registry import RLLIB_MODEL, _global_registry

from models.privileged_actor_model import PrivilegedActorModel
from models.gru_student_model import GRUStudentPrivilegedCriticModel

PRIVILEGED_ACTOR_MODEL_NAME = "privileged_actor_model"
GRU_STUDENT_MODEL_NAME = "gru_student_privileged_critic_model"


def register_privileged_actor_model(name: str = PRIVILEGED_ACTOR_MODEL_NAME) -> None:
    _global_registry.register(RLLIB_MODEL, name, PrivilegedActorModel)


def register_gru_student_model(name: str = GRU_STUDENT_MODEL_NAME) -> None:
    _global_registry.register(RLLIB_MODEL, name, GRUStudentPrivilegedCriticModel)
