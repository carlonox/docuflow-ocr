"""Learning subsystem: the part of docuflow that makes it self-improving."""

from docuflow.learning.learning_engine import LearningEngine  # noqa: F401
from docuflow.learning.advanced_learning import AdvancedLearning  # noqa: F401
from docuflow.learning.precision_monitor import PrecisionMonitor  # noqa: F401
from docuflow.learning.manual_feedback import ManualFeedback  # noqa: F401

__all__ = [
    "LearningEngine",
    "AdvancedLearning",
    "PrecisionMonitor",
    "ManualFeedback",
]
