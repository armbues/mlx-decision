"""Training: fit a decision model to labelled data, written as a fitted model.

``fit(base, data, output=...)`` runs the stages and reports progress as
``Event``s; a fitted model folder is the result, loaded like any model.
``evaluate(model, data, compare=...)`` scores models on labelled data.
"""

from .evaluation import EvalResult, ModelEval, evaluate
from .events import Cancelled, Event
from .fitting import FitResult, fit

__all__ = ["Cancelled", "EvalResult", "Event", "FitResult", "ModelEval", "evaluate", "fit"]
