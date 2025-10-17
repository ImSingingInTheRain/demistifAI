"""Page exports for the demistifAI multi-page app."""

from .home import home
from .start_your_machine import start_your_machine
from .prepare_data import prepare_data
from .train import train
from .evaluate import evaluate
from .use import use
from .model_card import model_card

__all__ = [
    "home",
    "start_your_machine",
    "prepare_data",
    "train",
    "evaluate",
    "use",
    "model_card",
]
