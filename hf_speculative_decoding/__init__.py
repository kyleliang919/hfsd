from transformers.generation.utils import GenerationMixin
from .speculative_decoding import speculative_generate
from .logits_processor import *
GenerationMixin.speculative_generate = speculative_generate
__version__ = "0.1.0"