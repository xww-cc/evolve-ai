"""
视觉模块 - Evolve-AI 视觉进化系统

提供视觉理解、推理和创造能力的进化模块
"""

from .vision_encoder import VisionEncoder
from .visual_reasoning import VisualReasoning
from .spatial_understanding import SpatialUnderstanding
from .vision_evolution import VisionEvolution

__all__ = [
    'VisionEncoder',
    'VisualReasoning', 
    'SpatialUnderstanding',
    'VisionEvolution'
]

__version__ = '1.0.0'
__author__ = 'Evolve-AI Team' 