"""
Módulo de utilidades para SDM-D Framework
"""

from .mask_utils import MaskProcessor
from .clip_utils import CLIPProcessor
from .label_utils import LabelGenerator
from .file_utils import FileManager
from .visualization_utils import VisualizationManager
from .logging_utils import SDMLogger, ProgressMonitor
from .avocado_analytics import AvocadoAnalytics

__all__ = [
    'MaskProcessor',
    'CLIPProcessor',
    'LabelGenerator',
    'FileManager',
    'VisualizationManager',
    'SDMLogger',
    'ProgressMonitor',
    'AvocadoAnalytics'
]

__version__ = '1.0.0'
__author__ = 'SDM-D Team'
__description__ = 'Utilidades modulares para el framework SDM-D con logging y analytics para avocados'