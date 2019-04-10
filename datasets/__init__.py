import os
import sys
_currDir = os.path.dirname(os.path.abspath(__file__))
BASE_DATA_FOLDER = os.path.join(_currDir, 'saved_datasets')
PATH_PERL_SCRIPTS_FOLDER = os.path.abspath(os.path.join(_currDir, '..', 'perl_scripts'))

try:
    os.mkdir(BASE_DATA_FOLDER)
except:pass

from .CIFAR10 import CIFAR10
from .ImageNet12 import ImageNet12

__all__ = ('CIFAR10', 'ImageNet12')