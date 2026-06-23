import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))

# Keep legacy ABounD imports working for VVCLIP_lib, prompt_generator, and utils.
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)
from . import VVCLIP_lib
from . import prompt_generator
from . import utils
