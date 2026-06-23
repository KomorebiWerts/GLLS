import sys
import os

# 1. 获取当前文件 (__init__.py) 所在的目录路径 (.../src/models/ABounD)
current_dir = os.path.dirname(os.path.abspath(__file__))

# 2. 将该目录加入到 sys.path 的最前面
# 这样 Python 就能直接找到该目录下的 VVCLIP_lib, prompt_generator 等模块
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)
from . import VVCLIP_lib
from . import prompt_generator
from . import utils
