'''
@Author: Clarence
@Date: 2024-10-8
@Description: This script is used to run the design process for the given stl.
'''

import os
import argparse
import subprocess
import sys
import time
import tqdm

project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_dir)

