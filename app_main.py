"""App Main

python3 app_main
"""
import os
from ml.framework.file_utils import FileUtils
from ml.ux import index

if __name__ == '__main__':
    FileUtils.mkdir('raw')
    index
