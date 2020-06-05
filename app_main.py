"""App Main

python3 app_main
"""
import os
from ml.framework.file_utils import FileUtils
FileUtils.mkdir('raw')
FileUtils.mkdir('clean')
FileUtils.mkdir('knn')
from ml.ux import index

if __name__ == '__main__':
    ""
