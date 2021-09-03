import os

PROJECT_DIR_PATH = os.path.dirname(__file__)
PROJECT_ROOT = os.path.dirname(PROJECT_DIR_PATH)

CACHE_PATH = os.path.join(PROJECT_ROOT, 'cache')
LOG_PATH = os.path.join(PROJECT_ROOT, 'logdir')

if not os.path.isdir(CACHE_PATH):
    os.mkdir(CACHE_PATH)
if not os.path.isdir(LOG_PATH):
    os.mkdir(LOG_PATH)


