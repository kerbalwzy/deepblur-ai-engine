import os

PROJECT_NAME = "deepblur-ai-engine"
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
ASSETS_DIR = os.path.join(PROJECT_DIR, "assets")
# 
SERVER_ADDRESS = "[::]:25629"
LOG_DIR = os.path.join(PROJECT_DIR, "logs")
LOG_PATH = os.path.join(LOG_DIR, "log.txt")

