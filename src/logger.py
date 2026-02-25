import os
import logging as lg
from datetime import datetime

logger = lg.getLogger(__name__)
logger.setLevel(lg.DEBUG)

LOG_FILE = f"{datetime.now().strftime('%m_%d_%y_%H_%M_%S')}.log"
logs_path = os.path.join(os.getcwd(), "Logs")
os.makedirs(logs_path, exist_ok=True)

LOG_FILE_PATH = os.path.join(logs_path, LOG_FILE)

## Creating handler and its formatter
handler = lg.FileHandler(LOG_FILE_PATH)
formatter = lg.Formatter(
    "[%(asctime)s] %(name)s %(levelno)s - %(levelname)s %(message)s"
)
handler.setFormatter(formatter)

## adding handler to the logger
logger.addHandler(handler)
