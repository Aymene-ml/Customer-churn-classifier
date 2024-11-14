import logging
import os
from datetime import datetime

# Generate a log filename with the current timestamp
LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
logs_path = os.path.join(os.getcwd(), 'logs')
os.makedirs(logs_path, exist_ok=True)

LOG_FILE_PATH = os.path.join(logs_path, LOG_FILE)

# Configure logging
logging.basicConfig(
    level=logging.INFO,                     
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    filename=LOG_FILE_PATH,                # Use dynamic log file path
    filemode='w'                           
)


