import json
import wandb
from utils.log_config import configure_logging

logger = configure_logging()

WANDB_ON = True
def log_stats(stats):
    if WANDB_ON:
        wandb.log(stats)
    else:
        logger.info(json.dumps(stats))