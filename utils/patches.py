import json
import wandb
from utils.logging import configure_logging

logger = configure_logging()

WANDB_ON = False
def log_stats(stats):
    if WANDB_ON:
        wandb.log(stats)
    else:
        logger.info(json.dumps(stats))