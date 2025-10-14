import json
import wandb
from utils.log_config import configure_logging

logger = configure_logging()

WANDB_ON = True
def log_stats(stats) -> None:
    if WANDB_ON:
        try:
            # Check if wandb is initialized
            if not wandb.run:
                logger.warning("Wandb is not initialized, falling back to console logging")
                logger.info(json.dumps(stats))
                return
            
            logger.info(f"Logging to wandb: {stats}")
            wandb.log(stats)
            logger.info("Successfully logged to wandb")
        except Exception as e:
            logger.error(f"Failed to log to wandb: {e}")
            logger.info(json.dumps(stats))
    else:
        logger.info(json.dumps(stats))