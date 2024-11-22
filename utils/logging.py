import logging
import structlog

# Configure standard logging
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# Configure structlog
structlog.configure(
    processors=[
        structlog.processors.KeyValueRenderer(
            key_order=["event", "model", "data_size"]
        ),
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
)


def get_structlog_logger():
    return structlog.get_logger()
