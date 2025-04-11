# logging_config.py

import sys
import structlog
import logging

def configure_logging():
    # Basic logging configuration using the standard library
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=logging.INFO,  # Set log level here
    )

    # Structlog configuration
    structlog.configure(
        processors=[
            structlog.stdlib.add_log_level,  # Include log level in the output
            structlog.processors.TimeStamper(fmt="%Y-%m-%d %H:%M"),  # Timestamp with minute precision
            # structlog.processors.JSONRenderer(),  # Render logs as JSON
            structlog.processors.StackInfoRenderer(),  # Include stack info for exceptions
            structlog.processors.format_exc_info,  # Format exception information
            structlog.dev.ConsoleRenderer()
        ],
        context_class=dict,  # Use a standard dict for context
        logger_factory=structlog.stdlib.LoggerFactory(),  # Use standard logging
        wrapper_class=structlog.stdlib.BoundLogger,  # For bound logging
        cache_logger_on_first_use=True,  # Cache logger for performance
    )

    return structlog.get_logger()  # Return the logger instance
