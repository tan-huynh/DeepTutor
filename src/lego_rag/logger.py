import logging

# Configure root logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

# Shared logger instance for the entire project
logger = logging.getLogger("lego_rag")
