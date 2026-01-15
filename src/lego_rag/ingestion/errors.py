class IngestionError(Exception):
    """Base class for ingestion-related errors."""
    pass


class UnsupportedFileTypeError(IngestionError):
    pass


class FileReadError(IngestionError):
    pass


class EmptyDocumentError(IngestionError):
    pass
