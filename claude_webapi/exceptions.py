"""
Custom exceptions for claude_webapi.
"""


class ClaudeWebAPIError(Exception):
    """Base exception for all claude_webapi errors."""


class AuthenticationError(ClaudeWebAPIError):
    """Raised when session credentials are invalid or expired."""


class APIError(ClaudeWebAPIError):
    """Raised when the Claude.ai API returns an unexpected status code."""

    def __init__(self, message: str, status_code: int | None = None):
        super().__init__(message)
        self.status_code = status_code


class TimeoutError(ClaudeWebAPIError):
    """Raised when a request to Claude.ai times out."""


class ConversationNotFoundError(ClaudeWebAPIError):
    """Raised when a referenced conversation UUID does not exist."""


class FileUploadError(ClaudeWebAPIError):
    """Raised when a file upload fails."""


class QuotaExceededError(ClaudeWebAPIError):
    """Raised when the Claude.ai message limit has been reached."""

