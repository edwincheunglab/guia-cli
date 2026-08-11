"""Errors raised by GUIA CLI's local A2A transport."""


class A2AServiceError(RuntimeError):
    """Raised when the local A2A service cluster cannot start or stop safely."""


class A2ADispatchError(RuntimeError):
    """Raised when an A2A domain-agent request fails."""


__all__ = ["A2ADispatchError", "A2AServiceError"]
