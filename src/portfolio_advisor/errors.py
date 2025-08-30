class PortfolioAdvisorError(Exception):
    """Base exception for PortfolioAdvisor."""


class ConfigurationError(PortfolioAdvisorError):
    """Raised when configuration is invalid or incomplete."""


class InputOutputError(PortfolioAdvisorError):
    """Raised for input/output related failures."""
