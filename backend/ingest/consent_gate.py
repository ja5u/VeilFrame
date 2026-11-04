"""Validates explicit consent before media processing."""


def verify_consent(consent_flag: bool, filename: str) -> None:
    """Reject processing unless the uploader explicitly confirms consent."""
    if consent_flag is not True:
        raise ValueError("Explicit user consent is required to process media.")


def check_consent_flag(file_path: str, consent_flag: bool) -> bool:
    """Compatibility wrapper for callers using the original scaffold API."""
    verify_consent(consent_flag, file_path)
    return True
