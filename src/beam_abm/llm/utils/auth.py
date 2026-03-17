"""Authentication utilities for LLM ABM package."""

import google.auth
import google.auth.transport.requests

from beam_abm.common.logging import get_logger

logger = get_logger(__name__)


def get_google_auth_token():
    """Get a fresh Google Cloud authentication token."""
    logger.info("Attempting to get Google Cloud authentication token")
    try:
        creds, project = google.auth.default()
        logger.debug(f"Found default credentials for project: {project}")

        auth_req = google.auth.transport.requests.Request()
        creds.refresh(auth_req)

        logger.info("Successfully refreshed Google Cloud authentication token")
        logger.debug(f"Token expiry: {creds.expiry}")
        return creds.token
    except Exception as e:
        logger.error(f"Failed to get Google Cloud authentication token: {e}")
        raise
