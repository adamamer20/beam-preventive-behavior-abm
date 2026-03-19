import os
from io import BytesIO

import pandas as pd
import requests

from beam_abm.common.logging import get_logger
from beam_abm.settings import API_CONFIG

logger = get_logger(__name__)


def load_raw_survey_df() -> pd.DataFrame:
    logger.info("Starting data loading process")

    # Share link and download link from centralized config
    share_url = API_CONFIG["share_url"]
    download_url = API_CONFIG["download_url"]

    if not share_url or not download_url:
        logger.warning("Share/Download URLs not configured; attempting local cached file")
        local_path = os.getenv("RAW_SURVEY_CSV", "data/1_CROSSCOUNTRY_translated.csv")
        if os.path.exists(local_path):
            logger.info(f"Loading local cached survey CSV: {local_path}")
            df = pd.read_csv(local_path)
            logger.success(f"Loaded cached CSV with {len(df)} rows and {len(df.columns)} columns from {local_path}")
            return df
        logger.error("Share URL and download URL must be configured or a local cached CSV must exist")
        raise ValueError("Share URL and download URL must be configured")

    logger.debug(f"Using share URL: {share_url[:50]}...")
    logger.debug(f"Using download URL: {download_url[:50]}...")

    # Step 1: Get the cookie from the share link
    logger.info("Step 1: Getting cookies from share link")
    session = requests.Session()

    try:
        response_share = session.get(share_url, allow_redirects=False)  # Disable redirects to inspect headers
        logger.debug(f"Share link response status: {response_share.status_code}")
    except requests.RequestException as e:
        logger.error(f"Failed to access share link: {e}")
        raise

    if response_share.status_code == 302:  # HTTP 302 indicates a redirect
        logger.info("Successfully received redirect response, extracting cookies")
        # Save the cookies from the response
        cookies = session.cookies
        logger.debug(f"Extracted {len(cookies)} cookies")

        # Step 2: Use the cookies to download the file
        logger.info("Step 2: Downloading file using cookies")
        try:
            response_download = session.get(download_url, cookies=cookies)
            logger.debug(f"Download response status: {response_download.status_code}")
            logger.debug(f"Content length: {len(response_download.content)} bytes")
        except requests.RequestException as e:
            logger.error(f"Failed to download file: {e}")
            raise

        # Check if the request was successful
        if response_download.status_code == 200:
            logger.info("File downloaded successfully, parsing as CSV")
            try:
                # Attempt to parse as CSV
                data = pd.read_csv(BytesIO(response_download.content))
                logger.success(f"Successfully loaded CSV with {len(data)} rows and {len(data.columns)} columns")
                logger.debug(f"DataFrame shape: {data.shape}")
                logger.debug(f"Column names: {list(data.columns)[:10]}...")  # Log first 10 columns
                return data
            except pd.errors.ParserError as e:
                logger.error(f"Failed to parse CSV data: {e}")
                raise
            except Exception as e:
                logger.error(f"Unexpected error while parsing CSV: {e}")
                raise
        else:
            logger.error(f"Failed to download the file. HTTP Status Code: {response_download.status_code}")
            raise requests.HTTPError(f"Download failed with status {response_download.status_code}")
    else:
        logger.error(f"Failed to get cookies from share link. HTTP Status Code: {response_share.status_code}")
        raise requests.HTTPError(f"Share link access failed with status {response_share.status_code}")

