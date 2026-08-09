"""Generic Selenium browser automation helpers.

The original production system drove a JSF portal with Selenium: search a
record, open a section, download an attachment. This module keeps those
helpers generic (wait, click, read tables, switch tabs, download files) so
any document source behind a web UI can be automated with the same toolkit.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import List, Optional

from selenium.webdriver.common.by import By
from selenium.webdriver.remote.webdriver import WebDriver
from selenium.webdriver.remote.webelement import WebElement
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait

logger = logging.getLogger(__name__)


def wait_for_element(
    driver: WebDriver,
    selector: str,
    by: str = By.CSS_SELECTOR,
    timeout: float = 10.0,
) -> Optional[WebElement]:
    """Wait for an element to be present and return it."""
    try:
        return WebDriverWait(driver, timeout).until(
            EC.presence_of_element_located((by, selector))
        )
    except Exception:  # noqa: BLE001 - Selenium raises several exception types
        logger.warning("Element not found within %.1fs: %s", timeout, selector)
        return None


def safe_click(
    driver: WebDriver,
    selector: str,
    by: str = By.CSS_SELECTOR,
    timeout: float = 10.0,
) -> bool:
    """Click an element, waiting for it to be clickable."""
    try:
        element = WebDriverWait(driver, timeout).until(
            EC.element_to_be_clickable((by, selector))
        )
        element.click()
        return True
    except Exception:  # noqa: BLE001
        logger.warning("Could not click element: %s", selector)
        return False


def table_texts(driver: WebDriver, table_id: str, column_index: int = 0) -> List[str]:
    """Read the text of one column from a table identified by id."""
    try:
        rows = driver.find_elements(By.CSS_SELECTOR, f"#{table_id} tr")
        texts = []
        for row in rows:
            cells = row.find_elements(By.TAG_NAME, "td")
            if len(cells) > column_index:
                texts.append(cells[column_index].text.strip())
        return texts
    except Exception:  # noqa: BLE001
        logger.warning("Could not read table %s", table_id)
        return []


def switch_to_new_tab(driver: WebDriver, timeout: float = 10.0) -> bool:
    """Switch to the most recently opened browser tab."""
    try:
        WebDriverWait(driver, timeout).until(
            lambda d: len(d.window_handles) > 1
        )
        driver.switch_to.window(driver.window_handles[-1])
        return True
    except Exception:  # noqa: BLE001
        return False


def close_extra_tabs(driver: WebDriver) -> None:
    """Close all tabs except the first and return to it."""
    handles = driver.window_handles
    if len(handles) > 1:
        driver.close()
        driver.switch_to.window(handles[0])


def download_latest_file(download_dir: str, before: Optional[List[str]] = None) -> Optional[Path]:
    """Return the newest file in a download directory.

    Args:
        download_dir: Directory where the browser saves downloads.
        before: Optional list of filenames seen before the download; the
            newest file not in that list is returned.

    Returns:
        Path of the newest file, or None when the directory is empty.
    """
    directory = Path(download_dir)
    if not directory.exists():
        return None

    candidates = [
        p for p in directory.iterdir()
        if p.is_file() and (before is None or p.name not in before)
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def wait_for_download(
    download_dir: str,
    before: Optional[List[str]] = None,
    timeout: float = 30.0,
    poll_interval: float = 1.0,
) -> Optional[Path]:
    """Wait until a new file appears in the download directory."""
    elapsed = 0.0
    while elapsed < timeout:
        latest = download_latest_file(download_dir, before)
        if latest is not None:
            return latest
        time.sleep(poll_interval)
        elapsed += poll_interval
    logger.warning("No new download after %.1fs", timeout)
    return None
