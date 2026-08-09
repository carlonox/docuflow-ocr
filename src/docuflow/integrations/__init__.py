"""Integrations: browser automation, spreadsheets, local storage."""

from docuflow.integrations.local_storage import LocalStorage  # noqa: F401
from docuflow.integrations.spreadsheet import SpreadsheetHandler  # noqa: F401

__all__ = ["LocalStorage", "SpreadsheetHandler"]
