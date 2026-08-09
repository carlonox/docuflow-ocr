"""Generic spreadsheet handler.

Reads rows and writes results to Google Sheets, CSV or JSON with the same
interface. The original production system used Google Sheets as its queue and
result store; this module keeps that pattern but makes the backend swappable
so pipelines can run offline or with any spreadsheet service.
"""

from __future__ import annotations

import csv
import json
import logging
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

logger = logging.getLogger(__name__)


class SpreadsheetHandler(ABC):
    """Base class for spreadsheet backends.

    Args:
        sheet_id: Identifier of the target sheet/table.
        sheet_name: Worksheet name.
    """

    def __init__(self, sheet_id: str, sheet_name: str = "Principal") -> None:
        self.sheet_id = sheet_id
        self.sheet_name = sheet_name

    @abstractmethod
    def get_all_rows(self) -> List[List[str]]:
        """Return all rows as lists of cell strings."""
        raise NotImplementedError

    @abstractmethod
    def update_row(self, row_number: int, columns: Dict[str, str]) -> None:
        """Write a mapping of column letter -> value into a row."""
        raise NotImplementedError

    # ------------------------------------------------------------------ #
    # Shared query helpers
    # ------------------------------------------------------------------ #

    def get_unprocessed_rows(
        self,
        result_columns: Sequence[str],
        start_row: int = 2,
        force_start_row: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Return rows where all result columns are empty.

        Args:
            result_columns: Column letters that mark a row as processed.
            start_row: First data row (1-indexed).
            force_start_row: When set, return rows from this row regardless
                of their processing state.

        Returns:
            List of dicts with ``row_number`` and ``data``.
        """
        rows = self.get_all_rows()
        unprocessed: List[Dict[str, Any]] = []

        for row_idx in range(start_row, len(rows) + 1):
            row = rows[row_idx - 1]
            if force_start_row or all(self._cell(row, col) == "" for col in result_columns):
                unprocessed.append({"row_number": row_idx, "data": row})
        return unprocessed

    def get_failed_rows(self, failure_column: str, marker: str) -> List[Dict[str, Any]]:
        """Return rows whose status column contains a failure marker."""
        rows = self.get_all_rows()
        failed: List[Dict[str, Any]] = []
        for row_idx, row in enumerate(rows[1:], start=2):
            if marker in self._cell(row, failure_column):
                failed.append({"row_number": row_idx, "data": row})
        return failed

    @staticmethod
    def _cell(row: Sequence[str], column_letter: str) -> str:
        """Read a cell by column letter (A=0, B=1, ...)."""
        index = sum(
            (ord(char) - 64) * (26 ** position)
            for position, char in enumerate(reversed(column_letter.upper()))
        ) - 1
        return row[index] if index < len(row) else ""


class GoogleSheetsHandler(SpreadsheetHandler):
    """Google Sheets backend using gspread.

    Credentials come from ``GOOGLE_APPLICATION_CREDENTIALS`` (service account
    JSON path) or the ``credentials_path`` config key. Never commit service
    account files to a repository.
    """

    def __init__(
        self,
        sheet_id: str,
        sheet_name: str = "Principal",
        credentials_path: Optional[str] = None,
    ) -> None:
        super().__init__(sheet_id, sheet_name)
        self.credentials_path = credentials_path or os.environ.get(
            "GOOGLE_APPLICATION_CREDENTIALS"
        )
        self._client = None
        self._worksheet = None

    def available(self) -> bool:
        return bool(self.credentials_path)

    def _lazy_load(self):
        if self._worksheet is not None:
            return
        if not self.credentials_path:
            raise FileNotFoundError(
                "Google Sheets credentials not configured; set "
                "GOOGLE_APPLICATION_CREDENTIALS or pass credentials_path"
            )
        import gspread
        from google.oauth2.service_account import Credentials

        credentials = Credentials.from_service_account_file(
            self.credentials_path,
            scopes=["https://www.googleapis.com/auth/spreadsheets"],
        )
        self._client = gspread.authorize(credentials)
        self._worksheet = self._client.open_by_key(self.sheet_id).worksheet(self.sheet_name)

    def get_all_rows(self) -> List[List[str]]:
        self._lazy_load()
        return self._worksheet.get_all_values()

    def update_row(self, row_number: int, columns: Dict[str, str]) -> None:
        self._lazy_load()
        for column, value in columns.items():
            if value:
                self._worksheet.update_acell(f"{column}{row_number}", value)


class CSVHandler(SpreadsheetHandler):
    """CSV backend: the offline alternative to Google Sheets.

    Args:
        path: CSV file path. Created when missing.
    """

    def __init__(self, path: str, sheet_name: str = "data") -> None:
        super().__init__(sheet_id=path, sheet_name=sheet_name)
        self.path = Path(path)

    def get_all_rows(self) -> List[List[str]]:
        if not self.path.exists():
            return []
        with self.path.open("r", encoding="utf-8", newline="") as fh:
            return list(csv.reader(fh))

    def update_row(self, row_number: int, columns: Dict[str, str]) -> None:
        rows = self.get_all_rows()
        while len(rows) < row_number:
            rows.append([])
        target = rows[row_number - 1]
        for column, value in columns.items():
            index = sum(
                (ord(char) - 64) * (26 ** position)
                for position, char in enumerate(reversed(column.upper()))
            ) - 1
            while len(target) <= index:
                target.append("")
            target[index] = value
        with self.path.open("w", encoding="utf-8", newline="") as fh:
            csv.writer(fh).writerows(rows)
        logger.info("CSV row %d updated", row_number)


class JSONHandler(SpreadsheetHandler):
    """JSON backend storing rows as a list of dicts.

    Args:
        path: JSON file path.
        columns: Ordered column names; row index maps to list position.
    """

    def __init__(self, path: str, columns: Sequence[str]) -> None:
        super().__init__(sheet_id=path, sheet_name="data")
        self.path = Path(path)
        self.columns = list(columns)

    def _load(self) -> List[Dict[str, Any]]:
        if not self.path.exists():
            return []
        with self.path.open("r", encoding="utf-8") as fh:
            return json.load(fh)

    def _save(self, rows: List[Dict[str, Any]]) -> None:
        with self.path.open("w", encoding="utf-8") as fh:
            json.dump(rows, fh, indent=2, ensure_ascii=False)

    def get_all_rows(self) -> List[List[str]]:
        return [
            [row.get(column, "") for column in self.columns]
            for row in self._load()
        ]

    def update_row(self, row_number: int, columns: Dict[str, str]) -> None:
        rows = self._load()
        while len(rows) < row_number:
            rows.append({})
        for column, value in columns.items():
            rows[row_number - 1][column] = value
        self._save(rows)
