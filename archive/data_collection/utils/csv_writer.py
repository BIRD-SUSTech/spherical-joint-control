"""统一 CSV 写入器。"""

from __future__ import annotations

import csv
import threading
from pathlib import Path
from typing import List, Optional


class CsvWriter:
    """参数化的 CSV 写入器，替换重复的 ImuCsvLogger / MocapCsvLogger 模式。"""

    def __init__(
        self,
        csv_path: Path,
        fieldnames: List[str],
        start_event: Optional[threading.Event] = None,
        flush_interval: int = 10,
    ):
        self._path = csv_path
        self._fieldnames = fieldnames
        self._start_event = start_event
        self._flush_interval = flush_interval
        self._file = None
        self._writer = None
        self._row_count = 0
        self._is_open = False

    def open(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        is_new = not self._path.exists() or self._path.stat().st_size == 0
        self._file = open(
            self._path, "a", newline="", encoding="utf-8", buffering=1024 * 1024,
        )
        self._writer = csv.DictWriter(self._file, fieldnames=self._fieldnames)
        if is_new:
            self._writer.writeheader()
            self._file.flush()
        self._is_open = True

    def write_row(self, row: dict) -> bool:
        if not self._is_open:
            return False
        if self._start_event is not None and not self._start_event.is_set():
            return False
        # 只保留 fieldnames 中声明的列
        filtered = {k: row.get(k, "") for k in self._fieldnames}
        self._writer.writerow(filtered)
        self._row_count += 1
        if self._row_count % self._flush_interval == 0:
            self._file.flush()
        return True

    def flush(self) -> None:
        if self._file:
            self._file.flush()

    def close(self) -> None:
        if self._file:
            self._file.flush()
            self._file.close()
            self._is_open = False

    @property
    def row_count(self) -> int:
        return self._row_count
