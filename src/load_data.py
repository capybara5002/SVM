"""Doc va kiem tra bo du lieu ReINTEL."""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import pandas as pd

from . import config


TEXT_COLUMN_CANDIDATES = {
    "text",
    "content",
    "sentence",
    "post",
    "post_text",
    "message",
    "news",
    "title",
    "noi_dung",
    "van_ban",
}
LABEL_COLUMN_CANDIDATES = {
    "label",
    "class",
    "target",
    "category",
    "reliability",
    "is_reliable",
    "nhan",
}


class DataConfigurationError(ValueError):
    """Loi cau hinh du lieu co thong bao de nguoi dung tu sua."""


def find_data_file(data_file: str | Path | None = None) -> Path:
    """Tim tep du lieu; khong tu chon neu co nhieu ung vien."""
    config.RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    selected = data_file if data_file is not None else config.DATA_FILE
    if selected:
        path = Path(selected)
        if not path.is_absolute():
            project_relative = config.BASE_DIR / path
            raw_relative = config.RAW_DATA_DIR / path
            path = project_relative if project_relative.exists() else raw_relative
        if not path.is_file():
            raise FileNotFoundError(
                f"Không tìm thấy tệp dữ liệu đã cấu hình: {path}. "
                "Hãy đặt ReINTEL vào data/raw/ và sửa DATA_FILE trong src/config.py."
            )
        return path

    candidates = sorted(
        path
        for path in config.RAW_DATA_DIR.rglob("*")
        if path.is_file() and path.suffix.lower() in config.SUPPORTED_EXTENSIONS
    )
    if not candidates:
        raise FileNotFoundError(
            "Chưa tìm thấy bộ dữ liệu ReINTEL. Hãy đặt tệp CSV/TSV/TXT/JSON/JSONL/ZIP "
            f"vào thư mục: {config.RAW_DATA_DIR} rồi chạy lại."
        )
    if len(candidates) > 1:
        names = "\n- ".join(str(path.relative_to(config.BASE_DIR)) for path in candidates)
        raise DataConfigurationError(
            "Tìm thấy nhiều tệp dữ liệu nên chương trình không tự ý chọn. "
            "Hãy đặt DATA_FILE trong src/config.py. Các tệp hiện có:\n- " + names
        )
    return candidates[0]


def _read_delimited(source: Path | io.BytesIO, suffix: str) -> pd.DataFrame:
    """Doc tep phan tach cot, uu tien UTF-8 va ho tro tu phat hien dau tach."""
    separator = "\t" if suffix == ".tsv" else None
    encodings = ("utf-8-sig", "utf-8")
    last_error: Exception | None = None
    for encoding in encodings:
        try:
            if hasattr(source, "seek"):
                source.seek(0)
            return pd.read_csv(
                source,
                sep=separator,
                engine="python" if separator is None else "c",
                encoding=encoding,
            )
        except (UnicodeDecodeError, pd.errors.ParserError) as error:
            last_error = error
    raise DataConfigurationError(f"Không thể đọc tệp dạng bảng bằng UTF-8: {last_error}")


def _read_json(source: Path | io.BytesIO, suffix: str) -> pd.DataFrame:
    if hasattr(source, "seek"):
        source.seek(0)
    try:
        return pd.read_json(source, lines=suffix == ".jsonl", encoding="utf-8")
    except ValueError as first_error:
        if suffix == ".json":
            if hasattr(source, "seek"):
                source.seek(0)
            try:
                return pd.read_json(source, lines=True, encoding="utf-8")
            except ValueError:
                pass
        raise DataConfigurationError(f"Không thể đọc cấu trúc JSON: {first_error}") from first_error


def _read_from_zip(path: Path) -> pd.DataFrame:
    with zipfile.ZipFile(path) as archive:
        members = [
            name
            for name in archive.namelist()
            if not name.endswith("/")
            and Path(name).suffix.lower() in config.SUPPORTED_EXTENSIONS - {".zip"}
        ]
        if len(members) != 1:
            raise DataConfigurationError(
                f"Tệp ZIP phải chứa đúng một tệp dữ liệu được hỗ trợ; hiện tìm thấy {len(members)}. "
                "Hãy giải nén vào data/raw/ và đặt DATA_FILE rõ ràng."
            )
        member = members[0]
        suffix = Path(member).suffix.lower()
        raw = io.BytesIO(archive.read(member))
        return _read_json(raw, suffix) if suffix in {".json", ".jsonl"} else _read_delimited(raw, suffix)


def load_dataset(path: Path) -> pd.DataFrame:
    """Doc mot tep du lieu da duoc chon vao DataFrame."""
    suffix = path.suffix.lower()
    if suffix == ".zip":
        dataframe = _read_from_zip(path)
    elif suffix in {".json", ".jsonl"}:
        dataframe = _read_json(path, suffix)
    elif suffix in {".csv", ".tsv", ".txt"}:
        dataframe = _read_delimited(path, suffix)
    else:
        raise DataConfigurationError(f"Định dạng {suffix} chưa được hỗ trợ.")

    if dataframe.empty:
        raise DataConfigurationError("Tệp dữ liệu rỗng, không có mẫu nào để xử lý.")
    return dataframe


def _resolve_column(
    columns: list[str], configured: str | None, candidates: set[str], purpose: str
) -> str:
    if configured:
        if configured not in columns:
            raise DataConfigurationError(
                f"Cột {purpose} được cấu hình là '{configured}' nhưng không tồn tại. "
                f"Các cột hiện có: {columns}"
            )
        return configured

    normalized = {str(column).strip().lower(): column for column in columns}
    matches = [normalized[name] for name in candidates if name in normalized]
    matches = list(dict.fromkeys(matches))
    if len(matches) == 1:
        return str(matches[0])
    reason = "không có tên cột phù hợp" if not matches else f"có nhiều cột có thể phù hợp: {matches}"
    raise DataConfigurationError(
        f"Không thể xác định chắc chắn cột {purpose} ({reason}). "
        f"Các cột hiện có: {columns}. Hãy sửa {'TEXT_COLUMN' if purpose == 'văn bản' else 'LABEL_COLUMN'} "
        "trong src/config.py."
    )


def identify_columns(dataframe: pd.DataFrame) -> tuple[str, str]:
    columns = [str(column) for column in dataframe.columns]
    text_column = _resolve_column(columns, config.TEXT_COLUMN, TEXT_COLUMN_CANDIDATES, "văn bản")
    label_column = _resolve_column(columns, config.LABEL_COLUMN, LABEL_COLUMN_CANDIDATES, "nhãn")
    if text_column == label_column:
        raise DataConfigurationError("Cột văn bản và cột nhãn không được trùng nhau.")
    return text_column, label_column


def load_reintel(data_file: str | Path | None = None) -> tuple[pd.DataFrame, Path, str, str]:
    path = find_data_file(data_file)
    dataframe = load_dataset(path)
    text_column, label_column = identify_columns(dataframe)
    return dataframe, path, text_column, label_column


def print_dataset_preview(dataframe: pd.DataFrame) -> None:
    print(f"Kích thước dữ liệu: {dataframe.shape}")
    print(f"Các cột: {list(dataframe.columns)}")
    print("5 dòng đầu:")
    preview = dataframe.head().copy()
    for column in preview.select_dtypes(include=["object", "string"]).columns:
        preview[column] = preview[column].map(
            lambda value: value
            if pd.isna(value) or len(str(value)) <= 300
            else str(value)[:300] + "..."
        )
    print(preview.to_string(index=False))
