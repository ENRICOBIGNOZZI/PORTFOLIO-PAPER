"""Download the three official JKP Common Task Framework tables from WRDS.

The script follows the official JKP dataset-access route and never stores credentials
in source code. Credentials are resolved from command-line/environment, then keyring,
then an interactive password prompt.
"""
from __future__ import annotations

import argparse
import getpass
import os
from pathlib import Path

import keyring
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from sqlalchemy import create_engine, text
from sqlalchemy.engine import URL

TABLES = {
    "ctff_features": "contrib_global_factor.ctff_features",
    "ctff_chars": "contrib_global_factor.ctff_chars",
    "ctff_daily_ret": "contrib_global_factor.ctff_daily_ret",
}


def credentials(username_arg: str | None) -> tuple[str, str]:
    username = username_arg or os.getenv("WRDS_USERNAME")
    if not username:
        username = input("WRDS username: ").strip()
    password = os.getenv("WRDS_PASSWORD")
    if not password:
        try:
            password = keyring.get_password("wrds", username)
        except Exception:
            password = None
    if not password:
        password = getpass.getpass("WRDS password: ")
    if not username or not password:
        raise RuntimeError("WRDS credentials are required")
    return username, password


def stream_table(conn, sql_table: str, path: Path, chunksize: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    writer: pq.ParquetWriter | None = None
    rows = 0
    try:
        for chunk in pd.read_sql_query(text(f"SELECT * FROM {sql_table};"), conn, chunksize=chunksize):
            table = pa.Table.from_pandas(chunk, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(path, table.schema, compression="zstd")
            writer.write_table(table)
            rows += len(chunk)
            print(f"{path.name}: {rows:,} rows")
    finally:
        if writer is not None:
            writer.close()
    if writer is None:
        raise RuntimeError(f"WRDS query returned no rows for {sql_table}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--username", default=None)
    ap.add_argument("--out", type=Path, default=Path("data/raw"))
    ap.add_argument("--chunksize", type=int, default=500_000)
    ap.add_argument("--only", choices=["all", *TABLES.keys()], default="all")
    args = ap.parse_args()

    username, password = credentials(args.username)
    url = URL.create(
        "postgresql+psycopg2",
        username=username,
        password=password,
        host="wrds-pgdata.wharton.upenn.edu",
        port=9737,
        database="wrds",
        query={"sslmode": "require"},
    )
    engine = create_engine(url, pool_pre_ping=True)
    selected = TABLES if args.only == "all" else {args.only: TABLES[args.only]}
    with engine.connect() as conn:
        for local_name, sql_table in selected.items():
            stream_table(conn, sql_table, args.out / f"{local_name}.parquet", args.chunksize)


if __name__ == "__main__":
    main()
