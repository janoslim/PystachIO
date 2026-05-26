#!/usr/bin/env python3
"""
$ conda install duckdb

DGX
$ cd /raid/jluo/duckdb/
$ python ~/pytorchio/scripts/duckdb_to_parquet.py 
$ (After this, I will use this parquet as source for rewriting...)
$ e.g. CUSTOMER_parquet_rewrite.py

FNG
$ python ~/pytorchio/scripts/duckdb_to_parquet.py 
"""

TMP_DIR = "/home/jluo/duckdb.tmp/" # FIX OOM
DB_BASE_DIR = "/mnt/labstore/jluo/duckdb"
DB_LIST = ['tpch-sf500.db', 'tpch-sf600.db', 'tpch-sf700.db', 'tpch-sf800.db', 'tpch-sf900.db', 'tpch-sf1000.db']  #INPUT
OUTDIR_LIST = ['tpch-sf500', 'tpch-sf600', 'tpch-sf700', 'tpch-sf800', 'tpch-sf900', 'tpch-sf1000']

# TMP_DIR = "/tmp/" # FIX OOM
# DB_BASE_DIR = "/mnt/labstore/nboeschen/duckdb"
# DB_LIST = ['tpch-sf1.db', 'tpch-sf3.db', 'tpch-sf10.db', 'tpch-sf30.db', 'tpch-sf100.db', 'tpch-sf200.db', 'tpch-sf300.db', 'tpch-sf400.db']    # INPUT
# OUTDIR_LIST = ['tpch-sf1', 'tpch-sf3', 'tpch-sf10', 'tpch-sf30', 'tpch-sf100', 'tpch-sf200', 'tpch-sf300', 'tpch-sf400']      # OUTPUT


import duckdb
import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'tpch'))
from dict import TPCH_CATALOG

# I generate Parquet with DUCKDB
# Download with https://duckdb.org/docs/stable/extensions/tpch.html#pre-generated-data-sets

# Tables to export
# TABLES = ['customer']
TABLES = ['region', 'nation', 'supplier', 'customer', 'part', 'partsupp', 'orders', 'lineitem']
# TABLES = ['region', 'nation', 'supplier', 'customer', 'part', 'partsupp']
EXTRA_SQL_OUT = []
sorting = False
if sorting:
    OUTDIR_LIST = [f"{dir}_sorted" for dir in OUTDIR_LIST]

def export_tables(db_path: str, output_folder: str):
    os.makedirs(output_folder, exist_ok=True)
    con = duckdb.connect(database=db_path)
    con.execute("PRAGMA threads=64;") # FIX OOM
    con.execute(f"SET temp_directory = '{TMP_DIR}';") # FIX OOM
    # con.execute("SET memory_limit='600GB';") # FIX OOM
    try:
        for table in TABLES:
            out_file = os.path.join(output_folder, f"{table}.parquet")

            if table not in TPCH_CATALOG:
                # default export:
                copy_sql = f"COPY {table} TO '{out_file}' (FORMAT 'parquet');"
                print(f"[{table} TABLE] Default Exporting: {copy_sql}")
                con.execute(copy_sql)
                continue
            # else theres some additional info
            table_info = TPCH_CATALOG[table]

            MORE = ""
            if sorting and len(table_info["orderby"]) != 0:
                MORE = f"ORDER BY {','.join(table_info["orderby"])}"

            tmp_table = f"tmp_{table}"
            selected_categorized_columns = []
            for column_name in table_info["columns"]:
                if column_name not in table_info["categorize"]:
                    selected_categorized_columns.append(column_name)
                    continue
                # check that our list contains all variants:
                distinct_count = con.execute(f"Select count(distinct {column_name}) from {table};").fetchone()[0]
                if distinct_count != len(table_info["categorize"][column_name]):
                    # only count is checked, not the actual values
                    print(f"WARN !!!!!! [{table} TABLE] categorize values dont match distinct count!: {len(table_info["categorize"][column_name])} vs {distinct_count=}")
                # create the case statement
                case_statements = []
                for i, categorical_val in enumerate(table_info["categorize"][column_name]):
                    case_statements.append(f"WHEN {column_name} = '{categorical_val}' THEN {i}")
                case_sql = "CASE " + " ".join(case_statements) + f" ELSE NULL END as {column_name}"
                selected_categorized_columns.append(case_sql)
            selected_categorized_columns = ", ".join(selected_categorized_columns)

            create_sql = f"CREATE TEMPORARY TABLE {tmp_table} AS SELECT {selected_categorized_columns} FROM {table} {MORE};"
            print(f"[{table} TABLE] temp table SQL: {create_sql}")

            if table_info["categorize"]:
                # The original file could be saved here!
                # This is currently not done, since its usually not needed.
                # Implement if necessary!
                pass

            con.execute(create_sql)
            copy_sql = f"COPY {tmp_table} TO '{out_file}' (FORMAT 'parquet');"
            print(f"[{table} TABLE] Copy to parquet SQL: {copy_sql}")
            con.execute(copy_sql)
            con.execute(f"DROP TABLE {tmp_table};")

            if "extrasql" in table_info:
                for extrasql in table_info["extrasql"]:
                    EXTRA_SQL_OUT.append(con.execute(extrasql).fetchone()[0])

    finally:
        con.close()

def main():
    # Expand relative DB paths by prefixing DB_BASE_DIR when set
    if DB_BASE_DIR:
        db_paths = [p if os.path.isabs(p) else os.path.join(DB_BASE_DIR, p) for p in DB_LIST]
        output_dirs = [os.path.join(DB_BASE_DIR, d) for d in OUTDIR_LIST]
    else:
        db_paths = DB_LIST
        output_dirs = OUTDIR_LIST

    for db_path, output_dir in zip(db_paths, output_dirs):
        print(f"Processing database: {db_path} -> {output_dir}")
        export_tables(db_path, output_dir)
        print("\n")

    if EXTRA_SQL_OUT:
        print("EXTRA_SQL_OUT:", EXTRA_SQL_OUT)

if __name__ == "__main__":
    main()
