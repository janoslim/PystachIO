import re
from all_queries_sql import QUERY_SQL
from dict import TPCH_CATALOG,TPCH_COLUMN_LOGICAL_SIZES


def get_col_names(sql_text):
    cols = set()
    prefixes = set(["l_","o_","c_","p_","ps_","s_","n_","r_"])
    for col in re.findall(r"[locpsnr]+\_[a-z]+", sql_text):
        # should learn the regex for this
        if col[:2] in prefixes or col[:3] in prefixes:
            cols.add(col)
    return cols


def query_scan_set_size(query,sf):

    columns_in_query = get_col_names(QUERY_SQL[query])
    print(columns_in_query)

    total = 0
    found_cols = 0
    for table_name, table_info in TPCH_CATALOG.items():
        attrib_size = 0
        for table_col in table_info["columns"]:
            if table_col not in columns_in_query:
                continue
            found_cols += 1
            if TPCH_COLUMN_LOGICAL_SIZES[table_col] == -1:
                print(f"WARN: {query=} size not set for column={table_col}")
            attrib_size += TPCH_COLUMN_LOGICAL_SIZES[table_col]
        table_scan_size = attrib_size * table_info["size"](sf)
        total += table_scan_size
        if table_scan_size == 0:
            continue
        print(f"\tTable[{table_name}]: {attrib_size=} * {table_info["size"](sf)} tuples = {table_scan_size} bytes, {table_scan_size/(1<<30):.3f} GiB")
    if found_cols != len(columns_in_query):
        print(f"WARN: Didnt find all columns ({found_cols}/{len(columns_in_query)}): {columns_in_query}")
    print(f"Query {query}: Total scan size= {total} bytes, {total/(1<<30):.3f} GiB")
    return total


if __name__ == '__main__':
    SF = 1000
    per_query = {query: query_scan_set_size(query,SF) for query in range(1,23)}
    total = sum(per_query.values())

    print(f"Scan heavy queries: {sorted(per_query.items(), key=lambda x: -x[1])[:5]}")
    print(f"Total scan set size for ALL queries: {total/(1<<30):.3f} GiB")
