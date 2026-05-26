import torch
import pandas as pd
import numpy as np


def info(scale_factor):
    print(f"ORDER: {scale_factor*1500000} tuples")
    print(f"LINEITEM: {scale_factor*1500000*5} tuples")
    print(f"PART: {scale_factor*200000} tuples")
    print(f"SUPP: {scale_factor*10000} tuples")
    print(f"PARTSUPP: {scale_factor*800000} tuples")
    print(f"CUST: {scale_factor*150000} tuples")
    print(f"NATION: {25} tuples")
    print(f"REGION: {5} tuples")


def get_cols(path, device, names, projection=None, dtype={}, partition_query=None, *args, **kwargs):
    # read and partition
    chunksize = (1<<18)
    if "nrows" in kwargs:
        chunksize = min(chunksize, kwargs["nrows"])
    gen = pd.read_csv(path, dtype=dtype, delimiter="|", chunksize=(1<<18), names=names,
                      usecols=lambda x: (not projection or x in projection) and x not in [],
                      *args, **kwargs)

    if partition_query is None:
        df = pd.concat((x for x in gen), ignore_index=True)
    else:
        df = pd.concat((x.query(partition_query) for x in gen), ignore_index=True)

    # print(df.columns)
    # print(df.head())
    # print(df.dtypes)
    cols = [torch.tensor(df[col_str].to_numpy(), device=device) for col_str in df.columns]
    return cols

def get_orders(rootpath, device, *args, **kwargs):
    ORDERSTATUS_DICT = {val: np.int32(idx) for idx,val in enumerate(["O","F"])}
    PRIORITY_DICT = {val: np.int32(idx) for idx,val in enumerate(["1-URGENT","2-HIGH","3-MEDIUM","4-NOT SPECIFIED","5-LOW"])}
    return get_cols(rootpath+"/orders.csv",
                    device,
                    ["o_orderkey","o_custkey","o_orderstatus","o_totalprice","o_orderdate","o_orderpriority","o_clerk","o_shippriority"],
                    dtype={"o_orderkey":np.int64, "o_totalprice":np.float64},
                    converters={"o_orderstatus": lambda x: ORDERSTATUS_DICT[x],
                                "o_orderdate":lambda x: pd.Timestamp(x).value,
                                "o_clerk":lambda x: np.int64(x[6:]),
                                "o_orderpriority":lambda x: PRIORITY_DICT[x]},
                    *args, **kwargs)

def get_customer(rootpath, device, *args, **kwargs):
    return get_cols(rootpath+"/customer.csv",
                    device,
                    ["c_custkey","c_name","c_address","c_nationkey","c_phone","c_acctbal","c_mktsegment"],
                    dtype={"c_custkey":np.int64},
                    *args, **kwargs)

def get_lineitem(rootpath, device, *args, **kwargs):
    RETURNFLAG_DICT = {val: np.int32(idx) for idx,val in enumerate(["N","R","A"])}
    LINESTATUS_DICT = {val: np.int32(idx) for idx,val in enumerate(["O","F"])}
    SHIPINSTR_DICT = {val: np.int32(idx) for idx,val in enumerate(["DELIVER IN PERSON","COLLECT COD","TAKE BACK RETURN", "NONE"])}
    SHIPMODE_DICT = {val: np.int32(idx) for idx,val in enumerate(["REG AIR","AIR","RAIL","SHIP","TRUCK","MAIL","FOB"])}
    return get_cols(rootpath+"/lineitem.csv",
                    device,
                    ["l_orderkey","l_partkey","l_suppkey","l_linenumber","l_quantity","l_extendedprice","l_discount","l_tax","l_returnflag","l_linestatus","l_shipdate","l_commitdate","l_receiptdate","l_shipinstruct","l_shipmode"],
                    dtype={"l_orderkey":np.int64, "l_extendedprice":np.float64, "l_tax":np.float64, "l_discount":np.float64},
                    converters={"l_returnflag":lambda x:RETURNFLAG_DICT[x],
                                "l_linestatus":lambda x:LINESTATUS_DICT[x],
                                "l_shipdate":lambda x: pd.Timestamp(x).value,
                                "l_commitdate":lambda x: pd.Timestamp(x).value,
                                "l_receiptdate":lambda x: pd.Timestamp(x).value,
                                "l_shipinstruct":lambda x: SHIPINSTR_DICT[x],
                                "l_shipmode":lambda x: SHIPMODE_DICT[x]},
                    *args, **kwargs)

if __name__ == '__main__':
    rootpath = "/mnt/labstore/nboeschen/tpch/csv1/"
    cpu = torch.device("cpu")
    # print(get_lineitem(rootpath, cpu, nrows=10))
    # print(get_lineitem(rootpath, cpu, projection=["l_orderkey","l_linenumber","l_quantity","l_extendedprice","l_discount"], nrows=10))
    print(get_lineitem(rootpath, cpu, projection=["l_orderkey","l_returnflag","l_linestatus"], nrows=10))
    # print(get_orders(rootpath, cpu, nrows=10))
