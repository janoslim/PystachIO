
from itertools import product


TPCH_CATALOG = {
    "lineitem": {
                    "columns": ["l_orderkey", "l_partkey", "l_suppkey", "l_linenumber", "l_quantity",
                             "l_extendedprice", "l_discount", "l_tax", "l_returnflag", "l_linestatus",
                             "l_shipdate", "l_commitdate", "l_receiptdate","l_shipinstruct","l_shipmode"],
                    "categorize": {
                                    "l_returnflag": ["A","N","R"],
                                    "l_linestatus": ["F","O"],
                                    "l_shipinstruct": ["COLLECT COD", "DELIVER IN PERSON", "NONE", "TAKE BACK RETURN"],
                                    "l_shipmode": ['AIR', 'FOB', 'MAIL', 'RAIL', 'REG AIR', 'SHIP', 'TRUCK']
                                },
                    "orderby": [],
                    "size": lambda sf: sf*1500000*5,
                },
    "orders":   {
                    "columns": ["o_orderkey", "o_custkey", "o_orderstatus", "o_totalprice", "o_orderdate", "o_orderpriority", "o_shippriority"],
                    "categorize": {
                                    "o_orderstatus": ["F","O","P"],
                                    "o_orderpriority": ["1-URGENT","2-HIGH","3-MEDIUM","4-NOT SPECIFIED","5-LOW"],
                    },
                    "orderby": [],
                    "size": lambda sf: sf*1500000,
                },
    "customer": {
                    "columns": ["c_custkey", "c_nationkey", "c_acctbal", "c_mktsegment"],
                    "categorize": {
                                    "c_mktsegment": ['AUTOMOBILE', 'BUILDING', 'FURNITURE', 'HOUSEHOLD', 'MACHINERY']
                    },
                    "orderby": [],
                    "extrasql": ["SELECT count(*) FROM customer WHERE c_mktsegment = 'BUILDING';"],
                    "size": lambda sf: sf*150000,
                },
    "part": {
                    "columns": ["p_partkey","p_name","p_mfgr","p_brand","p_type","p_size","p_container","p_retailprice"],
                    "categorize": {
                                    "p_mfgr": [f"Manufacturer#{i}" for i in range(1,6)],
                                    "p_brand": [f"Brand#{i}{j}" for i in range(1,6) for j in range(1,6)],
                                    "p_type": [" ".join(pair) for pair in product(['ECONOMY', 'LARGE', 'MEDIUM', 'PROMO', 'SMALL', 'STANDARD'],
                                                                                  ['ANODIZED', 'BRUSHED', 'BURNISHED', 'PLATED', 'POLISHED'],
                                                                                  ['BRASS', 'COPPER', 'NICKEL', 'STEEL', 'TIN'])],
                                    "p_container": [" ".join(pair) for pair in product(['JUMBO', 'LG', 'MED', 'SM', 'WRAP'],
                                                                                       ['BAG', 'BOX', 'CAN', 'CASE', 'DRUM', 'JAR', 'PACK', 'PKG'])],
                    },
                    "orderby": [],
                    "size": lambda sf: sf*200000,
                },
    "supplier": {
                    "columns": ["s_suppkey","s_name","s_address","s_nationkey","s_phone","s_acctbal","s_comment"],
                    "categorize": {},
                    "orderby": [],
                    "size": lambda sf: sf*10000,
                },
    "partsupp": {
                    "columns": ["ps_partkey","ps_suppkey","ps_availqty","ps_supplycost"],
                    "categorize": {},
                    "orderby": [],
                    "size": lambda sf: sf*800000,
                },
    "nation": {
                    "columns": ["n_nationkey", "n_name", "n_regionkey"],
                    "categorize": {
                                    "n_name": ['ALGERIA', 'ARGENTINA', 'BRAZIL', 'CANADA', 'CHINA', 'EGYPT',
                                                'ETHIOPIA', 'FRANCE', 'GERMANY', 'INDIA', 'INDONESIA', 'IRAN',
                                                'IRAQ', 'JAPAN', 'JORDAN', 'KENYA', 'MOROCCO', 'MOZAMBIQUE',
                                                'PERU', 'ROMANIA', 'RUSSIA', 'SAUDI ARABIA', 'UNITED KINGDOM',
                                                'UNITED STATES', 'VIETNAM']
                    },
                    "orderby": [],
                    "size": lambda sf: 25,
                },
    "region": {
                    "columns": ["r_regionkey", "r_name"],
                    "categorize": {
                                    "r_name": ["AFRICA","AMERICA","ASIA","EUROPE","MIDDLE EAST"]
                    },
                    "orderby": [],
                    "size": lambda sf: 5,
    }
}

TPCH_COLUMN_LOGICAL_SIZES = {
    "l_orderkey": 8, "l_partkey": 8, "l_suppkey": 8, "l_linenumber": 8, "l_quantity": 8,
     "l_extendedprice": 8, "l_discount": 8, "l_tax": 8, "l_returnflag": 4, "l_linestatus": 4,
     "l_shipdate": 4, "l_commitdate": 4, "l_receiptdate": 4,"l_shipinstruct": 4,"l_shipmode": 4,
     "o_orderkey": 8, "o_custkey": 8, "o_orderstatus": 4, "o_totalprice": 8, "o_orderdate": 4, "o_orderpriority": 4,
     "o_shippriority": 4,
     "c_custkey": 8, "c_nationkey": 4, "c_acctbal": 8, "c_mktsegment": 4,
     "p_partkey": 8,"p_name": -1,"p_mfgr": 4,"p_brand": 4,"p_type": 4,"p_size": 4,"p_container": 4,"p_retailprice": 8,
     "s_suppkey": 8,"s_name": -1,"s_address": -1,"s_nationkey": 4,"s_phone": -1,"s_acctbal": 8,"s_comment": -1,
     "ps_partkey": 8,"ps_suppkey": 8,"ps_availqty": 8,"ps_supplycost": 8,
     "n_nationkey": 4, "n_name": 4, "n_regionkey": 4,
     "r_regionkey": 4, "r_name" : 4,
}

def string_to_categoryidx(col_name,string_val):
    for _,table_info in TPCH_CATALOG.items():
        for catcol,vals in table_info["categorize"].items():
            if catcol == col_name:
                return vals.index(string_val)
    return -1

def categoryidx_to_string(col_name,category_idx):
    for _,table_info in TPCH_CATALOG.items():
        for catcol,vals in table_info["categorize"].items():
            if catcol == col_name:
                return vals[category_idx]
    return -1


def main():
    print("Table Catalog:")
    for table,table_info in TPCH_CATALOG.items():
        print(f"[{table} TABLE] columns {(table_info["columns"])}")
        for catcol,vals in table_info["categorize"].items():
            if sorted(vals) != vals:
                print(f"--> Consider fixing the values for {catcol} to have same sort order as strings: {sorted(vals)}")
            print(f"    Categorizing \"{catcol}\": {list(enumerate(vals))}")
        if (orderby := table_info["orderby"]):
            print(f"    Ordered By: {orderby}")
    # testing:
    # print(string_to_categoryidx("c_mktsegment","BUILDING")) # should be 0
    # print(string_to_categoryidx("c_mktsegment","FURNITURE")) # should be 2
    # print(categoryidx_to_string("c_mktsegment",3)) # should be "HOUSEHOLD"
    # print(categoryidx_to_string("l_shipinstruct",1)) # should be "COLLECT COD"
    # print(string_to_categoryidx("p_container","MED BOX"))
    # print(string_to_categoryidx("p_brand", "Brand#34"))
    print(string_to_categoryidx("p_type","ECONOMY ANODIZED STEEL"))
    # 'LG CASE', 'LG BOX', 'LG PACK', 'LG PKG'
    # print(string_to_categoryidx("l_shipinstruct", "DELIVER IN PERSON"))

if __name__ == "__main__":
    main()