# TPC-H Queries

Query 1:
    Tables: Lineitem
    Shuffled: -
    Plan: L

Query 2:
    Tables: Nation, Supplier, Part, Partsupp
    Plan: (N ⋈ S) ⋈ (P ⋈ PS)

Query 3:
    Tables: Lineitem, Orders, Customer
    Shuffled: L,O
    Plan: ((C ⋈ O) ⋈ L)
    Comment: Data-intensive also for shuffling.
    <!-- 2 Nodes: SHUFFLE_BUFFER_RIGHT=2 -->
    <!-- 3 Nodes: SHUFFLE_BUFFER_RIGHT=3 -->

Query 4:
    Tables: Lineitem, Orders
    Shuffled: L,O
    Plan: (L ⋈ O)
    Comment:

Query 5:
    Tables: Customer, Orders, Lineitem, Supplier, Nation
    Shuffled: L,O
    Plan: (N ⋈ S)  ⋈  ((C ⋈ O) ⋈ L)
    Comment: Similar to query 3
    <!-- 3 Nodes: SHUFFLE_BUFFER_LEFT=8 SHUFFLE_BUFFER_RIGHT=2 -->

Query 6:
    Tables: Lineitem
    Shuffled: -
    Plan: L

Query 7:
    Tables: Customer, Order, Supplier, Lineitem
    Shuffled: L,O
    Plan: C ⋈ (O ⋈ (S ⋈ L))

Query 8:
    Tables: Supplier, Nation, Customer, Order, Part, Lineitem
    Shuffled: L,O
    Plan: S  ⋈  (((N ⋈ C) ⋈ O) ⋈ (P ⋈ L))

Query 9:
    Tables: Part, Partsupp, Order, Supplier, Lineitem
    Shuffled: L,O
    Plan: (S ⋈ ((P ⋈ PS) ⋈ L)) ⋈ O

Query 10:
    Tables: Lineitem, Orders, Customer
    Shuffled: L,O
    Plan: ((C ⋈ O) ⋈ L)

Query 11:
    Tables: Supplier, Partsupp
    Shuffled: S,PS,S,PS
    Plan: (S ⋈ PS) ⋈ (S ⋈ PS)

Query 12:
    Tables: Lineitem, Orders
    Shuffled: L,O
    Plan: (L ⋈ O)
    Comment: Many probe kernels, because of non-unique lineitem build side
    <!-- 2 nodes: SHUFFLE_RECV_TYPE=COMB_RECV SHUFFLE_BUFFER_RIGHT=6 -->

Query 13:
    Tables: Customer, Order
    Shuffled: C,O
    Plan: (C ⋈ O)
    Comment: Final groupbys and aggregation should be profiled and optimized

Query 14:
    Tables: Lineitem, Part
    Shuffled: L,P
    Plan: (L ⋈ P)
    Comment: Mainly a storage benchmark -> Not data-intensive after filtering.
    <!-- 3 Nodes: SHUFFLE_BUFFER_LEFT=20 -->

Query 15:
    Tables: Lineitem, Supplier
    Shuffled: L
    Plan: S ⋈ (L)

Query 16:
    Tables: Supplier, Part, Partsupp
    Shuffled: P, PS
    Plan: S ⋈ (P ⋈ PS)

Query 17:
    Tables: Lineitem, Part
    Shuffled: L,P for first join, only L for second
    Plan: (P ⋈ L) ⋈ L
    Comment:

Query 18:
    Tables: Lineitem, Customer, Orders
    Shuffled: L, (C,O) on orderkey
    Plan: L ⋈ (C ⋈ O) 
    Comment:

Query 19:
    Tables: Lineitem
    Shuffled: L, P on partkey
    Plan: (L ⋈ P)
    Comment: Could check build on part table
    <!-- SHUFFLE_BUFFER_LEFT=16 SHUFFLE_BUFFER_RIGHT=2 -->

Query 20: NOT IMPLEMENTED
    Problem: String Operation "like"

Query 21: NOT IMPLEMENTED
    Problem: ?

Query 22: NOT IMPLEMENTED
    Problem: Substring Operation

# Problem types:
- string operations which can not be rewritten (e.g., using string dictionary values) or altered (e.g., changing predicates),
    without significantly changing the query plan or cardinalities