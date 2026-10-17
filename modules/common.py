from enum import Enum, auto

class SHUFFLE_TYPE(Enum):
    BLOCKING = auto()
    OVERLAPPED = auto()
    OVERLAPPED_GROUPS = auto()
    MIG = auto()
    GRACE = auto()  # SSD spill mode (patch 0003): GraceHashJoin; the overlapped reader as in OVERLAPPED

class SHUFFLE_PROBE_TYPE(Enum):
    MATCHES_ONLY = auto()
    FULL = auto()
    NOOP = auto()

class SHUFFLE_SEND_TYPE(Enum):
    SEND_PER_RANK = auto()
    COMB_SEND = auto()

class SHUFFLE_RECV_TYPE(Enum):
    RECV_PER_RANK = auto()
    COMB_RECV = auto()

class SHUFFLE_PROBE_SCHED(Enum):
    STANDARD = auto()
    INVERTED = auto()
