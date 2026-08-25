import numpy as np


# FastFiz default table
TABLE_WIDTH = 1.116
TABLE_LENGTH = 2.236

CORNER_POCKET_WIDTH = 0.11

CORNER_OFFSET = (
    CORNER_POCKET_WIDTH
    / np.sqrt(2.0)
    / 2.0
)


# FastFiz の POCKETED_* と同じ順番
#
# 2 (NW) -------- 3 (NE)
#    |              |
# 1 (W)          4 (E)
#    |              |
# 0 (SW) -------- 5 (SE)

POCKET_POSITIONS = {
    0: np.array(
        [CORNER_OFFSET, CORNER_OFFSET],
        dtype=np.float32,
    ),

    1: np.array(
        [0.0, TABLE_LENGTH / 2.0],
        dtype=np.float32,
    ),

    2: np.array(
        [
            CORNER_OFFSET,
            TABLE_LENGTH - CORNER_OFFSET,
        ],
        dtype=np.float32,
    ),

    3: np.array(
        [
            TABLE_WIDTH - CORNER_OFFSET,
            TABLE_LENGTH - CORNER_OFFSET,
        ],
        dtype=np.float32,
    ),

    4: np.array(
        [
            TABLE_WIDTH,
            TABLE_LENGTH / 2.0,
        ],
        dtype=np.float32,
    ),

    5: np.array(
        [
            TABLE_WIDTH - CORNER_OFFSET,
            CORNER_OFFSET,
        ],
        dtype=np.float32,
    ),
}


POCKET_NAMES = {
    0: "SW",
    1: "W",
    2: "NW",
    3: "NE",
    4: "E",
    5: "SE",
}