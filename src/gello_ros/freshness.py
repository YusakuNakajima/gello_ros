"""入力（joint state・力センサ・画像・Touch）が何秒前のものかを、受信時刻と header.stamp から決める。

受信が途切れた場合だけでなく、遅れて届き続ける（バッファに溜まった古い値が流れてくる）場合も古いとみなす。
"""

import math
from typing import Optional


def stamp_to_ns(stamp) -> int:
    """builtin_interfaces/Time を ns にする。"""
    return stamp.sec * 1_000_000_000 + stamp.nanosec


def input_age_s(now_ns: int, received_ns: Optional[int], stamp_ns: Optional[int]) -> float:
    """入力の古さ（s）。

    - 一度も受信していなければ inf
    - 受信からの経過時間と、header.stamp からの経過時間の大きいほう
    - header.stamp が 0（送信側が時刻を入れていない）なら受信時刻だけで決める
    - header.stamp が未来を指していても、その差の大きさを古さとして扱う（時計のずれを新しい値と取り違えない）
    """
    if received_ns is None:
        return math.inf
    age_ns = now_ns - received_ns
    if stamp_ns:
        age_ns = max(age_ns, abs(now_ns - stamp_ns))
    return age_ns * 1e-9


def stamp_went_backwards(previous_stamp_ns: Optional[int], stamp_ns: int) -> bool:
    """header.stamp が前の値より戻ったか（0 は時刻なしとして比較しない）。"""
    return bool(stamp_ns) and bool(previous_stamp_ns) and stamp_ns < previous_stamp_ns
