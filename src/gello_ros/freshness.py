"""入力（joint state・力センサ・画像・Touch）が何秒前のものかを、受信時刻と header.stamp から決める。

受信が途切れた場合だけでなく、遅れて届き続ける（バッファに溜まった古い値が流れてくる）場合も古いとみなす。
受信からの経過は壁時計（time.monotonic_ns）で測る。ROS 時刻は use_sim_time で止まったり戻ったりするので、
受信の途切れを見落とす。header.stamp との比較だけ ROS 時刻を使う（送信側も ROS 時刻で stamp を入れるため）。
"""

import math
import time
from typing import Optional


def stamp_to_ns(stamp) -> int:
    """builtin_interfaces/Time を ns にする。"""
    return stamp.sec * 1_000_000_000 + stamp.nanosec


def monotonic_ns() -> int:
    """受信時刻の記録に使う時計。"""
    return time.monotonic_ns()


def input_age_s(now_ros_ns: int, received_monotonic_ns: Optional[int], stamp_ns: Optional[int],
                now_monotonic_ns: Optional[int] = None) -> float:
    """入力の古さ（s）。

    - 一度も受信していなければ inf
    - 受信からの経過時間（monotonic）と、header.stamp からの経過時間（ROS 時刻）の大きいほう
    - header.stamp が 0（送信側が時刻を入れていない）なら受信時刻だけで決める
    - header.stamp が未来を指していても、その差の大きさを古さとして扱う（時計のずれを新しい値と取り違えない）
    """
    if received_monotonic_ns is None:
        return math.inf
    if now_monotonic_ns is None:
        now_monotonic_ns = monotonic_ns()
    age_ns = now_monotonic_ns - received_monotonic_ns
    if stamp_ns:
        age_ns = max(age_ns, abs(now_ros_ns - stamp_ns))
    return age_ns * 1e-9


def stamp_went_backwards(previous_stamp_ns: Optional[int], stamp_ns: int) -> bool:
    """header.stamp が前の値より戻ったか（0 は時刻なしとして比較しない）。"""
    return bool(stamp_ns) and bool(previous_stamp_ns) and stamp_ns < previous_stamp_ns


def sample_after_request(request_ros_ns: int, request_monotonic_ns: int,
                         received_monotonic_ns: Optional[int], stamp_ns: Optional[int]) -> bool:
    """止めると決めた後に計測されたサンプルか。

    受信が要求より後で、かつ header.stamp が要求時刻以降のものだけを認める。遅れて届いた古いサンプル
    （受信は後でも stamp が前）や、stamp のないサンプル（いつ計測したか分からない）は認めない。
    """
    if received_monotonic_ns is None or received_monotonic_ns <= request_monotonic_ns:
        return False
    return bool(stamp_ns) and stamp_ns >= request_ros_ns
