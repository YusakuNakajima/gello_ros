import math

import pytest

from gello_ros.freshness import input_age_s, stamp_went_backwards

S = 1_000_000_000


def test_never_received_is_infinitely_old():
    assert input_age_s(10 * S, None, None, now_monotonic_ns=10 * S) == math.inf


def test_receipt_age_without_stamp():
    assert input_age_s(10 * S, 9 * S, 0, now_monotonic_ns=10 * S) == pytest.approx(1.0)


def test_delayed_message_is_old_even_if_just_received():
    # 今受信したが、計測は 0.5 s 前（バッファに溜まっていた値）
    assert input_age_s(10 * S, 5 * S, int(9.5 * S), now_monotonic_ns=5 * S) == pytest.approx(0.5)


def test_future_stamp_counts_as_old():
    assert input_age_s(10 * S, 5 * S, int(10.3 * S), now_monotonic_ns=5 * S) == pytest.approx(0.3)


def test_receipt_age_ignores_a_paused_ros_clock():
    # use_sim_time で ROS 時刻が止まっていても、受信が途切れてからの壁時計の経過で古いと判定する
    paused_ros_ns = 10 * S
    assert input_age_s(paused_ros_ns, 5 * S, 0, now_monotonic_ns=7 * S) == pytest.approx(2.0)


def test_backwards_stamp():
    assert stamp_went_backwards(10 * S, 9 * S)
    assert not stamp_went_backwards(10 * S, 11 * S)
    assert not stamp_went_backwards(None, 9 * S)
    assert not stamp_went_backwards(10 * S, 0)
