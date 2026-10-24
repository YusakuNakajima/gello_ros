"""方策の推論を期限付きで動かす（推論が戻らなくても止められる）ことを確かめる。"""
import threading
import time
from types import SimpleNamespace

import pytest

from gello_ros.deadline_call import ABORTED, DONE, TIMEOUT, call_with_deadline


def test_returns_the_value_when_done():
    result = call_with_deadline(lambda: 42, 1.0)
    assert result.status == DONE
    assert result.value == 42


def test_gives_up_on_a_hung_call():
    never = threading.Event()
    start = time.monotonic()
    result = call_with_deadline(never.wait, 0.05)
    assert result.status == TIMEOUT
    assert time.monotonic() - start < 0.5
    never.set()


def test_stops_waiting_when_abort_is_requested():
    never = threading.Event()
    requested = {"stop": False}
    threading.Timer(0.02, lambda: requested.update(stop=True)).start()
    start = time.monotonic()
    result = call_with_deadline(never.wait, 10.0, should_abort=lambda: requested["stop"])
    assert result.status == ABORTED
    assert time.monotonic() - start < 0.5
    never.set()


def test_reraises_the_exception_from_the_call():
    def fail():
        raise RuntimeError("boom")
    with pytest.raises(RuntimeError, match="boom"):
        call_with_deadline(fail, 1.0)


@pytest.mark.parametrize("timeout_s", [0.0, -1.0, float("nan")])
def test_timeout_must_be_positive(timeout_s):
    with pytest.raises(ValueError):
        call_with_deadline(lambda: None, timeout_s)


def runner_node(act, stop_requested=lambda: False):
    calls = []
    node = SimpleNamespace(
        agent=SimpleNamespace(act=act),
        _stop_requested=stop_requested,
        _stop_and_hold=lambda: calls.append("stop_and_hold") or True,
        get_logger=lambda: SimpleNamespace(error=lambda *a, **k: calls.append("error"),
                                           info=lambda *a, **k: None),
    )
    return node, calls


def test_policy_act_returns_the_action_in_time():
    from test_stop_transitions import load_runner
    module = load_runner()
    node, calls = runner_node(lambda obs, t: {"ee_pos": [0.4, 0.0, 0.3]})
    assert module.AgentNode._policy_act(node, {}, 3, 0.2) == {"ee_pos": [0.4, 0.0, 0.3]}
    assert calls == []


def test_policy_act_holds_when_inference_hangs():
    from test_stop_transitions import load_runner
    module = load_runner()
    never = threading.Event()
    node, calls = runner_node(lambda obs, t: never.wait())
    start = time.monotonic()
    assert module.AgentNode._policy_act(node, {}, 3, 0.05) is None
    assert time.monotonic() - start < 0.5
    assert calls == ["error", "stop_and_hold"]
    never.set()


def test_policy_act_does_not_send_when_stop_arrives_with_the_result():
    from test_stop_transitions import load_runner
    module = load_runner()
    # 推論が終わった直後に終了要求を確かめる。そのとき要求があれば指令を返さない
    node, calls = runner_node(lambda obs, t: {"ee_pos": [0.4, 0.0, 0.3]}, stop_requested=lambda: True)
    assert module.AgentNode._policy_act(node, {}, 3, 0.2) is None
    assert calls == ["stop_and_hold"]
