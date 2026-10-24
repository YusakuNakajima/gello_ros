"""関数（方策の推論）を別スレッドで動かし、期限・終了要求で待つのをやめる。

推論が戻らない（GPU のハング・極端な遅延）と、呼び出し元のループは止める判定（Ctrl+C・入力の期限切れ）に
たどり着けない。ここでは推論を daemon スレッドで動かし、呼び出し元は短い間隔で終了要求を確かめながら待つ。
Python のスレッドは外から止められないので、期限を過ぎた推論はそのまま捨てる（daemon なのでプロセスの
終了は待たない）。捨てたあとは同じ方策でもう推論しない（内部状態が途中のままのため）。
"""

import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Optional

DONE = "done"
TIMEOUT = "timeout"
ABORTED = "aborted"


@dataclass
class DeadlineResult:
    status: str             # DONE / TIMEOUT / ABORTED
    value: Any = None       # DONE のときの戻り値
    elapsed_s: float = 0.0  # 待った時間（s）


def call_with_deadline(fn: Callable[[], Any], timeout_s: float,
                       should_abort: Optional[Callable[[], bool]] = None,
                       poll_interval_s: float = 0.005) -> DeadlineResult:
    """fn() を別スレッドで呼び、終わるか、timeout_s（s）を過ぎるか、should_abort() が True になるまで待つ。

    fn が例外を投げたら、呼び出し元のスレッドで同じ例外を投げ直す。
    """
    if not timeout_s > 0.0:
        raise ValueError(f"timeout_s must be > 0, got {timeout_s}")
    finished = threading.Event()
    outcome = {}

    def worker():
        try:
            outcome["value"] = fn()
        except BaseException as e:  # noqa: B902 呼び出し元で投げ直す
            outcome["error"] = e
        finally:
            finished.set()

    start = time.monotonic()
    threading.Thread(target=worker, name="deadline_call", daemon=True).start()
    while True:
        if finished.wait(timeout=poll_interval_s):
            elapsed_s = time.monotonic() - start
            if "error" in outcome:
                raise outcome["error"]
            return DeadlineResult(DONE, outcome.get("value"), elapsed_s)
        elapsed_s = time.monotonic() - start
        if should_abort is not None and should_abort():
            return DeadlineResult(ABORTED, None, elapsed_s)
        if elapsed_s > timeout_s:
            return DeadlineResult(TIMEOUT, None, elapsed_s)
