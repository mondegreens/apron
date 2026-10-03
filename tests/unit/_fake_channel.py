"""A paramiko-like SSH channel for tests: output must be read before the
remote command can finish, as with a real 2 MiB channel window."""

from __future__ import annotations

import threading


class FakeChannel:
    def __init__(
        self, out: bytes = b"", err: bytes = b"", code: int = 0, *, finishes: bool = True
    ) -> None:
        self._out = bytearray(out)
        self._err = bytearray(err)
        self._code = code
        self._finishes = finishes
        self.closed = False
        self.status_event = threading.Event()

    def _done(self) -> bool:
        # A writer blocked on a full window cannot exit until it is read.
        return self._finishes and not self._out and not self._err

    def recv_ready(self) -> bool:
        return bool(self._out)

    def recv(self, n: int) -> bytes:
        chunk, self._out = bytes(self._out[:n]), self._out[n:]
        return chunk

    def recv_stderr_ready(self) -> bool:
        return bool(self._err)

    def recv_stderr(self, n: int) -> bytes:
        chunk, self._err = bytes(self._err[:n]), self._err[n:]
        return chunk

    def exit_status_ready(self) -> bool:
        return self._done()

    @property
    def eof_received(self) -> bool:
        return self._done()

    def recv_exit_status(self) -> int:
        assert self._done(), "recv_exit_status before the command could finish: deadlock"
        return self._code

    def close(self) -> None:
        self.closed = True
