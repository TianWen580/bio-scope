from __future__ import annotations

import socket


_ORIGINAL_CREATE_CONNECTION = socket.create_connection
_ORIGINAL_CONNECT = socket.socket.connect
_ORIGINAL_CONNECT_EX = socket.socket.connect_ex


def _blocked_create_connection(
    address: object,
    timeout: float | None = None,
    source_address: object = None,
) -> socket.socket:
    _ = (address, timeout, source_address)
    raise RuntimeError('Network access is disabled by default in unit tests.')


def _blocked_connect(sock: socket.socket, address: object) -> None:
    _ = (sock, address)
    raise RuntimeError('Network access is disabled by default in unit tests.')


def _blocked_connect_ex(sock: socket.socket, address: object) -> int:
    _ = (sock, address)
    raise RuntimeError('Network access is disabled by default in unit tests.')


def pytest_sessionstart(session: object) -> None:
    _ = session
    socket.create_connection = _blocked_create_connection
    socket.socket.connect = _blocked_connect
    socket.socket.connect_ex = _blocked_connect_ex


def pytest_sessionfinish(session: object, exitstatus: int) -> None:
    _ = (session, exitstatus)
    socket.create_connection = _ORIGINAL_CREATE_CONNECTION
    socket.socket.connect = _ORIGINAL_CONNECT
    socket.socket.connect_ex = _ORIGINAL_CONNECT_EX
