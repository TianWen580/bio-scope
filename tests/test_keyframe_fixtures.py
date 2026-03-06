from __future__ import annotations

import socket

def test_no_network_fixture_guard() -> None:
    try:
        _ = socket.create_connection(('example.com', 443), timeout=0.01)
    except RuntimeError as err:
        assert 'Network access is disabled' in str(err)
    else:
        raise AssertionError('Expected network guard to block create_connection')
