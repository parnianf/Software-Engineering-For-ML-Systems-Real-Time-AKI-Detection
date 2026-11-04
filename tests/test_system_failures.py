import unittest
from unittest.mock import MagicMock, patch
import socket
from src.aki.app import run_loop


class TestSystemFailures(unittest.TestCase):

    @patch("src.aki.app.socket.create_connection")
    @patch("src.aki.app.time.sleep")  # Prevent actual waiting
    def test_tcp_disconnect_reconnect(self, mock_sleep, mock_connect):
        """
        Failure Test: Verify loop attempts to reconnect on socket failure.

        Scenario:
        1. Connects successfully -> Receives data -> Remote close
        2. Reconnects -> Fails (Network Error)
        3. Reconnects -> Success -> Receives data
        """
        # Setup dependencies
        store = MagicMock()
        predictor = MagicMock()
        pager = MagicMock()

        # --- Socket 1: Initial Success ---
        sock_1 = MagicMock()
        sock_1.__enter__.return_value = sock_1
        # Returns one message, then empty bytes (indicating remote close)
        sock_1.recv.side_effect = [
            b"\x0bMSH|^~\\&|...|||||ADT^A01\rPID|1||123\x1c\r",
            b""
        ]

        # --- Socket 2: Reconnection Success ---
        sock_2 = MagicMock()
        sock_2.__enter__.return_value = sock_2
        # Returns one message, then empty bytes
        sock_2.recv.side_effect = [
            b"\x0bMSH|^~\\&|...|||||ADT^A01\rPID|1||123\x1c\r",
            b""
        ]

        # --- Connection Lifecycle ---
        # 1. First call: Returns sock_1
        # 2. Second call: Raises ConnectionError (Simulates network down)
        # 3. Third call: Returns sock_2 (Simulates network restored)
        mock_connect.side_effect = [sock_1, socket.error("Network Down"), sock_2]

        # --- Loop Control ---
        # Stop the loop only after we've hit the error and tried to reconnect
        def stop_loop(*_):
            import src.aki.app
            # Check call count to ensure we went through the reconnect cycle
            if mock_connect.call_count >= 3:
                src.aki.app.SHUTDOWN_FLAG = True

        mock_sleep.side_effect = stop_loop

        # Run the loop
        try:
            run_loop(store, predictor, pager)
        except Exception as e:
            self.fail(f"Loop crashed: {e}")

        # Assertions
        # Ensure we actually tried to connect 3 times
        self.assertEqual(mock_connect.call_count, 3)

        # Ensure we slept (backed off) when the error occurred
        self.assertTrue(mock_sleep.called)