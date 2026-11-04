import unittest
from src.aki.mllp import read_messages, START_BLOCK, END_BLOCK, CARRIAGE_RETURN


class MockSocket:
    def __init__(self, data: bytes):
        self.data = data
        self.cursor = 0

    def recv(self, size):
        if self.cursor >= len(self.data):
            return b""
        chunk = self.data[self.cursor:self.cursor + size]
        self.cursor += size
        return chunk


class TestMLLP(unittest.TestCase):
    def test_valid_frame(self):
        content = "MSH|^~\\&|TEST"
        frame = START_BLOCK + content.encode("ascii") + END_BLOCK + CARRIAGE_RETURN
        msgs = list(read_messages(MockSocket(frame)))
        self.assertEqual(msgs, [content])

    def test_fragmented_frame(self):
        """Simulates TCP fragmentation (byte-by-byte delivery)."""
        content = "MSH|^~\\&|TEST"
        frame = START_BLOCK + content.encode("ascii") + END_BLOCK + CARRIAGE_RETURN

        sock = MockSocket(frame)
        # Force strict 1-byte reads
        original_recv = sock.recv
        sock.recv = lambda size: original_recv(1)

        msgs = list(read_messages(sock))
        self.assertEqual(msgs, [content])

    def test_junk_data_resilience(self):
        """The reader should ignore bytes outside the Start/End blocks."""
        junk = b"noise" + START_BLOCK + b"REAL" + END_BLOCK + CARRIAGE_RETURN + b"more_noise"
        msgs = list(read_messages(MockSocket(junk)))
        self.assertEqual(msgs, ["REAL"])