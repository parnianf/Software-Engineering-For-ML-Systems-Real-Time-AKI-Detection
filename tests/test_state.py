import unittest
import tempfile
import shutil
import json
import os
from pathlib import Path
from datetime import datetime
from src.aki.state import Patient, PatientStore
from src.aki.config import SETTINGS


class TestStatePersistence(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.path = Path(self.test_dir)
        SETTINGS.PERSISTENCE_PATH = self.path / "state.pkl"
        SETTINGS.HISTORY_PATH = self.path / "history.csv"
        SETTINGS.JOURNAL_PATH = self.path / "journal.jsonl"

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_patient_logic(self):
        """Unit test for in-memory patient sorting logic."""
        p = Patient(mrn="1")
        # Out of order insertion
        p.add_result(datetime(2024, 1, 5), 10.)
        p.add_result(datetime(2024, 1, 1), 5.)

        # Should sort automatically
        self.assertEqual(p.history[0].value, 5.)
        self.assertEqual(p.history[-1].value, 10.)

    def test_snapshot_roundtrip(self):
        """Verifies full state serialization and deserialization."""
        store = PatientStore()
        # Use log methods to populate (simulating app usage)
        store.log_demographics("123", sex='F', dob=None)

        store.save()

        # Reload
        new_store = PatientStore()
        new_store.hydrate()
        self.assertEqual(new_store.patients["123"].sex, 'F')

    def test_wal_write_and_replay(self):
        """
        Verify that operations are logged to WAL and replayed on hydration without a snapshot.
        """
        store = PatientStore()
        # (1) Log events
        store.log_demographics("WAL_PATIENT", sex="F", dob=datetime(1990, 1, 1))
        store.log_result("WAL_PATIENT", datetime(2024, 1, 1), 100.)

        # (2) Verify journal file content
        self.assertTrue(SETTINGS.JOURNAL_PATH.exists(), "Journal file was not created.")
        with open(SETTINGS.JOURNAL_PATH) as f:
            lines = f.readlines()
        self.assertEqual(len(lines), 2, "Expected 2 journal entries.")

        # Verify JSON structure
        rec1 = json.loads(lines[0])
        self.assertEqual(rec1['op'], 'demo')
        self.assertEqual(rec1['mrn'], 'WAL_PATIENT')

        # (3) Simulate Crash: Create new store without calling save()
        new_store = PatientStore()
        new_store.hydrate()

        # (4) Verify data recovered from WAL
        p = new_store.patients.get("WAL_PATIENT")
        self.assertIsNotNone(p, "Patient lost after crash simulation.")
        self.assertEqual(p.sex, "F")
        self.assertEqual(len(p.history), 1)
        self.assertEqual(p.history[0].value, 100.)

    def test_wal_truncation_on_snapshot(self):
        """Verify journal is cleared after a successful snapshot to prevent unbounded growth."""
        store = PatientStore()
        store.log_result("123", datetime(2024, 1, 1), 50.)

        # Pre-condition: Journal has data
        self.assertGreater(os.path.getsize(SETTINGS.JOURNAL_PATH), 0)

        # Action: Save Snapshot
        store.save()

        # Post-condition: Journal empty, Snapshot exists
        self.assertEqual(os.path.getsize(SETTINGS.JOURNAL_PATH), 0, "Journal was not truncated.")
        self.assertTrue(SETTINGS.PERSISTENCE_PATH.exists())

    def test_corrupt_wal_recovery(self):
        """
        Failure Test: Verify system falls back gracefully if state file is corrupt
        and handles garbage in the journal log.
        """
        # (1) Create a corrupt pickle file
        with open(SETTINGS.PERSISTENCE_PATH, "wb") as f:
            f.write(b"NOT_A_PICKLE_FILE")

        # (2) Create a mixed-quality journal
        SETTINGS.JOURNAL_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(SETTINGS.JOURNAL_PATH, 'w') as f:
            f.write(json.dumps({'op': 'demo', 'mrn': 'VALID_1'}) + '\n')
            f.write("THIS_IS_GARBAGE_JSON_DATA_FROM_DISK_CORRUPTION\n")  # Should be skipped
            f.write(json.dumps({'op': 'demo', 'mrn': 'VALID_2'}) + '\n')

        # (3) Attempt hydration
        store = PatientStore()
        try:
            store.hydrate()
        except Exception as e:
            self.fail(f"Store crashed on corrupt recovery: {e}")

        # (4) Verify valid data recovered despite corruption
        self.assertIn("VALID_1", store.patients)
        self.assertIn("VALID_2", store.patients)