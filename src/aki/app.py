"""
Main application entry point.
Orchestrates MLLP connection, state updates, and inference.
"""

import logging
import signal
import sys
import time
import socket

from src.aki.config import SETTINGS
from src.aki.state import PatientStore
from src.aki.features import compute_features
from src.aki.mllp import read_messages, create_ack
from src.aki.hl7 import HL7Message, parse_hl7_date
from src.aki.predictor import AKIPredictor
from src.aki.pager import PagerService
from src.aki.monitoring import (
    start_metrics_server,
    MESSAGES_TOTAL,
    BLOOD_TESTS_TOTAL,
    AKI_PREDICTIONS_TOTAL,
    MLLP_RECONNECTS_TOTAL,
    ACTIVE_CONNECTIONS,
    BLOOD_TEST_VALUES
)


SHUTDOWN_FLAG = False   # Global flag for graceful shutdown
logger = logging.getLogger(__name__)


def setup_logging() -> None:
    """Configures structured JSON-like logging."""
    root = logging.getLogger()
    root.setLevel(getattr(logging, SETTINGS.LOG_LEVEL, logging.INFO))

    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter('{"ts": "%(asctime)s", "lvl": "%(levelname)s", "msg": "%(message)s"}')
    handler.setFormatter(formatter)
    root.addHandler(handler)

    logging.getLogger("urllib3").setLevel(logging.WARNING)


def handle_adt(msg: HL7Message, store: PatientStore) -> None:
    """Handles Admission/Discharge/Transfer events."""
    patient = store.get_or_create(msg.mrn)
    msg_type = msg.message_type

    # A03 = Discharge
    if "A03" in msg_type:
        store.log_discharge(msg.mrn)
        logger.info(f"Discharge: {msg.mrn}")
    else:
        # Implicit re-admission if receiving new ADT data
        if patient.is_discharged:
            patient.is_discharged = False

        # Update demographics
        pid = msg.get_segment("PID")
        if pid:
            sex = pid[8] if len(pid) > 8 else None
            dob = parse_hl7_date(pid[7]) if len(pid) > 7 else None
            store.log_demographics(msg.mrn, sex=sex, dob=dob)

        logger.info(f"Admit/Update Demographics: {msg.mrn}")


def handle_oru(msg: HL7Message, store: PatientStore, predictor: AKIPredictor, pager: PagerService) -> None:
    """
    Handles Observation Result (ORU) events.
    Updates history and runs inference on new creatinine results.
    """
    # Do not predict for discharged patients
    patient = store.get_or_create(msg.mrn)
    if patient.is_discharged:
        logger.info(f"Ignoring ORU for discharged patient {msg.mrn}")
        return

    result = msg.get_obx_value("CREATININE")
    if not result:
        return

    val, obs_time = result
    BLOOD_TESTS_TOTAL.inc()
    BLOOD_TEST_VALUES.observe(val)

    arrival_ts = time.time()

    # Log result (WAL + state update)
    store.log_result(msg.mrn, obs_time, val)

    # Run inference
    features = compute_features(patient)
    if predictor.predict(features):
        AKI_PREDICTIONS_TOTAL.labels(result="positive").inc()
        pager.send_page(msg.mrn, obs_time, arrival_ts)
        logger.info(f"AKI DETECTED: {msg.mrn}")
    else:
        AKI_PREDICTIONS_TOTAL.labels(result="negative").inc()
        logger.info(f"AKI NEGATIVE: {msg.mrn}")


def process_message(raw: str, store: PatientStore, predictor: AKIPredictor, pager: PagerService) -> bytes:
    """Router for HL7 message types."""
    msg = HL7Message(raw)

    metric_label = "unknown"
    if msg.message_type:
        if "A01" in msg.message_type:
            metric_label = "admit"
        elif "A03" in msg.message_type:
            metric_label = "discharge"
        elif "ORU" in msg.message_type:
            metric_label = "blood_test"
        else:
            metric_label = "other_adt"
    MESSAGES_TOTAL.labels(message_type=metric_label).inc()

    if msg.mrn:
        if "ADT" in msg.message_type:
            handle_adt(msg, store)
        elif "ORU" in msg.message_type:
            handle_oru(msg, store, predictor, pager)

    return create_ack()


def run_loop(store: PatientStore, predictor: AKIPredictor, pager: PagerService) -> None:
    """
    Main connection loop.
    Handles socket persistence, message parsing, and error recovery.
    """
    host, port = SETTINGS.mllp_host_port
    processed_count = 0

    while not SHUTDOWN_FLAG:
        try:
            logger.info(f"Connecting to MLLP Source {host}:{port}...")

            # Set a 10-second timeout ONLY for the initial handshake
            with socket.create_connection((host, port), timeout=10) as sock:
                sock.settimeout(5.)

                # Turn on OS-level TCP Keep-Alive
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)

                # Configure Keep-Alive timing
                # After 60 seconds of silence, send a heartbeat every 10 seconds.
                # Drop the connection if 5 heartbeats fail in a row.
                if hasattr(socket, "TCP_KEEPIDLE"):
                    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPIDLE, 60)
                    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPINTVL, 10)
                    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPCNT, 5)

                logger.info("Connected. Waiting for messages...")
                ACTIVE_CONNECTIONS.set(1)
                MLLP_RECONNECTS_TOTAL.inc()

                for raw_msg in read_messages(sock):
                    if SHUTDOWN_FLAG: break

                    # Ignore heartbeat yields from timeouts
                    if raw_msg is None: continue

                    response = process_message(raw_msg, store, predictor, pager)
                    sock.sendall(response)

                    # Periodic snapshot (WAL is primary persistence)
                    processed_count += 1
                    if processed_count % SETTINGS.SAVE_INTERVAL == 0:
                        store.save()

            # Socket closed normally by server
            ACTIVE_CONNECTIONS.set(0)

            if not SETTINGS.RECONNECT:
                logger.info("Connection closed by server. Reconnect disabled.")
                break

            logger.warning("Connection closed. Reconnecting in 2s...")
            time.sleep(2)

        except (ConnectionError, socket.timeout, OSError) as e:
            ACTIVE_CONNECTIONS.set(0)
            if not SHUTDOWN_FLAG:
                logger.error(f"Network error: {e}. Retrying in 2s...")
                time.sleep(2)
        except Exception:
            ACTIVE_CONNECTIONS.set(0)
            logger.exception("Critical error in event loop.")
            time.sleep(2)


def signal_handler(*_) -> None:
    global SHUTDOWN_FLAG
    logger.info("Shutdown signal received.")
    SHUTDOWN_FLAG = True


def main() -> None:
    setup_logging()

    logger.info(f"Starting Prometheus server on port {SETTINGS.PROMETHEUS_PORT}")
    start_metrics_server()

    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)

    store = PatientStore()
    store.hydrate()

    predictor = AKIPredictor()
    pager = PagerService()

    try:
        run_loop(store, predictor, pager)
    finally:
        logger.info("Shutting down... Saving state.")
        store.save()
        pager.shutdown()
        logger.info("Goodbye.")


if __name__ == "__main__":
    main()