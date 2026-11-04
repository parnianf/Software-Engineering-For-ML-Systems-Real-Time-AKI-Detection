# Software Engineering For ML Systems: Real-Time AKI Detection With HL7, ONNX And Kubernetes

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![ONNX Runtime](https://img.shields.io/badge/ONNX%20Runtime-005CED?logo=onnx&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-2496ED?logo=docker&logoColor=white)
![Kubernetes](https://img.shields.io/badge/Kubernetes-326CE5?logo=kubernetes&logoColor=white)
![Prometheus](https://img.shields.io/badge/Prometheus-E6522C?logo=prometheus&logoColor=white)

A containerised, real-time machine learning service that detects **acute kidney injury (AKI)** in hospital patients from a live stream of HL7 messages and pages the clinical team. It receives admissions, discharges and blood-test results over MLLP, keeps a crash-safe record of every patient's creatinine history, scores each new result with an XGBoost model served through ONNX Runtime, and sends an alert to the hospital pager. It is deployed on Kubernetes and monitored with Prometheus. Software Engineering for ML Systems module, MSc Computing (Artificial Intelligence & Machine Learning), Imperial College London, 2026.

On the simulated hospital stream the system reaches an **F3 score of 0.973** (recall 99.1%, precision 83.6%) with a 90th-percentile alert latency of about **5 ms**.

## How it works

```
 HL7 / MLLP stream                                                    HTTP
 (admit, discharge,   ┌────────────┐   ┌───────────────┐   ┌──────────────┐   ┌──────────────┐
  creatinine result) ─▶  mllp.py   ├──▶│    hl7.py     ├──▶│   state.py   ├──▶│ features.py  │
                      │ framing +  │   │ message parse │   │ patient store│   │ 10 features  │
                      │ ACK        │   └───────────────┘   │ + WAL journal│   └──────┬───────┘
                      └────────────┘                       └──────────────┘          │
                                                                                     ▼
                      Prometheus ◀── monitoring.py        ┌──────────────┐   ┌──────────────┐
                                                          │  pager.py    │◀──┤ predictor.py │
                                                          │ retries,     │   │ ONNX Runtime │
                                                          │ thread pool  │   │ + threshold  │
                                                          └──────────────┘   └──────────────┘
```

1. **Receive.** `mllp.py` reads MLLP-framed HL7 messages from a TCP socket (with TCP keep-alive), and acknowledges each one. `hl7.py` parses admission/discharge/transfer (ADT) and observation-result (ORU) messages.
2. **Track state.** `state.py` keeps every patient's demographics and creatinine history in memory. Discharged patients are ignored until readmitted, and out-of-order results are re-sorted by time.
3. **Featurise.** `features.py` turns a patient's history into a 10-feature vector: age, sex, baseline (lowest creatinine so far), current, peak, volatility (sample standard deviation), number of results, velocity (change per day since the previous result), ratio to baseline and absolute increase over baseline.
4. **Predict.** `predictor.py` runs the exported XGBoost model with ONNX Runtime and flags AKI when the positive-class probability reaches the tuned decision threshold (0.3175, chosen to favour recall, which the F3 metric rewards).
5. **Alert.** `pager.py` posts the patient's MRN and test time to the pager endpoint from a thread pool, so slow pager responses never block the message stream.

## Reliability

- **Write-ahead logging.** Every state change is appended to a journal and `fsync`-ed before it is used. A snapshot of the full state is written every 100 messages (to a temporary file that is then moved into place), after which the journal is truncated.
- **Crash recovery.** On start-up the service loads the latest snapshot (or, on first start, the hospital's historical results), then replays the journal on top, skipping corrupt lines. A pod can be killed at any moment without losing a result.
- **Resilient paging.** Up to three attempts with exponential back-off on timeouts and server errors; client errors (4xx) are not retried; every outcome is counted.
- **Resilient input.** Automatic reconnection to the message source, a connection timeout, OS-level keep-alive, and graceful shutdown on `SIGTERM`.
- **Recovery tool.** `scripts/recovery.py` replays a plain-text HL7 dump through the same code path, to catch up on messages missed during an outage (alerts are re-sent and the state is saved).

## Results

Evaluation on a replay of the simulated hospital message stream (`scripts/evaluate.py` compares the pages in the application log with the ground-truth AKI events):

| True positives | False positives | False negatives | Precision | Recall | **F3** | p90 latency | Run time |
|---|---|---|---|---|---|---|---|
| 1,483 | 291 | 13 | 83.6% | 99.1% | **0.973** | 0.005 s | 1 m 23 s |

F3 weights recall nine times as much as precision, which matches the clinical priority: a missed AKI case is far costlier than an extra page. True negatives are not counted because every blood test that does not trigger a page is a negative.

## Monitoring

An embedded Prometheus endpoint (port 8000) exposes:

| Metric | Type | Meaning |
|---|---|---|
| `aki_messages_received_total` | counter | HL7 messages received, by type |
| `aki_blood_tests_total` | counter | creatinine results processed |
| `aki_predictions_total` | counter | model outcomes (positive / negative) |
| `aki_pager_requests_total` | counter | pager requests (success / error / timeout) |
| `aki_mllp_reconnects_total`, `aki_mllp_errors_total` | counter | connection health |
| `aki_e2e_latency_seconds` | histogram | time from a result arriving to a successful page |
| `aki_blood_test_values` | histogram | distribution of creatinine values |
| `aki_active_connections` | gauge | live MLLP connections (0 or 1) |

`kubernetes/prometheus.yaml` deploys Prometheus with four alert rules: **MLLPConnectionLost** (no connection for 2 minutes), **SLALatencyBreach** (p90 latency above 3 s), **PagerRequestsFailing** (any pager errors or timeouts) and **MLLPNetworkUnstable** (frequent reconnects or socket errors).

## Repository structure

```
├── src/aki/
│   ├── app.py              # entry point: connection loop, message routing, graceful shutdown
│   ├── config.py           # settings from environment variables, model feature order
│   ├── mllp.py             # MLLP framing and acknowledgements
│   ├── hl7.py              # HL7 message parsing
│   ├── state.py            # patient store, write-ahead log, snapshots, recovery
│   ├── features.py         # 10-feature vector
│   ├── predictor.py        # ONNX Runtime inference and threshold
│   ├── pager.py            # non-blocking HTTP pager client with retries
│   ├── monitoring.py       # Prometheus metrics
│   └── convert_model.py    # converts the trained XGBoost model to ONNX
├── model/                  # exported model (model.onnx) and decision threshold
├── tests/                  # unit, integration and failure-injection tests (20 tests)
├── scripts/                # run_tests.sh, evaluate.py, recovery.py
├── kubernetes/             # deployment (with persistent volume) and Prometheus
├── Dockerfile
└── requirements.txt
```

## Running it

**Tests** (Python 3.10+; they need no external services):

```bash
pip install -r requirements.txt
./scripts/run_tests.sh
```

They cover HL7 and MLLP parsing, feature computation, journal replay and snapshot recovery, an end-to-end message-to-page flow, and failure scenarios such as dropped connections and reconnects.

**Configuration** is read from environment variables:

| Variable | Default | Purpose |
|---|---|---|
| `MLLP_ADDRESS` | `localhost:8440` | host and port of the HL7 message source |
| `PAGER_ADDRESS` | `localhost:8441` | host and port of the pager (`POST /page`, body `mrn,timestamp`) |
| `PROMETHEUS_PORT` | `8000` | metrics port |
| `MODEL_PATH`, `THRESHOLD_PATH` | `/model/...` | ONNX model and threshold files |
| `HISTORY_PATH` | `/data/history.csv` | historical creatinine results used on first start |
| `PERSISTENCE_PATH`, `JOURNAL_PATH` | `/state/...` | snapshot and journal files (mount a persistent volume) |
| `SAVE_INTERVAL` | `100` | messages between snapshots |
| `LOG_LEVEL`, `RECONNECT` | `INFO`, `true` | logging level; reconnect when the source closes |

**Docker.** The image expects the hospital's historical results at `data/history.csv`, which is not stored in this repository:

```bash
docker build -t aki-detection .
docker run --rm -p 8000:8000 -v "$(pwd)/state":/state \
  -e MLLP_ADDRESS=<message-source>:8440 -e PAGER_ADDRESS=<pager>:8441 aki-detection
```

**Kubernetes.** `kubernetes/deployment.yaml` runs a single replica (strategy `Recreate`, because the state files are single-writer) with a persistent volume for the snapshot and journal, an init container that provides the history file, and resource limits. Replace the `<registry>` image placeholders and the simulator address with your own, then:

```bash
kubectl apply -f kubernetes/deployment.yaml
kubectl apply -f kubernetes/prometheus.yaml
```

## What is not in this repository

- **Data and simulator.** The hospital history, message stream, ground-truth labels and the message simulator were provided by the module and are not redistributed.
- **Model training.** The model was trained offline; only the exported ONNX model and its threshold are included. `convert_model.py` documents the pickle-to-ONNX conversion (it needs the trained pickle, `joblib` and `onnxmltools`).

## Team

This project was developed in a team of four.
