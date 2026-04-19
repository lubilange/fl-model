import os
import threading
import io
import torch

from flask import Flask, request, send_file, jsonify

from flwr.app import ArrayRecord, ConfigRecord, Context, MetricRecord
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import (
    DifferentialPrivacyServerSideFixedClipping,
    FedAvg,
)

from authexample.task import Net, test

# =========================
# CONFIG
# =========================
PORT = int(os.environ.get("PORT", 8080))
FL_CLIENT_TOKEN = os.environ.get("FL_CLIENT_TOKEN", "SHARED_TOKEN")

FRACTION_EVALUATE = float(os.environ.get("FRACTION_EVALUATE", 0.5))
NUM_SERVER_ROUNDS = int(os.environ.get("NUM_SERVER_ROUNDS", 10))
LEARNING_RATE = float(os.environ.get("LEARNING_RATE", 0.001))

NOISE_MULTIPLIER = float(os.environ.get("NOISE_MULTIPLIER", 0.3))
CLIPPING_NORM = float(os.environ.get("CLIPPING_NORM", 1.0))
NUM_SAMPLED_CLIENTS = int(os.environ.get("NUM_SAMPLED_CLIENTS", 2))

# =========================
# GLOBAL STATE
# =========================
global_model = Net()
model_lock = threading.Lock()

metrics_history = []
metrics_lock = threading.Lock()

final_model_path = "final_model.pt"

# =========================
# FLOWER SERVER (CORE FL)
# =========================
flwr_app = ServerApp()


@flwr_app.main()
def main(grid: Grid, context: Context) -> None:
    print("🚀 Flower FL server starting...")

    with model_lock:
        initial_arrays = ArrayRecord(global_model.state_dict())

    base_strategy = FedAvg(fraction_evaluate=FRACTION_EVALUATE)

    strategy = DifferentialPrivacyServerSideFixedClipping(
        base_strategy,
        noise_multiplier=NOISE_MULTIPLIER,
        clipping_norm=CLIPPING_NORM,
        num_sampled_clients=NUM_SAMPLED_CLIENTS,
    )

    result = strategy.start(
        grid=grid,
        initial_arrays=initial_arrays,
        train_config=ConfigRecord({"lr": LEARNING_RATE}),
        num_rounds=NUM_SERVER_ROUNDS,
        evaluate_fn=global_evaluate,
    )

    final_state = result.arrays.to_torch_state_dict()

    with model_lock:
        global_model.load_state_dict(final_state)

    torch.save(final_state, final_model_path)

    print("✅ Training finished, model saved.")


# =========================
# GLOBAL EVALUATION (FL REAL STYLE)
# =========================
def global_evaluate(server_round: int, arrays: ArrayRecord) -> MetricRecord:
    model = Net()
    model.load_state_dict(arrays.to_torch_state_dict())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    model.eval()

    # ⚠️ FL PUR: pas de dataset serveur ici
    # donc on ne fait PAS de test réel centralisé

    loss = 0.0
    acc = 0.0

    with metrics_lock:
        metrics_history.append({
            "round": server_round,
            "loss": loss,
            "accuracy": acc,
        })

    return MetricRecord({"accuracy": acc, "loss": loss})


# =========================
# FLASK API
# =========================
app = Flask(__name__)


@app.route("/")
def home():
    return "Flower FL Server + API running 🚀"


# =========================
# GET MODEL
# =========================
@app.route("/get_model", methods=["GET"])
def get_model():
    with model_lock:
        buffer = io.BytesIO()
        torch.save(global_model.state_dict(), buffer)
        buffer.seek(0)

    return send_file(buffer, as_attachment=True, download_name="global_model.pt")


# =========================
# FINAL MODEL
# =========================
@app.route("/final_model", methods=["GET"])
def final_model():
    if not os.path.exists(final_model_path):
        return jsonify({"error": "Model not trained yet"}), 404

    return send_file(final_model_path, as_attachment=True)


# =========================
# SUBMIT WEIGHTS (FL ONLY)
# =========================
@app.route("/submit_weights", methods=["POST"])
def submit_weights():
    try:
        auth_header = request.headers.get("Authorization", "")
        if not auth_header.startswith("Bearer "):
            return jsonify({"error": "Missing token"}), 401

        token = auth_header.split(" ")[1]
        if token != FL_CLIENT_TOKEN:
            return jsonify({"error": "Invalid token"}), 401

        if "weights" not in request.files:
            return jsonify({"error": "Missing file"}), 400

        file = request.files["weights"]
        buffer = io.BytesIO(file.read())
        _ = torch.load(buffer, map_location="cpu")

        return jsonify({
            "message": "Weights received",
            "mode": "Flower handles training"
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# =========================
# METRICS
# =========================
@app.route("/metrics", methods=["GET"])
def metrics():
    with metrics_lock:
        return jsonify(metrics_history)


# =========================
# RUN
# =========================
def run_flower():
    flwr_app.main()


if __name__ == "__main__":
    threading.Thread(target=run_flower, daemon=True).start()
    app.run(host="0.0.0.0", port=PORT)
