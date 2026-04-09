# server_render.py
import os
import threading
import torch
from flask import Flask, request, send_file, jsonify
from flwr.app import ArrayRecord, ConfigRecord, Context, MetricRecord
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import DifferentialPrivacyServerSideFixedClipping, FedAvg
from authexample.task import Net
import io

# --- PORT Render ---
PORT = int(os.environ.get("PORT", 8080))

# --- Config Flower & DP ---
FRACTION_EVALUATE = float(os.environ.get("FRACTION_EVALUATE", 0.5))
NUM_SERVER_ROUNDS = int(os.environ.get("NUM_SERVER_ROUNDS", 10))
LEARNING_RATE = float(os.environ.get("LEARNING_RATE", 0.001))
NOISE_MULTIPLIER = float(os.environ.get("NOISE_MULTIPLIER", 0.3))
CLIPPING_NORM = float(os.environ.get("CLIPPING_NORM", 1.0))
NUM_SAMPLED_CLIENTS = int(os.environ.get("NUM_SAMPLED_CLIENTS", 2))

# --- Flower App ---
flwr_app = ServerApp()

# Modèle global accessible aux clients
global_model = Net()

@flwr_app.main()
def main(grid: Grid, context: Context) -> None:
    arrays = ArrayRecord(global_model.state_dict())
    base_strategy = FedAvg(fraction_evaluate=FRACTION_EVALUATE)
    dp_strategy = DifferentialPrivacyServerSideFixedClipping(
        base_strategy,
        noise_multiplier=NOISE_MULTIPLIER,
        clipping_norm=CLIPPING_NORM,
        num_sampled_clients=NUM_SAMPLED_CLIENTS,
    )

    dp_strategy.start(
        grid=grid,
        initial_arrays=arrays,
        train_config=ConfigRecord({"lr": LEARNING_RATE}),
        num_rounds=NUM_SERVER_ROUNDS,
        evaluate_fn=global_evaluate,
    )

def global_evaluate(server_round: int, arrays: ArrayRecord) -> MetricRecord:
    # Placeholder, à remplacer par vraie évaluation si besoin
    return MetricRecord({"accuracy": 0.0, "loss": 0.0})

# --- Flask ---
app = Flask(__name__)
# --- Token client autorisé (partagé pour tous les clients) ---
ALLOWED_TOKEN = os.environ.get("FL_CLIENT_TOKEN", "SHARED_TOKEN")

@app.route("/")
def home():
    return "Flower server is running 🚀"

@app.route("/get_model", methods=["GET"])
def get_model():
    buffer = io.BytesIO()
    torch.save(global_model.state_dict(), buffer)
    buffer.seek(0)
    return send_file(buffer, download_name="global_model.pt", as_attachment=True)

@app.route("/submit_weights", methods=["POST"])
def submit_weights():
    try:
        # =========================
        # 1. CHECK TOKEN (ZERO TRUST)
        # =========================
        auth_header = request.headers.get("Authorization", "")

        if not auth_header.startswith("Bearer "):
            return jsonify({"error": "Missing Bearer token"}), 401

        token = auth_header.split(" ")[1]

        SERVER_TOKEN = os.environ.get("FL_CLIENT_TOKEN", "SHARED_TOKEN")

        if token != SERVER_TOKEN:
            return jsonify({"error": "Invalid token"}), 401

        # =========================
        # 2. CHECK FILE
        # =========================
        if "weights" not in request.files:
            return jsonify({"error": "No weights file provided"}), 400

        file = request.files["weights"]

        buffer = io.BytesIO(file.read())
        client_state = torch.load(buffer, map_location="cpu")

        # =========================
        # 3. FEDAVG SIMPLE MERGE
        # =========================
        with torch.no_grad():
            for key in global_model.state_dict().keys():
                global_model.state_dict()[key].copy_(
                    0.5 * global_model.state_dict()[key] +
                    0.5 * client_state[key]
                )

        return jsonify({"message": "Weights received and merged!"}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# --- Lancer Flower en parallèle ---
def run_flower():
    flwr_app.main()

if __name__ == "__main__":
    threading.Thread(target=run_flower, daemon=True).start()
    app.run(host="0.0.0.0", port=PORT)
