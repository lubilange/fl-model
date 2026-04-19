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

from authexample.task import Net, load_centralized_dataset, test

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
# GLOBAL STATE (THREAD SAFE)
# =========================
global_model = Net()
model_lock = threading.Lock()

# =========================
# FLOWER SERVER
# =========================
flwr_app = ServerApp()


@flwr_app.main()
def main(grid: Grid, context: Context) -> None:
    print("🚀 Starting Flower server...")

    # Initial model
    with model_lock:
        arrays = ArrayRecord(global_model.state_dict())

    # FedAvg
    base_strategy = FedAvg(fraction_evaluate=FRACTION_EVALUATE)

    # DP wrapper
    dp_strategy = DifferentialPrivacyServerSideFixedClipping(
        base_strategy,
        noise_multiplier=NOISE_MULTIPLIER,
        clipping_norm=CLIPPING_NORM,
        num_sampled_clients=NUM_SAMPLED_CLIENTS,
    )

    # Start FL
    result = dp_strategy.start(
        grid=grid,
        initial_arrays=arrays,
        train_config=ConfigRecord({"lr": LEARNING_RATE}),
        num_rounds=NUM_SERVER_ROUNDS,
        evaluate_fn=global_evaluate,
    )

    # Save final model
    print("💾 Saving final model...")
    final_state = result.arrays.to_torch_state_dict()

    with model_lock:
        global_model.load_state_dict(final_state)

    torch.save(final_state, "final_model.pt")
    print("✅ Training complete!")


# =========================
# REAL EVALUATION
# =========================
def global_evaluate(server_round: int, arrays: ArrayRecord) -> MetricRecord:
    print(f"📊 Evaluating round {server_round}...")

    model = Net()
    model.load_state_dict(arrays.to_torch_state_dict())

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    test_loader = load_centralized_dataset()

    loss, acc = test(model, test_loader, device)

    print(f"➡️ Round {server_round} | Loss: {loss:.4f} | Acc: {acc:.4f}")

    return MetricRecord({"accuracy": acc, "loss": loss})


# =========================
# FLASK API
# =========================
app = Flask(__name__)


@app.route("/")
def home():
    return "Flower + Flask server running 🚀"


# -------------------------
# GET GLOBAL MODEL
# -------------------------
@app.route("/get_model", methods=["GET"])
def get_model():
    with model_lock:
        buffer = io.BytesIO()
        torch.save(global_model.state_dict(), buffer)
        buffer.seek(0)

    return send_file(buffer, download_name="global_model.pt", as_attachment=True)


# -------------------------
# SUBMIT CLIENT WEIGHTS
# -------------------------
@app.route("/submit_weights", methods=["POST"])
def submit_weights():
    try:
        # 🔐 AUTH
        auth_header = request.headers.get("Authorization", "")
        if not auth_header.startswith("Bearer "):
            return jsonify({"error": "Missing Bearer token"}), 401

        token = auth_header.split(" ")[1]
        if token != FL_CLIENT_TOKEN:
            return jsonify({"error": "Invalid token"}), 401

        # 📂 FILE CHECK
        if "weights" not in request.files:
            return jsonify({"error": "No weights file provided"}), 400

        file = request.files["weights"]
        buffer = io.BytesIO(file.read())
        client_state = torch.load(buffer, map_location="cpu")

        # 🔄 SAFE MERGE (simple fallback, NOT main FL)
        with model_lock:
            current_state = global_model.state_dict()

            for key in current_state.keys():
                current_state[key] = (
                    0.9 * current_state[key] + 0.1 * client_state[key]
                )

            global_model.load_state_dict(current_state)

        return jsonify({"message": "Weights merged (fallback mode)"}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# =========================
# RUN BOTH SERVERS
# =========================
def run_flower():
    flwr_app.main()


if __name__ == "__main__":
    # Run Flower in background
    threading.Thread(target=run_flower, daemon=True).start()

    # Run Flask
    app.run(host="0.0.0.0", port=PORT)
