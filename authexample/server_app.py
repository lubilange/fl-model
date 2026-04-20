import os
import threading
import io
import torch

from torch.utils.data import DataLoader, TensorDataset
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

flower_running = False


# =========================
# FLOWER SERVER
# =========================
flwr_app = ServerApp()


@flwr_app.main()
def main(grid: Grid, context: Context) -> None:
    global flower_running
    flower_running = True

    print("\n🚀 ===== FLOWER SERVER STARTED =====")
    print(f"⚙️ Rounds: {NUM_SERVER_ROUNDS}")
    print(f"👥 Clients: {NUM_SAMPLED_CLIENTS}")
    print(f"📉 Noise: {NOISE_MULTIPLIER} | Clipping: {CLIPPING_NORM}\n")

    with model_lock:
        initial_arrays = ArrayRecord(global_model.state_dict())

    strategy = DifferentialPrivacyServerSideFixedClipping(
        FedAvg(),
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

    print("\n✅ ===== TRAINING FINISHED =====")
    print("📦 Model saved:", final_model_path)


# =========================
# GLOBAL EVALUATION
# =========================
def global_evaluate(server_round: int, arrays: ArrayRecord) -> MetricRecord:

    model = Net()
    model.load_state_dict(arrays.to_torch_state_dict())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    # fake dataset
    X = torch.randn(200, 7)
    y = torch.randint(0, 2, (200,))

    loader = DataLoader(
        TensorDataset(X, y),
        batch_size=32,
        shuffle=False
    )

    loss, acc = test(model, loader, device)

    print(f"📊 ROUND {server_round} | LOSS={loss:.4f} | ACC={acc:.4f}")

    with metrics_lock:
        metrics_history.append({
            "round": server_round,
            "loss": float(loss),
            "accuracy": float(acc),
        })

    return MetricRecord({"loss": loss, "accuracy": acc})


# =========================
# FLASK API
# =========================
app = Flask(__name__)


@app.route("/")
def home():
    return "🔥 FL Server Running"


@app.route("/status")
def status():
    return jsonify({
        "flower_running": flower_running,
        "rounds": len(metrics_history),
        "last_metrics": metrics_history[-1] if metrics_history else None,
        "model_saved": os.path.exists(final_model_path)
    })


@app.route("/metrics")
def metrics():
    with metrics_lock:
        return jsonify(metrics_history)


@app.route("/get_model")
def get_model():
    with model_lock:
        buffer = io.BytesIO()
        torch.save(global_model.state_dict(), buffer)
        buffer.seek(0)

    return send_file(buffer, download_name="global_model.pt", as_attachment=True)


@app.route("/final_model")
def final_model():
    if not os.path.exists(final_model_path):
        return jsonify({"error": "Model not trained yet"}), 404

    return send_file(final_model_path, as_attachment=True)


@app.route("/submit_weights", methods=["POST"])
def submit_weights():
    print("📤 client sent weights")

    try:
        token = request.headers.get("Authorization", "").replace("Bearer ", "")

        if token != FL_CLIENT_TOKEN:
            return jsonify({"error": "Invalid token"}), 401

        if "weights" not in request.files:
            return jsonify({"error": "Missing file"}), 400

        file = request.files["weights"]
        _ = torch.load(io.BytesIO(file.read()), map_location="cpu")

        print("✔ weights received (NOT used by Flower, info only)")

        return jsonify({"message": "received"}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# =========================
# RUN
# =========================
def run_flower():
    flwr_app.main()


if __name__ == "__main__":
    print("🔥 Starting server...")
    threading.Thread(target=run_flower, daemon=True).start()
    app.run(host="0.0.0.0", port=PORT)
