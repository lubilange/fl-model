import os
import threading
import io
import torch
import flwr as fl

from torch.utils.data import DataLoader, TensorDataset
from flask import Flask, jsonify, send_file

from flwr.server.strategy import FedAvg
from authexample.task import Net, test

# =========================
# CONFIG
# =========================
GRPC_PORT = "0.0.0.0:8080"
HTTP_PORT = int(os.environ.get("PORT", 5000))

NUM_SERVER_ROUNDS = 10

# =========================
# GLOBAL STATE
# =========================
global_model = Net()
model_lock = threading.Lock()

metrics_history = []
metrics_lock = threading.Lock()

final_model_path = "final_model.pt"

# =========================
# STRATEGY WITH EVAL
# =========================
def evaluate(server_round, parameters, config):

    model = Net()
    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = {k: torch.tensor(v) for k, v in params_dict}
    model.load_state_dict(state_dict)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    # fake dataset
    X = torch.randn(200, 7)
    y = torch.randint(0, 2, (200,))

    loader = DataLoader(TensorDataset(X, y), batch_size=32)

    loss, acc = test(model, loader, device)

    print(f"📊 ROUND {server_round} | LOSS={loss:.4f} | ACC={acc:.4f}")

    with metrics_lock:
        metrics_history.append({
            "round": server_round,
            "loss": float(loss),
            "accuracy": float(acc),
        })

    return float(loss), {"accuracy": float(acc)}

# =========================
# START FLOWER SERVER
# =========================
def start_flower():

    strategy = FedAvg(
        evaluate_fn=evaluate
    )

    print("🚀 Flower gRPC server started on", GRPC_PORT)

    fl.server.start_server(
        server_address=GRPC_PORT,
        config=fl.server.ServerConfig(num_rounds=NUM_SERVER_ROUNDS),
        strategy=strategy,
    )

# =========================
# FLASK API (DASHBOARD)
# =========================
app = Flask(__name__)

@app.route("/")
def home():
    return "🔥 FL Server Running (HTTP)"

@app.route("/metrics")
def metrics():
    with metrics_lock:
        return jsonify(metrics_history)

@app.route("/status")
def status():
    return jsonify({
        "rounds": len(metrics_history),
        "last_metrics": metrics_history[-1] if metrics_history else None
    })

@app.route("/final_model")
def final_model():

    if not os.path.exists(final_model_path):
        return jsonify({"error": "Model not ready"}), 404

    return send_file(final_model_path, as_attachment=True)

# =========================
# RUN BOTH
# =========================
if __name__ == "__main__":

    # 🔥 Flower server thread
    threading.Thread(target=start_flower, daemon=True).start()

    # 🌐 Flask HTTP server
    print("🌐 Flask server started on port", HTTP_PORT)
    app.run(host="0.0.0.0", port=HTTP_PORT)
