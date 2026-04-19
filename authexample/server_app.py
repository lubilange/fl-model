import os
import threading
import io
import torch
import copy
from flask import Flask, request, send_file, jsonify

from authexample.task import Net, test


# =========================
# CONFIG
# =========================
PORT = int(os.environ.get("PORT", 8080))
FL_CLIENT_TOKEN = os.environ.get("FL_CLIENT_TOKEN", "SHARED_TOKEN")
NUM_CLIENTS_EXPECTED = int(os.environ.get("NUM_CLIENTS_EXPECTED", 2))


# =========================
# GLOBAL MODEL
# =========================
global_model = Net()
model_lock = threading.Lock()


# =========================
# STORAGE CLIENT WEIGHTS
# =========================
client_weights_buffer = []
buffer_lock = threading.Lock()


metrics_history = []
metrics_lock = threading.Lock()

final_model_path = "final_model.pt"


# =========================
# FEDAVG FUNCTION
# =========================
def fedavg(weight_list):
    avg_weights = copy.deepcopy(weight_list[0])

    for key in avg_weights.keys():
        for i in range(1, len(weight_list)):
            avg_weights[key] += weight_list[i][key]
        avg_weights[key] = avg_weights[key] / len(weight_list)

    return avg_weights


# =========================
# FLASK APP
# =========================
app = Flask(__name__)


@app.route("/")
def home():
    return "🔥 Manual FL Server (FedAvg) Running"


# =========================
# SUBMIT WEIGHTS
# =========================
@app.route("/submit_weights", methods=["POST"])
def submit_weights():
    global client_weights_buffer

    print("📤 Received client weights")

    try:
        token = request.headers.get("Authorization", "").replace("Bearer ", "")
        if token != FL_CLIENT_TOKEN:
            return jsonify({"error": "Invalid token"}), 401

        if "weights" not in request.files:
            return jsonify({"error": "Missing file"}), 400

        file = request.files["weights"]
        state_dict = torch.load(io.BytesIO(file.read()), map_location="cpu")

        with buffer_lock:
            client_weights_buffer.append(state_dict)

        print(f"✔ Stored weights: {len(client_weights_buffer)}")

        # =========================
        # WHEN ALL CLIENTS ARRIVE → AGGREGATE
        # =========================
        if len(client_weights_buffer) >= NUM_CLIENTS_EXPECTED:

            print("🔥 Aggregating FedAvg...")

            new_global = fedavg(client_weights_buffer)

            with model_lock:
                global_model.load_state_dict(new_global)

            torch.save(new_global, final_model_path)

            print("✅ Global model updated")

            client_weights_buffer = []  # reset buffer

        return jsonify({"message": "weights received"}), 200

    except Exception as e:
        print("❌ ERROR:", e)
        return jsonify({"error": str(e)}), 500


# =========================
# GET MODEL
# =========================
@app.route("/get_model")
def get_model():
    with model_lock:
        buffer = io.BytesIO()
        torch.save(global_model.state_dict(), buffer)
        buffer.seek(0)

    return send_file(buffer, download_name="global_model.pt", as_attachment=True)


# =========================
# FINAL MODEL
# =========================
@app.route("/final_model")
def final_model():
    if not os.path.exists(final_model_path):
        return jsonify({"error": "Model not trained yet"}), 404

    return send_file(final_model_path, as_attachment=True)


# =========================
# STATUS
# =========================
@app.route("/status")
def status():
    return jsonify({
        "clients_received": len(client_weights_buffer),
        "expected_clients": NUM_CLIENTS_EXPECTED,
        "model_saved": os.path.exists(final_model_path)
    })


# =========================
# RUN
# =========================
if __name__ == "__main__":
    print("🔥 Starting FedAvg server...")
    app.run(host="0.0.0.0", port=PORT)
