import os
import threading
import io
import torch
from flask import Flask, request, send_file, jsonify

from authexample.task import Net

# =========================
# CONFIG
# =========================
PORT = int(os.environ.get("PORT", 8080))
FL_CLIENT_TOKEN = os.environ.get("FL_CLIENT_TOKEN", "SHARED_TOKEN")

# =========================
# GLOBAL STATE
# =========================
global_model = Net()
model_lock = threading.Lock()

client_updates = []   # 🔥 STOCKAGE DES CLIENTS
round_lock = threading.Lock()

# =========================
# FLASK
# =========================
app = Flask(__name__)


@app.route("/")
def home():
    return "FL Server Running 🚀"


# =========================
# GET GLOBAL MODEL
# =========================
@app.route("/get_model", methods=["GET"])
def get_model():
    with model_lock:
        buffer = io.BytesIO()
        torch.save(global_model.state_dict(), buffer)
        buffer.seek(0)

    return send_file(buffer, as_attachment=True, download_name="global_model.pt")


# =========================
# RECEIVE CLIENT WEIGHTS
# =========================
@app.route("/submit_weights", methods=["POST"])
def submit_weights():
    try:
        # auth
        token = request.headers.get("Authorization", "").replace("Bearer ", "")
        if token != FL_CLIENT_TOKEN:
            return jsonify({"error": "Unauthorized"}), 401

        file = request.files["weights"]
        buffer = io.BytesIO(file.read())
        client_state = torch.load(buffer, map_location="cpu")

        # stocker update client
        with round_lock:
            client_updates.append(client_state)

        return jsonify({"message": "Received"}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# =========================
# AGGREGATION (FEDAVG)
# =========================
@app.route("/aggregate", methods=["POST"])
def aggregate():
    global global_model

    with round_lock:
        if len(client_updates) == 0:
            return jsonify({"error": "No updates"}), 400

        avg_state = {}

        # init
        for k in client_updates[0].keys():
            avg_state[k] = torch.zeros_like(client_updates[0][k])

        # sum
        for update in client_updates:
            for k in update:
                avg_state[k] += update[k]

        # mean
        for k in avg_state:
            avg_state[k] /= len(client_updates)

        # update global model
        with model_lock:
            global_model.load_state_dict(avg_state)

        client_updates.clear()

    return jsonify({"message": "Model aggregated ✔"}), 200


# =========================
# RUN
# =========================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT)
