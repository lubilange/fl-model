import os
import threading
import io
import torch
import random

from flask import Flask, request, send_file, jsonify
from authexample.task import Net, test

# =========================
# CONFIG
# =========================
PORT = int(os.environ.get("PORT", 8080))
FL_CLIENT_TOKEN = os.environ.get("FL_CLIENT_TOKEN", "SHARED_TOKEN")

CLIENT_FRACTION = 0.5

# =========================
# GLOBAL STATE
# =========================
global_model = Net()
model_lock = threading.Lock()

client_updates = []
registered_clients = set()

round_id = 0
model_version = 0

round_lock = threading.Lock()

final_metrics = {
    "accuracy": [],
    "loss": []
}

MODEL_PATH = "global_model.pt"

# =========================
# FLASK
# =========================
app = Flask(__name__)


@app.route("/")
def home():
    return "FL Server Running 🚀"


# =========================
# GET MODEL (avec metadata)
# =========================
@app.route("/get_model", methods=["GET"])
def get_model():
    with model_lock:
        buffer = io.BytesIO()
        torch.save({
            "model": global_model.state_dict(),
            "round": round_id,
            "version": model_version
        }, buffer)
        buffer.seek(0)

    return send_file(buffer, as_attachment=True, download_name="global_model.pt")


# =========================
# CLIENT WEIGHTS
# =========================
@app.route("/submit_weights", methods=["POST"])
def submit_weights():
    try:
        token = request.headers.get("Authorization", "").replace("Bearer ", "")
        if token != FL_CLIENT_TOKEN:
            return jsonify({"error": "Unauthorized"}), 401

        client_id = request.form.get("client_id", "unknown")

        file = request.files["weights"]
        buffer = io.BytesIO(file.read())
        client_state = torch.load(buffer, map_location="cpu")

        with round_lock:
            registered_clients.add(client_id)
            client_updates.append({
                "client_id": client_id,
                "state": client_state
            })

        return jsonify({"message": "Received", "round": round_id}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# =========================
# CLIENT SAMPLING (50%)
# =========================
def sample_clients():
    clients = list(registered_clients)
    if len(clients) == 0:
        return []
    return random.sample(clients, max(1, int(len(clients) * CLIENT_FRACTION)))


# =========================
# AGGREGATION FEDAVG
# =========================
@app.route("/aggregate", methods=["POST"])
def aggregate():
    global round_id, model_version, global_model

    with round_lock:
        if len(client_updates) == 0:
            return jsonify({"error": "No updates"}), 400

        round_id += 1
        model_version += 1

        selected_clients = sample_clients()

        selected_updates = [
            u["state"] for u in client_updates
            if u["client_id"] in selected_clients
        ]

        if len(selected_updates) == 0:
            return jsonify({"error": "No sampled clients"}), 400

        avg_state = {}
        for k in selected_updates[0].keys():
            avg_state[k] = torch.zeros_like(selected_updates[0][k])

        for update in selected_updates:
            for k in update:
                avg_state[k] += update[k]

        for k in avg_state:
            avg_state[k] /= len(selected_updates)

        with model_lock:
            global_model.load_state_dict(avg_state)
            torch.save(global_model.state_dict(), MODEL_PATH)

        client_updates.clear()

    return jsonify({
        "message": "Aggregation done ✔",
        "round": round_id,
        "version": model_version,
        "clients_used": len(selected_updates)
    })


# =========================
# VALIDATION REAL (FL STYLE)
# =========================
@app.route("/validate", methods=["POST"])
def validate():
    try:
        with model_lock:
            model = Net()
            model.load_state_dict(global_model.state_dict())

        device = torch.device("cpu")

        # ⚠️ à remplacer par ton vrai dataset test
        test_loader = None

        if test_loader is None:
            return jsonify({"error": "No test dataset configured"}), 400

        loss, acc = test(model, test_loader, device)

        final_metrics["accuracy"].append(acc)
        final_metrics["loss"].append(loss)

        return jsonify({
            "round": round_id,
            "accuracy": acc,
            "loss": loss
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# =========================
# FINAL REPORT (FIN FL)
# =========================
@app.route("/final_report", methods=["GET"])
def final_report():

    if len(final_metrics["accuracy"]) == 0:
        return jsonify({"error": "No metrics yet"}), 400

    return jsonify({
        "final_accuracy": sum(final_metrics["accuracy"]) / len(final_metrics["accuracy"]),
        "final_loss": sum(final_metrics["loss"]) / len(final_metrics["loss"]),
        "rounds": round_id,
        "model_version": model_version
    })


# =========================
# RUN
# =========================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT)
