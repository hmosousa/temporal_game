import json
import re
from pathlib import Path

from flask import Flask, jsonify, render_template, request

from src.base import Timeline
from src.data import load_qtimelines

ROOT_DIR = Path(__file__).parent
ASSETS_DIR = ROOT_DIR / "assets"

app = Flask(__name__)

Q_TIMELINES = load_qtimelines("train", augment=False, use_cache=False)


def highlight_entities(text):
    pattern = r"<(\w+\d+)>(.*?)</\1>"
    return re.sub(pattern, r'<span class="entity \1">\2</span>', text)


def build_entities_dict(text):
    pattern = r"<(\w+\d+)>(.*?)</\1>"
    groups = re.findall(pattern, text)
    eid2name = {eid: name for eid, name in groups}
    return eid2name


def order_entities_by_appearance(context, entities):
    entity_positions = {}
    for entity in entities:
        match = re.search(f"<{entity}>(.*?)</{entity}>", context)
        if match:
            entity_positions[entity] = match.start()

    return sorted(entities, key=lambda e: entity_positions.get(e, float("inf")))


def _build_data(data):
    data["eid2name"] = build_entities_dict(data["context"])
    data["ordered_entities"] = order_entities_by_appearance(
        data["context"], data["entities"]
    )
    data["context"] = highlight_entities(data["context"])
    return data


@app.route("/")
def index():
    with open(ASSETS_DIR / "sample_data.json", "r") as f:
        data = json.load(f)
    data = _build_data(data)
    return render_template("index.html", data=data)


@app.route("/api/data", methods=["GET"])
def get_context():
    with open(ASSETS_DIR / "sample_data.json", "r") as f:
        data = json.load(f)
    data = _build_data(data)
    return jsonify(data)


@app.route("/api/temporal_closure", methods=["POST"])
def temporal_closure():
    data = request.json
    relations = data.get("timeline", [])
    app.logger.info(f"Received relations: {relations}")

    timeline = Timeline.from_relations(relations)
    closed_timeline = timeline.closure()  # Compute the temporal closure
    closed_relations = closed_timeline.to_dict()["relations"]
    app.logger.info(f"Computed timeline: {json.dumps(closed_relations, indent=2)}")

    return jsonify({"timeline": closed_relations})


@app.route("/questions")
def questions():
    # Load a sample question from the QTimelines dataset
    question_data = Q_TIMELINES[0]
    question_data["text"] = highlight_entities(question_data["text"])
    return render_template("questions.html", question=question_data)


@app.route("/api/submit_answer", methods=["POST"])
def submit_answer():
    data = request.json
    user_answer = data.get("answer")
    question = data.get("question")
    app.logger.info(f"User answered: {user_answer} for question: {question['text']}")

    # Here you can add logic to evaluate the answer or store it
    return jsonify({"status": "success", "message": "Answer submitted successfully"})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)
