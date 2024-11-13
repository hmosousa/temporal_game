import json
import re
import uuid
from datetime import datetime
from pathlib import Path

import datasets
from flask import Flask, jsonify, render_template, request, session

from src.base import Timeline
from src.data import load_qtimelines

ROOT_DIR = Path(__file__).parent
ASSETS_DIR = ROOT_DIR / "assets"
ANNOTATION_DIR = ROOT_DIR / "annotations"
ANNOTATION_DIR.mkdir(exist_ok=True)

app = Flask(__name__)
app.secret_key = "your-secret-key-here"  # Required for session management


def q_timelines_sample(n_samples=100):
    data = load_qtimelines("train", augment=False, use_cache=False)
    before_data = data.filter(lambda x: x["label"] == "<").select(range(n_samples // 4))
    after_data = data.filter(lambda x: x["label"] == ">").select(range(n_samples // 4))
    equal_data = data.filter(lambda x: x["label"] == "=").select(range(n_samples // 4))
    none_data = data.filter(lambda x: x["label"] == "").select(
        n_samples - 3 * (n_samples // 4)
    )
    sample = datasets.concatenate_datasets(
        [before_data, after_data, equal_data, none_data]
    )
    return sample


Q_TIMELINES = q_timelines_sample()


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


def get_session_id():
    if "session_id" not in session:
        session["session_id"] = str(uuid.uuid4())
        session["annotated_questions"] = []
    return session["session_id"]


def get_next_question():
    annotated = set(session.get("annotated_questions", []))
    available_indices = [i for i in range(len(Q_TIMELINES)) if i not in annotated]

    if not available_indices:
        return None

    return min(available_indices)


def save_annotation(session_id, question_idx, user_answer, correct_answer):
    annotation_file = ANNOTATION_DIR / f"{session_id}.jsonl"

    annotation = {
        "timestamp": datetime.now().isoformat(),
        "question_idx": question_idx,
        "user_answer": user_answer,
        "correct_answer": correct_answer,
        "question_text": Q_TIMELINES[question_idx]["text"],
    }

    with open(annotation_file, "a") as f:
        f.write(json.dumps(annotation) + "\n")


def load_user_annotations(user_id):
    """Load previously annotated questions for this user."""
    annotation_file = ANNOTATION_DIR / f"{user_id}.jsonl"
    annotated_questions = []

    if annotation_file.exists():
        with open(annotation_file) as f:
            for line in f:
                try:
                    annotation = json.loads(line.strip())
                    annotated_questions.append(annotation["question_idx"])
                except json.JSONDecodeError:
                    continue

    return annotated_questions


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


@app.route("/questions", methods=["GET", "POST"])
def questions():
    """Handle both the login form and questions display."""
    # If no user_id provided, show login form
    if request.method == "GET":
        return render_template("login.html")

    # Handle POST request (form submission)
    user_id = request.form.get("user_id")
    if not user_id:
        return render_template("login.html", error="Please enter a user ID")

    # Load previously annotated questions for this user
    annotated_questions = load_user_annotations(user_id)

    # Get next available question
    available_indices = [
        i for i in range(len(Q_TIMELINES)) if i not in annotated_questions
    ]
    if not available_indices:
        return render_template("completed.html", user_id=user_id)

    idx = min(available_indices)
    question_data = Q_TIMELINES[idx]
    question_data["text"] = highlight_entities(question_data["text"])

    return render_template(
        "questions.html",
        question=question_data,
        current_idx=idx,
        n_questions=len(Q_TIMELINES),
        user_id=user_id,
    )


@app.route("/api/submit_answer", methods=["POST"])
def submit_answer():
    data = request.json
    user_id = data.get("user_id")
    user_answer = data.get("answer")
    question_idx = data.get("question_idx")

    if not user_id:
        return jsonify({"status": "error", "message": "No user ID provided"}), 401

    # Get correct answer from dataset
    correct_answer = Q_TIMELINES[question_idx]["label"]

    # Save the annotation
    save_annotation(user_id, question_idx, user_answer, correct_answer)

    # Get next available question
    annotated_questions = load_user_annotations(user_id)
    available_indices = [
        i for i in range(len(Q_TIMELINES)) if i not in annotated_questions
    ]
    next_idx = min(available_indices) if available_indices else None

    return jsonify(
        {"status": "success", "correct_answer": correct_answer, "next_idx": next_idx}
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)
