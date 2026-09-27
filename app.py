"""
Flask web app for Student Placement Prediction.
"""
from flask import Flask, render_template, request, jsonify
from src.predict import predict_placement
import os

app = Flask(__name__)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json
        result = predict_placement({
            "cgpa":                       float(data.get("cgpa", 7.0)),
            "aptitude_score":             float(data.get("aptitude", 60)),
            "communication_skill_score":  float(data.get("communication", 60)),
            "coding_skill_score":         float(data.get("coding", 60)),
            "logical_reasoning_score":    float(data.get("logical", 60)),
            "internships_count":          int(data.get("internships", 0)),
            "projects_count":             int(data.get("projects", 0)),
            "certifications_count":       int(data.get("certifications", 0)),
            "backlogs":                   int(data.get("backlogs", 0)),
            "attendance_percentage":      float(data.get("attendance", 75)),
            "mock_interview_score":       float(data.get("mock_interview", 60)),
            "hackathons_participated":    int(data.get("hackathons", 0)),
            "github_repos":               int(data.get("github", 0)),
            "linkedin_connections":       int(data.get("linkedin", 0)),
            "extracurricular_score":      float(data.get("extracurricular", 50)),
            "leadership_score":           float(data.get("leadership", 50)),
            "study_hours_per_day":        float(data.get("study_hours", 3)),
        })
        return jsonify({"success": True, **result})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400


if __name__ == "__main__":
    if not os.path.exists("models/placement_model.pkl"):
        print("⚠️  No model found. Run: python -m src.train_model")
    app.run(debug=True, port=5000)