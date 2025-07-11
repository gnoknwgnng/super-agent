import streamlit as st
import pandas as pd
from difflib import SequenceMatcher
import matplotlib.pyplot as plt

# === Dummy Test Data ===
test_cases = [
    {"id": 1, "question": "What is the capital of India?", "expected": "New Delhi",
     "answer": "New Delhi is the capital of India.", "agent": "GPT-3.5"},
    {"id": 2, "question": "What is the capital of France?", "expected": "Paris",
     "answer": "It is Paris.", "agent": "Claude-3"},
    {"id": 3, "question": "What is 5 multiplied by 6?", "expected": "30",
     "answer": "The answer is 30.", "agent": "Mistral"},
    {"id": 4, "question": "Who discovered gravity?", "expected": "Isaac Newton",
     "answer": "Newton discovered gravity.", "agent": "Claude-3"},
    {"id": 5, "question": "What is the square root of 81?", "expected": "9",
     "answer": "The square root of 81 is 9.", "agent": "GPT-3.5"},
    {"id": 6, "question": "What is the capital of Italy?", "expected": "Rome",
     "answer": "The capital is Milan.", "agent": "Mistral"}
]

# === Evaluation Function ===
def evaluate(expected, answer):
    expected = expected.lower().strip()
    answer = answer.lower().strip()
    similarity = SequenceMatcher(None, expected, answer).ratio()
    passed = expected in answer or similarity > 0.75
    return round(similarity, 2), passed, "Correct" if passed else "Incorrect"

# === Evaluate ===
evaluated = []
for test in test_cases:
    score, passed, feedback = evaluate(test["expected"], test["answer"])
    evaluated.append({
        "Agent": test["agent"],
        "Question": test["question"],
        "Expected": test["expected"],
        "Answer": test["answer"],
        "Score": score,
        "Pass": passed,
        "Feedback": feedback
    })

df = pd.DataFrame(evaluated)

# === Analytics Summary ===
summary = df.groupby("Agent").agg({
    "Pass": ["sum", "count"],
    "Score": "mean"
})
summary.columns = ["Correct", "Total", "Avg Score"]
summary["Accuracy (%)"] = (summary["Correct"] / summary["Total"]) * 100

# === Streamlit UI ===
st.set_page_config(page_title="Super Agent Analytics", layout="wide")
st.title("🧠 Super Agent Testing Dashboard")

st.subheader("📊 Agent-wise Summary")
st.dataframe(summary.style.format({"Avg Score": "{:.2f}", "Accuracy (%)": "{:.2f}"}))

st.subheader("📋 Detailed Evaluation")
st.dataframe(df)

# === Visualization ===
st.subheader("📈 Accuracy Comparison Chart")
fig, ax = plt.subplots(figsize=(7, 4))
ax.bar(summary.index, summary["Accuracy (%)"], color=["green", "blue", "orange"])
ax.set_ylabel("Accuracy (%)")
ax.set_ylim(0, 100)
ax.set_title("Agent Accuracy")
st.pyplot(fig)
