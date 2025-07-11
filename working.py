import pandas as pd
from difflib import SequenceMatcher
import matplotlib.pyplot as plt

# === Step 1: Dummy Test Data ===
test_cases = [
    {
        "id": 1,
        "question": "What is the capital of India?",
        "expected": "New Delhi",
        "answer": "New Delhi is the capital of India.",
        "agent": "GPT-3.5"
    },
    {
        "id": 2,
        "question": "What is the capital of France?",
        "expected": "Paris",
        "answer": "It is Paris.",
        "agent": "Claude-3"
    },
    {
        "id": 3,
        "question": "What is 5 multiplied by 6?",
        "expected": "30",
        "answer": "The answer is 30.",
        "agent": "Mistral"
    },
    {
        "id": 4,
        "question": "Who discovered gravity?",
        "expected": "Isaac Newton",
        "answer": "Newton discovered gravity.",
        "agent": "Claude-3"
    },
    {
        "id": 5,
        "question": "What is the square root of 81?",
        "expected": "9",
        "answer": "The square root of 81 is 9.",
        "agent": "GPT-3.5"
    },
    {
        "id": 6,
        "question": "What is the capital of Italy?",
        "expected": "Rome",
        "answer": "The capital is Milan.",
        "agent": "Mistral"
    }
]

# === Step 2: Evaluate Each Answer ===
def evaluate(expected, answer):
    expected = expected.strip().lower()
    answer = answer.strip().lower()
    similarity = SequenceMatcher(None, expected, answer).ratio()
    passed = expected in answer or similarity > 0.75
    return {
        "score": round(similarity, 2),
        "pass": passed,
        "feedback": "Correct" if passed else "Incorrect"
    }

results = []

for test in test_cases:
    eval_result = evaluate(test["expected"], test["answer"])
    results.append({
        "id": test["id"],
        "question": test["question"],
        "expected": test["expected"],
        "answer": test["answer"],
        "agent": test["agent"],
        **eval_result
    })

# === Step 3: DataFrame + Analytics ===
df = pd.DataFrame(results)

# Overall Accuracy
total = len(df)
correct = df["pass"].sum()
accuracy = correct / total * 100
print(f"\n✅ Overall Accuracy: {accuracy:.2f}%")

# Agent-wise Summary
agent_stats = df.groupby("agent").agg({
    "pass": ["sum", "count"],
    "score": "mean"
})
agent_stats.columns = ["correct", "total", "avg_score"]
agent_stats["accuracy (%)"] = (agent_stats["correct"] / agent_stats["total"]) * 100
print("\n📊 Agent Stats:\n", agent_stats)

# === Step 4: Save CSVs ===
df.to_csv("dummy_agent_evaluation.csv", index=False)
agent_stats.to_csv("dummy_agent_summary.csv")

# === Step 5: Visualization ===
plt.figure(figsize=(8, 5))
plt.bar(agent_stats.index, agent_stats["accuracy (%)"], color=["green", "blue", "orange"])
plt.title("Agent Accuracy Comparison")
plt.ylabel("Accuracy (%)")
plt.ylim(0, 100)
plt.grid(axis='y')
plt.savefig("agent_accuracy_chart.png")
plt.show()
