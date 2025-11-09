import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ---------------------------
# LOAD CSV
# ---------------------------
CSV_FILE = "evaluation_results.csv"

if not os.path.exists(CSV_FILE):
    raise FileNotFoundError("evaluation_results.csv not found!")

df = pd.read_csv(CSV_FILE)

# ---------------------------
# SPLIT LLaMA & MISTRAL RESULTS
# ---------------------------
split_index = df[df["Question"].isna()].index.tolist()

llama_df = df.iloc[:split_index[0]].copy()
mistral_df = df.iloc[split_index[0]+1:split_index[1]].copy()

print("✅ Loaded LLAMA rows :", len(llama_df))
print("✅ Loaded Mistral rows:", len(mistral_df))

# ----------------------------------------------
# CREATE OUTPUT FOLDER FOR PLOTS
# ----------------------------------------------
os.makedirs("plots", exist_ok=True)

# ----------------------------------------------
# AVERAGE METRICS BAR CHART
# ----------------------------------------------
metrics = ["Correctness", "Relevance", "Fluency", "Helpfulness"]

llama_avg = llama_df[metrics].mean()
mistral_avg = mistral_df[metrics].mean()

comparison_df = pd.DataFrame({
    "LLaMA 3.2": llama_avg,
    "Mistral": mistral_avg
})

plt.figure(figsize=(10,5))
comparison_df.plot(kind="bar")
plt.title("Average Score Comparison")
plt.ylabel("Score (1–10)")
plt.xticks(rotation=0)
plt.savefig("plots/avg_scores.png")
plt.show()

# ----------------------------------------------
# RESPONSE TIME LINE GRAPH
# ----------------------------------------------
plt.figure(figsize=(12,5))
plt.plot(llama_df["Response Time (s)"], marker='o', label="LLaMA 3.2")
plt.plot(mistral_df["Response Time (s)"], marker='o', label="Mistral")

plt.title("Response Time per Question")
plt.xlabel("Question Number")
plt.ylabel("Time (seconds)")
plt.legend()
plt.grid(True)
plt.savefig("plots/response_times.png")
plt.show()

# ----------------------------------------------
# HEATMAP — QUESTION WISE SCORES
# ----------------------------------------------
heat_df = pd.DataFrame({
    "LLaMA Avg Score": llama_df["Avg Score"].tolist(),
    "Mistral Avg Score": mistral_df["Avg Score"].tolist(),
})

plt.figure(figsize=(8,6))
sns.heatmap(heat_df, annot=True, cmap="coolwarm")
plt.title("Question-wise Score Comparison")
plt.savefig("plots/score_heatmap.png")
plt.show()

# ----------------------------------------------
# SCATTER — Correctness vs Response Time
# ----------------------------------------------
plt.figure(figsize=(10,5))
plt.scatter(llama_df["Correctness"], llama_df["Response Time (s)"], label="LLaMA 3.2")
plt.scatter(mistral_df["Correctness"], mistral_df["Response Time (s)"], label="Mistral")

plt.xlabel("Correctness Score")
plt.ylabel("Response Time (Seconds)")
plt.title("Correctness vs Response Time")
plt.legend()
plt.grid(True)
plt.savefig("plots/correctness_vs_time.png")
plt.show()

# ----------------------------------------------
# BEST & WORST QUESTIONS
# ----------------------------------------------
print("\n🔥 BEST questions for LLaMA:")
print(llama_df.nlargest(3, "Avg Score")[["Question", "Avg Score"]])

print("\n🔥 BEST questions for Mistral:")
print(mistral_df.nlargest(3, "Avg Score")[["Question", "Avg Score"]])

print("\n💀 WORST questions for LLaMA:")
print(llama_df.nsmallest(3, "Avg Score")[["Question", "Avg Score"]])

print("\n💀 WORST questions for Mistral:")
print(mistral_df.nsmallest(3, "Avg Score")[["Question", "Avg Score"]])

print("\n✅ All graphs saved in /plots folder.")
