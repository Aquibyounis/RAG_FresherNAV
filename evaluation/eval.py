import os
import time
import csv
import json
import subprocess
from main1 import run_query as llama_query
from main2 import run_query as mistral_query

# -------------------------------
# CONFIG
# -------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_FILE = os.path.join(BASE_DIR, "evaluation_results.csv")

# -------------------------------
# UPDATED EVALUATION QUESTIONS
# -------------------------------
questions = [
    "Does VIT-AP allow course registration directly through VTOP or is it done through a unique mail link each semester?",
    "Does every branch in VIT-AP follow FFCS or only selected schools?",
    "Is there a fixed maximum credit limit for all students, or does it change every semester by email?",

    "Do allowed and not-allowed exam items remain constant every semester, or does VIT-AP send fresh instructions before each CAT/FAT?",
    "Does VIT-AP provide CGPA-based attendance relaxation for CGPA > 9.0?",

    "Is ReFAT a fixed policy every semester, or does it vary and require checking email/mentor?",

    "Can course registration be done from VTOP > Exam Schedule?",

    "Are visitors and parents allowed inside hostels at VIT-AP?",
    "What color ID tag do day scholars use when entering campus?",
    "Until what time are visitors allowed inside VIT-AP campus?",

    "Are evening buses available for all students or only registered day scholars?",
    "Where is the Transport Office located on the campus?",

    "Does VIT-AP use a permanent course registration link every semester?",
    "Is add/drop always guaranteed, or only if announced via email?",
    "Do exam rules follow a single standard list, or does it depend on the mail circular for that semester?"
]

# -------------------------------
# Ollama Judge Function
# -------------------------------
def ollama_judge_score(question, answer, model_name):
    print(f"    🧪 Scoring answer using Llama2 judge for {model_name}...")

    prompt = f"""
You are an expert evaluator. Evaluate the model's answer to the question.
Score from 1 to 10 for:

1. Correctness
2. Relevance
3. Fluency
4. Helpfulness

Return ONLY JSON like: {{"correctness":X,"relevance":Y,"fluency":Z,"helpfulness":W}}

Question: {question}
Answer: {answer}
Model: {model_name}
"""

    try:
        result = subprocess.run(
            ["ollama", "run", "llama2"],
            input=prompt.encode("utf-8"),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True
        )
        content = result.stdout.decode("utf-8").strip()

        start = content.find("{")
        end = content.rfind("}") + 1
        json_str = content[start:end]

        scores = json.loads(json_str)

        print(f"    ✅ Score: {scores}")
        return scores

    except Exception as e:
        print(f"    ❌ Scoring failed for {model_name}: {e}")
        return {"correctness": 0, "relevance": 0, "fluency": 0, "helpfulness": 0}

# -------------------------------
# Evaluate One Model
# -------------------------------
def evaluate_model(model_name, query_func):
    print(f"\n===============================")
    print(f"🔥 STARTING EVALUATION FOR: {model_name}")
    print(f"===============================\n")

    results = []
    total_time = 0
    total_scores = {"correctness":0, "relevance":0, "fluency":0, "helpfulness":0}

    for i, q in enumerate(questions, start=1):
        print(f"\n🔹 [{model_name}] Question {i}/{len(questions)}:")
        print(f"    ❓ {q}")

        start = time.time()
        try:
            print(f"    🤖 Generating answer...")
            answer = query_func(q)
        except Exception as e:
            answer = f"Error: {e}"
        end = time.time()

        duration = round(end - start, 2)
        print(f"    ⏱️ Time taken: {duration}s")
        print(f"    💬 Answer: {answer[:120]}{'...' if len(answer)>120 else ''}")

        scores = ollama_judge_score(q, answer, model_name)
        avg_score = round(sum(scores.values())/4, 3)

        print(f"    ✅ Avg Score: {avg_score}")

        results.append({
            "Question": q,
            "Answer": answer,
            "Response Time (s)": duration,
            "Correctness": scores["correctness"],
            "Relevance": scores["relevance"],
            "Fluency": scores["fluency"],
            "Helpfulness": scores["helpfulness"],
            "Avg Score": avg_score
        })

        total_time += duration
        for k in total_scores:
            total_scores[k] += scores[k]

    num_questions = len(questions)
    avg = {k: round(total_scores[k]/num_questions, 3) for k in total_scores}
    avg["avg_time"] = round(total_time / num_questions, 2)

    print(f"\n✅ FINISHED MODEL: {model_name}")
    print(f"✅ Average Scores: {avg}\n")

    return results, avg

# -------------------------------
# Run Evaluations
# -------------------------------
print("\n🚀 STARTING FULL MODEL EVALUATION...\n")

print("⏳ Evaluating LLaMA 3.2 (RAG)...")
llama_results, llama_avg = evaluate_model("LLaMA 3.2", llama_query)

print("⏳ Evaluating Mistral (RAG)...")
mistral_results, mistral_avg = evaluate_model("Mistral", mistral_query)

# -------------------------------
# Save CSV
# -------------------------------
print("\n💾 Saving results to CSV...")

with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as csvfile:
    fieldnames = [
        "Question",
        "Answer",
        "Response Time (s)",
        "Correctness",
        "Relevance",
        "Fluency",
        "Helpfulness",
        "Avg Score"
    ]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()

    for r in llama_results:
        writer.writerow(r)

    writer.writerow({})

    for r in mistral_results:
        writer.writerow(r)

    writer.writerow({})

    writer.writerow({
        "Question": "AVERAGES (LLaMA 3.2)",
        "Response Time (s)": llama_avg["avg_time"],
        "Correctness": llama_avg["correctness"],
        "Relevance": llama_avg["relevance"],
        "Fluency": llama_avg["fluency"],
        "Helpfulness": llama_avg["helpfulness"],
        "Avg Score": round((llama_avg["correctness"]+llama_avg["relevance"]+llama_avg["fluency"]+llama_avg["helpfulness"])/4, 3)
    })

    writer.writerow({
        "Question": "AVERAGES (Mistral)",
        "Response Time (s)": mistral_avg["avg_time"],
        "Correctness": mistral_avg["correctness"],
        "Relevance": mistral_avg["relevance"],
        "Fluency": mistral_avg["fluency"],
        "Helpfulness": mistral_avg["helpfulness"],
        "Avg Score": round((mistral_avg["correctness"]+mistral_avg["relevance"]+mistral_avg["fluency"]+mistral_avg["helpfulness"])/4, 3)
    })

print(f"\n✅ Evaluation complete! Results saved to {OUTPUT_FILE}")
