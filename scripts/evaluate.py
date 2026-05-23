"""
scripts/evaluate.py
Evaluation script for KudiWise AI.

Task A metrics:
  - ROUGE-L (review text quality vs reference reviews)
  - RMSE    (predicted rating vs actual dataset rating)

Task B metrics:
  - Hit Rate@10 (does the right item appear in top-10?)

Usage:
  python scripts/evaluate.py --endpoint http://localhost:8000 --samples 20
  Results saved to: data/eval_results.json
"""
import argparse
import json
import math
import logging
import random
from pathlib import Path

import requests
import numpy as np
from rouge_score import rouge_scorer

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
logger = logging.getLogger("kudiwise.eval")

# ─── Sample test cases ────────────────────────────────────────────────────────

TASK_A_TEST_CASES = [
    {"item_name": "Wireless Earbuds",     "price_ngn": 18000, "true_rating": 3},
    {"item_name": "Power Bank 20000mAh",  "price_ngn": 12000, "true_rating": 4},
    {"item_name": "Textbook (Used)",      "price_ngn":  3500, "true_rating": 5},
    {"item_name": "Laptop Cooling Pad",   "price_ngn":  6000, "true_rating": 4},
    {"item_name": "Cheap Android Phone",  "price_ngn": 45000, "true_rating": 2},
    {"item_name": "Extension Cable",      "price_ngn":  2500, "true_rating": 5},
    {"item_name": "Branded Backpack",     "price_ngn": 22000, "true_rating": 2},
    {"item_name": "Scientific Calculator","price_ngn":  4000, "true_rating": 5},
    {"item_name": "Noise Cancel Headset", "price_ngn": 35000, "true_rating": 1},
    {"item_name": "USB Flash Drive 32GB", "price_ngn":  2800, "true_rating": 5},
]

REFERENCE_REVIEWS = [
    "Decent sound but overpriced for a student on a tight budget.",
    "Very useful for outages, charges phone twice. Worth every naira.",
    "Best value for exam prep. Helped me pass my semester.",
    "Keeps laptop cool during long study sessions, solid buy.",
    "Too expensive for what you get. Not student friendly at all.",
    "Every hostel needs one. Cheap, reliable, essential.",
    "Nice bag but the price no make sense for student wallet.",
    "Essential for engineering students. Perfect value.",
    "Beautiful but my account wept. Cannot recommend for survival mode.",
    "Small, fast, stores everything. Best ₦2800 I ever spent.",
]

TASK_B_TEST_CASES = [
    {"need": "cheap study materials for engineering exam", "expected_domain": "goodreads"},
    {"need": "affordable food near campus",                "expected_domain": "yelp"},
    {"need": "budget laptop accessories",                  "expected_domain": "amazon"},
    {"need": "self improvement books under 5000 naira",    "expected_domain": "goodreads"},
    {"need": "productivity tools for student",             "expected_domain": "amazon"},
]

STANDARD_PERSONA = {
    "student_level": "300L",
    "field_of_study": "Computer Science",
    "weekly_budget_ngn": 8000,
    "urgency": "moderate",
    "location": "Lagos",
}


# ─── Task A evaluation ────────────────────────────────────────────────────────

def evaluate_task_a(endpoint: str, n: int) -> dict:
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    rouge_scores, rmse_values = [], []
    cases = TASK_A_TEST_CASES[:n]

    for i, case in enumerate(cases):
        try:
            resp = requests.post(f"{endpoint}/review", json={
                "persona": STANDARD_PERSONA,
                "item_name": case["item_name"],
                "price_ngn": case["price_ngn"],
            }, timeout=30)
            resp.raise_for_status()
            data = resp.json()

            # ROUGE-L against reference
            reference = REFERENCE_REVIEWS[i % len(REFERENCE_REVIEWS)]
            score = scorer.score(reference, data["review"])
            rouge_scores.append(score["rougeL"].fmeasure)

            # RMSE
            rmse_values.append((data["rating"] - case["true_rating"]) ** 2)

            logger.info(
                "Task A [%d] %s → rating=%s ROUGE-L=%.3f",
                i, case["item_name"], data["rating"], score["rougeL"].fmeasure,
            )
        except Exception as exc:
            logger.error("Task A case %d failed: %s", i, exc)

    avg_rouge = float(np.mean(rouge_scores)) if rouge_scores else 0.0
    rmse = float(math.sqrt(np.mean(rmse_values))) if rmse_values else 0.0

    return {
        "task": "A",
        "n_samples": len(cases),
        "avg_rouge_l": round(avg_rouge, 4),
        "rmse": round(rmse, 4),
        "samples_evaluated": len(rouge_scores),
    }


# ─── Task B evaluation ────────────────────────────────────────────────────────

def evaluate_task_b(endpoint: str, n: int) -> dict:
    hits = []
    cases = TASK_B_TEST_CASES[:n]

    for case in cases:
        try:
            resp = requests.post(f"{endpoint}/recommend", json={
                "persona": STANDARD_PERSONA,
                "need": case["need"],
            }, timeout=30)
            resp.raise_for_status()
            data = resp.json()

            domains_returned = [r["domain"] for r in data["recommendations"]]
            hit = case["expected_domain"] in domains_returned
            hits.append(int(hit))

            logger.info(
                "Task B [%s] hit=%s domains=%s",
                case["need"][:40], hit, set(domains_returned),
            )
        except Exception as exc:
            logger.error("Task B case failed: %s", exc)
            hits.append(0)

    hit_rate = float(np.mean(hits)) if hits else 0.0
    return {
        "task": "B",
        "n_samples": len(cases),
        "hit_rate_at_10": round(hit_rate, 4),
        "samples_evaluated": len(hits),
    }


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="KudiWise AI evaluator")
    parser.add_argument("--endpoint", default="http://localhost:8000")
    parser.add_argument("--samples",  type=int, default=10)
    parser.add_argument("--output",   default="data/eval_results.json")
    args = parser.parse_args()

    logger.info("Evaluating against: %s", args.endpoint)

    task_a = evaluate_task_a(args.endpoint, args.samples)
    task_b = evaluate_task_b(args.endpoint, args.samples)

    results = {
        "task_a": task_a,
        "task_b": task_b,
        "summary": {
            "rouge_l": task_a["avg_rouge_l"],
            "rmse": task_a["rmse"],
            "hit_rate_at_10": task_b["hit_rate_at_10"],
        },
    }

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)

    logger.info("─── RESULTS ───────────────────────────────")
    logger.info("Task A ROUGE-L : %.4f", task_a["avg_rouge_l"])
    logger.info("Task A RMSE    : %.4f", task_a["rmse"])
    logger.info("Task B Hit@10  : %.4f", task_b["hit_rate_at_10"])
    logger.info("Saved to: %s", args.output)


if __name__ == "__main__":
    main()
