"""
scripts/evaluate.py
Evaluation script for KudiWise AI.

Task A metrics:
  - ROUGE-L (review text quality vs reference reviews)
  - RMSE    (predicted rating vs actual dataset rating)

Task B metrics:
  - Hit Rate@10 (does the right item appear in top-10?)

Usage:
  python scripts/evaluate.py --endpoint https://kudiwise-ai-990094999937.europe-west1.run.app --samples 20
  Results saved to: data/eval_results.json
"""
import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict

import numpy as np
import requests
from requests.exceptions import RequestException
from rouge_score import rouge_scorer

# ─── Logging ─────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("kudiwise.eval")

# ─── Sample Test Cases ────────────────────────────────────────────────────────

TASK_A_TEST_CASES = [
    {
        "item_name": "Wireless Earbuds", "price_ngn": 18000, "true_rating": 3,
        "reference": "Decent sound but overpriced for a student on a tight budget."
    },
    {
        "item_name": "Power Bank 20000mAh", "price_ngn": 12000, "true_rating": 4,
        "reference": "Very useful for outages, charges phone twice. Worth every naira."
    },
    {
        "item_name": "Textbook (Used)", "price_ngn": 3500, "true_rating": 5,
        "reference": "Best value for exam prep. Helped me pass my semester."
    },
    {
        "item_name": "Laptop Cooling Pad", "price_ngn": 6000, "true_rating": 4,
        "reference": "Keeps laptop cool during long study sessions, solid buy."
    },
    {
        "item_name": "Cheap Android Phone", "price_ngn": 45000, "true_rating": 2,
        "reference": "Too expensive for what you get. Not student friendly at all."
    },
    {
        "item_name": "Extension Cable", "price_ngn": 2500, "true_rating": 5,
        "reference": "Every hostel needs one. Cheap, reliable, essential."
    },
    {
        "item_name": "Branded Backpack", "price_ngn": 22000, "true_rating": 2,
        "reference": "Nice bag but the price no make sense for student wallet."
    },
    {
        "item_name": "Scientific Calculator", "price_ngn": 4000, "true_rating": 5,
        "reference": "Essential for engineering students. Perfect value."
    },
    {
        "item_name": "Noise Cancel Headset", "price_ngn": 35000, "true_rating": 1,
        "reference": "Beautiful but my account wept. Cannot recommend for survival mode."
    },
    {
        "item_name": "USB Flash Drive 32GB", "price_ngn": 2800, "true_rating": 5,
        "reference": "Small, fast, stores everything. Best ₦2800 I ever spent."
    },
]

TASK_B_TEST_CASES = [
    {"need": "cheap study materials for engineering exam", "expected_domain": "goodreads"},
    {"need": "affordable food near campus",                "expected_domain": "yelp"},
    {"need": "budget laptop accessories",                  "expected_domain": "amazon"},
    {"need": "self improvement books under 5000 naira",    "expected_domain": "goodreads"},
    {"need": "productivity tools for student",             "expected_domain": "amazon"},
]

# NOTE: If you are getting 422 errors, it is highly likely that one of these fields
# does not match your Pydantic "Persona" model in core/models.py
STANDARD_PERSONA = {
    "student_level": "300",  # Changed from "300L" to match the Pydantic Enum
    "field_of_study": "Computer Science",
    "weekly_budget_ngn": 8000,
    "urgency": "moderate",
    "location": "Lagos",
}


# ─── Evaluator Class ─────────────────────────────────────────────────────────

class KudiWiseEvaluator:
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()
        self.scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

    def evaluate_task_a(self, n_samples: int) -> Dict[str, Any]:
        rouge_scores, rmse_values = [], []
        cases = TASK_A_TEST_CASES[:n_samples]

        for i, case in enumerate(cases, 1):
            try:
                resp = self.session.post(
                    f"{self.base_url}/review",
                    json={
                        "persona": STANDARD_PERSONA,
                        "item_name": case["item_name"],
                        "price_ngn": case["price_ngn"],
                    },
                    timeout=30
                )

                # Explicitly catch and print FastAPI Validation Errors
                if resp.status_code == 422:
                    logger.error("Task A Validation Error (422) on case %d: %s", i, resp.text)
                    continue

                resp.raise_for_status()
                data = resp.json()

                # ROUGE-L vs Reference
                score = self.scorer.score(case["reference"], data.get("review", ""))
                rouge_l_f1 = score["rougeL"].fmeasure
                rouge_scores.append(rouge_l_f1)

                # RMSE
                squared_error = (data.get("rating", 0) - case["true_rating"]) ** 2
                rmse_values.append(squared_error)

                logger.info("Task A [%d/%d] | %s → rating=%s, ROUGE-L=%.3f",
                            i, len(cases), case["item_name"][:20], data.get("rating"), rouge_l_f1)

            except RequestException as exc:
                logger.error("Task A case %d failed: %s", i, exc)
            except Exception as exc:
                logger.error("Task A case %d unexpected error: %s", i, exc)

        avg_rouge = float(np.mean(rouge_scores)) if rouge_scores else 0.0
        rmse = float(np.sqrt(np.mean(rmse_values))) if rmse_values else 0.0

        return {
            "task": "A",
            "n_samples": len(cases),
            "avg_rouge_l": round(avg_rouge, 4),
            "rmse": round(rmse, 4),
            "samples_evaluated": len(rouge_scores),
        }

    def evaluate_task_b(self, n_samples: int) -> Dict[str, Any]:
        hits = []
        cases = TASK_B_TEST_CASES[:n_samples]

        for i, case in enumerate(cases, 1):
            try:
                resp = self.session.post(
                    f"{self.base_url}/recommend",
                    json={
                        "persona": STANDARD_PERSONA,
                        "need": case["need"],
                    },
                    timeout=30
                )

                # Explicitly catch and print FastAPI Validation Errors
                if resp.status_code == 422:
                    logger.error("Task B Validation Error (422) on case %d: %s", i, resp.text)
                    continue

                resp.raise_for_status()
                data = resp.json()

                domains_returned = [r.get("domain", "") for r in data.get("recommendations", [])]
                hit = case["expected_domain"] in domains_returned
                hits.append(int(hit))

                logger.info("Task B [%d/%d] | %s... → hit=%s (domains=%s)",
                            i, len(cases), case["need"][:30], hit, set(domains_returned))

            except RequestException as exc:
                logger.error("Task B case %d failed: %s", i, exc)
            except Exception as exc:
                logger.error("Task B case %d unexpected error: %s", i, exc)

        hit_rate = float(np.mean(hits)) if hits else 0.0
        return {
            "task": "B",
            "n_samples": len(cases),
            "hit_rate_at_10": round(hit_rate, 4),
            "samples_evaluated": len(hits),
        }


# ─── Execution Pipeline ──────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="KudiWise AI Evaluation Test Suite")
    parser.add_argument("--endpoint", default="https://kudiwise-ai-990094999937.europe-west1.run.app", help="Base URL of the KudiWise API")
    parser.add_argument("--samples", type=int, default=10, help="Number of samples to evaluate per task")
    parser.add_argument("--output", default="data/eval_results.json", help="Path to save the JSON results")
    args = parser.parse_args()

    logger.info("Starting evaluation against endpoint: %s", args.endpoint)
    evaluator = KudiWiseEvaluator(args.endpoint)

    logger.info("Running Task A Evaluator (Review Generation)...")
    task_a_results = evaluator.evaluate_task_a(args.samples)

    logger.info("Running Task B Evaluator (Recommendation Matching)...")
    task_b_results = evaluator.evaluate_task_b(args.samples)

    # Compile Results
    results_payload = {
        "task_a": task_a_results,
        "task_b": task_b_results,
        "summary": {
            "rouge_l": task_a_results["avg_rouge_l"],
            "rmse": task_a_results["rmse"],
            "hit_rate_at_10": task_b_results["hit_rate_at_10"],
        },
    }

    # Output to File
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results_payload, f, indent=2)

    # Output to Console
    logger.info("\n" + "═" * 45)
    logger.info(" 📊 KUDIWISE EVALUATION SUMMARY ")
    logger.info("═" * 45)
    logger.info(" Task A - ROUGE-L     : %.4f", task_a_results["avg_rouge_l"])
    logger.info(" Task A - RMSE        : %.4f", task_a_results["rmse"])
    logger.info(" Task B - Hit Rate@10 : %.4f", task_b_results["hit_rate_at_10"])
    logger.info("═" * 45)
    logger.info(" Results successfully saved to: %s", args.output)


if __name__ == "__main__":
    main()