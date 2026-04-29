"""
Mumzworld Gift Finder — Evaluation Suite
Rubric: relevance, budget adherence, hallucination, uncertainty handling, Arabic quality, schema validity
Run: python evals.py
"""

import json
import time
from gift_finder import find_gifts, GiftFinderResult

# ── Test cases ────────────────────────────────────────────────────────────────
# Each case: query, expected_behavior, budget_aed (or None), expect_results (True/False),
#            notes for grader
TEST_CASES = [
    # ── Happy path ────────────────────────────────────────────────────────────
    {
        "id": "TC01",
        "category": "happy_path",
        "query": "Thoughtful gift for a friend with a 6-month-old, under 200 AED",
        "expect_results": True,
        "budget_aed": 200,
        "notes": "Classic use case. Should return 2-4 relevant products all under AED 200.",
    },
    {
        "id": "TC02",
        "category": "happy_path",
        "query": "Baby shower gift for a new mom expecting twins, budget 350 AED",
        "expect_results": True,
        "budget_aed": 350,
        "notes": "Budget stated. Should prioritise gift sets or versatile items.",
    },
    {
        "id": "TC03",
        "category": "happy_path",
        "query": "Gift for a toddler who loves building blocks and is 2 years old",
        "expect_results": True,
        "budget_aed": None,
        "notes": "Age-specific. Should surface DUPLO or wooden blocks.",
    },
    {
        "id": "TC04",
        "category": "happy_path",
        "query": "Self-care gift for a pregnant friend who loves natural skincare",
        "expect_results": True,
        "budget_aed": None,
        "notes": "Mom-focused, not baby-focused. Should return maternity skincare items.",
    },
    {
        "id": "TC05",
        "category": "happy_path",
        "query": "Safe car seat for a newborn under 1000 AED",
        "expect_results": True,
        "budget_aed": 1000,
        "notes": "Safety-critical category. Should return infant car seats, all under AED 1000.",
    },

    # ── Budget-edge cases ─────────────────────────────────────────────────────
    {
        "id": "TC06",
        "category": "budget_edge",
        "query": "Best stroller under 50 AED",
        "expect_results": False,
        "budget_aed": 50,
        "notes": "Impossible budget for strollers. Model should indicate it cannot help or no products fit.",
    },
    {
        "id": "TC07",
        "category": "budget_edge",
        "query": "Premium car seat, money is no object",
        "expect_results": True,
        "budget_aed": None,
        "notes": "No budget constraint. Should surface top-rated car seats.",
    },

    # ── Vague / ambiguous ─────────────────────────────────────────────────────
    {
        "id": "TC08",
        "category": "ambiguous",
        "query": "Something nice for a baby",
        "expect_results": True,
        "budget_aed": None,
        "notes": "Very vague. Should return results but with low-medium confidence and note ambiguity.",
    },
    {
        "id": "TC09",
        "category": "ambiguous",
        "query": "Gift for my sister",
        "expect_results": True,
        "budget_aed": None,
        "notes": "No age, no context. Model should ask or state low confidence. Still attempt recommendations.",
    },

    # ── Out of scope / refusal ────────────────────────────────────────────────
    {
        "id": "TC10",
        "category": "out_of_scope",
        "query": "Best laptop for a university student under 3000 AED",
        "expect_results": False,
        "budget_aed": 3000,
        "notes": "Completely out of scope. Model must set unable_to_help=true. Must not hallucinate electronics.",
    },
    {
        "id": "TC11",
        "category": "out_of_scope",
        "query": "Recommend a good restaurant in Dubai for a baby shower dinner",
        "expect_results": False,
        "budget_aed": None,
        "notes": "Service request, not a product. Should gracefully decline.",
    },

    # ── Multilingual / Arabic quality ─────────────────────────────────────────
    {
        "id": "TC12",
        "category": "multilingual",
        "query": "هدية لأم جديدة وطفلها عمره شهر واحد، الميزانية 250 درهم",
        "expect_results": True,
        "budget_aed": 250,
        "notes": "Query in Arabic. Model should handle gracefully. Arabic output must be natural.",
    },
]

# ── Rubric scorers ─────────────────────────────────────────────────────────────

def score_budget_adherence(result: GiftFinderResult, budget_aed: float | None) -> tuple[int, str]:
    """Check all recommended products are within budget. Score 0-2."""
    if budget_aed is None:
        return 2, "No budget constraint — N/A"
    if not result.success or not result.parsed:
        return 0, "No parsed output"
    recs = result.parsed.get("recommendations", [])
    if not recs:
        return 2, "No recommendations to check"
    over_budget = [r for r in recs if r.get("price_aed", 9999) > budget_aed * 1.05]
    if over_budget:
        names = [r.get("name_en", r.get("product_id", "?")) for r in over_budget]
        return 0, f"Over budget: {names}"
    return 2, "All within budget"


def score_hallucination(result: GiftFinderResult) -> tuple[int, str]:
    """Check product IDs exist in catalog. Score 0-2."""
    if not result.success or not result.parsed:
        return 0, "No parsed output"
    errors = [e for e in result.validation_errors if "hallucinated" in e.lower()]
    if errors:
        return 0, f"Hallucinated product(s): {errors}"
    return 2, "No hallucination detected"


def score_schema_validity(result: GiftFinderResult) -> tuple[int, str]:
    """Score 0-2 based on validation errors."""
    if result.parsed is None:
        return 0, "JSON parse failed"
    if not result.validation_errors:
        return 2, "Schema valid"
    minor = [e for e in result.validation_errors if "minor" in e.lower()]
    if len(result.validation_errors) <= 1:
        return 1, f"Minor schema issue: {result.validation_errors}"
    return 0, f"Multiple schema errors: {result.validation_errors}"


def score_refusal_handling(result: GiftFinderResult, expect_results: bool) -> tuple[int, str]:
    """Score 0-2: did model correctly refuse or provide results as expected?"""
    if not result.parsed:
        return 0, "No parsed output"
    did_refuse = result.parsed.get("unable_to_help", False)
    recs = result.parsed.get("recommendations", [])

    if not expect_results:
        # We expected a refusal
        if did_refuse or len(recs) == 0:
            return 2, "Correctly declined out-of-scope request"
        return 0, "Should have declined but returned results"
    else:
        # We expected results
        if did_refuse and len(recs) == 0:
            return 0, "Declined when it should have helped"
        if len(recs) >= 1:
            return 2, f"Returned {len(recs)} recommendations as expected"
        return 1, "Returned fewer recommendations than expected"


def score_arabic_present(result: GiftFinderResult) -> tuple[int, str]:
    """Check Arabic fields are non-empty and look like Arabic. Score 0-2."""
    if not result.parsed:
        return 0, "No parsed output"
    recs = result.parsed.get("recommendations", [])
    if not recs:
        return 1, "No recommendations — Arabic N/A"
    arabic_score = 0
    for rec in recs:
        ar = rec.get("why_ar", "")
        # Very basic check: Arabic Unicode range u0600-u06FF
        has_arabic = any('\u0600' <= c <= '\u06FF' for c in ar)
        if has_arabic and len(ar) > 20:
            arabic_score += 1
    if arabic_score == len(recs):
        return 2, "Arabic present and plausible in all recommendations"
    elif arabic_score > 0:
        return 1, f"Arabic present in {arabic_score}/{len(recs)} recommendations"
    return 0, "No Arabic text detected"


# ── Run evals ─────────────────────────────────────────────────────────────────
def run_evals(verbose: bool = True):
    results_summary = []
    total_score = 0
    max_score = 0

    print("=" * 70)
    print("  MUMZWORLD GIFT FINDER — EVALUATION SUITE")
    print("=" * 70)
    print(f"  Running {len(TEST_CASES)} test cases\n")

    for tc in TEST_CASES:
        if verbose:
            print(f"  [{tc['id']}] {tc['query'][:60]}{'...' if len(tc['query']) > 60 else ''}")

        try:
            result = find_gifts(tc["query"])
            time.sleep(1)  # Respect rate limits
        except Exception as e:
            print(f"    ❌ Error: {e}\n")
            results_summary.append({"id": tc["id"], "error": str(e), "score": 0, "max": 10})
            continue

        # Score dimensions (each 0-2, total max 10)
        s1, r1 = score_schema_validity(result)
        s2, r2 = score_hallucination(result)
        s3, r3 = score_budget_adherence(result, tc["budget_aed"])
        s4, r4 = score_refusal_handling(result, tc["expect_results"])
        s5, r5 = score_arabic_present(result)

        case_score = s1 + s2 + s3 + s4 + s5
        case_max = 10
        total_score += case_score
        max_score += case_max

        if verbose:
            icon = "✅" if case_score >= 8 else ("⚠ " if case_score >= 5 else "❌")
            print(f"    {icon} Score: {case_score}/{case_max}")
            print(f"       Schema:    {s1}/2 — {r1}")
            print(f"       Hallucin:  {s2}/2 — {r2}")
            print(f"       Budget:    {s3}/2 — {r3}")
            print(f"       Refusal:   {s4}/2 — {r4}")
            print(f"       Arabic:    {s5}/2 — {r5}")
            if result.parsed and result.parsed.get("unable_to_help"):
                print(f"       Model said: '{result.parsed.get('unable_reason', '')[:80]}'")
            print()

        results_summary.append({
            "id": tc["id"],
            "category": tc["category"],
            "query": tc["query"],
            "score": case_score,
            "max": case_max,
            "pct": round(case_score / case_max * 100),
            "schema": f"{s1}/2",
            "hallucination": f"{s2}/2",
            "budget": f"{s3}/2",
            "refusal": f"{s4}/2",
            "arabic": f"{s5}/2",
            "notes": tc["notes"],
        })

    pct = round(total_score / max_score * 100) if max_score else 0
    print("=" * 70)
    print(f"  TOTAL: {total_score}/{max_score}  ({pct}%)")
    print()

    # Category breakdown
    cats = {}
    for r in results_summary:
        cat = r.get("category", "unknown")
        if cat not in cats:
            cats[cat] = {"score": 0, "max": 0}
        cats[cat]["score"] += r.get("score", 0)
        cats[cat]["max"] += r.get("max", 10)

    print("  By category:")
    for cat, v in cats.items():
        cp = round(v["score"] / v["max"] * 100) if v["max"] else 0
        print(f"    {cat:<20} {v['score']}/{v['max']}  ({cp}%)")

    print()

    # Known failure modes
    failed = [r for r in results_summary if r.get("score", 0) < 6]
    if failed:
        print(f"  ⚠ Cases scoring < 6/10 ({len(failed)}):")
        for f in failed:
            print(f"    [{f['id']}] {f.get('query', '')[:50]} — {f['score']}/{f['max']}")

    # Save JSON report
    with open("eval_results.json", "w", encoding="utf-8") as f:
        json.dump({
            "total_score": total_score,
            "max_score": max_score,
            "pct": pct,
            "cases": results_summary,
        }, f, ensure_ascii=False, indent=2)

    print("\n  Full results saved to eval_results.json")
    print("=" * 70)
    return total_score, max_score


if __name__ == "__main__":
    run_evals(verbose=True)
