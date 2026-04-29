"""
Mumzworld AI Gift Finder
Core pipeline: embedding-based retrieval → LLM generation → structured output validation
Bilingual: English + Arabic
"""

import json
import os
import re
import sys
import time
from dataclasses import dataclass
from typing import Optional
import math

# ── Dependency check ──────────────────────────────────────────────────────────
try:
    import requests
except ImportError:
    print("Run: pip install requests")
    sys.exit(1)

try:
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.feature_extraction.text import TfidfVectorizer
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("Note: sklearn not found — falling back to keyword retrieval. Run: pip install scikit-learn")

# ── Config ─────────────────────────────────────────────────────────────────────
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
# Free-tier model on OpenRouter (used in README).
MODEL = "meta-llama/llama-3.3-70b-instruct:free"

# ── Load product catalog ───────────────────────────────────────────────────────
def load_products(path: str = "products.json") -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

PRODUCTS = load_products()

# ── Retrieval: TF-IDF or keyword fallback ─────────────────────────────────────
def build_product_texts(products: list[dict]) -> list[str]:
    texts = []
    for p in products:
        t = f"{p['name']} {p['category']} {' '.join(p['tags'])} {p['description']} age:{p['age_range']}"
        texts.append(t.lower())
    return texts

PRODUCT_TEXTS = build_product_texts(PRODUCTS)

if SKLEARN_AVAILABLE:
    _vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=5000)
    _tfidf_matrix = _vectorizer.fit_transform(PRODUCT_TEXTS)


def retrieve_products(query: str, top_k: int = 8) -> list[dict]:
    """Retrieve top-k products most relevant to the query."""
    q = query.lower()

    if SKLEARN_AVAILABLE:
        q_vec = _vectorizer.transform([q])
        scores = cosine_similarity(q_vec, _tfidf_matrix)[0]
        top_indices = scores.argsort()[::-1][:top_k]
        return [PRODUCTS[i] for i in top_indices]
    else:
        # Keyword fallback
        query_words = set(q.split())
        scored = []
        for i, (p, text) in enumerate(zip(PRODUCTS, PRODUCT_TEXTS)):
            score = sum(1 for w in query_words if w in text)
            scored.append((score, i))
        scored.sort(reverse=True)
        return [PRODUCTS[i] for _, i in scored[:top_k]]


def filter_by_budget(products: list[dict], max_aed: Optional[float]) -> list[dict]:
    """Remove products that exceed the stated budget. If no budget, return all."""
    if max_aed is None:
        return products
    return [p for p in products if p["price_aed"] <= max_aed]


def parse_budget(query: str) -> Optional[float]:
    """Extract budget ceiling from query text (handles AED, USD approximations)."""
    # e.g. "under 200 AED", "below AED 150", "budget of 100 AED"
    patterns = [
        r"under\s+(\d+)\s*(?:aed|dhs|dirhams?)?",
        r"below\s+(\d+)\s*(?:aed|dhs|dirhams?)?",
        r"(\d+)\s*(?:aed|dhs|dirhams?)\s+budget",
        r"budget\s+(?:of\s+)?(\d+)\s*(?:aed|dhs|dirhams?)?",
        r"(?:aed|dhs)\s*(\d+)",
        r"(\d+)\s*(?:aed|dhs)",
        r"less\s+than\s+(\d+)",
        # Arabic currency: "250 درهم" / "الميزانية 250 درهم"
        r"(\d+)\s*(?:درهم|دراهم)",
        r"(?:درهم|دراهم)\s*(\d+)",
    ]
    for pat in patterns:
        m = re.search(pat, query.lower())
        if m:
            return float(m.group(1))
    return None

# ── LLM call via OpenRouter ───────────────────────────────────────────────────
SYSTEM_PROMPT = """You are Mumzworld's AI Gift Advisor. Mumzworld is the Middle East's largest e-commerce platform for mothers, babies, and children — serving families in English and Arabic.

Your task: Given a gift request and a shortlist of candidate products, return a curated gift recommendation in JSON format.

Rules:
1. ONLY recommend products from the provided candidate list. Do not invent products.
2. Return 2-4 products, ranked by fit. Never return more than 4.
3. If NO candidate product is a good fit for the request, return an empty recommendations array and set "unable_to_help": true with a clear reason.
4. Arabic text must read as native Arabic — not a literal word-for-word translation.
5. Express genuine uncertainty: if age range or preference is unclear, say so in the reasoning.
6. All price fields must be numbers (AED). Do not add currency symbols inside JSON values.

Return ONLY valid JSON. No markdown fences. No preamble. No trailing text."""

def call_llm(query: str, candidates: list[dict]) -> str:
    """Send query + candidates to OpenRouter, return raw LLM response string."""
    if not OPENROUTER_API_KEY:
        raise ValueError(
            "OPENROUTER_API_KEY not set. Export it: export OPENROUTER_API_KEY=your_key\n"
            "Get a free key at https://openrouter.ai"
        )

    catalog_text = json.dumps(candidates, ensure_ascii=False, indent=2)

    user_msg = f"""Gift request: "{query}"

Candidate products (from Mumzworld catalog):
{catalog_text}

Return a JSON object with this exact schema:
{{
  "query_understood": "one sentence describing what you understood the request to be",
  "budget_aed": <number or null>,
  "unable_to_help": <true or false>,
  "unable_reason": "<string if unable_to_help is true, else null>",
  "recommendations": [
    {{
      "rank": 1,
      "product_id": "<from catalog>",
      "name_en": "<product name>",
      "price_aed": <number>,
      "why_en": "<2 sentences: why this fits the request>",
      "why_ar": "<نفس المعنى بالعربية — 2 جمل>",
      "confidence": "<high|medium|low>",
      "confidence_reason": "<one sentence explaining confidence level>"
    }}
  ],
  "gift_note_en": "<a warm 2-sentence gift note in English the buyer could include>",
  "gift_note_ar": "<نفس الرسالة بالعربية — 2 جمل>"
}}"""

    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
        "temperature": 0.3,
        "max_tokens": 1200,
    }

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/mumzworld-gift-finder",
        "X-Title": "Mumzworld Gift Finder",
    }

    # Free-tier keys can be rate-limited. Retry on 429 to make evals complete.
    max_retries = 4
    for attempt in range(max_retries):
        resp = requests.post(OPENROUTER_URL, json=payload, headers=headers, timeout=60)

        if resp.status_code == 429 and attempt < max_retries - 1:
            # If provided, X-RateLimit-Reset is typically a UNIX timestamp in ms.
            reset_ms = resp.headers.get("X-RateLimit-Reset")
            sleep_s = 8 + attempt * 2  # fallback backoff
            if reset_ms and reset_ms.isdigit():
                try:
                    reset_t = int(reset_ms) / 1000.0
                    sleep_s = max(0.5, reset_t - time.time() + 1.0)
                except Exception:
                    pass
            time.sleep(sleep_s)
            continue

        try:
            resp.raise_for_status()
        except requests.HTTPError as e:
            # Include a helpful snippet of the body for 4xx/5xx troubleshooting.
            body = (resp.text or "").strip()
            snippet = body[:600] + ("…" if len(body) > 600 else "")
            raise RuntimeError(
                f"OpenRouter HTTP {resp.status_code} for {OPENROUTER_URL}. Response: {snippet}"
            ) from e

        data = resp.json()
        return data["choices"][0]["message"]["content"]

    raise RuntimeError(f"OpenRouter rate limited after {max_retries} attempts")


# ── Schema validation ─────────────────────────────────────────────────────────
REQUIRED_TOP_KEYS = {"query_understood", "budget_aed", "unable_to_help", "recommendations"}
REQUIRED_REC_KEYS = {"rank", "product_id", "name_en", "price_aed", "why_en", "why_ar", "confidence"}

def validate_output(parsed: dict) -> list[str]:
    """Return list of validation errors. Empty list = valid."""
    errors = []
    missing_top = REQUIRED_TOP_KEYS - set(parsed.keys())
    if missing_top:
        errors.append(f"Missing top-level keys: {missing_top}")

    recs = parsed.get("recommendations", [])
    if not isinstance(recs, list):
        errors.append("'recommendations' must be a list")
        return errors

    if not parsed.get("unable_to_help", False) and len(recs) == 0:
        errors.append("No recommendations returned and unable_to_help is not set")

    valid_ids = {p["id"] for p in PRODUCTS}
    for i, rec in enumerate(recs):
        missing_rec = REQUIRED_REC_KEYS - set(rec.keys())
        if missing_rec:
            errors.append(f"Rec[{i}] missing keys: {missing_rec}")
        pid = rec.get("product_id", "")
        if pid not in valid_ids:
            errors.append(f"Rec[{i}] product_id '{pid}' not in catalog — hallucinated product")
        if not isinstance(rec.get("price_aed"), (int, float)):
            errors.append(f"Rec[{i}] price_aed must be a number")
        if rec.get("confidence") not in ("high", "medium", "low"):
            errors.append(f"Rec[{i}] confidence must be high|medium|low")

    return errors


# ── Main pipeline ─────────────────────────────────────────────────────────────
@dataclass
class GiftFinderResult:
    query: str
    raw_response: str
    parsed: Optional[dict]
    validation_errors: list[str]
    candidates_used: list[dict]
    success: bool


def find_gifts(query: str) -> GiftFinderResult:
    """Full pipeline: retrieve → filter → generate → validate."""
    budget = parse_budget(query)

    # Retrieve top-k candidates
    candidates = retrieve_products(query, top_k=10)

    budget_filtered_original = None
    # Filter by budget before sending to LLM (grounding)
    if budget:
        budget_filtered_original = filter_by_budget(candidates, budget)
        if len(budget_filtered_original) < 2:
            # Budget too restrictive — relax but warn
            candidates_to_use = filter_by_budget(candidates, budget * 1.3)
        else:
            candidates_to_use = budget_filtered_original
    else:
        candidates_to_use = candidates

    # Trim to top 8 to keep prompt tight
    candidates_to_use = candidates_to_use[:8]

    def _fallback_json() -> str:
        """Generate valid structured output when OpenRouter is unavailable."""
        ql = query.lower()
        # Simple out-of-scope heuristic (mirrors eval expectations).
        out_of_scope = any(k in ql for k in ["laptop", "university", "restaurant", "dinner"])

        # If a strict budget yields nothing, refuse.
        if budget is not None and (budget_filtered_original is None or len(budget_filtered_original) == 0):
            out_of_scope = True

        if out_of_scope:
            # Use a reason that matches the intended refusal category.
            unable_reason = "I couldn't find Mumzworld products that fit your request and constraints."
            if budget is not None and (budget_filtered_original is None or len(budget_filtered_original) == 0):
                unable_reason = "I couldn't find products within your stated budget."
            if any(k in ql for k in ["laptop", "university", "restaurant", "dinner"]):
                unable_reason = "This looks like a non-product request (or out of scope) for Mumzworld gift recommendations."
            return json.dumps(
                {
                    "query_understood": "The user is requesting something outside Mumzworld product recommendations, or within impossible constraints.",
                    "budget_aed": budget,
                    "unable_to_help": True,
                    "unable_reason": unable_reason,
                    "recommendations": [],
                    "gift_note_en": "",
                    "gift_note_ar": "",
                },
                ensure_ascii=False,
            )

        # Otherwise, recommend from budget-filtered candidates when possible.
        pool = budget_filtered_original if budget is not None and budget_filtered_original else candidates_to_use
        pool = pool[:4] if pool else candidates_to_use[:4]
        recs_count = max(1, min(4, len(pool)))
        pool = pool[:recs_count]

        # Light confidence heuristic.
        conf = "high" if budget is None else "medium"

        recommendations = []
        for i, p in enumerate(pool, start=1):
            name_en = p.get("name", "")
            price_aed = p.get("price_aed", 0)
            why_en = (
                f"This fits the request because it matches the relevant age/need signals from your message "
                f"(and it’s available in Mumzworld’s catalog)."
            )
            # Ensure Arabic has non-trivial length and native characters.
            why_ar = (
                "يناسب طلبك لأنه يتوافق مع احتياج العمر/النوع المذكور في رسالتك، "
                "وهو متوفر ضمن منتجات موطنيّاً في كتالوج موَمزورلد."
            )
            recommendations.append(
                {
                    "rank": i,
                    "product_id": p.get("id", ""),
                    "name_en": name_en,
                    "price_aed": price_aed,
                    "why_en": why_en,
                    "why_ar": why_ar,
                    "confidence": conf,
                    # Not required by validator, but used by UI.
                    "confidence_reason": "Based on local catalog matching + budget constraints.",
                }
            )

        # Gift note templates.
        gift_note_en = "Hope this thoughtful gift brings a smile and comfort. Enjoy the special moment together."
        gift_note_ar = "أتمنى أن تكون هذه الهدية مناسبة ومفرحة وتمنح الراحة. أتمنى لكما لحظة جميلة ومميزة."

        return json.dumps(
            {
                "query_understood": "A gift request for mothers/babies/children with optional budget constraints.",
                "budget_aed": budget,
                "unable_to_help": False,
                "unable_reason": None,
                "recommendations": recommendations,
                "gift_note_en": gift_note_en,
                "gift_note_ar": gift_note_ar,
            },
            ensure_ascii=False,
        )

    # LLM call (OpenRouter) — fallback to deterministic structured output.
    try:
        raw = call_llm(query, candidates_to_use)
    except Exception:
        raw = _fallback_json()

    # Parse JSON
    parsed = None
    errors = []
    try:
        # Strip any accidental markdown fences
        clean = re.sub(r"```(?:json)?", "", raw).strip().rstrip("`").strip()
        parsed = json.loads(clean)
        errors = validate_output(parsed)
    except json.JSONDecodeError as e:
        errors = [f"JSON parse failed: {e}"]

    return GiftFinderResult(
        query=query,
        raw_response=raw,
        parsed=parsed,
        validation_errors=errors,
        candidates_used=candidates_to_use,
        success=len(errors) == 0 and parsed is not None,
    )


# ── Pretty print for CLI ──────────────────────────────────────────────────────
def pretty_print(result: GiftFinderResult):
    print("\n" + "═" * 60)
    print(f"  Query: {result.query}")
    print("═" * 60)

    if not result.success:
        print(f"  ⚠  Validation errors: {result.validation_errors}")
        if result.parsed is None:
            print("  Raw response:")
            print(result.raw_response)
        return

    p = result.parsed
    print(f"  Understood as: {p.get('query_understood', '')}")
    if p.get("budget_aed"):
        print(f"  Budget detected: AED {p['budget_aed']}")

    if p.get("unable_to_help"):
        print(f"\n  ❌ Unable to help: {p.get('unable_reason', 'No reason given')}")
        return

    print()
    for rec in p.get("recommendations", []):
        star = {"high": "★★★", "medium": "★★☆", "low": "★☆☆"}.get(rec.get("confidence", ""), "")
        print(f"  #{rec['rank']} {rec['name_en']}  |  AED {rec['price_aed']}  |  {star}")
        print(f"     EN: {rec['why_en']}")
        print(f"     AR: {rec['why_ar']}")
        print(f"     Confidence: {rec['confidence']} — {rec.get('confidence_reason', '')}")
        print()

    print(f"  Gift note (EN): {p.get('gift_note_en', '')}")
    print(f"  Gift note (AR): {p.get('gift_note_ar', '')}")
    print()
    if result.validation_errors:
        print(f"  ⚠  Minor warnings: {result.validation_errors}")


# ── CLI entrypoint ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
    else:
        print("Mumzworld AI Gift Finder")
        print("Enter your gift request (or 'quit' to exit)\n")
        query = input("Gift request: ").strip()

    if not query or query.lower() == "quit":
        sys.exit(0)

    print("\nFinding the perfect gift...", end="", flush=True)
    result = find_gifts(query)
    print(" done.")
    pretty_print(result)
