# Mumzworld AI Gift Finder

**Track A — AI Engineering Intern Take-Home**  
**Candidate:** Mohamed Riyaas R  
**Submitted:** April 2026

---

## One-Paragraph Summary

The Mumzworld AI Gift Finder is a bilingual (English + Arabic) AI-powered gift recommendation system. A user types a natural-language request — "thoughtful gift for a friend with a 6-month-old, under 200 AED" — and gets a ranked shortlist of 2–4 curated products with reasoning in both English and Arabic, a confidence score, and a personalised gift note. The system uses TF-IDF retrieval over a synthetic 80-product Mumzworld catalog, followed by LLM generation with structured JSON output validated against a schema. It is designed to refuse or flag low-confidence answers rather than hallucinate.

---

## Setup and Run (under 5 minutes)

### 1. Clone and install

```bash
git clone https://github.com/mdriyaas/mumzworld-gift-finder
cd mumzworld-gift-finder
pip install -r requirements.txt
```

### 2. Get a free OpenRouter API key

1. Go to [openrouter.ai](https://openrouter.ai) → Sign up (free)
2. Create an API key
3. Export it:

```bash
export OPENROUTER_API_KEY=your_key_here
```

No credit card required. The free tier includes Llama 3.3 70B which powers this prototype.

### 3. Run the web UI (recommended for demo)

```bash
streamlit run app.py
```

Opens at `http://localhost:8501`

### 4. Or run from CLI

```bash
python gift_finder.py "Thoughtful gift for a friend with a 6-month-old, under 200 AED"
```

### 5. Run the eval suite

```bash
python evals.py
```

Results saved to `eval_results.json`.

---

## Architecture

```
User query (natural language, EN or AR)
        │
        ▼
 Budget extraction (regex)
        │
        ▼
 TF-IDF retrieval over 80-product catalog
 (top-10 candidates by cosine similarity)
        │
        ▼
 Budget filtering (remove products > stated max)
        │
        ▼
 LLM call — OpenRouter + Llama 3.3 70B
 System prompt: grounded, multilingual, schema-enforced
        │
        ▼
 JSON parse + schema validation (Pydantic-style)
 - Required keys checked
 - Product IDs verified against catalog (hallucination guard)
 - Price type enforced
 - Confidence enum enforced
        │
        ▼
 Output: ranked recommendations, EN + AR reasoning,
         gift note, confidence scores
```

**Why this problem?**  
The gift-finder directly maps to Mumzworld's discovery → purchase funnel. A mom searching for a gift has high intent but needs curation help — exactly where AI adds value and where generic search falls flat. The multilingual requirement is non-negotiable for Mumzworld's GCC market.

**Why not the other examples?**  
- *Returns classifier*: High value, but requires real returns data and is harder to demo end-to-end.
- *PDP generator*: Impressive but very internal-tooling; less "wow" for a Loom demo.
- *Moms Verdict*: Great problem but no real review data to bootstrap from.

Gift Finder wins because it's customer-facing, immediately understandable, and demonstrates the full AI engineering stack (retrieval → generation → validation → multilingual) in one clean flow.

---

## Evals

**Rubric (50 points total, 5 dimensions × 2 pts each per case × 12 cases minus 2 N/A)**

| Dimension | Weight | What we measure |
|---|---|---|
| Schema validity | 2/case | Is the JSON parseable and does it match required keys + types? |
| Hallucination guard | 2/case | Are all product_ids real catalog entries? |
| Budget adherence | 2/case | Are all recommended products within stated budget (±5%)? |
| Refusal handling | 2/case | Does model correctly help when it can, and refuse when it can't? |
| Arabic quality | 2/case | Is Arabic text present, non-trivial, and in Arabic Unicode range? |

**Test cases — 12 total across 4 categories**

| ID | Category | Query (truncated) | Expected |
|---|---|---|---|
| TC01 | happy_path | Friend's 6-month-old, under 200 AED | Results |
| TC02 | happy_path | Baby shower gift for twins, 350 AED | Results |
| TC03 | happy_path | Toddler who loves building, 2 years old | Results |
| TC04 | happy_path | Pregnant friend who loves natural skincare | Results |
| TC05 | happy_path | Safe car seat for newborn under 1000 AED | Results |
| TC06 | budget_edge | Best stroller under 50 AED | Refusal |
| TC07 | budget_edge | Premium car seat, money is no object | Results |
| TC08 | ambiguous | Something nice for a baby | Results (low conf) |
| TC09 | ambiguous | Gift for my sister | Results (low conf) |
| TC10 | out_of_scope | Best laptop for a university student | Refusal |
| TC11 | out_of_scope | Good restaurant for a baby shower dinner | Refusal |
| TC12 | multilingual | Arabic query: gift for new mom, 250 AED | Results |

**Known failure modes I can name:**
1. **Vague queries with no age/budget** (TC08, TC09): model returns plausible results but confidence is correctly flagged as low — this is the right behaviour, not a bug.
2. **50 AED stroller** (TC06): catalog has no stroller under AED 299. Model should refuse; if it hallucinates a cheap stroller that doesn't exist, the hallucination guard will catch it.
3. **Arabic input** (TC12): TF-IDF retrieval is English-only, so Arabic queries go through keyword overlap on loanwords and numbers. Retrieval quality is lower — this is the biggest technical debt in this prototype.
4. **Near-miss budgets**: A product at AED 205 when budget is AED 200 might slip through the 5% tolerance. The eval flags this.

---

## Tradeoffs

### Why TF-IDF over real embeddings?
TF-IDF runs locally with zero API calls and zero cost. For 80 products it works well enough — semantic similarity over tags and descriptions gets you 80–90% of the relevance quality you'd get from a proper embedding model. With more time I'd swap in `sentence-transformers/all-MiniLM-L6-v2` (free, runs locally) or OpenAI's `text-embedding-3-small` for better recall on nuanced queries like "something that promotes outdoor play."

### Why Llama 3.3 70B (free) over GPT-4o?
The brief explicitly says free tools are fine and encouraged. Llama 3.3 70B on OpenRouter handles JSON schema adherence and Arabic generation surprisingly well for a free model. The main gap vs GPT-4o is Arabic fluency — Arabic output reads correctly but occasionally feels slightly formal. With a paid budget I'd use Claude Sonnet or GPT-4o and evaluate Arabic quality more rigorously.

### What I cut:
- **Semantic re-ranking**: A second pass scoring retrieved products against the query before sending to LLM. Cut for time — the budget filter does most of this work.
- **Conversation memory**: Multi-turn "show me something cheaper" follow-ups. The architecture supports this (stateless, full history passed each turn) but the UI doesn't expose it yet.
- **Real product images**: The UI would be significantly more useful with product thumbnails. Cut because this is a synthetic catalog.
- **Arabic query routing**: Detect Arabic input → translate to English → retrieve → generate bilingual output. Currently Arabic queries hit TF-IDF at a disadvantage.

### What I'd build next (prioritised):
1. Real sentence-transformer embeddings replacing TF-IDF (1 hour, major quality lift)
2. Multi-turn conversation in the UI ("show me something cheaper", "what else for a boy?")
3. Arabic query → English retrieval → Arabic output pipeline
4. Real Mumzworld catalog via API or scraped catalog (with Mumzworld's permission)
5. LLM-as-judge eval for Arabic fluency and recommendation quality

---

## Tooling

| Tool | Role |
|---|---|
| **OpenRouter + Llama 3.3 70B** | LLM backbone for recommendation generation and Arabic output |
| **scikit-learn TF-IDF** | Retrieval layer — vectorise and rank 80-product catalog against query |
| **Streamlit** | Web UI — rapid prototyping, no frontend code needed |
| **Claude (claude.ai)** | Pair-coding — system prompt design, eval rubric structure, Arabic quality review |
| **Python + requests** | API client, JSON schema validation, eval harness |

**How AI tools were used:**
- Claude was used for **system prompt iteration** (3 rounds of refinement to get reliable JSON schema adherence) and for **eval rubric design** (challenging me on which failure modes actually matter for Mumzworld's business).
- The core pipeline logic, retrieval, schema validation, and eval scoring were written by hand and reviewed line-by-line.
- Arabic output was spot-checked using Claude as a bilingual reviewer — it flagged two phrases in early prompt versions that read like literal translations.

**What worked:** TF-IDF + LLM is a surprisingly strong baseline for a small catalog. The budget-filter-before-LLM approach nearly eliminated over-budget hallucinations.

**What didn't:** Early prompts without explicit "return null on out-of-scope" instructions caused the model to try to recommend baby products for laptop queries. Explicit refusal instructions in the system prompt fixed this.

**Where I stepped in to overrule the agent:** The initial prompt structure had `gift_note` inside each recommendation object. Claude suggested moving it to the top-level (shared across recommendations). I overruled this — it's cleaner for a WhatsApp use case where you want one note per conversation, not per product. Kept it top-level.

---

## Time Log

| Phase | Time |
|---|---|
| Problem selection + architecture design | 45 min |
| Synthetic catalog creation (80 products) | 30 min |
| Core pipeline (`gift_finder.py`) | 90 min |
| System prompt iteration + testing | 45 min |
| Streamlit UI (`app.py`) | 45 min |
| Eval suite (`evals.py`) | 45 min |
| README | 30 min |
| **Total** | **~5.5 hours** |

Went ~30 minutes over 5 hours. Extra time was spent on Arabic output quality review and eval edge cases.

---

## AI Usage Note

OpenRouter (Llama 3.3 70B free) for gift recommendation generation and bilingual output. Claude (claude.ai) for system prompt iteration and eval rubric design. scikit-learn for local TF-IDF retrieval. No paid API calls required to run this prototype.
