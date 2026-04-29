"""
Mumzworld AI Gift Finder — Streamlit UI
Run: streamlit run app.py
"""

import streamlit as st
import json
from gift_finder import find_gifts, PRODUCTS

st.set_page_config(
    page_title="Mumzworld Gift Finder",
    page_icon="🎁",
    layout="centered",
)

# ── Custom CSS for Mumzworld feel ─────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #fefefe; }
    .stTextInput > div > div > input {
        border: 2px solid #e91e8c;
        border-radius: 8px;
        font-size: 15px;
    }
    .rec-card {
        background: #fff8fc;
        border: 1px solid #f5c0de;
        border-radius: 10px;
        padding: 16px 20px;
        margin-bottom: 14px;
    }
    .rec-title { font-size: 17px; font-weight: 700; color: #1a1a2e; }
    .rec-price { font-size: 14px; color: #e91e8c; font-weight: 600; }
    .why-en { font-size: 13px; color: #333; margin-top: 6px; }
    .why-ar { font-size: 13px; color: #555; text-align: right; direction: rtl; margin-top: 4px; }
    .gift-note-box {
        background: #fff3e0;
        border-left: 4px solid #ff9800;
        padding: 12px 16px;
        border-radius: 6px;
        margin-top: 20px;
    }
    .validation-error { color: #c62828; font-size: 12px; }
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
col1, col2 = st.columns([1, 6])
with col1:
    st.markdown("## 🎁")
with col2:
    st.markdown("## Mumzworld Gift Finder")
    st.markdown("*The AI gift advisor for every mother, baby, and child in the Middle East*")

st.divider()

# ── Input ─────────────────────────────────────────────────────────────────────
st.markdown("### Tell me what you're looking for")
st.markdown("Use natural language — be as specific or vague as you like.")

example_queries = [
    "Thoughtful gift for a friend with a 6-month-old, under 200 AED",
    "Something special for a new mom — not for the baby, for her",
    "Gift for a toddler who loves building things, under 150 AED",
    "Best car seat for a newborn under 1000 AED",
    "Baby shower gift for twins, budget 300 AED",
    "I need a gift for a pregnant friend who loves self-care",
    "Something for a kid who is 3 years old and obsessed with animals",
]

with st.expander("💡 Try an example query"):
    for ex in example_queries:
        if st.button(ex, key=ex):
            st.session_state["query_input"] = ex

query = st.text_input(
    "Your gift request",
    placeholder="e.g. Thoughtful gift for a friend with a 6-month-old, under 200 AED",
    key="query_input",
    label_visibility="collapsed",
)

col_btn, col_info = st.columns([2, 5])
with col_btn:
    search_clicked = st.button("🎁 Find Gifts", type="primary", use_container_width=True)
with col_info:
    st.markdown(
        "<small style='color:#888'>Searches across 80 Mumzworld products · Responds in English + Arabic</small>",
        unsafe_allow_html=True,
    )

# ── Results ───────────────────────────────────────────────────────────────────
if search_clicked and query.strip():
    with st.spinner("Finding the perfect gift..."):
        result = find_gifts(query.strip())

    if not result.success or result.parsed is None:
        st.error("Something went wrong parsing the response.")
        with st.expander("Debug info"):
            st.text(result.raw_response)
            st.json(result.validation_errors)
    else:
        p = result.parsed

        st.markdown(f"**Understood as:** _{p.get('query_understood', '')}_")
        if p.get("budget_aed"):
            st.markdown(f"**Budget detected:** AED {p['budget_aed']:.0f}")

        if p.get("unable_to_help"):
            st.warning(f"**Sorry, I couldn't help with this one:** {p.get('unable_reason', '')}")
            st.markdown("Try rephrasing with a specific age range or occasion.")
        else:
            recs = p.get("recommendations", [])
            if not recs:
                st.warning("No recommendations returned. Try broadening your request.")
            else:
                st.markdown(f"### {len(recs)} Gift Recommendation{'s' if len(recs) > 1 else ''}")

                confidence_icon = {"high": "🟢", "medium": "🟡", "low": "🔴"}

                for rec in recs:
                    conf = rec.get("confidence", "low")
                    icon = confidence_icon.get(conf, "⚪")
                    with st.container():
                        st.markdown(f"""
<div class="rec-card">
  <div class="rec-title">#{rec['rank']} {rec['name_en']}</div>
  <div class="rec-price">AED {rec['price_aed']:.0f} &nbsp;|&nbsp; {icon} {conf.capitalize()} confidence</div>
  <div class="why-en">🇬🇧 {rec['why_en']}</div>
  <div class="why-ar">🇦🇪 {rec['why_ar']}</div>
  <div style="font-size:11px;color:#999;margin-top:6px">{rec.get('confidence_reason','')}</div>
</div>
""", unsafe_allow_html=True)

            # Gift notes
            gift_en = p.get("gift_note_en")
            gift_ar = p.get("gift_note_ar")
            if gift_en or gift_ar:
                st.markdown("### ✉️ Gift Note")
                st.markdown(f"""
<div class="gift-note-box">
  <div style="font-size:13px;color:#333">🇬🇧 {gift_en}</div>
  <div style="font-size:13px;color:#555;text-align:right;direction:rtl;margin-top:8px">🇦🇪 {gift_ar}</div>
</div>
""", unsafe_allow_html=True)

        # Validation warnings
        if result.validation_errors:
            with st.expander("⚠ Schema warnings"):
                for e in result.validation_errors:
                    st.markdown(f'<div class="validation-error">• {e}</div>', unsafe_allow_html=True)

        # Candidates used (debug)
        with st.expander("🔍 Products searched (retrieval candidates)"):
            st.markdown(f"Retrieved {len(result.candidates_used)} candidates via TF-IDF similarity:")
            for c in result.candidates_used:
                st.markdown(f"- **{c['name']}** (AED {c['price_aed']}) — {c['category']}")

elif search_clicked and not query.strip():
    st.warning("Please enter a gift request first.")

# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.markdown(
    "<small style='color:#aaa'>Prototype · 80-product synthetic catalog · Powered by OpenRouter + Llama 3.3 70B</small>",
    unsafe_allow_html=True,
)
