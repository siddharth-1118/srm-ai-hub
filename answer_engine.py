"""
Answer synthesis for the SRM Admissions AI Assistant.

Retrieval (rag_engine.py) finds the brochure chunks most relevant to a
question. This module turns those chunks into a *natural-language answer*
instead of dumping raw PDF text:

* Local mode (default, fully offline): extracts the most query-relevant
  sentences from the top chunks, deduplicates near-duplicates, and renders
  program lists as proper bullet lists.
* LLM mode (optional): if an OpenAI-compatible API key is configured
  (LLM_API_KEY / OPENAI_API_KEY env var, or the sidebar settings), the top
  chunks are passed as grounding context and the model writes a fluent,
  grounded answer. Falls back to local mode on any error.
"""

import json
import os
import re
import urllib.request

# ---------------------------------------------------------------------------
# Sentence splitting
# ---------------------------------------------------------------------------

# Protect abbreviations before splitting so "B.Tech", "M.Sc", "Ph.D", "e.g."
# are not broken into separate "sentences".
_ABBREV_PROTECT = re.compile(r'\b([A-Za-z])\.(?=\S)')
_SENT_SPLIT = re.compile(r'(?<=[.!?])\s+(?=[A-Z0-9])')
_PHONE_RE = re.compile(r'\b0\d{2}\s*-\s*\d{4}\s*\d{4}\b')


def _split_sentences(text):
    """Split text into sentences, keeping abbreviations intact."""
    protected = _ABBREV_PROTECT.sub(r'\1', text)
    return [s.strip() for s in _SENT_SPLIT.split(protected) if s.strip()]


def _score_sentence(sentence, tokens):
    """Count how many query tokens (with light plural handling) appear."""
    norm = re.sub(r'[^a-z0-9 ]', '', sentence.lower())
    score = 0
    for t in tokens:
        if not t or len(t) < 2:
            continue
        variants = {t, t + 's'}
        if t.endswith('s') and len(t) > 3 and not t.endswith(('ss', 'us', 'is')):
            variants.add(t[:-1])
        if any(
            re.search(r'(?<![a-z0-9])' + re.escape(v) + r'(?![a-z0-9])', norm)
            for v in variants
        ):
            score += 1
    return score


def _best_sentences(chunk, tokens, max_sentences=3):
    """Return the highest-scoring sentences of a chunk, in score order."""
    scored = [
        (s, _score_sentence(s, tokens))
        for s in _split_sentences(chunk["text"])
    ]
    scored.sort(key=lambda x: x[1], reverse=True)
    return [s for s, sc in scored if sc >= 1][:max_sentences]


def _section_title(chunk):
    kind = chunk.get("kind")
    return {
        "ug_engineering": "B.Tech Programs",
        "ug_overview": "Undergraduate Programs",
        "pg_overview": "Postgraduate Programs",
        "pg_school": "Postgraduate Programs",
    }.get(kind, chunk.get("doc_title") or "Programs")


def _phone_dedupe_key(sentence):
    m = _PHONE_RE.search(sentence)
    return m.group(0) if m else None


# ---------------------------------------------------------------------------
# Local synthesis
# ---------------------------------------------------------------------------

_PROGRAM_WORDS = re.compile(
    r'speciali[sz]|course|program|degree|major|branch|school|offer|available|list'
)
# List answers need explicit list intent ("what/which/all/show me the programs").
# A yes/no question like "is there a semester abroad program" must NOT become a
# bulleted program list just because it contains the word "program".
_LIST_INTENT = re.compile(
    r'\b(list|all|what|which|show|tell|name|available|offered|options|overview)\b'
)

# Words so generic they appear in nearly every brochure chunk; a sentence that
# matches ONLY these is not evidence the sentence answers the question.
_GENERIC_TOKENS = {
    "srm", "srmist", "institute", "offer", "offers", "offered",
    "program", "programs", "course", "courses", "student", "students",
    "admission", "admissions", "available", "study", "college", "colleges",
    "know", "want", "tell", "please", "also", "per", "including", "many",
}


def _top_chunks(results, min_frac=0.35, max_chunks=3):
    """Return the strongest results, dropping far-below-top noise chunks."""
    if not results:
        return []
    top_score = results[0]["score"]
    out = []
    for res in results:
        if res["score"] < min_frac * top_score:
            break
        out.append(res)
        if len(out) >= max_chunks:
            break
    return out


def _compose_list_answer(results, engine, query, max_sections=4, max_bullets=30):
    """Compose a bulleted program-list answer from curated list chunks."""
    tokens = engine._query_tokens(query)
    lines = []
    sections = 0
    for res in results:
        chunk = res["chunk"]
        text = chunk["text"]
        if chunk.get("curated") and " | " in text:
            # Only render sections that actually relate to the question.
            if _score_sentence(text, tokens) < 1:
                continue
            body = text.split(": ", 1)[1] if ": " in text else text
            items = [i.strip() for i in body.split("|") if i.strip()]
            if len(items) > 1:
                header = chunk.get("school") or _section_title(chunk)
                lines.append(f"**{header}**")
                lines.extend(f"- {i}" for i in items[:max_bullets])
                if len(items) > max_bullets:
                    lines.append(f"- …and {len(items) - max_bullets} more")
                sections += 1
                if sections >= max_sections:
                    break
                continue
        # Prose chunks: only trusted curated overviews make clean list intros,
        # and only for broad "list everything" style questions.
        if chunk.get("curated") and engine.is_list_query(query):
            for s in _best_sentences(chunk, tokens, max_sentences=2):
                lines.append(s)
            sections += 1
            if sections >= max_sections:
                break
    return "\n".join(lines) if lines else None


def _token_weights(tokens, chunks):
    """Weight query tokens by rarity across the whole knowledge base: rare
    tokens drive selection, generic words like 'srm' or 'offer' (present in
    almost every chunk) don't."""
    norm_chunks = [
        re.sub(r'[^a-z0-9 ]', '', c["text"].lower()) for c in chunks
    ]
    df = {}
    for t in tokens:
        if not t or len(t) < 2:
            continue
        variants = {t, t + 's'}
        if t.endswith('s') and len(t) > 3 and not t.endswith(('ss', 'us', 'is')):
            variants.add(t[:-1])
        df[t] = sum(
            1 for text in norm_chunks
            if any(
                re.search(r'(?<![a-z0-9])' + re.escape(v) + r'(?![a-z0-9])', text)
                for v in variants
            )
        )
    return {t: 1.0 / (1.0 + df.get(t, 0)) for t in tokens}


def _score_sentence_weighted(sentence, tokens, weights):
    """Weighted sentence score: sum of rarity weights of matched tokens."""
    norm = re.sub(r'[^a-z0-9 ]', '', sentence.lower())
    total = 0.0
    for t in tokens:
        if not t or len(t) < 2:
            continue
        variants = {t, t + 's'}
        if t.endswith('s') and len(t) > 3 and not t.endswith(('ss', 'us', 'is')):
            variants.add(t[:-1])
        if any(
            re.search(r'(?<![a-z0-9])' + re.escape(v) + r'(?![a-z0-9])', norm)
            for v in variants
        ):
            total += weights.get(t, 0.0)
    return total


def _compose_fact_answer(query, results, engine, max_sentences=3):
    """Compose a factual answer from the best matching sentences.

    Hand-curated chunks (the FAQ/ground-truth entries) always win over raw
    PDF text: if any curated sentence matches the question, only curated
    sentences are used, ranked by rarity-weighted query overlap.
    """
    tokens = engine._query_tokens(query)
    weights = _token_weights(tokens, engine.chunks)
    specific = [t for t in tokens if t not in _GENERIC_TOKENS]

    def _meaningful(sentence):
        """True unless the sentence matches only generic words."""
        if _score_sentence_weighted(sentence, specific, weights) > 0:
            return True
        return _score_sentence(sentence, specific) + \
            _score_sentence(sentence, [t for t in tokens if t in _GENERIC_TOKENS]) >= 2

    curated_cands, raw_cands = [], []
    for rank, res in enumerate(results):
        chunk = res["chunk"]
        for s in _split_sentences(chunk["text"]):
            score = _score_sentence_weighted(s, tokens, weights)
            if score <= 0 or not _meaningful(s):
                continue
            (curated_cands if chunk.get("curated") else raw_cands).append(
                (score, rank, s)
            )
    candidates = (curated_cands or raw_cands)
    candidates.sort(key=lambda x: (x[0], -x[1]), reverse=True)

    chosen = []
    seen = set()
    seen_phones = set()
    for _, _, s in candidates:
        key = re.sub(r'[^a-z0-9]', '', s.lower())[:80]
        if key in seen:
            continue
        phone = _phone_dedupe_key(s)
        if phone is not None:
            if phone in seen_phones:
                continue  # same phone repeated in a second sentence
            seen_phones.add(phone)
        seen.add(key)
        chosen.append(s)
        if len(chosen) >= max_sentences:
            break
    if not chosen:
        return None
    return " ".join(chosen)


# ---------------------------------------------------------------------------
# Targeted extractors: answer common questions directly with the exact fact
# ---------------------------------------------------------------------------

_PHONE_QUERY_RE = re.compile(r'helpline|phone|contact|toll\s*free|reach')
_PLACEMENT_QUERY_RE = re.compile(
    r'placement|salary|lpa|package|recruit|job\s*offer|companies\s*visited|highest\s*offer'
)


def _extract_phone_answer(query, results):
    """Answer helpline/contact queries with the exact phone number."""
    if not _PHONE_QUERY_RE.search(query.lower()):
        return None
    for res in results:
        m = _PHONE_RE.search(res["chunk"]["text"])
        if m:
            return (
                f"The SRM admissions helpline number is **{m.group(0).strip()}**. "
                "You can also apply online at https://applications.srmist.edu.in."
            )
    return None


def _extract_placement_answer(query, results):
    """Answer placement queries with the brochure's placement statistics."""
    if not _PLACEMENT_QUERY_RE.search(query.lower()):
        return None
    for res in results:
        text = res["chunk"]["text"]
        if "Placement Statistics 2025" not in text:
            continue
        m = re.search(
            r'Total Job Offers received by students: ([\d,]+)\+?\. '
            r'Highest salary package offered: ([\d]+) LPA\. '
            r'International / Global Job Offers: ([\d,]+)\+?\. '
            r'Jobs with package above 20 LPA: ([\d,]+)\+?\. '
            r'Total companies that visited campus for recruitment: ([\d,]+)\+?\. '
            r'Global Capability Centres \(GCC\) that hired: ([\d,]+)\+?\.',
            text,
        )
        if not m:
            continue
        offers, highest, intl, above20, companies, gcc = m.groups()
        return (
            f"Per the brochure's **Placement Statistics 2025**, SRMIST recorded "
            f"**{offers}+ job offers** with a **highest salary package of {highest} LPA**. "
            f"There were {intl}+ international offers, {above20}+ offers above 20 LPA, "
            f"{companies}+ companies visited the campus, and {gcc}+ Global Capability "
            f"Centres (GCC) hired students."
        )
    return None


_HOSTEL_QUERY_RE = re.compile(
    r'hostel|room\s+fee|accommodation\s+fee|room\s+rent',
    re.IGNORECASE
)


_HOSTEL_FEE_ROW_RE = re.compile(
    r'((?:AC|Non-AC|Non AC)[\w\s,./()-]*?(?:Sharing|sharing))\s*'
    r'(?:\(([^)]+)\)\s*)?'
    r'(?:([A-Za-z][A-Za-z0-9 .()-]+?):\s*)?'
    r'Hostel\s+Rs\s*([\d,]+)\s*\+\s*Mess\s+Rs\s*([\d,]+)\s*=\s*Total\s+Rs\s*([\d,]+)',
    re.IGNORECASE,
)


def _format_hostel_table(text, filter_name=None):
    """Parse curated hostel text and render as a markdown table, optionally filtering by specific hostel name."""
    # Extract title (everything before the first colon before a fee entry)
    title_match = re.match(r'([^:]+):', text)
    title = title_match.group(1).strip() if title_match else "Hostel Fees"
   
    rows = _HOSTEL_FEE_ROW_RE.findall(text)
    if not rows:
        return None

    # If filtering, check if any row's hostel name matches the filter
    if filter_name:
        has_match = False
        norm_filter = re.sub(r'[^a-z0-9]', '', filter_name.lower())
        for room_type, hostel_name, hostel_name2, hostel_fees, mess_fees, total_fees in rows:
            name = (hostel_name or hostel_name2 or "").strip()
            norm_name = re.sub(r'[^a-z0-9]', '', name.lower())
            if norm_filter in norm_name or norm_name in norm_filter:
                has_match = True
                break
        if not has_match:
            return None

    lines = [f"### {title}", ""]
    lines.append("| Room Type & Sharing | Hostel Name | Hostel Fees | Mess Fees | Total Fees |")
    lines.append("|---|---|---|---|---|")
    seen_rows = set()
    rows_added = 0
    for room_type, hostel_name, hostel_name2, hostel_fees, mess_fees, total_fees in rows:
        name = (hostel_name or hostel_name2 or "-").strip()
        # Fix known OCR quirks
        if name.lower() == "bunker cot":
            name = "Meenakshi"
            
        if filter_name:
            norm_name = re.sub(r'[^a-z0-9]', '', name.lower())
            norm_filter = re.sub(r'[^a-z0-9]', '', filter_name.lower())
            if norm_filter not in norm_name and norm_name not in norm_filter:
                continue

        row_key = f"{room_type.strip().lower()}|{name.lower()}"
        if row_key in seen_rows:
            continue
        seen_rows.add(row_key)
        lines.append(
            f"| {room_type.strip()} | {name} | Rs {hostel_fees} | Rs {mess_fees} | Rs {total_fees} |"
        )
        rows_added += 1
    
    if rows_added == 0:
        return None

    # Extract footer notes (Laundry, booking date, etc.)
    note_match = re.search(r'(Boys|Girls)?\s*Laundry[^.]*Optional', text, re.IGNORECASE)
    if note_match:
        lines.append(f"\n*{note_match.group(0).strip()}*")
    booking_match = re.search(r'Booking starts?[^.]+\.', text, re.IGNORECASE)
    if not booking_match:
        booking_match = re.search(r'Booking opens[^.]+\.', text, re.IGNORECASE)
    if booking_match:
        lines.append(f"*{booking_match.group(0).strip()}*")
    
    return "\n".join(lines)


def _extract_hostel_answer(query, results):
    """Answer hostel fee queries by collecting curated hostel fee chunks."""
    if not _HOSTEL_QUERY_RE.search(query.lower()):
        return None

    # Check if a specific hostel name is mentioned in the query
    hostel_words = [w for w in re.findall(r'\b[a-zA-Z]+\b', query.lower()) 
                    if w not in ["what", "is", "the", "fees", "of", "boys", "girls", "hostel", "accommodation", "room", "sharing", "rates", "srm", "srmist"]]
    if hostel_words:
        matched_in_chunks = False
        for word in hostel_words:
            for res in results:
                if word in res["chunk"]["text"].lower():
                    matched_in_chunks = True
                    break
            if matched_in_chunks:
                break
        
        if not matched_in_chunks:
            potential_names = [w for w in hostel_words if w not in ["srm", "srmist", "campus", "kattankulathur", "fees", "fee"]]
            if potential_names:
                name_cap = " ".join([w.capitalize() for w in potential_names])
                return (
                    f"I couldn't find the fees for **{name_cap} Hostel** in the provided SRM circulars.\n\n"
                    "The available circulars only list fees for these boys hostels:\n"
                    "- **Pierre Fauchard (PF)**\n"
                    "- **N Block**\n"
                    "- **Green Pearl (off-campus)**\n"
                    "- **Adhiyaman**\n"
                    "- **Oori**\n"
                    "- **Kaari**\n"
                    "- **Nelson Mandela**\n"
                    "- **Sannasi**\n"
                    "- **JA Block 2 (off-campus)**"
                )

    # Detect if a known hostel is explicitly requested to filter rows
    KNOWN_HOSTELS_LOWER = [
        "pierre fauchard", "pf", "oori", "kaari", "adhiyaman", "n block", "n-block", 
        "green pearl", "ja block", "sannasi", "nelson mandela", "premium boys", 
        "malligai", "senbagam", "kopperundevi", "esq", "kalpana chawla", "meenakshi", 
        "thamarai", "mullai"
    ]
    filter_name = None
    query_lower = query.lower()
    for kh in KNOWN_HOSTELS_LOWER:
        if kh in query_lower:
            filter_name = kh
            break

    hostel_chunks = []
    for res in results:
        chunk = res["chunk"]
        if chunk.get("kind") == "hostel_fees" and chunk.get("curated"):
            hostel_chunks.append(chunk["text"])
    if not hostel_chunks:
        # Fall back to any chunk mentioning hostel fees with actual numbers
        for res in results:
            text = res["chunk"]["text"]
            if "Total Fees" in text or "HOSTEL" in text.upper():
                hostel_chunks.append(text)
    if not hostel_chunks:
        return None
    # Deduplicate near-duplicates (by first 80 chars normalized)
    seen = set()
    unique = []
    for text in hostel_chunks:
        key = re.sub(r'\s+', ' ', text[:80]).lower().strip()
        if key not in seen:
            seen.add(key)
            unique.append(text)
    # Try to render as tables, show at most 4 to avoid overwhelming output
    parts = []
    for text in unique[:4]:
        table = _format_hostel_table(text, filter_name=filter_name)
        if table:
            parts.append(table)
        else:
            if not filter_name:
                parts.append(text)
    return "\n\n".join(parts) if parts else None


def synthesize_answer(query, results, engine):
    """Turn retrieved chunks into a natural-language answer (offline)."""
    if not results:
        return None

    # Targeted extractors first: exact, direct answers for common questions.
    for extractor in (_extract_phone_answer, _extract_placement_answer, _extract_hostel_answer):
        answer = extractor(query, results)
        if answer:
            return answer

    listy = engine.is_list_query(query) or (
        bool(_PROGRAM_WORDS.search(query.lower()))
        and bool(_LIST_INTENT.search(query.lower()))
    )
    if listy:
        answer = _compose_list_answer(results, engine, query)
        if answer:
            return answer

    answer = _compose_fact_answer(query, results, engine)
    if not answer:
        # Last resort: the single best-matching chunk, cleaned up.
        answer = results[0]["chunk"]["text"]
    return answer


# ---------------------------------------------------------------------------
# Small talk: greetings, thanks, goodbyes and help requests get friendly
# replies instead of the "not in the brochure" fallback message.
# ---------------------------------------------------------------------------

_GREETING_RE = re.compile(
    r'^(hi+|hii*|hello+|hey+|yo+|namaste|greetings|howdy|'
    r'good\s*(morning|afternoon|evening|day|night))'
    r'(\s+(there|everyone|guys|sir|mam|maam|friend))?$'
)
_HOW_ARE_YOU_RE = re.compile(
    r'^(how\s+(are\s+you|r\s+u|have\s+you\s+been|\'s\s+it\s+going|do\s+you\s+do)|'
    r'what\'?s\s+(up|your\s+name)|sup|yo|how\s+are\s+things|'
    r'what\s+are\s+you\s+up\s+to|how\s+is\s+life|how\s+is\s+your\s+day|'
    r'how\s+was\s+your\s+day|how\s+do\s+you\s+feel|are\s+you\s+ok|'
    r'are\s+you\s+fine|you\s+good|r\s+u\s+ok|r\s+u\s+fine)$'
)
_THANKS_RE = re.compile(
    r'^(thanks+|thank\s*you+|thx|ty|appreciate|nice|cool|awesome|great|'
    r'good|perfect|excellent|wonderful|amazing|fantastic|brilliant|'
    r'that\s+(helps|works|is\s+great|is\s+good|is\s+perfect))'
    r'(\s+(a\s+lot|so\s+much|very\s+much|for\s+.*))?$'
)
_BYE_RE = re.compile(
    r'^(bye+|goodbye+|see\s+you(\s+(later|soon))?|cya|take\s+care|'
    r'gotta\s+go|gtg|talk\s+to\s+you\s+later|catch\s+you\s+later|'
    r'ill\s+be\s+back|i\'?ll\s+be\s+back|farewell|adios|ciao|sayonara)$'
)
_INTRO_RE = re.compile(
    r'^(who\s+are\s+you|what\s+(can|do)\s+you\s+do|what\s+are\s+you|'
    r'how\s+do\s+you\s+work|what\s+is\s+your\s+purpose|what\s+is\s+your\s+role|'
    r'what\s+is\s+your\s+name|what\s+(are|do)\s+you\s+know|'
    r'what\s+documents\s+do\s+you\s+have|'
    r'help|help\s+me|can\s+you\s+help(\s+me)?|what\s+can\s+i\s+ask|'
    r'what\s+questions\s+can\s+i\s+ask|what\s+should\s+i\s+ask|'
    r'what\s+topics(\s+do\s+you\s+know)?|what\s+information|'
    r'tell\s+me\s+about\s+yourself|introduce\s+yourself|'
    r'about\s+you|your\s+capabilities)$'
)

_GREETING_MSG = (
    "Hello! 👋 I'm the SRM Admissions AI Assistant. I answer questions using "
    "SRM documents — the Admission Brochure 2026-27 (programs, entrance exams, "
    "scholarships, placements, fees, campuses and contact details) and hostel "
    "circulars/fee structures for various batches."
    "\n\nHere are some things I can help you with:\n"
    "🎓 **Admissions** — Programs, entrance exams, eligibility, cutoffs\n"
    "💰 **Fees & Scholarships** — Tuition fees, hostel fees, scholarship schemes\n"
    "🏢 **Hostels** — Booking schedules, room types, fee structures for all batches\n"
    "📊 **Placements** — Salary packages, top recruiters, job offers\n"
    "📞 **Contact** — Helpline numbers, how to apply\n"
    "\nWhat would you like to know?"
)
_HOW_ARE_YOU_MSG = (
    "I'm doing great, thanks for asking! 😊 I'm here to help you with "
    "anything about SRM admissions, hostels, fees, placements and more."
    "\n\nYou can ask me things like:\n"
    "- What are the B.Tech programs at SRM?\n"
    "- What are the hostel fees for first year boys?\n"
    "- What is the placement salary?\n"
    "- What scholarships does SRM offer?\n"
    "\nWhat would you like to know?"
)
_THANKS_MSG = (
    "You're welcome! 😊 Happy to help! If you have more questions about SRM "
    "admissions, hostels, fees, or placements — just ask! I'm here to help."
)
_BYE_MSG = (
    "Goodbye! 👋 It was great talking to you! Come back anytime you have "
    "questions about SRM admissions, hostels, fees, or anything else. Good luck! 🍀"
)
_INTRO_MSG = (
    "I'm the SRM Admissions AI Assistant, built on the SRM Admission Brochure "
    "2026-27 and hostel circulars. Here's what I know about:\n"
    "\n🎓 **Programs** — B.Tech, MBA, M.Sc, MBBS, BDS, Law, Pharmacy and more\n"
    "📝 **Entrance Exams** — SRMJEEE, SRMJEEH, SRMJEEL, SRMJEEM, NATA, NEET\n"
    "💰 **Fees** — Hostel fees, mess fees for all batches (2023-2027)\n"
    "🏢 **Hostels** — Booking schedules, room types, fee structures\n"
    "📊 **Placements** — 14,000+ job offers, 65 LPA highest salary\n"
    "🏆 **Scholarships** — Founder's, Merit, Sports, Defence and more\n"
    "📞 **Contact** — Helpline: 080-6908 7000\n"
    "\nGo ahead — ask me anything!"
)


def handle_smalltalk(query):
    """Return a friendly reply for greetings/thanks/goodbyes/help, else None."""
    q = re.sub(r'\s+', ' ', query.strip().lower().rstrip('!.,?'))
    if _GREETING_RE.fullmatch(q):
        return _GREETING_MSG
    if _HOW_ARE_YOU_RE.fullmatch(q):
        return _HOW_ARE_YOU_MSG
    if _THANKS_RE.fullmatch(q):
        return _THANKS_MSG
    if _BYE_RE.fullmatch(q):
        return _BYE_MSG
    if _INTRO_RE.fullmatch(q):
        return _INTRO_MSG
    return None


# ---------------------------------------------------------------------------
# Optional LLM generation (OpenAI-compatible chat completions)
# ---------------------------------------------------------------------------

DEFAULT_BASE_URL = "https://api.openai.com/v1"
DEFAULT_MODEL = "gpt-4o-mini"


def llm_configured(config=None):
    """True when an API key is available (sidebar config or env var)."""
    if config and config.get("api_key"):
        return True
    return bool(os.environ.get("LLM_API_KEY") or os.environ.get("OPENAI_API_KEY"))


def llm_answer(query, results, config=None, timeout=60):
    """Generate a grounded answer from the retrieved chunks via chat API."""
    api_key = (config or {}).get("api_key") or \
        os.environ.get("LLM_API_KEY") or os.environ.get("OPENAI_API_KEY")
    base_url = (config or {}).get("base_url") or \
        os.environ.get("LLM_BASE_URL", DEFAULT_BASE_URL)
    model = (config or {}).get("model") or \
        os.environ.get("LLM_MODEL", DEFAULT_MODEL)

    context = "\n\n".join(
        f"[Excerpt {i + 1} — Page {r['chunk']['page']}]\n{r['chunk']['text']}"
        for i, r in enumerate(results[:8])
    )
    system = (
        "You are an admissions assistant for SRM Institute of Science and "
        "Technology (SRMIST). Answer the user's question using ONLY the brochure "
        "excerpts provided below. Be concise, accurate and friendly, in the tone "
        "of a helpful admissions officer. If the excerpts do not contain the "
        "answer, say so honestly and suggest where to check. Do not invent "
        "facts. Mention the page number in parentheses when you use an excerpt."
    )
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {
                "role": "user",
                "content": f"Brochure excerpts:\n\n{context}\n\nQuestion: {query}",
            },
        ],
        "temperature": 0.2,
        "max_tokens": 500,
    }
    req = urllib.request.Request(
        f"{base_url.rstrip('/')}/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return data["choices"][0]["message"]["content"].strip()

def generate_answer(query, results, engine, config=None, chat_history=None):
    # Run targeted extractors first for exact rendering (e.g. hostel fee tables, placement stats)
    for extractor in (_extract_phone_answer, _extract_placement_answer, _extract_hostel_answer):
        ans = extractor(query, results)
        if ans:
            return ans, False

    # Route broad listing queries to heuristic list composer for instant, 100% accurate results
    listy = engine.is_list_query(query) or (
        bool(_PROGRAM_WORDS.search(query.lower()))
        and bool(_LIST_INTENT.search(query.lower()))
    )
    if listy:
        ans = _compose_list_answer(results, engine, query)
        if ans:
            return ans, False

    if llm_configured(config):
        try:
            return llm_answer(query, results, config), True
        except Exception as e:  # noqa: BLE001 - fall back to offline synthesis
            print(f"LLM API answer failed ({e}); using local synthesis.")
            
    # Try local offline LLM
    try:
        from llm_local import generate_local_answer
        ans = generate_local_answer(query, results, chat_history)
        if ans:
            return ans, True
    except Exception as e:
        print(f"Local offline LLM failed ({e}); using heuristic synthesis.")
        
    return synthesize_answer(query, results, engine), False


if __name__ == "__main__":
    import sys

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    from rag_engine import RAGEngine

    engine = RAGEngine()
    engine.build_or_load_index()
    queries = [
        "what is the admission helpline number",
        "what are the fees for btech",
        "placement statistics and highest salary",
        "what are the MBA specializations",
        "all pg programs in srm institute of science and technology",
        "what are the B.Tech programs",
        "hello",
    ]
    for q in queries:
        res = engine.search(q, top_k=8)
        ans, used_llm = generate_answer(q, res, engine)
        print("=" * 60)
        print(f"Q: {q}  (mode: {'LLM' if used_llm else 'local'})")
        print("A:", ans if ans else "(no answer)")
        print()
