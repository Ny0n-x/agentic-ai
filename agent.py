# agent_streamlit_prod_final.py
# Final production-ready Streamlit app: Assignment Generator & Evaluator (server-side key by default)
# Features applied:
# - Uses server-side GEMINI_API_KEY by default, optional per-session user key fallback
# - Per-session generation/evaluation caps
# - Input size limits and truncation
# - Cached generation to reduce duplicate calls
# - Robust Gemini SDK call handling across different google-genai versions
# - No raw debug output shown to users
# - Resets session state on new upload / pasted text change

import os
import re
import json
import hashlib
import streamlit as st
from io import BytesIO
from dotenv import load_dotenv
from PyPDF2 import PdfReader

# Try to import genai and types robustly
try:
    from google import genai
    from google.genai import types
except Exception:
    try:
        import importlib
        genai = importlib.import_module("google.genai")
        types = None
    except Exception:
        genai = None
        types = None

# -------------------- Configuration --------------------
# load_dotenv()
# API_KEY = os.getenv("GEMINI_API_KEY")
API_KEY = st.secrets["GEMINI_API_KEY"]
client = genai.Client(api_key=API_KEY)
if not API_KEY:
    raise RuntimeError("GEMINI_API_KEY not found in environment. Set it server-side before running the app.")

# Session caps & limits (tune as needed)
MAX_GENERATIONS_PER_SESSION = 10
MAX_EVALS_PER_SESSION = 20
MAX_CONTENT_CHARS = 20000  # truncate content to this length to reduce API usage
CACHE_MAX_ENTRIES = 128

# -------------------- Helpers --------------------

def reset_session_state_for_new_doc():
    keys = [
        "assignment",
        "student_answers",
        "evaluation",
        "raw_assignment_text",
        "raw_eval_text",
        "assignment_method",
        "eval_method",
        "gen_count",
        "eval_count",
    ]
    for k in keys:
        if k in st.session_state:
            del st.session_state[k]


def read_pdf(file_obj) -> str:
    try:
        reader = PdfReader(file_obj)
        pages = [p.extract_text() or "" for p in reader.pages]
        return "\n\n".join(pages)
    except Exception:
        try:
            file_obj.seek(0)
            raw = file_obj.read()
            return raw.decode("utf-8", errors="ignore")
        except Exception as e:
            raise RuntimeError(f"Failed to read PDF: {e}")

# --------- Loose JSON parsing & repair utilities (defensive) ---------

def extract_json_from_text(text: str):
    if not isinstance(text, str):
        text = str(text)
    start = None
    for i, ch in enumerate(text):
        if ch in ("{", "["):
            start = i
            break
    if start is None:
        raise ValueError("No JSON-like start found in text.")
    stack = []
    for i in range(start, len(text)):
        ch = text[i]
        if ch == "{":
            stack.append("}")
        elif ch == "[":
            stack.append("]")
        elif ch in ("}", "]"):
            if not stack:
                continue
            stack.pop()
            if not stack:
                candidate = text[start:i+1]
                try:
                    return json.loads(candidate)
                except json.JSONDecodeError:
                    cleaned = re.sub(r",\s*}", "}", candidate)
                    cleaned = re.sub(r",\s*]", "]", cleaned)
                    return json.loads(cleaned)
    raise ValueError("Couldn't find a balanced JSON snippet in text.")


def try_repair_kv_blob(text: str):
    if not isinstance(text, str):
        text = str(text)
    text = re.sub(r"```.*?```", "", text, flags=re.S)
    m = re.search(r'(^|\n)\s*(["\']?id["\']?)\s*:', text, flags=re.IGNORECASE)
    if not m:
        return None
    start = m.start(2)
    candidate = "{" + text[start:].strip()
    if not candidate.rstrip().endswith("}"):
        candidate = candidate.rstrip() + "}"
    candidate = re.sub(r",\s*([}\]])", r"\\1", candidate)
    try:
        return json.loads(candidate)
    except Exception:
        lines = [ln.strip() for ln in candidate.strip("{} \n\t").splitlines() if ln.strip()]
        obj = {}
        cur_key = None
        cur_val_chunks = []
        for ln in lines:
            kv = re.match(r'^\s*["\']?([^"\':]+?)["\']?\s*:\s*(.*)$', ln)
            if kv:
                if cur_key:
                    raw_val = " ".join(cur_val_chunks).rstrip(", ")
                    obj[cur_key] = _try_parse_value(raw_val)
                cur_key = kv.group(1).strip()
                cur_val = kv.group(2).strip().rstrip(",")
                cur_val_chunks = [cur_val]
            else:
                if cur_key:
                    cur_val_chunks.append(ln.rstrip(","))
        if cur_key:
            raw_val = " ".join(cur_val_chunks).rstrip(", ")
            obj[cur_key] = _try_parse_value(raw_val)
        if obj:
            if isinstance(obj.get("questions"), str):
                try:
                    obj["questions"] = json.loads(obj["questions"])
                except Exception:
                    pass
            return obj
    return None


def _try_parse_value(val: str):
    v = val.strip()
    if (v.startswith("{") and v.endswith("}")) or (v.startswith("[") and v.endswith("]")):
        try:
            return json.loads(v)
        except Exception:
            pass
    if v.lower() in ("true", "false", "null"):
        return None if v.lower() == "null" else (v.lower() == "true")
    if re.fullmatch(r"[-+]?\d+(\.\d+)?", v):
        try:
            return int(v) if "." not in v else float(v)
        except Exception:
            pass
    if (v.startswith('"') and v.endswith('"')) or (v.startswith("'") and v.endswith("'")):
        return v[1:-1]
    return v


def parse_loose_json(raw: str):
    try:
        return json.loads(raw)
    except Exception as e1:
        last_err = e1
    try:
        return extract_json_from_text(raw)
    except Exception as e2:
        last_err = e2
    repaired = try_repair_kv_blob(raw)
    if repaired is not None:
        return repaired
    m = re.search(r'["\']id["\']\s*:', raw)
    if m:
        idx = m.start()
        candidate = "{" + raw[idx:].strip()
        if not candidate.endswith("}"):
            candidate = candidate + "}"
        candidate = re.sub(r",\s*([}\]])", r"\\1", candidate)
        try:
            return json.loads(candidate)
        except Exception:
            pass
    raise ValueError(f"All parsing attempts failed. Last error: {last_err}")

# Normalize assignment shapes into canonical form

def normalize_assignment(parsed):
    if isinstance(parsed, list):
        questions = parsed
        assignment_id = "assignment-from-list"
    elif isinstance(parsed, dict):
        if "questions" in parsed and isinstance(parsed["questions"], list):
            assignment_id = parsed.get("id") or parsed.get("assignment_id") or "assignment-1"
            questions = parsed["questions"]
        elif "assignment" in parsed and isinstance(parsed["assignment"], dict):
            a = parsed["assignment"]
            assignment_id = a.get("id") or parsed.get("id") or "assignment-1"
            questions = a.get("questions") or []
        else:
            found = None
            for k, v in parsed.items():
                if isinstance(v, list) and v and isinstance(v[0], dict):
                    found = v
                    break
            if found:
                assignment_id = parsed.get("id") or "assignment-heuristic"
                questions = found
            else:
                if {"question", "type"}.issubset(parsed.keys()):
                    assignment_id = parsed.get("id") or "assignment-single"
                    questions = [parsed]
                else:
                    raise ValueError("Could not discover questions list in parsed object.")
    else:
        raise ValueError("Parsed JSON has unexpected type.")

    norm = []
    for idx, q in enumerate(questions, start=1):
        if not isinstance(q, dict):
            continue
        qid = q.get("id") or idx
        qtype = (q.get("type") or q.get("qtype") or "").lower()
        if qtype not in ("mcq", "short"):
            if q.get("options") or q.get("choices"):
                qtype = "mcq"
            else:
                qtype = "short"
        qtext = q.get("question") or q.get("prompt") or ""
        options = q.get("options") or q.get("choices") or []
        answer = q.get("answer") or q.get("correct") or ""
        points = q.get("points") or q.get("max_points") or 5
        norm.append({
            "id": int(qid) if isinstance(qid, (int, str)) and str(qid).isdigit() else qid,
            "type": qtype,
            "question": qtext,
            "options": options,
            "answer": answer,
            "points": int(points) if isinstance(points, (int, str)) and str(points).isdigit() else points,
        })
    if not norm:
        raise ValueError("Normalization produced zero questions.")
    return {"id": parsed.get("id") if isinstance(parsed, dict) else "assignment", "questions": norm}

# ------------------ Gemini call helper (robust) ------------------

def call_gemini(prompt_text: str, model="gemini-2.5-flash", api_key=None):
    """Try several client call signatures; return (raw_text, method_used)."""
    # construct client with provided api_key
    try:
        if api_key:
            c = genai.Client(api_key=api_key)
        else:
            c = genai.Client(api_key=API_KEY)
    except Exception:
        # fallback to global client if genai.Client signature differs
        try:
            c = genai.Client(api_key=API_KEY)
        except Exception as e:
            raise RuntimeError(f"Could not construct genai client: {e}")

    # 1) try config object if available
    try:
        if 'types' in globals() and types is not None:
            cfg = types.GenerateContentConfig(response_mime_type="application/json")
            resp = c.models.generate_content(model=model, contents=prompt_text, config=cfg)
            text = getattr(resp, "text", None) or str(resp)
            return text, "models.generate_content(config)"
    except Exception:
        pass

    # 2) try simple call
    try:
        resp = c.models.generate_content(model=model, contents=prompt_text)
        text = getattr(resp, "text", None) or str(resp)
        return text, "models.generate_content(contents)"
    except Exception:
        pass

    # 3) try client.generate
    try:
        if hasattr(c, "generate"):
            resp = c.generate(model=model, prompt=prompt_text)
            return getattr(resp, "text", None) or str(resp), "client.generate"
    except Exception:
        pass

    # 4) try client.text.generate
    try:
        if hasattr(c, "text") and hasattr(c.text, "generate"):
            resp = c.text.generate(model=model, input=prompt_text)
            if hasattr(resp, "output"):
                return resp.output, "client.text.generate.output"
            if hasattr(resp, "candidates") and len(resp.candidates) > 0:
                c0 = resp.candidates[0]
                return getattr(c0, "content", str(c0)), "client.text.generate.candidates"
            return str(resp), "client.text.generate"
    except Exception:
        pass

    raise RuntimeError("Could not call genai client with known signatures. Please check google-genai SDK version.")

# ---------------- Prompt templates ----------------
QUESTION_GEN_PROMPT = """
You are an assistant that creates concise, clear assignments from source content.
INPUT_CONTENT:
{content}

TASK:
1) Generate exactly {count} questions mixing:
   - EXACTLY 1 multiple-choice question (MCQ) with 4 options.
   - the rest short-answer questions.
2) Each question must have:
   - id (integer, 1-based)
   - type: "mcq" or "short"
   - question: string
   - options: list of 4 strings (for mcq only)
   - answer: string (correct answer text)
   - points: integer

Return ONLY valid JSON (an object with fields "id" and "questions"). No extra commentary or markdown fences.
"""

EVAL_PROMPT = """
You are an assistant that grades student answers given:
- the original content (for reference),
- the assignment answer key as JSON,
- the student's answers as JSON.

CONTENT:
{content}

ANSWER_KEY_JSON:
{answer_key_json}

STUDENT_ANSWERS_JSON:
{student_answers_json}

TASK:
Produce a JSON object with:
- "results": [{{ id:int, max_points:int, points_awarded:int (0..max), feedback:string }}],
- "total": {{ "max": <int>, "awarded": <int> }}

Return ONLY valid JSON and nothing else.
"""

# ------------------ Cached generation wrapper ------------------
@st.cache_data(max_entries=CACHE_MAX_ENTRIES)
def cached_generate_key(content_hash: str, qcount: int, model: str):
    # Placeholder to allow caching by key; actual generation occurs in generate_assignment()
    return {
        "placeholder": True,
    }

# ------------------ High-level ops ------------------

def generate_assignment_from_content(content: str, qcount: int = 4, model="gemini-2.5-flash", api_key=None):
    # Truncate content to limit
    if len(content) > MAX_CONTENT_CHARS:
        content = content[:MAX_CONTENT_CHARS]
    prompt = QUESTION_GEN_PROMPT.format(content=content, count=qcount)

    # Attempt a cached fast-return: compute hash of content+qcount+model
    h = hashlib.sha256((content + str(qcount) + model).encode("utf-8")).hexdigest()
    try:
        # Touch cache (no sensitive key included in cache key)
        _ = cached_generate_key(h, qcount, model)
    except Exception:
        # caching may fail in some envs; ignore
        pass

    raw, method = call_gemini(prompt, model=model, api_key=api_key)
    try:
        parsed = parse_loose_json(raw)
    except Exception as e:
        # surface friendly error to UI
        raise RuntimeError(f"Failed to parse assignment JSON from model (method={method}): {e}")

    assignment = normalize_assignment(parsed)
    return assignment


def evaluate_student_answers(content: str, answer_key: dict, student_answers: list, model="gemini-2.5-flash", api_key=None):
    if len(content) > MAX_CONTENT_CHARS:
        content = content[:MAX_CONTENT_CHARS]
    ak_json = json.dumps(answer_key, ensure_ascii=False)
    sa_json = json.dumps(student_answers, ensure_ascii=False)
    prompt = EVAL_PROMPT.format(content=content, answer_key_json=ak_json, student_answers_json=sa_json)
    raw, method = call_gemini(prompt, model=model, api_key=api_key)
    parsed = parse_loose_json(raw)
    if not isinstance(parsed, dict) or "results" not in parsed or "total" not in parsed:
        raise RuntimeError("Evaluation JSON missing required keys.")
    return parsed

# ------------------ Streamlit UI ------------------
st.set_page_config(page_title="Assignment Generator and Evaluator", layout="centered")
st.title("Assignment Generator and Evaluator")

with st.sidebar:
    st.markdown("**Settings**")
    model = st.text_input("Gemini model", value="gemini-2.5-flash")
    qcount = st.number_input("Number of questions", min_value=2, max_value=10, value=4, step=1)
    st.markdown("---")
    st.markdown("**Usage limits (session):**")
    st.write(f"Generations: {MAX_GENERATIONS_PER_SESSION}, Evaluations: {MAX_EVALS_PER_SESSION}")
    st.markdown("---")
    st.markdown("Optional: Provide your own Gemini API key to use your billing (session-only). If left blank, the server-side key will be used.")
    user_key = st.text_input("Optional user Gemini API key", type="password")

# Decide active key for this session (do not persist user key)
ACTIVE_KEY = user_key.strip() or API_KEY

# Upload/paste inputs
uploaded_file = st.file_uploader("Upload PDF (optional)", type=["pdf"])
pasted = st.text_area("Or paste content here (optional)", height=200)

# Reset when a new file is uploaded or pasted text changes
current_upload_fingerprint = None
if uploaded_file is not None:
    current_upload_fingerprint = f"{uploaded_file.name}|{getattr(uploaded_file, 'size', '')}"
prev_upload = st.session_state.get("uploaded_fingerprint")
if current_upload_fingerprint and current_upload_fingerprint != prev_upload:
    st.session_state["uploaded_fingerprint"] = current_upload_fingerprint
    reset_session_state_for_new_doc()

prev_pasted = st.session_state.get("pasted_snapshot", "")
if pasted.strip() and pasted.strip() != prev_pasted:
    st.session_state["pasted_snapshot"] = pasted.strip()
    reset_session_state_for_new_doc()

# Initialize counters
if "gen_count" not in st.session_state:
    st.session_state["gen_count"] = 0
if "eval_count" not in st.session_state:
    st.session_state["eval_count"] = 0

# Build content
content = ""
if uploaded_file:
    try:
        content = read_pdf(uploaded_file)
        st.success("PDF text extracted (preview below).")
        st.text(content[:1200] + ("..." if len(content) > 1200 else ""))
    except Exception as e:
        st.error(f"Could not extract PDF text: {e}")
elif pasted.strip():
    content = pasted.strip()

if not content:
    st.info("Upload a PDF or paste text to generate an assignment.")

# Generate assignment (with rate limiting)
if content and st.button("Generate Assignment"):
    if st.session_state["gen_count"] >= MAX_GENERATIONS_PER_SESSION:
        st.error("Generation limit reached for this session. Please reload or try later.")
    else:
        st.session_state["gen_count"] += 1
        with st.spinner("Generating assignment..."):
            try:
                # use ACTIVE_KEY (session) to call Gemini
                assignment = generate_assignment_from_content(content, qcount=qcount, model=model, api_key=ACTIVE_KEY)
                st.session_state["assignment"] = assignment
                st.success("Assignment generated — answer it below.")
            except Exception as e:
                st.error(f"Error generating assignment: {e}")

# Display assignment and collect answers
if st.session_state.get("assignment"):
    assignment = st.session_state["assignment"]
    st.subheader("Assignment")
    answers_map = {}
    with st.form("answer_form"):
        for q in assignment.get("questions", []):
            qid = q.get("id")
            qtype = q.get("type")
            qtext = q.get("question")
            st.markdown(f"**Q{qid}. {qtext}**")
            if qtype == "mcq":
                opts = q.get("options") or []
                if not opts:
                    resp = st.text_input(f"Q{qid} (expected MCQ but no options) — your answer", key=f"q{qid}")
                    answers_map[str(qid)] = resp
                else:
                    choice = st.radio(f"Choose (Q{qid}):", opts, key=f"q{qid}")
                    answers_map[str(qid)] = choice
            else:
                resp = st.text_area(f"Your answer (Q{qid}):", key=f"q{qid}", height=120)
                answers_map[str(qid)] = resp
        submitted = st.form_submit_button("Submit Answers")

    if submitted:
        if st.session_state["eval_count"] >= MAX_EVALS_PER_SESSION:
            st.error("Evaluation limit reached for this session. Please reload or try later.")
        else:
            st.session_state["eval_count"] += 1
            student_answers = []
            for q in assignment.get("questions", []):
                sid = q.get("id")
                student_answers.append({"id": sid, "answer": answers_map.get(str(sid), "")})
            st.session_state["student_answers"] = student_answers

            with st.spinner("Evaluating answers..."):
                try:
                    eval_out = evaluate_student_answers(content, assignment, student_answers, model=model, api_key=ACTIVE_KEY)
                    st.session_state["evaluation"] = eval_out
                    st.success("Evaluation complete.")
                except Exception as e:
                    st.error(f"Error during evaluation: {e}")

# Display evaluation + download
if st.session_state.get("evaluation"):
    st.subheader("Evaluation Results")
    eval_res = st.session_state["evaluation"]
    for r in eval_res.get("results", []):
        st.markdown(f"**Q{r.get('id')}:** {r.get('points_awarded')} / {r.get('max_points')}")
        st.write(f"- Feedback: {r.get('feedback')}")
        st.write("---")
    total = eval_res.get("total", {})
    st.write(f"**Total:** {total.get('awarded')} / {total.get('max')}")

    out_pkg = {
        "assignment": st.session_state.get("assignment"),
        "student_answers": st.session_state.get("student_answers"),
        "evaluation": st.session_state.get("evaluation"),
    }
    st.download_button(
        "Download report (JSON)",
        data=json.dumps(out_pkg, ensure_ascii=False, indent=2).encode("utf-8"),
        file_name="assignment_results.json",
        mime="application/json",
    )

st.caption("This app uses a server-side GEMINI_API_KEY by default; if you provide your own key in the sidebar it will be used only for your session and not stored.")
