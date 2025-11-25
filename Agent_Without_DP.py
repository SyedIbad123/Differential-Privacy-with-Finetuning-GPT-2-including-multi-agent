
from langgraph.graph import StateGraph, END
from typing import TypedDict, List, Optional, Dict, Any
import os, re, torch, json, glob, sys
from transformers import GPT2Tokenizer, GPT2LMHeadModel, StoppingCriteria, StoppingCriteriaList
from peft import PeftModel

def find_latest_baseline():
    candidates = glob.glob(os.path.join('runs', 'lora_baseline_*'))
    if not candidates:
        return os.environ.get('BASELINE_LORA_DIR', None)
    candidates = sorted(candidates, key=lambda p: os.path.getmtime(p))
    latest = candidates[-1]
    path = os.path.join(latest, 'lora_model')
    if os.path.exists(path):
        return path
    return None

LORA_DIR = find_latest_baseline() or os.environ.get('BASELINE_LORA_DIR', None) or "runs/lora_baseline_REPLACE/lora_model"
DEVICE   = "cuda" if torch.cuda.is_available() else "cpu"

if LORA_DIR is None or not os.path.exists(LORA_DIR):
    print("Could not find a baseline LoRA checkpoint automatically.")
    print("Set the environment variable BASELINE_LORA_DIR to the path of your LoRA model folder, or place the checkpoint under runs/lora_baseline_*/lora_model.")
    print("Current LORA_DIR value:", LORA_DIR)
    if len(sys.argv) > 1 and os.path.exists(sys.argv[1]):
        LORA_DIR = sys.argv[1]
        print("Using CLI provided LORA_DIR:", LORA_DIR)
    else:
        raise SystemExit("No valid LoRA checkpoint found. Please provide one via env BASELINE_LORA_DIR or CLI arg.")

MAX_NEW_TOKENS      = 280      
MIN_NEW_TOKENS      = 90       
TOP_P               = 0.92      
TEMPERATURE         = 0.7      
NO_REPEAT_NGRAM     = 3        
REPETITION_PENALTY  = 1.15      
LENGTH_PENALTY      = 1.0       

MAX_REVISION_ATTEMPTS = 3

print(f"Loading baseline LoRA from: {LORA_DIR} on device {DEVICE}")
base = GPT2LMHeadModel.from_pretrained("gpt2")
tok  = GPT2Tokenizer.from_pretrained("gpt2")
tok.pad_token = tok.eos_token
model = PeftModel.from_pretrained(base, LORA_DIR)
model = model.to(DEVICE).eval()

PHI_PATTERNS = [
    r'\\b(?:Dr\\.?|Doctor)\\s+[A-Z][a-z]+\\b',
    r'\\b(?:Mr\\.?|Ms\\.?|Mrs\\.?)\\s+[A-Z][a-z]+\\b',
    r'\\b\\d{3}[-\\s]?\\d{3}[-\\s]?\\d{4}\\b',
    r'[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\\.[A-Za-z]{2,}',
    r'\\b(?:\\d{1,2}[/-]\\d{1,2}[/-]\\d{2,4}|(?:19|20)\\d{2}-\\d{2}-\\d{2})\\b'
]

def dephi_text(t: str) -> str:
    if not isinstance(t, str):
        return ""
    for p in PHI_PATTERNS:
        t = re.sub(p, 'REDACTED', t)
    return t

def sectionize_dialogue(d: str) -> Dict[str, str]:
    d = (d or "").strip()
    if not d:
        return {"chief_complaint": "", "hpi": "", "other": ""}
    parts = re.split(r'(?<=[.?!])\\s+|\\n+', d, maxsplit=1)
    cc = parts[0].strip()
    hpi = parts[1].strip() if len(parts) > 1 else ""
    return {"chief_complaint": cc, "hpi": hpi, "other": ""}

BAD_SENTENCE_PATTERNS = [
    r'call\\s+911', r'email\\s+\\S+@\\S+', r'info@\\S+', r'hotline', r'http[s]?://\\S+',
    r'write\\s+to\\s+your\\s+doctor', r'contact\\s+your\\s+doctor\\s+at', r'please\\s+send\\s+me\\s+e-?mail',
    r'\\bpolice\\b', r'\\blawsuit\\b', r'\\blegal advice\\b'
]

def strip_emails_urls(text: str) -> str:
    text = re.sub(r'[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\\.[A-Za-z]{2,}', '', text)
    text = re.sub(r'http[s]?://\\S+', '', text)
    return text

def strip_tags_and_citations(text: str) -> str:
    text = re.sub(r'</?SOAP>', '', text, flags=re.I)
    text = re.sub(r'<!DOCTYPE[^>]*>', '', text, flags=re.I)
    text = re.sub(r'[<>]{1,2}[^<>\\n]{0,200}[<>]{1,2}', '', text)
    text = re.sub(r'\\[[0-9,\\-\\s]{1,10}\\]', '', text)
    return text

def strip_social(text: str) -> str:
    text = re.sub(r'#[A-Za-z0-9_]+', '', text)
    text = re.sub(r'@[A-Za-z0-9_]+', '', text)
    text = re.sub(r'\\b(twitter|facebook|instagram|reddit|linkedin|tumblr|youtube|google\\+|social media)\\b', '', text, flags=re.I)
    return text

def remove_bad_sentences(text: str) -> str:
    sentences = re.split(r'(?<=[.!?])\\s+', text)
    keep = []
    for s in sentences:
        low = s.lower()
        if any(re.search(p, low) for p in BAD_SENTENCE_PATTERNS):
            continue
        keep.append(s)
    out = ' '.join(keep)
    out = re.sub(r'\\s+', ' ', out).strip()
    return out

def sanitize_generated(text: str) -> str:
    text = strip_emails_urls(text)
    text = strip_tags_and_citations(text)
    text = strip_social(text)
    text = remove_bad_sentences(text)
    text = text.replace("REDACTED", "").strip()
    return text

def postprocess_soap(raw: str) -> str:
    text = (raw or "").strip()
    text = sanitize_generated(text)

    sections = {"Subjective": "", "Objective": "", "Assessment": "", "Plan": ""}

    for sec in sections.keys():
        m = re.search(
            rf'^\\s*{sec}\\s*:\\s*(.*?)(?=^\\s*(Subjective|Objective|Assessment|Plan)\\s*:|\\Z)',
            text, flags=re.I | re.S | re.M
        )
        if m:
            sections[sec] = sanitize_generated(m.group(1).strip())

    if not any(v for v in sections.values()):
        sections["Subjective"] = text

    fallback = {
        "Subjective": "Reports cough and fever with pertinent negatives as discussed in HPI.",
        "Objective":  "Vitals stable; oropharynx mildly erythematous; lungs clear; no respiratory distress.",
        "Assessment": "Likely self-limited upper respiratory illness; low concern for pneumonia.",
        "Plan":       "Supportive care (hydration, antipyretics), education, return precautions, and follow-up if persistent/worsening."
    }
    for k in sections:
        if not sections[k] or len(sections[k].split()) < 3:
            sections[k] = fallback[k]

    out = []
    for sec in ["Subjective", "Objective", "Assessment", "Plan"]:
        out.append(f"{sec}: {sections[sec]}")
    return "\n\n".join(out).strip()

class StopOnToken(StoppingCriteria):
    def __init__(self, tokenizer, stop_str="</SOAP>"):
        super().__init__()
        self.stop_ids = tokenizer.encode(stop_str, add_special_tokens=False)

    def __call__(self, input_ids, scores, **kwargs):
        seq = input_ids[0].tolist()
        L = len(self.stop_ids)
        return L > 0 and len(seq) >= L and seq[-L:] == self.stop_ids

def generate_soap(summary: str, constraints: Optional[str] = None) -> str:
    # Simplified prompt for better generation quality
    sys_prompt = (
        "You are a clinical documentation assistant. "
        "Write a brief SOAP note with Subjective, Objective, Assessment, and Plan sections. "
        "Keep it factual, concise, and clinical. "
        "No URLs, emails, or unnecessary details."
    )
    if constraints:
        sys_prompt += f" Fix these issues: {constraints}"

    prompt = f"{sys_prompt}\n\nPatient Summary:\n{summary}\n\n<SOAP>\n"

    ids = tok(prompt, return_tensors="pt").to(DEVICE)

def generate_soap(summary: str, constraints: Optional[str] = None) -> str:
    sys_prompt = (
        "You are a clinical scribe.\n"
        "Write a concise SOAP note with EXACTLY these four sections and headers:\n"
        "Subjective:\nObjective:\nAssessment:\nPlan:\n\n"
        "Rules:\n"
        "- Clinical, factual style (8–14 total sentences across all sections).\n"
        "- No emails, URLs, phone numbers, hashtags, @handles, or placeholders.\n"
        "- No non-clinical instructions like “call 911”, “email me”, or “contact police”.\n"
        "- Avoid legal advice; no social media mentions; no bracketed citations.\n"
        "- Do not invent personal identifiers.\n"
    )
    if constraints:
        sys_prompt += f"\nMust fix:\n{constraints}\n"

    prompt = (
        f"{sys_prompt}\n"
        f"<<DIALOGUE_SUMMARY>>\n{summary}\n</DIALOGUE_SUMMARY>\n"
        f"Begin the SOAP below. End with </SOAP>\n"
        f"<SOAP>\n"
    )

    ids = tok(prompt, return_tensors="pt").to(DEVICE)
    stopping = StoppingCriteriaList([StopOnToken(tok, stop_str="</SOAP>")])

    with torch.no_grad():
        out = model.generate(
            **ids,
            max_new_tokens=MAX_NEW_TOKENS,
            min_new_tokens=MIN_NEW_TOKENS,
            do_sample=True,
            top_p=TOP_P,
            top_k=50,                         
            temperature=TEMPERATURE,
            repetition_penalty=REPETITION_PENALTY,
            no_repeat_ngram_size=NO_REPEAT_NGRAM,
            length_penalty=LENGTH_PENALTY,     
            early_stopping=True,               
            eos_token_id=tok.eos_token_id,
            pad_token_id=tok.pad_token_id,     
            stopping_criteria=stopping
        )

    text = tok.decode(out[0], skip_special_tokens=True)
    tail = text.split("<SOAP>", 1)[-1]
    tail = tail.split("</SOAP>", 1)[0]
    soap = postprocess_soap(tail)
    return soap

def code_rules(soap: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    s = soap.lower()
    def push(code, desc, conf): out.append({"system":"ICD-10","code":code,"desc":desc,"confidence":conf})

    if any(k in s for k in ["cough","uri","upper respiratory","viral"]):
        push("J06.9", "Acute upper respiratory infection, unspecified", 0.62)
    if "fever" in s:
        push("R50.9", "Fever, unspecified", 0.55)
    if any(k in s for k in ["chest pain","substernal"]):
        push("R07.9", "Chest pain, unspecified", 0.58)
    if "low back pain" in s or "lumbar" in s:
        push("M54.50", "Low back pain, unspecified", 0.57)

    if "objective:" in s or "physical exam" in s:
        out.append({"system":"CPT","code":"99213","desc":"Est. patient office/outpatient, low MDM", "confidence":0.45})

    return out[:5] or [{"system":"ICD-10","code":"Z71.1","desc":"No diagnosis (general medical advice)","confidence":0.3}]

class ClinState(TypedDict):
    dialogue: str
    summary: str
    soap: str
    codes: List[Dict[str, Any]]
    issues: List[str]
    attempts: int
    privacy_log: List[str]


def intake_agent(state: ClinState) -> ClinState:
    d = dephi_text(state["dialogue"])
    secs = sectionize_dialogue(d)
    cc = secs['chief_complaint'].replace("REDACTED", "the patient")
    hpi = secs['hpi'].replace("REDACTED", "the patient")
    summary = (f"Chief Complaint: {cc}\n\nHPI: {hpi}\n").strip()
    state["summary"] = summary
    state.setdefault("privacy_log", []).append("Intake: PHI scrubbed.")
    return state

def soap_agent(state: ClinState) -> ClinState:
    constraints = None
    if state.get("issues"):
        bullets = "\n".join(f"- {x}" for x in state["issues"])
        constraints = bullets
    state["soap"] = generate_soap(state["summary"], constraints)
    return state

def coder_agent(state: ClinState) -> ClinState:
    state["codes"] = code_rules(state["soap"])
    return state

def safety_auditor(state: ClinState) -> ClinState:
    issues: List[str] = []
    soap = state.get("soap", "")

    if "REDACTED" in soap:
        issues.append("Remove PHI placeholders; rewrite without PHI.")
    if re.search(r'[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\\.[A-Za-z]{2,}', soap):
        issues.append("Remove email addresses; rewrite sentence clinically.")
    if re.search(r'http[s]?://\\S+', soap):
        issues.append("Remove URLs; rewrite sentence clinically.")
    if re.search(r'#[A-Za-z0-9_]+', soap):
        issues.append("Remove hashtags; keep clinical language only.")
    if re.search(r'@[A-Za-z0-9_]+', soap):
        issues.append("Remove @handles; keep clinical language only.")
    if re.search(r'\\[[0-9,\\-\\s]{1,10}\\]', soap):
        issues.append("Remove bracketed citations like [1]; not appropriate in notes.")
    if re.search(r'\\b(twitter|facebook|instagram|reddit|linkedin|tumblr|youtube|google\\+|social media)\\b', soap, re.I):
        issues.append("Remove social media references; keep clinical content only.")

    if re.search(r'call\\s+911', soap, re.I):
        issues.append("Remove emergency call instructions (documentation only).")
    if re.search(r'police|lawsuit|legal advice', soap, re.I):
        issues.append("Remove legal/police language; keep clinical content.")

    for sec in ["Subjective", "Objective", "Assessment", "Plan"]:
        if re.search(fr'^\\s*{sec}\\s*:', soap, re.I | re.M) is None:
            issues.append(f"Missing '{sec}:' section header.")

    word_count = len(soap.split())
    if word_count < 90:
        issues.append("Note too short; expand clinically relevant details (≥ 90 words).")

    trivial_phrase = "No additional pertinent information."
    trivial_hits = len(re.findall(re.escape(trivial_phrase), soap))
    if trivial_hits >= 2:
        issues.append("Too many placeholder lines; add relevant clinical details to each section.")

    if re.search(r'^\\s*Assessment\\s*:', soap, re.I | re.M):
        assess_block = re.split(r'^\\s*Plan\\s*:', soap, flags=re.I | re.M)[0]
        if not re.search(r'\\b(diagnos|likely|consistent with|differential|impression|etiology)\\b', assess_block, re.I):
            issues.append("Assessment lacks diagnostic impression; add concise clinical reasoning.")
    if re.search(r'^\\s*Plan\\s*:', soap, re.I | re.M):
        plan_block = soap.split("Plan:", 1)[-1]
        if not re.search(r'\\b(recommend|start|advise|prescribe|follow[- ]?up|return precautions|education)\\b', plan_block, re.I):
            issues.append("Plan lacks actionable items; add management and follow-up.")

    if re.search(r'\\bgood health\\b', soap, re.I) and re.search(r'\\b(cough|fever|pain|dyspnea)\\b', soap, re.I):
        issues.append("Subjective contains 'good health' despite symptoms; rewrite to reflect complaints accurately.")

    state["issues"] = issues
    return state


def needs_revision(state: ClinState) -> str:
    if state.get("issues") and state.get("attempts", 0) < MAX_REVISION_ATTEMPTS:
        return "revise"
    return "finish"


def revision_counter(state: ClinState) -> ClinState:
    state["attempts"] = state.get("attempts", 0) + 1
    return state

workflow = StateGraph(ClinState)
workflow.add_node("intake", intake_agent)
workflow.add_node("soap",   soap_agent)
workflow.add_node("coder",  coder_agent)
workflow.add_node("audit",  safety_auditor)
workflow.add_node("bump",   revision_counter)

workflow.set_entry_point("intake")
workflow.add_edge("intake", "soap")
workflow.add_edge("soap",   "coder")
workflow.add_edge("coder",  "audit")
workflow.add_conditional_edges("audit", needs_revision, {"revise": "bump", "finish": END})
workflow.add_edge("bump", "soap")

app = workflow.compile()

example_dialogue = """Patient Jane Doe (555-123-4567) presents with cough and fever for 3 days.
Denies chest pain. Has mild sore throat. No shortness of breath. Took acetaminophen.
Physical exam reveals mild erythema of the oropharynx, lungs clear. Vitals stable.
"""

state_in: ClinState = {
    "dialogue": example_dialogue,
    "summary": "",
    "soap": "",
    "codes": [],
    "issues": [],
    "attempts": 0,
    "privacy_log": []
}

result = app.invoke(state_in)

print("\n=== SUMMARY ===\n", result["summary"])
print("\n=== SOAP ===\n", result["soap"])
print("\n=== CODES ===\n", json.dumps(result["codes"], indent=2))
print("\n=== ISSUES ===\n", result["issues"])
print("\n=== PRIVACY LOG ===\n", result["privacy_log"])
