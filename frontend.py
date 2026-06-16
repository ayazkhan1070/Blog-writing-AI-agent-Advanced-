from __future__ import annotations

import json
import os
import re
import zipfile
from datetime import date
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, Optional, List, Iterator, Tuple

import pandas as pd
import streamlit as st

# -----------------------------
# Import your compiled LangGraph app
# -----------------------------
from backend2 import app, _safe_filename

if "user_email" not in st.session_state:
    st.session_state["user_email"] = None
if "login_step" not in st.session_state:
    st.session_state["login_step"] = "email"
if "temp_email" not in st.session_state:
    st.session_state["temp_email"] = ""
if "sent_otp" not in st.session_state:
    st.session_state["sent_otp"] = ""
if "topic_prefill" not in st.session_state:
    st.session_state["topic_prefill"] = ""


# -----------------------------
# Active Session & State Database
# -----------------------------
SESSIONS_FILE = Path("users") / "active_sessions.json"

def get_session_email(session_id: str) -> Optional[str]:
    if not SESSIONS_FILE.exists():
        return None
    try:
        data = json.loads(SESSIONS_FILE.read_text(encoding="utf-8"))
        return data.get(session_id)
    except Exception:
        return None

def create_session(email: str) -> str:
    import uuid
    session_id = str(uuid.uuid4())
    data = {}
    if SESSIONS_FILE.exists():
        try:
            data = json.loads(SESSIONS_FILE.read_text(encoding="utf-8"))
        except Exception:
            pass
    data[session_id] = email
    SESSIONS_FILE.parent.mkdir(parents=True, exist_ok=True)
    SESSIONS_FILE.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return session_id

def delete_session(session_id: str):
    if not SESSIONS_FILE.exists():
        return
    try:
        data = json.loads(SESSIONS_FILE.read_text(encoding="utf-8"))
        if session_id in data:
            del data[session_id]
            SESSIONS_FILE.write_text(json.dumps(data, indent=2), encoding="utf-8")
    except Exception:
        pass


# -----------------------------
# User Access Status Database
# -----------------------------
STATUS_FILE = Path("users") / "user_status.json"

def get_user_status(email: str) -> str:
    email_clean = email.strip().lower()
    if email_clean == "ayazbnk0107@gmail.com":
        return "admin"
    if not STATUS_FILE.exists():
        return "pending"
    try:
        data = json.loads(STATUS_FILE.read_text(encoding="utf-8"))
        return data.get(email_clean, "pending")
    except Exception:
        return "pending"

def set_user_status(email: str, status: str):
    email_clean = email.strip().lower()
    if email_clean == "ayazbnk0107@gmail.com":
        return
    data = {}
    if STATUS_FILE.exists():
        try:
            data = json.loads(STATUS_FILE.read_text(encoding="utf-8"))
        except Exception:
            pass
    data[email_clean] = status
    STATUS_FILE.parent.mkdir(parents=True, exist_ok=True)
    STATUS_FILE.write_text(json.dumps(data, indent=2), encoding="utf-8")


# -----------------------------
# Automated Quality Evaluation
# -----------------------------
def evaluate_blog_coherence(markdown_content: str) -> dict:
    if not markdown_content:
        return {"score": 0, "comment": "Empty content."}
    has_headings = "##" in markdown_content or "#" in markdown_content
    has_min_length = len(markdown_content) > 200
    score = 1 if (has_headings and has_min_length) else 0
    return {
        "score": score,
        "comment": "Pass: Structured document with headings." if score else "Fail: Lacks formatting or is too short."
    }

def evaluate_blog_relevance(topic: str, markdown_content: str) -> dict:
    if not markdown_content:
        return {"score": 0, "comment": "Empty content."}
    try:
        from langchain_openai import ChatOpenAI
        from langchain_core.messages import HumanMessage
        
        judge = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        prompt = f"""Evaluate if this generated blog content is relevant to the topic: "{topic}".
        
        Blog Content:
        {markdown_content[:1500]}
        
        Respond with exactly '1' if it is relevant, or '0' if it is not relevant. Just output the number."""
        
        res = judge.invoke([HumanMessage(content=prompt)]).content.strip()
        score = 1 if "1" in res else 0
        return {
            "score": score,
            "comment": "Pass: Content matches topic." if score else "Fail: Off-topic or low-relevance content."
        }
    except Exception as e:
        return {"score": 0, "comment": f"Error running LLM relevance check: {e}"}


def evaluate_blog_faithfulness(evidence: list, markdown_content: str) -> dict:
    if not markdown_content:
        return {"score": 0, "comment": "Empty content."}
    if not evidence:
        return {"score": 1, "comment": "Not applicable (No research evidence collected; Closed Book mode)."}
        
    try:
        from langchain_openai import ChatOpenAI
        from langchain_core.messages import HumanMessage
        
        evidence_text = "\n".join(
            f"- {e.get('title', 'Evidence')}: {e.get('snippet', '')} ({e.get('url', '')})"
            for e in evidence if isinstance(e, dict)
        )
        
        judge = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        prompt = f"""You are a fact-checker. Assess if the claims in the generated blog are supported by the provided evidence.
        
        Evidence:
        {evidence_text[:2000]}
        
        Blog Content:
        {markdown_content[:1500]}
        
        Respond with exactly '1' if all major claims/facts in the blog are supported by the evidence (or if there are no direct factual contradictions).
        Respond with '0' if the blog contains hallucinated facts, false metrics, or incorrect descriptions not found in the evidence.
        Just output the number '1' or '0'."""
        
        res = judge.invoke([HumanMessage(content=prompt)]).content.strip()
        score = 1 if "1" in res else 0
        return {
            "score": score,
            "comment": "Pass: No hallucinations or contradictions found." if score else "Fail: Contains claims not backed by research evidence."
        }
    except Exception as e:
        return {"score": 0, "comment": f"Error running faithfulness check: {e}"}


def evaluate_blog_seo(topic: str, markdown_content: str) -> dict:
    if not markdown_content:
        return {"score": 0, "comment": "Empty content."}
    
    score = 0
    checks = []
    
    # 1. H1 Check
    has_h1 = any(line.strip().startswith("# ") for line in markdown_content.splitlines())
    if has_h1:
        score += 25
        checks.append("✅ Main Heading (H1) present")
    else:
        checks.append("❌ Missing Main Heading (H1)")
        
    # 2. H2 Check
    has_h2 = any(line.strip().startswith("## ") for line in markdown_content.splitlines())
    if has_h2:
        score += 25
        checks.append("✅ Subheadings (H2) present")
    else:
        checks.append("❌ Missing Subheadings (H2)")
        
    # 3. Keyword Check (checks if some words of topic are in the headings/content)
    words = [w.lower() for w in re.split(r'\W+', topic) if len(w) > 3]
    found_words = [w for w in words if w in markdown_content.lower()]
    keyword_score = len(found_words) / len(words) if words else 1.0
    if keyword_score >= 0.5:
        score += 25
        checks.append("✅ Topic keywords used in content")
    else:
        checks.append("❌ Low usage of topic keywords")
        
    # 4. Image Alt text check
    has_images = "![" in markdown_content
    if has_images:
        images = re.findall(r'!\[(.*?)\]\((.*?)\)', markdown_content)
        all_have_alt = all(bool(alt.strip()) for alt, url in images) if images else False
        if all_have_alt:
            score += 25
            checks.append("✅ All images have Alt text description")
        else:
            checks.append("❌ Some images are missing Alt text descriptions")
    else:
        # If no images, reward formatting/link check
        has_links = "[" in markdown_content and "](" in markdown_content
        if has_links:
            score += 25
            checks.append("✅ Formatting includes reference links")
        else:
            checks.append("⚠️ No images or reference links found")
            
    return {
        "score": score,
        "comment": " | ".join(checks)
    }


def save_user_state(email: str, last_out: Any, topic: str = ""):
    safe_email = re.sub(r'[^a-zA-Z0-9_.-]', '_', email)
    state_file = Path("users") / safe_email / "session_state.json"
    state_file.parent.mkdir(parents=True, exist_ok=True)
    try:
        state_file.write_text(json.dumps({
            "last_out": last_out,
            "topic": topic
        }, indent=2), encoding="utf-8")
    except Exception as e:
        print(f"[Error] Failed to save user state: {e}")

def load_user_state(email: str) -> Tuple[Optional[Any], str]:
    safe_email = re.sub(r'[^a-zA-Z0-9_.-]', '_', email)
    state_file = Path("users") / safe_email / "session_state.json"
    if not state_file.exists():
        return None, ""
    try:
        data = json.loads(state_file.read_text(encoding="utf-8"))
        return data.get("last_out"), data.get("topic", "")
    except Exception:
        return None, ""


def extract_name_from_email(email: str) -> str:
    if not email:
        return "User"
    username = email.split("@")[0]
    parts = re.split(r"[^a-zA-Z]+", username)
    parts = [p.capitalize() for p in parts if p]
    name = " ".join(parts)
    return name or username.capitalize()


# Restore session from URL parameters if available
session_param = st.query_params.get("session")
if session_param and st.session_state["user_email"] is None:
    email = get_session_email(session_param)
    if email:
        st.session_state["user_email"] = email
        last_out, topic_val = load_user_state(email)
        st.session_state["last_out"] = last_out
        st.session_state["topic_prefill"] = ""


def send_email_notification(to_email: str, subject: str, body: str) -> bool:
    import smtplib
    from email.mime.text import MIMEText
    from email.mime.multipart import MIMEMultipart
    
    smtp_user = os.environ.get("SMTP_USER")
    smtp_pass = os.environ.get("SMTP_PASSWORD")
    
    if not smtp_user or not smtp_pass:
        # Fallback logging to console so the user is never locked out during testing
        print(f"\n========================================================")
        print(f"[SMTP OFFLINE - EMAIL SIMULATION]")
        print(f"To: {to_email}")
        print(f"Subject: {subject}")
        print(f"Body: {body}")
        print(f"========================================================\n")
        return False
        
    try:
        msg = MIMEMultipart()
        msg['From'] = smtp_user
        msg['To'] = to_email
        msg['Subject'] = subject
        msg.attach(MIMEText(body, 'plain'))
        
        server = smtplib.SMTP_SSL("smtp.gmail.com", 465, timeout=10)
        server.login(smtp_user, smtp_pass)
        server.sendmail(smtp_user, to_email, msg.as_string())
        server.close()
        return True
    except Exception as e:
        print(f"[SMTP Error] Failed to send email to {to_email}: {e}")
        return False

# -----------------------------
# Helpers
# -----------------------------
def safe_slug(title: str) -> str:
    s = title.strip().lower()
    s = re.sub(r"[^a-z0-9 _-]+", "", s)
    s = re.sub(r"\s+", "_", s).strip("_")
    return s or "blog"


def bundle_zip(md_text: str, md_filename: str, images_dir: Path) -> bytes:
    buf = BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr(md_filename, md_text.encode("utf-8"))

        if images_dir.exists() and images_dir.is_dir():
            for p in images_dir.rglob("*"):
                if p.is_file():
                    # Map the user's specific image path to a standard 'images/filename' inside the ZIP
                    arcname = Path("images") / p.relative_to(images_dir)
                    z.write(p, arcname=str(arcname))
    return buf.getvalue()


def images_zip(images_dir: Path) -> Optional[bytes]:
    if not images_dir.exists() or not images_dir.is_dir():
        return None
    buf = BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for p in images_dir.rglob("*"):
            if p.is_file():
                z.write(p, arcname=str(p))
    return buf.getvalue()


def try_stream(graph_app, inputs: Dict[str, Any]) -> Iterator[Tuple[str, Any]]:
    """
    Stream graph progress if available; else invoke.
    Yields ("updates"/"values"/"final", payload).
    """
    try:
        for step in graph_app.stream(inputs, stream_mode="updates"):
            yield ("updates", step)
        yield ("final", None)
        return
    except Exception:
        pass

    try:
        for step in graph_app.stream(inputs, stream_mode="values"):
            yield ("values", step)
        yield ("final", None)
        return
    except Exception:
        pass

    out = graph_app.invoke(inputs)
    yield ("final", out)


def extract_latest_state(current_state: Dict[str, Any], step_payload: Any) -> Dict[str, Any]:
    if isinstance(step_payload, dict):
        if len(step_payload) == 1 and isinstance(next(iter(step_payload.values())), dict):
            inner = next(iter(step_payload.values()))
            current_state.update(inner)
        else:
            current_state.update(step_payload)
    return current_state


# -----------------------------
# Markdown renderer that supports local images
# -----------------------------
_MD_IMG_RE = re.compile(r"!\[(?P<alt>[^\]]*)\]\((?P<src>[^)]+)\)")
_CAPTION_LINE_RE = re.compile(r"^\*(?P<cap>.+)\*$")


def _resolve_image_path(src: str) -> Path:
    src = src.strip().lstrip("./")
    return Path(src).resolve()


def render_markdown_with_local_images(md: str):
    matches = list(_MD_IMG_RE.finditer(md))
    if not matches:
        st.markdown(md, unsafe_allow_html=False)
        return

    parts: List[Tuple[str, str]] = []
    last = 0
    for m in matches:
        before = md[last : m.start()]
        if before:
            parts.append(("md", before))

        alt = (m.group("alt") or "").strip()
        src = (m.group("src") or "").strip()
        parts.append(("img", f"{alt}|||{src}"))
        last = m.end()

    tail = md[last:]
    if tail:
        parts.append(("md", tail))

    i = 0
    while i < len(parts):
        kind, payload = parts[i]

        if kind == "md":
            st.markdown(payload, unsafe_allow_html=False)
            i += 1
            continue

        alt, src = payload.split("|||", 1)

        caption = None
        if i + 1 < len(parts) and parts[i + 1][0] == "md":
            nxt = parts[i + 1][1].lstrip()
            if nxt.strip():
                first_line = nxt.splitlines()[0].strip()
                mcap = _CAPTION_LINE_RE.match(first_line)
                if mcap:
                    caption = mcap.group("cap").strip()
                    rest = "\n".join(nxt.splitlines()[1:])
                    parts[i + 1] = ("md", rest)

        if src.startswith("http://") or src.startswith("https://"):
            try:
                st.image(src, caption=caption or (alt or None), use_container_width=False)
            except Exception as e:
                st.error(f"Failed to load image from URL: {src}\nError: {e}")
        else:
            img_path = _resolve_image_path(src)
            if img_path.exists() and img_path.is_file():
                try:
                    st.image(str(img_path), caption=caption or (alt or None), use_container_width=False)
                except Exception as e:
                    st.error(f"Failed to load image: {img_path}\nError: {e}")
            else:
                st.warning(f"Image not found: `{src}` (looked for `{img_path}`)")

        i += 1


# -----------------------------
# ✅ NEW: Past blogs helpers
# -----------------------------
def list_past_blogs(email: Optional[str] = None) -> List[Path]:
    """
    Returns .md files in the user's specific directory (or current directory if none specified), newest first.
    """
    if email:
        safe_email = re.sub(r'[^a-zA-Z0-9_.-]', '_', email)
        blogs_dir = Path("users") / safe_email / "blogs"
    else:
        blogs_dir = Path(".")

    if not blogs_dir.exists():
        return []

    files = [p for p in blogs_dir.glob("*.md") if p.is_file()]
    files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return files


def read_md_file(p: Path) -> str:
    return p.read_text(encoding="utf-8", errors="replace")


def extract_title_from_md(md: str, fallback: str) -> str:
    """
    Use first '# ' heading as title if present.
    """
    for line in md.splitlines():
        if line.startswith("# "):
            t = line[2:].strip()
            return t or fallback
    return fallback


# -----------------------------
# Streamlit UI
# -----------------------------# -----------------------------
# Streamlit UI Setup
# -----------------------------
st.set_page_config(page_title="AI Blog Agent - LangGraph Workspace", layout="wide")

# Inject premium CSS for login screen and overall aesthetics
st.markdown(
    """
    <style>
    /* Premium Login Card Styles */
    .login-container {
        display: flex;
        justify-content: center;
        align-items: center;
        padding: 40px;
        margin-top: 5%;
    }
    .login-card {
        background: linear-gradient(135deg, #1e1b4b 0%, #0f172a 100%);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 24px;
        padding: 50px 40px;
        max-width: 460px;
        box-shadow: 0 20px 50px rgba(0, 0, 0, 0.5);
        text-align: center;
        color: #f8fafc;
        backdrop-filter: blur(20px);
    }
    .login-title {
        font-size: 32px;
        font-weight: 800;
        margin-bottom: 8px;
        background: linear-gradient(90deg, #c084fc 0%, #6366f1 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .login-subtitle {
        font-size: 14px;
        color: #94a3b8;
        margin-bottom: 35px;
    }
    /* Streamlit Customizations */
    div.stButton > button:first-child {
        background: linear-gradient(135deg, #a855f7 0%, #6366f1 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 12px 24px;
        font-weight: 600;
        font-size: 16px;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        box-shadow: 0 4px 15px rgba(99, 102, 241, 0.25);
    }
    div.stButton > button:first-child:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(99, 102, 241, 0.4);
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Authentication logic
if st.session_state["user_email"] is None:
    # Centered Glassmorphic Login Screen
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown('<div class="login-container">', unsafe_allow_html=True)
        st.markdown('<div class="login-card">', unsafe_allow_html=True)
        st.markdown('<div class="login-title">AI Blog Agent</div>', unsafe_allow_html=True)
        
        if st.session_state["login_step"] == "email":
            st.markdown('<div class="login-subtitle">Enter your email. A 6-digit access code will be sent to verify your identity.</div>', unsafe_allow_html=True)
            email_val = st.text_input("Email Address", placeholder="name@example.com", label_visibility="collapsed")
            st.markdown('<div style="margin-top: 15px;"></div>', unsafe_allow_html=True)
            submit_email = st.button("Send Access Code", use_container_width=True)
            
            if submit_email:
                email_clean = email_val.strip().lower()
                # Simple email validation
                if email_clean and "@" in email_clean and "." in email_clean:
                    import random
                    otp = str(random.randint(100000, 999999))
                    st.session_state["temp_email"] = email_clean
                    st.session_state["sent_otp"] = otp
                    
                    # Send code to the user and a notification to the owner
                    owner_email = os.environ.get("NOTIFICATION_EMAIL", "ayazbnk0107@gmail.com")
                    
                    user_sent = send_email_notification(
                        to_email=email_clean,
                        subject="AI Blog Agent - Verification Code",
                        body=f"Your verification code is: {otp}\n\nThis code is valid for single use."
                    )
                    
                    send_email_notification(
                        to_email=owner_email,
                        subject="[AI Blog Agent] New Login Attempt",
                        body=f"User {email_clean} is attempting to login.\nVerification Code (OTP) Sent: {otp}"
                    )
                    
                    st.session_state["login_step"] = "otp"
                    st.toast("Verification code sent!")
                    st.rerun()
                else:
                    st.error("Please enter a valid email address.")
                    
        elif st.session_state["login_step"] == "otp":
            st.markdown(f'<div class="login-subtitle">We sent a 6-digit access code to <b>{st.session_state["temp_email"]}</b>. Enter it below to log in.</div>', unsafe_allow_html=True)
            otp_val = st.text_input("6-digit Access Code", placeholder="XXXXXX", label_visibility="collapsed")
            st.markdown('<div style="margin-top: 15px;"></div>', unsafe_allow_html=True)
            submit_otp = st.button("Verify & Enter", use_container_width=True)
            
            # Check if SMTP is not configured to show a helper message for local testing
            smtp_user = os.environ.get("SMTP_USER")
            smtp_pass = os.environ.get("SMTP_PASSWORD")
            if not smtp_user or not smtp_pass:
                st.warning(f"⚠️ SMTP not configured. Logged the OTP code to your console/terminal for local testing.\n\nCode is: `{st.session_state['sent_otp']}`")
                
            if submit_otp:
                if otp_val.strip() == st.session_state["sent_otp"]:
                    user_email = st.session_state["temp_email"]
                    st.session_state["user_email"] = user_email
                    
                    # Every time a user logs in, reset their status to "pending" (except for admin)
                    if get_user_status(user_email) != "admin":
                        set_user_status(user_email, "pending")
                    
                    # Create persistent session and set URL query param
                    session_id = create_session(user_email)
                    st.query_params["session"] = session_id
                    
                    # Load user's last saved state
                    last_out, topic_val = load_user_state(user_email)
                    st.session_state["last_out"] = last_out
                    st.session_state["topic_prefill"] = ""
                    
                    # Notify owner of successful login
                    owner_email = os.environ.get("NOTIFICATION_EMAIL", "ayazbnk0107@gmail.com")
                    send_email_notification(
                        to_email=owner_email,
                        subject="[AI Blog Agent] Successful User Login",
                        body=f"User {user_email} has successfully authenticated and entered the workspace."
                    )
                    
                    # Reset login steps
                    st.session_state["login_step"] = "email"
                    st.session_state["temp_email"] = ""
                    st.session_state["sent_otp"] = ""
                    name = extract_name_from_email(user_email)
                    st.success(f"Welcome {name}!")
                    st.rerun()
                else:
                    st.error("Incorrect verification code. Please check and try again.")
            
            st.markdown('<div style="margin-top: 10px;"></div>', unsafe_allow_html=True)
            if st.button("← Back to Email", use_container_width=True):
                st.session_state["login_step"] = "email"
                st.session_state["sent_otp"] = ""
                st.rerun()
                
        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    st.stop()

# Validate logged-in user status (non-admin users)
user_email = st.session_state["user_email"]
status = get_user_status(user_email)
if status == "pending":
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown('<div class="login-container">', unsafe_allow_html=True)
        st.markdown('<div class="login-card">', unsafe_allow_html=True)
        st.markdown('<div class="login-title">Access Pending</div>', unsafe_allow_html=True)
        st.markdown('<div class="login-subtitle">Your account is pending administrator approval. Please wait for access to be granted.</div>', unsafe_allow_html=True)
        if st.button("🚪 Logout / Use another email", use_container_width=True):
            session_param = st.query_params.get("session")
            if session_param:
                delete_session(session_param)
            st.query_params.clear()
            st.session_state["user_email"] = None
            st.session_state["last_out"] = None
            st.session_state["topic_prefill"] = ""
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    st.stop()

elif status == "rejected":
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown('<div class="login-container">', unsafe_allow_html=True)
        st.markdown('<div class="login-card">', unsafe_allow_html=True)
        st.markdown('<div class="login-title">Access Denied</div>', unsafe_allow_html=True)
        st.markdown('<div class="login-subtitle">Your registration request was rejected or revoked by the administrator.</div>', unsafe_allow_html=True)
        if st.button("🚪 Logout / Use another email", use_container_width=True):
            session_param = st.query_params.get("session")
            if session_param:
                delete_session(session_param)
            st.query_params.clear()
            st.session_state["user_email"] = None
            st.session_state["last_out"] = None
            st.session_state["topic_prefill"] = ""
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    st.stop()

# Main Workspace
user_email = st.session_state["user_email"] or ""
safe_email = re.sub(r'[^a-zA-Z0-9_.-]', '_', user_email)
user_blogs_dir = Path("users") / safe_email / "blogs" if safe_email else Path(".")
user_images_dir = Path("users") / safe_email / "images" if safe_email else Path("images")

st.title("Blog Writing Agent Workspace")

with st.sidebar:
    name = extract_name_from_email(user_email)
    st.markdown(f"👋 **Welcome {name}!**")
    st.caption(f"👤 `{user_email}`")
    if st.button("🚪 Logout", use_container_width=True):
        session_param = st.query_params.get("session")
        if session_param:
            delete_session(session_param)
        st.query_params.clear()
        st.session_state["user_email"] = None
        st.session_state["last_out"] = None
        st.session_state["topic_prefill"] = ""
        st.rerun()
        
    st.divider()
    st.header("Generate New Blog")
    topic = st.text_area(
        "Topic",
        value=st.session_state.get("topic_prefill", ""),
        height=120,
    )
    as_of = st.date_input("As-of date", value=date.today())
    run_btn = st.button("🚀 Generate Blog", type="primary", use_container_width=True)

    st.divider()
    st.subheader("Your Saved Blogs")

    past_files = list_past_blogs(user_email)
    if not past_files:
        st.caption("No saved blogs found yet.")
        selected_md_file = None
    else:
        # Build labels from file name + (optional) parsed title
        options: List[str] = []
        file_by_label: Dict[str, Path] = {}
        for p in past_files[:50]:
            try:
                md_text = read_md_file(p)
                title = extract_title_from_md(md_text, p.stem)
            except Exception:
                title = p.stem
            label = f"{title}  ·  {p.name}"
            options.append(label)
            file_by_label[label] = p

        selected_label = st.radio(
            "Select a blog to load",
            options=options,
            index=0,
            label_visibility="collapsed",
        )
        selected_md_file = file_by_label.get(selected_label)

        if st.button("📂 Load selected blog", use_container_width=True):
            if selected_md_file:
                md_text = read_md_file(selected_md_file)
                json_file = selected_md_file.with_suffix('.json')
                
                state_loaded = False
                if json_file.exists():
                    try:
                        loaded_state = json.loads(json_file.read_text(encoding="utf-8"))
                        loaded_state["final"] = md_text
                        state_loaded = True
                        
                        # Evaluate on the fly if metrics are missing or incomplete
                        metrics = loaded_state.get("metrics", {})
                        if not metrics or "faithfulness" not in metrics or "seo" not in metrics:
                            topic_name = extract_title_from_md(md_text, selected_md_file.stem)
                            coh = evaluate_blog_coherence(md_text)
                            rel = evaluate_blog_relevance(topic_name, md_text)
                            faith = evaluate_blog_faithfulness(loaded_state.get("evidence", []), md_text)
                            seo = evaluate_blog_seo(topic_name, md_text)
                            loaded_state["metrics"] = {
                                "coherence": coh,
                                "relevance": rel,
                                "faithfulness": faith,
                                "seo": seo
                            }
                            try:
                                json_file.write_text(json.dumps(loaded_state, indent=2), encoding="utf-8")
                            except Exception:
                                pass
                        st.session_state["last_out"] = loaded_state
                    except Exception as e:
                        pass
                        
                if not state_loaded:
                    topic_name = extract_title_from_md(md_text, selected_md_file.stem)
                    coh = evaluate_blog_coherence(md_text)
                    rel = evaluate_blog_relevance(topic_name, md_text)
                    faith = evaluate_blog_faithfulness([], md_text)
                    seo = evaluate_blog_seo(topic_name, md_text)
                    st.session_state["last_out"] = {
                        "plan": None,
                        "evidence": [],
                        "image_specs": [],
                        "final": md_text,
                        "metrics": {
                            "coherence": coh,
                            "relevance": rel,
                            "faithfulness": faith,
                            "seo": seo
                        }
                    }
                st.session_state["topic_prefill"] = extract_title_from_md(md_text, selected_md_file.stem)
                save_user_state(user_email, st.session_state["last_out"], st.session_state["topic_prefill"])
                st.rerun()

# Storage for latest run
if "last_out" not in st.session_state:
    st.session_state["last_out"] = None

# Layout
if user_email == "ayazbnk0107@gmail.com":
    tabs = st.tabs(
        ["👥 Manage Users", "🧩 Plan", "🔎 Evidence", "📝 Markdown Preview", "🖼️ Images", "🧾 Logs", "📊 Quality Metrics"]
    )
    tab_users = tabs[0]
    tab_plan, tab_evidence, tab_preview, tab_images, tab_logs, tab_metrics = tabs[1:]
    
    with tab_users:
        st.subheader("👥 User Access Control")
        st.caption("Approve, reject, or revoke access for registered users.")
        
        # Load all user statuses
        if STATUS_FILE.exists():
            try:
                user_data = json.loads(STATUS_FILE.read_text(encoding="utf-8"))
            except Exception:
                user_data = {}
        else:
            user_data = {}
            
        if not user_data:
            st.info("No registered users found yet. Users will appear here after attempting to log in.")
        else:
            # Group users by status
            pending_users = [u for u, s in user_data.items() if s == "pending"]
            approved_users = [u for u, s in user_data.items() if s == "approved"]
            rejected_users = [u for u, s in user_data.items() if s == "rejected"]
            
            # Pending approvals
            st.markdown("### ⏳ Pending Approval")
            if not pending_users:
                st.caption("No pending user requests.")
            else:
                for u in pending_users:
                    col_u, col_app, col_rej = st.columns([3, 1, 1])
                    with col_u:
                        st.markdown(f"**{u}**")
                    with col_app:
                        if st.button("Approve", key=f"app_{u}", type="primary"):
                            set_user_status(u, "approved")
                            st.success(f"Approved {u}")
                            st.rerun()
                    with col_rej:
                        if st.button("Reject", key=f"rej_{u}"):
                            set_user_status(u, "rejected")
                            st.error(f"Rejected {u}")
                            st.rerun()
            st.divider()
            
            # Approved users
            st.markdown("### 🟢 Approved Access")
            if not approved_users:
                st.caption("No approved users yet.")
            else:
                for u in approved_users:
                    col_u, col_act = st.columns([4, 1])
                    with col_u:
                        st.write(u)
                    with col_act:
                        if st.button("Revoke Access", key=f"rev_{u}"):
                            set_user_status(u, "rejected")
                            st.warning(f"Revoked access for {u}")
                            st.rerun()
            st.divider()
            
            # Rejected users
            st.markdown("### 🔴 Rejected / Revoked Access")
            if not rejected_users:
                st.caption("No rejected users.")
            else:
                for u in rejected_users:
                    col_u, col_act = st.columns([4, 1])
                    with col_u:
                        st.write(u)
                    with col_act:
                        if st.button("Re-Approve", key=f"reapp_{u}"):
                            set_user_status(u, "approved")
                            st.success(f"Approved {u}")
                            st.rerun()
            
            # Centralized Customer Feedback list for the admin
            st.divider()
            st.markdown("### 💬 Received User Feedbacks")
            st.caption("Review feedback and ratings submitted by users on their generated blog posts.")
            
            admin_safe = re.sub(r'[^a-zA-Z0-9_.-]', '_', "ayazbnk0107@gmail.com")
            admin_feedbacks_path = Path("users") / admin_safe / "received_feedbacks.json"
            
            feedbacks_list = []
            if admin_feedbacks_path.exists():
                try:
                    feedbacks_list = json.loads(admin_feedbacks_path.read_text(encoding="utf-8"))
                except Exception:
                    feedbacks_list = []
                    
            if not feedbacks_list:
                st.caption("No user feedback submitted yet.")
            else:
                for idx, f in enumerate(reversed(feedbacks_list)):
                    rating_emoji = "👍" if f.get("rating") == "positive" else "👎"
                    bg_color = "#d4edda" if f.get("rating") == "positive" else "#f8d7da"
                    text_color = "#155724" if f.get("rating") == "positive" else "#721c24"
                    
                    st.markdown(
                        f"""
                        <div style="background-color: {bg_color}; padding: 12px; border-radius: 8px; margin-bottom: 10px; border: 1px solid {text_color}33;">
                            <span style="color: {text_color}; font-weight: bold; font-size: 1.1em;">{rating_emoji} {f.get('rating').title()}</span><br>
                            <span style="color: #333; font-weight: bold;">User:</span> <span style="color: #555;">{f.get('user')}</span><br>
                            <span style="color: #333; font-weight: bold;">Blog:</span> <span style="color: #555;">{f.get('blog')}</span><br>
                            <span style="color: #333; font-weight: bold;">Comment:</span> <span style="color: #222; font-style: italic;">"{f.get('comment', 'No comments provided')}"</span><br>
                            <span style="color: #888; font-size: 0.85em;">Submitted on: {f.get('timestamp')}</span>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
else:
    tabs = st.tabs(
        ["🧩 Plan", "🔎 Evidence", "📝 Markdown Preview", "🖼️ Images", "🧾 Logs", "📊 Quality Metrics"]
    )
    tab_plan, tab_evidence, tab_preview, tab_images, tab_logs, tab_metrics = tabs

logs: List[str] = []

# Execution Logic
if run_btn:
    if not topic.strip():
        st.error("Please enter a topic before generating.")
        st.stop()
        
    inputs = {
        "topic": topic.strip(),
        "as_of": as_of.isoformat(),
        "user_email": user_email,
        "evidence": []
    }
    
    with st.status("🚀 Running Blog Writer...", expanded=True) as status_box:
        st.session_state["last_out"] = {}
        for kind, payload in try_stream(app, inputs):
            if kind == "updates":
                st.session_state["last_out"] = extract_latest_state(st.session_state["last_out"], payload)
                step_name = list(payload.keys())[0] if isinstance(payload, dict) else str(payload)
                msg = f"Update from step: {step_name}"
                logs.append(msg)
                status_box.update(label=f"Running step: {step_name}")
                st.write(msg)
            elif kind == "values":
                st.session_state["last_out"] = extract_latest_state(st.session_state["last_out"], payload)
            elif kind == "final":
                if payload is not None:
                    st.session_state["last_out"] = payload
                else:
                    payload = st.session_state["last_out"]
                status_box.update(label="Finished!", state="complete")
                st.write("Done!")
                
                # Calculate quality metrics
                with st.spinner("Calculating blog quality metrics..."):
                    final_md = payload.get("final") or payload.get("merged_md") or ""
                    coh = evaluate_blog_coherence(final_md)
                    rel = evaluate_blog_relevance(topic.strip(), final_md)
                    faith = evaluate_blog_faithfulness(payload.get("evidence", []), final_md)
                    seo = evaluate_blog_seo(topic.strip(), final_md)
                    payload["metrics"] = {
                        "coherence": coh,
                        "relevance": rel,
                        "faithfulness": faith,
                        "seo": seo
                    }
                    st.session_state["last_out"]["metrics"] = payload["metrics"]
                
                # Append execution logs to payload
                payload["logs"] = logs
                st.session_state["last_out"]["logs"] = logs
                
                # Save full state to JSON companion file in user folder
                try:
                    plan_obj = payload.get("plan")
                    title = plan_obj.blog_title if hasattr(plan_obj, "blog_title") else (plan_obj.get("blog_title", "blog") if isinstance(plan_obj, dict) else "blog")
                    
                    json_path = user_blogs_dir / f"{_safe_filename(title).replace('.md', '.json')}"
                    
                    import copy
                    save_payload = copy.deepcopy(payload)
                    if hasattr(save_payload.get("plan"), "model_dump"):
                        save_payload["plan"] = save_payload["plan"].model_dump()
                    if save_payload.get("evidence") is not None:
                        save_payload["evidence"] = [e.model_dump() if hasattr(e, "model_dump") else e for e in save_payload["evidence"]]
                        
                    json_path.parent.mkdir(parents=True, exist_ok=True)
                    json_path.write_text(json.dumps(save_payload, indent=2), encoding="utf-8")
                    
                    # Save user session state (short-term memory database)
                    save_user_state(user_email, save_payload, topic)
                except Exception as e:
                    st.toast(f"Note: Could not save JSON state companion file: {e}")

# Rendering Logic
last_out = st.session_state.get("last_out")
if last_out:
    # 1. Preview
    with tab_preview:
        md = last_out.get("final") or last_out.get("merged_md")
        if md:
            render_markdown_with_local_images(md)
            
            st.divider()
            # ZIP Download
            plan_obj = last_out.get("plan")
            title = "blog"
            if plan_obj and hasattr(plan_obj, "blog_title"):
                title = plan_obj.blog_title
            elif isinstance(plan_obj, dict) and "blog_title" in plan_obj:
                title = plan_obj["blog_title"]
                
            md_filename = f"{_safe_filename(title)}"
            zip_bytes = bundle_zip(md, md_filename, user_images_dir)
            st.download_button(
                label="📦 Download ZIP (Markdown + Images)",
                data=zip_bytes,
                file_name=f"{_safe_filename(title).replace('.md', '.zip')}",
                mime="application/zip",
                use_container_width=True
            )
        else:
            st.info("No markdown content generated yet.")

    # 2. Plan
    with tab_plan:
        plan = last_out.get("plan")
        if plan:
            if hasattr(plan, "model_dump"):
                st.json(plan.model_dump())
            else:
                st.json(plan)
        else:
            st.info("No plan generated.")

    # 3. Evidence
    with tab_evidence:
        evidence = last_out.get("evidence") or []
        md = last_out.get("final") or last_out.get("merged_md") or ""
        
        # Extract inline links from the markdown content
        inline_links = []
        if md:
            # Match [Anchor Text](URL) but exclude images (which start with !)
            found_links = re.findall(r'(?<!\!)\[([^\]]+)\]\((https?://[^\)]+)\)', md)
            for title, url in found_links:
                # Avoid duplicate URLs
                if not any(e.get("url") == url if isinstance(e, dict) else getattr(e, "url", "") == url for e in evidence) and not any(l["url"] == url for l in inline_links):
                    inline_links.append({
                        "title": title.strip(),
                        "url": url.strip(),
                        "snippet": "Reference link found in the blog content."
                    })
                    
        all_sources = list(evidence) + inline_links
        
        if all_sources:
            st.subheader("🔎 Research Evidence & Reference Links")
            st.markdown("Here are the sources and reference links gathered during the research phase or referenced in the blog content:")
            
            for idx, e in enumerate(all_sources):
                title = ""
                url = ""
                snippet = ""
                if isinstance(e, dict):
                    title = e.get("title") or e.get("url") or f"Source #{idx+1}"
                    url = e.get("url") or ""
                    snippet = e.get("snippet") or ""
                else:
                    title = getattr(e, "title", None) or getattr(e, "url", None) or f"Source #{idx+1}"
                    url = getattr(e, "url", "")
                    snippet = getattr(e, "snippet", "")
                
                if url:
                    st.markdown(f"🔗 **[{title}]({url})**")
                else:
                    st.markdown(f"📄 **{title}**")
                if snippet:
                    st.caption(snippet)
                st.write("")
                
            with st.expander("🛠️ Raw Evidence JSON"):
                st.json([e.model_dump() if hasattr(e, "model_dump") else e for e in all_sources])
        else:
            st.info("No research evidence or reference links found.")

    # 4. Images
    with tab_images:
        image_specs = last_out.get("image_specs")
        if image_specs:
            st.subheader("🖼️ Planned Image Specifications")
            st.markdown("Here is the metadata and generation prompt for the blog image(s):")
            for idx, spec in enumerate(image_specs):
                st.markdown(f"**Image #{idx+1} Specs:**")
                st.json(spec.model_dump() if hasattr(spec, "model_dump") else spec)
        else:
            st.info("No images planned/generated.")

    # 5. Logs
    with tab_logs:
        all_logs = last_out.get("logs") or logs
        if all_logs:
            for l in all_logs:
                st.text(l)
        else:
            st.info("No logs captured.")

    # 6. Quality Metrics
    with tab_metrics:
        st.subheader("📊 Blog Quality Evaluation Metrics")
        st.caption("Automated evaluation metrics checking structure, alignment, factual accuracy, and SEO readiness.")
        
        metrics = last_out.get("metrics")
        if metrics:
            # Row 1: Coherence & Relevance
            col_coh, col_rel = st.columns(2)
            with col_coh:
                coh = metrics.get("coherence", {})
                score = coh.get("score", 0)
                if score == 1:
                    st.success("🟢 **Coherence: Pass**")
                else:
                    st.error("🔴 **Coherence: Fail**")
                st.info(coh.get("comment", ""))
            with col_rel:
                rel = metrics.get("relevance", {})
                score = rel.get("score", 0)
                if score == 1:
                    st.success("🟢 **Relevance: Pass**")
                else:
                    st.error("🔴 **Relevance: Fail**")
                st.info(rel.get("comment", ""))
                
            st.divider()
            
            # Row 2: Faithfulness & SEO Score
            col_faith, col_seo = st.columns(2)
            with col_faith:
                faith = metrics.get("faithfulness", {})
                score = faith.get("score", 0)
                if score == 1:
                    st.success("🟢 **Faithfulness: Pass**")
                else:
                    st.error("🔴 **Faithfulness: Fail**")
                st.info(faith.get("comment", ""))
            with col_seo:
                seo = metrics.get("seo", {})
                score = seo.get("score", 0)
                st.metric(label="📈 SEO Score", value=f"{score}/100")
                # Split checks for display
                comments = seo.get("comment", "").split(" | ")
                for c in comments:
                    st.caption(c)
        else:
            st.info("No metrics calculated for this blog yet.")

        # Human Feedback Loop
        st.divider()
        st.subheader("💬 Human Feedback Loop")
        st.caption("Help us improve the AI blog writer by providing your rating and review on this post.")
        
        feedback = last_out.get("user_feedback", {})
        existing_rating = feedback.get("rating", None)
        existing_comment = feedback.get("comment", "")
        
        if existing_rating:
            rating_emoji = "👍" if existing_rating == "positive" else "👎"
            st.success(f"**Saved Feedback:** {rating_emoji} {existing_rating.title()}" + (f" — *\"{existing_comment}\"*" if existing_comment else ""))
            
        with st.form(key="feedback_form", clear_on_submit=False):
            # Select rating
            rating_options = ["👍 Positive", "👎 Negative"]
            default_index = 0 if existing_rating == "positive" else (1 if existing_rating == "negative" else 0)
            
            feedback_rating = st.radio(
                "Overall Quality Rating:",
                options=rating_options,
                index=default_index,
                horizontal=True
            )
            
            feedback_comment = st.text_area(
                "Feedback / Suggestions for improvement:",
                value=existing_comment,
                placeholder="What could be improved? (e.g., tone, code samples, image prompt)..."
            )
            
            submit_feedback = st.form_submit_button("Submit Feedback", use_container_width=True)
            
            if submit_feedback:
                rating_str = "positive" if "Positive" in feedback_rating else "negative"
                
                # Save to st.session_state["last_out"]
                st.session_state["last_out"]["user_feedback"] = {
                    "rating": rating_str,
                    "comment": feedback_comment.strip(),
                    "timestamp": date.today().isoformat()
                }
                
                # Save to JSON companion file
                plan_obj = last_out.get("plan")
                title = "blog"
                if plan_obj and hasattr(plan_obj, "blog_title"):
                    title = plan_obj.blog_title
                elif isinstance(plan_obj, dict) and "blog_title" in plan_obj:
                    title = plan_obj["blog_title"]
                
                # Save using same logic as in runner
                import copy
                save_payload = copy.deepcopy(st.session_state["last_out"])
                if hasattr(save_payload.get("plan"), "model_dump"):
                    save_payload["plan"] = save_payload["plan"].model_dump()
                if save_payload.get("evidence") is not None:
                    save_payload["evidence"] = [e.model_dump() if hasattr(e, "model_dump") else e for e in save_payload["evidence"]]
                
                from pathlib import Path
                safe_email = re.sub(r'[^a-zA-Z0-9_.-]', '_', user_email)
                blogs_dir = Path("users") / safe_email / "blogs"
                json_path = blogs_dir / f"{_safe_filename(title).replace('.md', '.json')}"
                
                try:
                    json_path.parent.mkdir(parents=True, exist_ok=True)
                    json_path.write_text(json.dumps(save_payload, indent=2), encoding="utf-8")
                    
                    # Also save a copy to the admin's central database
                    try:
                        admin_safe = re.sub(r'[^a-zA-Z0-9_.-]', '_', "ayazbnk0107@gmail.com")
                        admin_feedbacks_path = Path("users") / admin_safe / "received_feedbacks.json"
                        admin_feedbacks_path.parent.mkdir(parents=True, exist_ok=True)
                        
                        admin_feedbacks = []
                        if admin_feedbacks_path.exists():
                            try:
                                admin_feedbacks = json.loads(admin_feedbacks_path.read_text(encoding="utf-8"))
                            except Exception:
                                admin_feedbacks = []
                                
                        # Remove existing feedback for this blog by this user to avoid duplicates
                        admin_feedbacks = [f for f in admin_feedbacks if not (f.get("user") == user_email and f.get("blog") == title)]
                        
                        # Append new feedback
                        admin_feedbacks.append({
                            "user": user_email,
                            "blog": title,
                            "rating": rating_str,
                            "comment": feedback_comment.strip(),
                            "timestamp": date.today().isoformat()
                        })
                        
                        admin_feedbacks_path.write_text(json.dumps(admin_feedbacks, indent=2), encoding="utf-8")
                    except Exception as ae:
                        print(f"[Error] Failed to save copy of feedback to admin: {ae}")
                    
                    # Also save short-term session state to remember feedback on reload
                    topic_prefill = st.session_state.get("topic_prefill", "")
                    save_user_state(user_email, st.session_state["last_out"], topic_prefill)
                    
                    st.success("Feedback submitted successfully!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error saving feedback: {e}")
else:
    # Default Dashboard when nothing is loaded
    with tab_preview:
        name = extract_name_from_email(user_email)
        st.subheader(f"👋 Welcome {name}!")
        st.markdown("Use the sidebar on the left to **Generate a New Blog**, or select one of your **Saved Blogs** below to view and continue working on it.")
        
        st.divider()
        st.subheader("📚 Your Past Blogs")
        
        past_files = list_past_blogs(user_email)
        if not past_files:
            st.info("You haven't generated any blogs yet. Type a topic in the sidebar to get started!")
        else:
            import time
            for idx, p in enumerate(past_files):
                try:
                    md_text = read_md_file(p)
                    title = extract_title_from_md(md_text, p.stem)
                except Exception:
                    title = p.stem
                
                col_title, col_action = st.columns([4, 1])
                with col_title:
                    st.write(f"📄 **{title}**")
                    st.caption(f"File: `{p.name}`  ·  Last modified: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(p.stat().st_mtime))}")
                with col_action:
                    if st.button("📂 Load Blog", key=f"load_dashboard_{idx}_{p.stem}"):
                        md_text = read_md_file(p)
                        json_file = p.with_suffix('.json')
                        
                        state_loaded = False
                        if json_file.exists():
                            try:
                                loaded_state = json.loads(json_file.read_text(encoding="utf-8"))
                                loaded_state["final"] = md_text
                                state_loaded = True
                                
                                # Evaluate on the fly if metrics are missing or incomplete
                                metrics = loaded_state.get("metrics", {})
                                if not metrics or "faithfulness" not in metrics or "seo" not in metrics:
                                    topic_name = extract_title_from_md(md_text, p.stem)
                                    coh = evaluate_blog_coherence(md_text)
                                    rel = evaluate_blog_relevance(topic_name, md_text)
                                    faith = evaluate_blog_faithfulness(loaded_state.get("evidence", []), md_text)
                                    seo = evaluate_blog_seo(topic_name, md_text)
                                    loaded_state["metrics"] = {
                                        "coherence": coh,
                                        "relevance": rel,
                                        "faithfulness": faith,
                                        "seo": seo
                                    }
                                    try:
                                        json_file.write_text(json.dumps(loaded_state, indent=2), encoding="utf-8")
                                    except Exception:
                                        pass
                                st.session_state["last_out"] = loaded_state
                            except Exception:
                                pass
                                
                        if not state_loaded:
                            topic_name = extract_title_from_md(md_text, p.stem)
                            coh = evaluate_blog_coherence(md_text)
                            rel = evaluate_blog_relevance(topic_name, md_text)
                            faith = evaluate_blog_faithfulness([], md_text)
                            seo = evaluate_blog_seo(topic_name, md_text)
                            st.session_state["last_out"] = {
                                "plan": None,
                                "evidence": [],
                                "image_specs": [],
                                "final": md_text,
                                "metrics": {
                                    "coherence": coh,
                                    "relevance": rel,
                                    "faithfulness": faith,
                                    "seo": seo
                                }
                            }
                        st.session_state["topic_prefill"] = extract_title_from_md(md_text, p.stem)
                        save_user_state(user_email, st.session_state["last_out"], st.session_state["topic_prefill"])
                        st.rerun()
                st.write("")

    with tab_plan:
        st.info("Select or generate a blog to view its planning details.")
    with tab_evidence:
        st.info("Select or generate a blog to view research evidence.")
    with tab_images:
        st.info("Select or generate a blog to view image placements.")
    with tab_logs:
        st.info("Select or generate a blog to view run logs.")
    with tab_metrics:
        st.info("Select or generate a blog to view quality metrics.")
