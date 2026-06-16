from __future__ import annotations

import operator
import os
import re
import base64
import time
import traceback
from datetime import date, timedelta
from pathlib import Path
from typing import TypedDict, List, Optional, Literal, Annotated

from dotenv import load_dotenv
load_dotenv()

from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_community.tools.tavily_search import TavilySearchResults
from openai import OpenAI

# =====================================================================
# 1) Schemas
# =====================================================================
class Task(BaseModel):
    id: int
    title: str
    goal: str = Field(..., description="One sentence describing what the reader should be able to do/understand after this section.")
    bullets: List[str] = Field(..., min_length=2, max_length=4, description="2-4 concrete, non-overlapping subpoints.")
    target_words: int = Field(..., description="Target word count for this section (80-200).")
    tags: List[str] = Field(default_factory=list)
    requires_research: bool = False
    requires_citations: bool = False
    requires_code: bool = False


class Plan(BaseModel):
    blog_title: str
    audience: str
    tone: str
    blog_kind: Literal["explainer", "tutorial", "news_roundup", "comparison", "system_design"] = "explainer"
    constraints: List[str] = Field(default_factory=list)
    tasks: List[Task]


class EvidenceItem(BaseModel):
    title: str
    url: str
    published_at: Optional[str] = None
    snippet: Optional[str] = None
    source: Optional[str] = None


class RouterDecision(BaseModel):
    needs_research: bool
    mode: Literal["closed_book", "hybrid", "open_book"]
    queries: List[str] = Field(default_factory=list)


class EvidencePack(BaseModel):
    evidence: List[EvidenceItem] = Field(default_factory=list)


class ImageSpec(BaseModel):
    placeholder: str = Field(..., description="e.g. [[IMAGE_1]]")
    filename: str = Field(..., description="Save under images/, e.g. qkv_flow.png")
    alt: str
    caption: str
    prompt: str = Field(..., description="Prompt to send to the image model.")
    size: Literal["1024x1024", "1024x1536", "1536x1024"] = "1024x1024"
    quality: Literal["low", "medium", "high"] = "medium"


class GlobalImagePlan(BaseModel):
    md_with_placeholders: str
    images: List[ImageSpec] = Field(default_factory=list)


# =====================================================================
# State
# =====================================================================
class State(TypedDict):
    topic: str
    mode: str
    needs_research: bool
    queries: List[str]
    evidence: List[EvidenceItem]
    plan: Optional[Plan]
    as_of: str
    recency_days: int
    sections: Annotated[list, operator.add]
    merged_md: str
    md_with_placeholders: str
    image_specs: list
    final: str
    user_email: Optional[str]



# =====================================================================
# 2) LLM
# =====================================================================
llm_smart = ChatOpenAI(model="gpt-4o")
llm_fast = ChatOpenAI(model="gpt-4o-mini")
llm = llm_smart       # writing (workers)
llm_aux = llm_fast    # auxiliary: router, research, orchestrator, image planning


# =====================================================================
# 3) Router
# =====================================================================
ROUTER_SYSTEM = """Routing module for a blog planner.
Decide if web research is needed.

Modes:
- closed_book: evergreen topics, no research needed.
- hybrid: mostly evergreen but needs fresh examples/tools.
- open_book: volatile/news topics.

If needs_research=true, output 3-5 specific search queries.
"""

def router_node(state: State) -> dict:
    decider = llm_aux.with_structured_output(RouterDecision)
    decision = decider.invoke([
        SystemMessage(content=ROUTER_SYSTEM),
        HumanMessage(content=f"Topic: {state['topic']}"),
    ])
    
    # Always force research to gather reference links and enable faithfulness evaluation
    needs_research = True
    mode = "hybrid" if decision.mode == "closed_book" else decision.mode
    queries = decision.queries[:5]
    if not queries:
        queries = [state["topic"], f"{state['topic']} features", f"latest {state['topic']} examples"]
        
    return {
        "needs_research": needs_research,
        "mode": mode,
        "queries": queries,
    }

def route_next(state: State) -> str:
    return "research" if state["needs_research"] else "orchestrator"


# =====================================================================
# 4) Research (Tavily)
# =====================================================================
def _tavily_search(query: str, max_results: int = 3) -> List[dict]:
    tool = TavilySearchResults(max_results=max_results)
    results = tool.invoke({"query": query})
    normalized: List[dict] = []
    for r in results or []:
        normalized.append({
            "title": r.get("title") or "",
            "url": r.get("url") or "",
            "snippet": (r.get("content") or r.get("snippet") or "")[:200],
            "published_at": r.get("published_date") or r.get("published_at"),
            "source": r.get("source"),
        })
    return normalized


RESEARCH_SYSTEM = """Synthesize raw search results into a deduplicated EvidenceItem list.
Rules: only include items with a url. Prefer authoritative sources. Keep snippets under 100 chars. Deduplicate by URL.
"""

def research_node(state: State) -> dict:
    queries = (state.get("queries", []) or [])[:5]
    max_results = 3
    raw_results: List[dict] = []
    for q in queries:
        raw_results.extend(_tavily_search(q, max_results=max_results))
    if not raw_results:
        return {"evidence": []}
    extractor = llm_aux.with_structured_output(EvidencePack)
    pack = extractor.invoke([
        SystemMessage(content=RESEARCH_SYSTEM),
        HumanMessage(content=f"Raw results:\n{raw_results}"),
    ])
    dedup = {}
    for e in pack.evidence:
        if e.url:
            dedup[e.url] = e
    return {"evidence": list(dedup.values())}


# =====================================================================
# 5) Orchestrator (Plan)
# =====================================================================
ORCH_SYSTEM = """You are a technical writer. Produce a concise outline for a blog post.

Requirements:
- Create 3-4 sections (tasks).
- Each task: goal (1 sentence), 2-3 bullets (concrete, non-overlapping), target_words (80-200).
- Include at least one section with requires_code=True.
- closed_book: evergreen only. hybrid: use evidence for fresh claims, mark requires_research/citations=True. open_book: set blog_kind="news_roundup", summarize events only.
- Output must match Plan schema.
"""

def orchestrator_node(state: State) -> dict:
    planner = llm_aux.with_structured_output(Plan)
    evidence = state.get("evidence", [])
    mode = state.get("mode", "closed_book")
    plan = planner.invoke([
        SystemMessage(content=ORCH_SYSTEM),
        HumanMessage(content=(
            f"Topic: {state['topic']}\nMode: {mode}\n"
            f"Evidence:\n{[e.model_dump() for e in evidence][:10]}"
        )),
    ])
    return {"plan": plan}


# =====================================================================
# 6) Fanout
# =====================================================================
def fanout(state: State):
    return [
        Send("worker", {
            "task": task.model_dump(),
            "topic": state["topic"],
            "mode": state["mode"],
            "plan": state["plan"].model_dump(),
            "evidence": [e.model_dump() for e in state.get("evidence", [])][:12],
        })
        for task in state["plan"].tasks
    ]


# =====================================================================
# 7) Worker (write one section)
# =====================================================================
WORKER_SYSTEM = """Write ONE section of a technical blog in Markdown. Be concise.

Rules:
- Cover ALL bullets in order. Stay within target_words (+/-15%).
- Output ONLY section content. Start with '## <Title>'.
- If blog_kind=="news_roundup": summarize events, do NOT write tutorials.
- If mode==open_book or requires_citations: cite evidence URLs as Markdown links.
- If requires_code: include one minimal code snippet.
- Short paragraphs, no fluff.
"""

def worker_node(payload: dict) -> dict:
    task = Task(**payload["task"])
    plan = Plan(**payload["plan"])
    evidence = [EvidenceItem(**e) for e in payload.get("evidence", [])]
    topic = payload["topic"]
    mode = payload.get("mode", "closed_book")

    bullets_text = "\n- " + "\n- ".join(task.bullets)
    evidence_text = ""
    if evidence:
        evidence_text = "\n".join(
            f"- {e.title} | {e.url}"
            for e in evidence[:12]
        )

    section_md = llm.invoke([
        SystemMessage(content=WORKER_SYSTEM),
        HumanMessage(content=(
            f"Blog: {plan.blog_title} | Audience: {plan.audience} | "
            f"Tone: {plan.tone} | Kind: {plan.blog_kind}\n"
            f"Topic: {topic} | Mode: {mode}\n"
            f"Section: {task.title} | Goal: {task.goal}\n"
            f"Target words: {task.target_words} | "
            f"citations: {task.requires_citations} | code: {task.requires_code}\n"
            f"Bullets:{bullets_text}\n"
            f"Evidence:\n{evidence_text}"
        )),
    ]).content.strip()

    return {"sections": [(task.id, section_md)]}


# =====================================================================
# 8) Reducer with Images
# =====================================================================
def merge_content(state: State) -> dict:
    plan = state["plan"]
    ordered_sections = [md for _, md in sorted(state["sections"], key=lambda x: x[0])]
    body = "\n\n".join(ordered_sections).strip()
    merged_md = f"# {plan.blog_title}\n\n{body}\n"
    return {"merged_md": merged_md}


DECIDE_IMAGES_SYSTEM = """You MUST plan exactly 1 high-quality technical diagram, infographic, or conceptual illustration for this blog to visually explain the topic.
Insert the placeholder [[IMAGE_1]] at the most appropriate place in the text (e.g., after the introduction or in a key technical section).
You must return exactly 1 ImageSpec in the 'images' list. Do not return 0 images.
Make sure the prompt is descriptive and detailed for the DALL-E image generation model.
"""

def decide_images(state: State) -> dict:
    planner = llm_aux.with_structured_output(GlobalImagePlan)
    merged_md = state["merged_md"]
    plan = state["plan"]
    assert plan is not None

    image_plan = planner.invoke([
        SystemMessage(content=DECIDE_IMAGES_SYSTEM),
        HumanMessage(content=(
            f"Blog kind: {plan.blog_kind}\nTopic: {state['topic']}\n\n{merged_md}"
        )),
    ])

    # Enforce exactly 1 image spec
    if not image_plan.images:
        fallback_spec = ImageSpec(
            placeholder="[[IMAGE_1]]",
            filename=f"{_safe_filename(plan.blog_title).replace('.md', '')}_diagram.png",
            alt=f"Illustration for {plan.blog_title}",
            caption=f"Visual guide to {plan.blog_title}",
            prompt=f"A professional, clean modern infographic/diagram depicting: {state['topic']}. Sleek technical vector graphic style, high resolution, dark gradient background.",
            size="1024x1024",
            quality="medium"
        )
        image_plan.images = [fallback_spec]
        
        # Insert placeholder [[IMAGE_1]] after the intro (find the second heading or after 2 paragraphs)
        lines = merged_md.splitlines()
        inserted = False
        for i, line in enumerate(lines):
            if i > 2 and line.startswith("## "):
                lines.insert(i, "\n[[IMAGE_1]]\n")
                inserted = True
                break
        if not inserted:
            lines.append("\n[[IMAGE_1]]\n")
        image_plan.md_with_placeholders = "\n".join(lines)
    else:
        # Keep only the first image spec
        image_plan.images = [image_plan.images[0]]
        # Ensure the placeholder [[IMAGE_1]] is actually in the markdown, otherwise append it
        if "[[IMAGE_1]]" not in image_plan.md_with_placeholders:
            lines = image_plan.md_with_placeholders.splitlines()
            inserted = False
            for i, line in enumerate(lines):
                if i > 2 and line.startswith("## "):
                    lines.insert(i, "\n[[IMAGE_1]]\n")
                    inserted = True
                    break
            if not inserted:
                lines.append("\n[[IMAGE_1]]\n")
            image_plan.md_with_placeholders = "\n".join(lines)

    return {
        "md_with_placeholders": image_plan.md_with_placeholders,
        "image_specs": [img.model_dump() for img in image_plan.images],
    }


def _openai_generate_image_bytes(prompt: str) -> bytes:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set.")
    client = OpenAI(api_key=api_key)
    response = client.images.generate(
        model="gpt-image-1",
        prompt=prompt,
        size="1024x1024"
    )
    image_b64 = response.data[0].b64_json
    if not image_b64:
        raise RuntimeError("No image returned from OpenAI.")
    return base64.b64decode(image_b64)


def _safe_filename(title: str) -> str:
    """Turn a blog title into a safe .md filename preserving casing and spaces."""
    safe = re.sub(r'[\\/*?:"<>|]', '', title).strip()
    return f"{safe}.md"


def generate_and_place_images(state: State) -> dict:
    plan = state["plan"]
    assert plan is not None
    md = state.get("md_with_placeholders") or state["merged_md"]
    image_specs = state.get("image_specs", []) or []

    user_email = state.get("user_email")
    if user_email:
        # Sanitize email for folder name
        safe_email = re.sub(r'[^a-zA-Z0-9_.-]', '_', user_email)
        user_dir = Path("users") / safe_email
        blogs_dir = user_dir / "blogs"
        images_dir = user_dir / "images"
    else:
        blogs_dir = Path(".")
        images_dir = Path("images")

    blogs_dir.mkdir(parents=True, exist_ok=True)

    if not image_specs:
        filename = _safe_filename(plan.blog_title)
        out_path = blogs_dir / filename
        out_path.write_text(md, encoding="utf-8")
        return {"final": md}

    images_dir.mkdir(parents=True, exist_ok=True)

    for spec in image_specs:
        placeholder = spec["placeholder"]
        filename = spec["filename"]
        out_path = images_dir / filename

        if not out_path.exists():
            try:
                img_bytes = _openai_generate_image_bytes(spec["prompt"])
                out_path.write_bytes(img_bytes)
            except Exception as e:
                prompt_block = (
                    f"> **[IMAGE GENERATION FAILED]** {spec.get('caption','')}\n>\n"
                    f"> **Alt:** {spec.get('alt','')}\n>\n"
                    f"> **Prompt:** {spec.get('prompt','')}\n>\n"
                    f"> **Error:** {e}\n"
                )
                md = md.replace(placeholder, prompt_block)
                continue

        if user_email:
            img_md = (
                f"![{spec['alt']}](users/{safe_email}/images/{filename})\n"
                f"*{spec['caption']}*"
            )
        else:
            img_md = (
                f"![{spec['alt']}](images/{filename})\n"
                f"*{spec['caption']}*"
            )
        md = md.replace(placeholder, img_md)

    filename = _safe_filename(plan.blog_title)
    out_path = blogs_dir / filename
    out_path.write_text(md, encoding="utf-8")
    return {"final": md}


# =====================================================================
# 9) Build graphs
# =====================================================================
# Reducer subgraph
reducer_graph = StateGraph(State)
reducer_graph.add_node("merge_content", merge_content)
reducer_graph.add_node("decide_images", decide_images)
reducer_graph.add_node("generate_and_place_images", generate_and_place_images)
reducer_graph.add_edge(START, "merge_content")
reducer_graph.add_edge("merge_content", "decide_images")
reducer_graph.add_edge("decide_images", "generate_and_place_images")
reducer_graph.add_edge("generate_and_place_images", END)
reducer_subgraph = reducer_graph.compile()

# Main graph
g = StateGraph(State)
g.add_node("router", router_node)
g.add_node("research", research_node)
g.add_node("orchestrator", orchestrator_node)
g.add_node("worker", worker_node)
g.add_node("reducer", reducer_subgraph)

g.add_edge(START, "router")
g.add_conditional_edges("router", route_next, {"research": "research", "orchestrator": "orchestrator"})
g.add_edge("research", "orchestrator")
g.add_conditional_edges("orchestrator", fanout, ["worker"])
g.add_edge("worker", "reducer")
g.add_edge("reducer", END)

app = g.compile()


# =====================================================================
# 10) Runner
# =====================================================================
def run(topic: str, as_of: Optional[str] = None):
    if as_of is None:
        as_of = date.today().isoformat()

    state = {
        "topic": topic,
        "mode": "",
        "needs_research": False,
        "queries": [],
        "evidence": [],
        "plan": None,
        "as_of": as_of,
        "recency_days": 7,
        "sections": [],
        "merged_md": "",
        "md_with_placeholders": "",
        "image_specs": [],
        "final": "",
    }

    last_exc = None
    try:
        print("[1/1] Running pipeline with gpt-4o ...")
        result = app.invoke(state)
        print("\n[OK] Blog generated successfully!")
        if result.get("final"):
            print("\n--- Preview (first 500 chars) ---")
            print(result["final"][:500])
            print("...")
        return result
    except Exception as e:
        last_exc = e
        print("Primary app.invoke failed:", e)
        traceback.print_exc()

    # Fallback: retry with gpt-4o-mini
    try:
        global llm
        print("\n[Fallback] Retrying with gpt-4o-mini ...")
        llm = llm_fast
        time.sleep(1)
        result = app.invoke(state)
        print("\n[OK] Blog generated successfully (fallback)!")
        if result.get("final"):
            print("\n--- Preview (first 500 chars) ---")
            print(result["final"][:500])
            print("...")
        return result
    except Exception as e2:
        last_exc = e2
        print("Fallback failed:", e2)
        traceback.print_exc()

    raise last_exc


if __name__ == "__main__":
    run("the state of vector databases in 2024")
