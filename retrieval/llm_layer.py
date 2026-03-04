"""
LLM Integration Layer
======================
Uses Google Gemini (free tier) for:
  1. Query understanding & normalization
  2. Skill extraction (hard skills, soft skills, job level, domains)
  3. Multi-domain classification → which test types are needed
  4. Re-ranking of top retrieval candidates with explanation

Why Gemini?
- Free tier: 15 RPM, 1M tokens/day (sufficient for this task)
- Strong instruction-following for structured JSON extraction
- API compatible with standard REST

Environment variables required:
  GEMINI_API_KEY=your_key_here
"""

import os
import json
import logging
import re
import time
from typing import Optional
import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_URL = (
    "https://generativelanguage.googleapis.com/v1beta/models/"
    "gemini-1.5-flash:generateContent"
)
MAX_RETRIES = 3
RETRY_DELAY = 2.0


# ─────────────────────────────────────────────
#  Test type reference (from assignment doc)
# ─────────────────────────────────────────────
TEST_TYPE_DESCRIPTIONS = {
    "Ability & Aptitude": "Cognitive ability, numerical, verbal, logical, abstract reasoning",
    "Biodata & Situational Judgement": "Situational judgement tests, biodata, scenario-based",
    "Competencies": "Competency frameworks, behavioral competencies, leadership competencies",
    "Development & 360": "360-degree feedback, development assessments, self-assessment",
    "Assessment Exercises": "Role plays, in-tray exercises, group exercises, presentations",
    "Knowledge & Skills": "Technical knowledge, job-specific skills, coding, programming",
    "Personality & Behavior": "Personality questionnaires, behavioral styles, values, motivation",
    "Simulations": "Work simulations, job simulations, virtual assessment centers",
}


def _call_gemini(prompt: str, temperature: float = 0.1) -> Optional[str]:
    """Call Gemini API and return text response."""
    if not GEMINI_API_KEY:
        logger.warning("GEMINI_API_KEY not set — LLM features disabled")
        return None

    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": temperature,
            "maxOutputTokens": 1024,
        },
    }
    headers = {"Content-Type": "application/json"}
    url = f"{GEMINI_URL}?key={GEMINI_API_KEY}"

    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.post(url, json=payload, headers=headers, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            return data["candidates"][0]["content"]["parts"][0]["text"]
        except (requests.RequestException, KeyError, IndexError) as e:
            logger.warning(f"Gemini attempt {attempt+1}/{MAX_RETRIES} failed: {e}")
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY * (attempt + 1))
    return None


def _parse_json_response(text: str) -> Optional[dict]:
    """Extract JSON from LLM response (handles markdown code fences)."""
    # Strip markdown fences
    text = re.sub(r'```(?:json)?\s*', '', text).strip().rstrip('`').strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Try extracting JSON object
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
    logger.warning(f"Failed to parse JSON from: {text[:200]}")
    return None


# ─────────────────────────────────────────────
#  1. Query Understanding & Normalization
# ─────────────────────────────────────────────

QUERY_UNDERSTANDING_PROMPT = """You are an expert in psychometric assessments and HR hiring.

Analyze the following hiring query or job description and extract structured information.

Query: {query}

Return ONLY a valid JSON object with these exact keys:
{{
  "normalized_query": "cleaned, focused version of the query for retrieval",
  "job_title": "inferred job title or null",
  "job_level": "entry/mid/senior/manager/executive or null",
  "hard_skills": ["list of technical skills, tools, programming languages mentioned"],
  "soft_skills": ["list of behavioral/interpersonal skills mentioned"],
  "domains": ["broad domains: technical, behavioral, cognitive, leadership, etc."],
  "is_multi_domain": true/false,
  "key_requirements": ["top 3-5 most important requirements for this role"]
}}

Be precise. Only include what is actually mentioned or strongly implied."""


def understand_query(query: str) -> dict:
    """
    Use LLM to extract structured understanding from a raw query.

    Falls back to a simple heuristic if LLM is unavailable.
    """
    prompt = QUERY_UNDERSTANDING_PROMPT.format(query=query[:3000])
    response = _call_gemini(prompt, temperature=0.0)

    if response:
        parsed = _parse_json_response(response)
        if parsed:
            logger.info(f"Query understood: domains={parsed.get('domains')}, "
                       f"multi_domain={parsed.get('is_multi_domain')}")
            return parsed

    # Fallback: heuristic extraction
    logger.info("LLM unavailable — using heuristic query understanding")
    return _heuristic_understand(query)


def _heuristic_understand(query: str) -> dict:
    """Simple rule-based fallback when LLM is unavailable."""
    query_lower = query.lower()

    TECH_KEYWORDS = {
        "java", "python", "sql", "javascript", "js", "typescript", "c++", "c#",
        "react", "angular", "node", "aws", "azure", "gcp", "docker", "kubernetes",
        "coding", "programming", "software", "developer", "engineer", "data",
        "machine learning", "ml", "ai", "devops", "cloud", "database", "api",
    }
    BEHAVIORAL_KEYWORDS = {
        "collaborat", "communicat", "leadership", "teamwork", "interpersonal",
        "personality", "behavioral", "stakeholder", "manag", "motivat",
        "emotional intelligence", "conflict", "negotiat", "influenc",
    }
    COGNITIVE_KEYWORDS = {
        "cognitive", "reasoning", "verbal", "numerical", "logical", "aptitude",
        "analytical", "problem solving", "critical thinking", "abstract",
    }

    hard_skills = [kw for kw in TECH_KEYWORDS if kw in query_lower]
    soft_skills = [kw for kw in BEHAVIORAL_KEYWORDS if kw in query_lower]
    domains = []
    if hard_skills:
        domains.append("technical")
    if soft_skills:
        domains.append("behavioral")
    if any(kw in query_lower for kw in COGNITIVE_KEYWORDS):
        domains.append("cognitive")
    if not domains:
        domains.append("general")

    return {
        "normalized_query": query,
        "job_title": None,
        "job_level": None,
        "hard_skills": hard_skills,
        "soft_skills": soft_skills,
        "domains": domains,
        "is_multi_domain": len(domains) > 1,
        "key_requirements": [],
    }


# ─────────────────────────────────────────────
#  2. Test Type Mapping
# ─────────────────────────────────────────────

TYPE_MAPPING_PROMPT = """You are an expert in psychometric assessment design.

Given this job analysis:
{query_understanding}

And these available assessment test types:
{test_types}

Determine which test types are relevant for this query. For multi-domain queries, ensure multiple types are included.

Return ONLY a valid JSON object:
{{
  "required_test_types": ["list of relevant test type names from the provided list"],
  "primary_types": ["most important 1-2 types"],
  "secondary_types": ["supporting types"],
  "reasoning": "brief explanation"
}}

The "required_test_types" should be the union of primary and secondary types."""


def map_to_test_types(query_understanding: dict) -> dict:
    """Map extracted query understanding to relevant SHL test types."""
    type_list = "\n".join(
        f"- {name}: {desc}"
        for name, desc in TEST_TYPE_DESCRIPTIONS.items()
    )
    prompt = TYPE_MAPPING_PROMPT.format(
        query_understanding=json.dumps(query_understanding, indent=2),
        test_types=type_list,
    )
    response = _call_gemini(prompt, temperature=0.0)

    if response:
        parsed = _parse_json_response(response)
        if parsed:
            return parsed

    # Fallback: rule-based mapping
    return _heuristic_type_mapping(query_understanding)


def _heuristic_type_mapping(qu: dict) -> dict:
    """Rule-based test type mapping from query understanding."""
    domains = qu.get("domains", [])
    hard_skills = qu.get("hard_skills", [])
    soft_skills = qu.get("soft_skills", [])

    primary = []
    secondary = []

    if "technical" in domains or hard_skills:
        primary.append("Knowledge & Skills")
        secondary.append("Ability & Aptitude")

    if "behavioral" in domains or soft_skills:
        primary.append("Personality & Behavior")
        secondary.append("Competencies")

    if "cognitive" in domains:
        primary.append("Ability & Aptitude")

    if "leadership" in domains:
        primary.append("Competencies")
        secondary.append("Personality & Behavior")

    if not primary:
        primary = ["Ability & Aptitude", "Personality & Behavior"]
        secondary = ["Knowledge & Skills"]

    all_types = list(dict.fromkeys(primary + secondary))  # dedupe preserving order
    return {
        "required_test_types": all_types,
        "primary_types": primary,
        "secondary_types": secondary,
        "reasoning": "Heuristic mapping based on domain keywords",
    }


# ─────────────────────────────────────────────
#  3. LLM Re-ranker
# ─────────────────────────────────────────────

RERANK_PROMPT = """You are an expert in HR assessment design and talent management.

Job Query: {query}

Query Analysis:
{query_understanding}

Here are candidate assessments retrieved from SHL's catalog (pre-ranked by semantic similarity):
{candidates}

Re-rank these assessments based on:
1. Direct relevance to the specific job requirements
2. Coverage of both technical AND behavioral needs (if multi-domain query)
3. Appropriate test types for the role
4. Practical suitability (duration, remote support)

Return ONLY a valid JSON object:
{{
  "ranked_ids": [list of assessment indices 0-N in re-ranked order, best first],
  "reasoning": "brief explanation of ranking decisions"
}}

Include ALL provided indices. Put most relevant first."""


def llm_rerank(
    query: str,
    query_understanding: dict,
    candidates: list[dict],
    top_k: int = 10,
) -> list[dict]:
    """
    Re-rank retrieval candidates using LLM judgment.

    The LLM sees: query, structured query analysis, and candidate list.
    It returns a re-ordered index list.
    """
    if not candidates:
        return []

    if not GEMINI_API_KEY:
        logger.info("LLM reranking unavailable — returning retrieval order")
        return candidates[:top_k]

    # Build candidate summary for prompt (keep concise to fit context)
    candidate_strs = []
    for i, c in enumerate(candidates[:20]):  # max 20 candidates to LLM
        desc_preview = (c.get("description") or "")[:150]
        types = ", ".join(c.get("test_type", []))
        candidate_strs.append(
            f"[{i}] Name: {c['name']}\n"
            f"    Types: {types}\n"
            f"    Duration: {c.get('duration', 'N/A')} min | "
            f"Remote: {c.get('remote_support', 'N/A')}\n"
            f"    Description: {desc_preview}"
        )

    prompt = RERANK_PROMPT.format(
        query=query[:1000],
        query_understanding=json.dumps(
            {k: v for k, v in query_understanding.items() if k != "normalized_query"},
            indent=2
        ),
        candidates="\n\n".join(candidate_strs),
    )

    response = _call_gemini(prompt, temperature=0.1)

    if response:
        parsed = _parse_json_response(response)
        if parsed and "ranked_ids" in parsed:
            ranked_ids = parsed["ranked_ids"]
            reranked = []
            seen = set()
            for idx in ranked_ids:
                if isinstance(idx, int) and 0 <= idx < len(candidates) and idx not in seen:
                    reranked.append(candidates[idx])
                    seen.add(idx)
            # Append any missing candidates not in ranked_ids
            for i, c in enumerate(candidates):
                if i not in seen:
                    reranked.append(c)
            logger.info(f"LLM reranked {len(reranked)} candidates. "
                       f"Reason: {parsed.get('reasoning', '')[:100]}")
            return reranked[:top_k]

    logger.warning("LLM reranking failed — falling back to retrieval order")
    return candidates[:top_k]


# ─────────────────────────────────────────────
#  4. URL Content Fetcher (for JD URLs)
# ─────────────────────────────────────────────

def fetch_url_content(url: str) -> str:
    """
    Fetch and extract text content from a URL (job description page).
    Used when the input is a URL rather than raw text.
    """
    try:
        import requests
        from bs4 import BeautifulSoup

        headers = {"User-Agent": "Mozilla/5.0"}
        resp = requests.get(url, headers=headers, timeout=15)
        resp.raise_for_status()

        soup = BeautifulSoup(resp.text, "html.parser")
        # Remove script/style elements
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()

        text = soup.get_text(separator=" ", strip=True)
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        return text[:5000]  # cap to avoid token limits

    except Exception as e:
        logger.error(f"Failed to fetch URL {url}: {e}")
        return ""


def is_url(text: str) -> bool:
    """Check if the input looks like a URL."""
    return bool(re.match(r'https?://', text.strip()))
