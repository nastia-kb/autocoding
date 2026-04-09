"""
Pass 2 — Classification Agent
Assigns topics from the canonical list to each survey response.
Supports single-response and batched processing.
"""

import json
import anthropic
from dataclasses import dataclass, field, asdict
import streamlit as st

from dotenv import load_dotenv
import os

load_dotenv()

cl_api_key = st.secrets["API_KEY"]
client = anthropic.Anthropic(api_key=cl_api_key)  # reads ANTHROPIC_API_KEY from environment


@dataclass
class ClassifiedResponse:
    response_id: int | str
    original_text: str
    assigned_topics: list[str]          # list of topic ids
    confidence: str                     # "high" | "medium" | "low"                    
    error: str | None = field(default=None)


def _build_topic_reference(topics: list[dict]) -> str:
    """Format the topic list into a compact reference block for the prompt."""
    lines = []
    for t in topics:
        lines.append(f"- {t['id']}: {t['label']} — {t['description']}")
    return "\n".join(lines)


def classify_batch(
    responses: list[dict],          # [{"id": ..., "text": ...}, ...]
    topics: list[dict],             # from Pass 1 result["topics"]
    batch_size: int = 10,
) -> list[ClassifiedResponse]:
    """
    Classify a batch of responses against the canonical topic list.
    Sends up to `batch_size` responses per API call to reduce cost.

    Args:
        responses:   List of dicts with "id" and "text" keys.
        topics:      Topic list from Pass 1.
        batch_size:  Number of responses per API call (5–10 recommended).

    Returns:
        List of ClassifiedResponse objects, one per input response.
    """
    topic_ids = {t["id"] for t in topics}
    topic_reference = _build_topic_reference(topics)
    results: list[ClassifiedResponse] = []

    system_prompt = """You are a qualitative research analyst classifying
open-ended survey responses against a fixed set of topics. Return ONLY valid
JSON — no preamble, no markdown fences, no explanation outside the JSON."""

    # Process in batches
    for batch_start in range(0, len(responses), batch_size):
        batch = responses[batch_start : batch_start + batch_size]

        numbered = "\n\n".join(
            f"Response #{r['id']}:\n{r['text']}" for r in batch
        )

        user_prompt = f"""Classify each survey response below against the topic list.

TOPIC LIST:
{topic_reference}

INSTRUCTIONS:
- Assign 1–4 topic ids per response (use only ids from the list above).
- Set confidence to "Высокое", "Среднее", or "Низкое".
- If a response is blank, off-topic, or unclassifiable, assign an empty topics
  list and confidence "low".

Return a JSON array — one object per response, in the same order:
[
  {{
    "id": <response id>,
    "assigned_topics": ["topic_id_1", "topic_id_2"],
    "confidence": "high"
  }},
  ...
]

RESPONSES TO CLASSIFY:
{numbered}"""

        try:
            response = client.messages.create(
                model="claude-haiku-4-5-20251001",   # Haiku is fast and cheap for classification
                max_tokens=1000,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}],
            )

            raw = response.content[0].text.strip()

            # Strip accidental markdown fences
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            raw = raw.strip()

            batch_results = json.loads(raw)

            for item, original in zip(batch_results, batch):
                # Sanitise: drop any topic ids not in the canonical list
                valid_topics = [t for t in item.get("assigned_topics", []) if t in topic_ids]
                results.append(
                    ClassifiedResponse(
                        response_id=original["id"],
                        original_text=original["text"],
                        assigned_topics=valid_topics,
                        confidence=item.get("confidence", "low")
                    )
                )

        except Exception as e:
            # On error, mark each response in the batch as failed
            for original in batch:
                results.append(
                    ClassifiedResponse(
                        response_id=original["id"],
                        original_text=original["text"],
                        assigned_topics=[],
                        confidence="low",
                        error=str(e),
                    )
                )

    return results


def classify_all(
    raw_texts: list[str],
    topics: list[dict],
    batch_size: int = 10,
    progress_callback=None,            # optional fn(current, total)
) -> list[ClassifiedResponse]:
    """
    Convenience wrapper: takes a plain list of strings (no id dicts needed).

    Args:
        raw_texts:         Plain list of response strings.
        topics:            Topic list from Pass 1.
        batch_size:        Responses per API call.
        progress_callback: Optional callable(current_index, total).

    Returns:
        List of ClassifiedResponse in the same order as raw_texts.
    """
    responses = [{"id": i, "text": text} for i, text in enumerate(raw_texts)]
    results = []

    for batch_start in range(0, len(responses), batch_size):
        batch = responses[batch_start : batch_start + batch_size]
        batch_results = classify_batch(batch, topics, batch_size=len(batch))
        results.extend(batch_results)
        if progress_callback:
            progress_callback(min(batch_start + batch_size, len(responses)), len(responses))

    return results


# ── Example usage ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import pandas as pd

    # Load topics saved by Pass 1
    with open("topics.json") as f:
        topic_data = json.load(f)
    topics = topic_data["topics"]

    sample_responses = [
        "The price is too high compared to competitors.",
        "Customer support took three days to reply — very frustrating.",
        "Love the product but the mobile app crashes constantly.",
        "Shipping was fast and packaging was excellent.",
        "I couldn't find how to cancel my subscription anywhere.",
        "The quality has dropped since the last update.",
        "Great value for money, I'd recommend it to friends.",
        "The website is confusing and hard to navigate.",
    ]

    print("Running Pass 2 — Classification...\n")

    def on_progress(current, total):
        print(f"  Classified {current}/{total} responses...")

    results = classify_all(
        raw_texts=sample_responses,
        topics=topics,
        batch_size=5,
        progress_callback=on_progress,
    )

    print("\nResults:")
    for r in results:
        topics_str = ", ".join(r.assigned_topics) if r.assigned_topics else "(none)"
        print(f"  [{r.response_id}] [{r.confidence}] Topics: {topics_str}")
        if r.error:
            print(f"         ERROR: {r.error}")

    # Export to CSV via pandas
    rows = [asdict(r) for r in results]
    for row in rows:
        row["assigned_topics"] = ", ".join(row["assigned_topics"])
    df = pd.DataFrame(rows)
    df.to_csv("classified_responses.csv", index=False)
    print("\nExported to classified_responses.csv")
