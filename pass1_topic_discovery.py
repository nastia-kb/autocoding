"""
Pass 1 — Topic Discovery Agent
Reads a sample of survey responses and returns a canonical list of topics.
"""

import json
import anthropic
from dotenv import load_dotenv
import os

load_dotenv()

cl_api_key = os.getenv("ANTHROPIC_API_KEY")

client = anthropic.Anthropic(api_key=cl_api_key)  # reads ANTHROPIC_API_KEY from environment


def discover_topics(
    responses: list[str],
    sample_size: int = 50,
    min_topics: int = 5,
    max_topics: int = 15,
    question_context: str = "",
) -> dict:
    """
    Analyse a sample of open-ended survey responses and return a
    canonical topic list with a short description for each topic.

    Args:
        responses:        Full list of raw response strings.
        sample_size:      How many responses to sample for discovery (default 50).
        min_topics:       Minimum number of topics to generate.
        max_topics:       Maximum number of topics to generate.
        question_context: Optional — the original survey question, for context.

    Returns:
        {
            "topics": [
                {"id": "pricing", "label": "Pricing", "description": "..."},
                ...
            ],
            "model_notes": "..."   # any caveats the model flagged
        }
    """
    # Sample responses to keep token usage low
    import random
    sample = responses[:sample_size] if len(responses) <= sample_size else random.sample(responses, sample_size)

    numbered = "\n".join(f"{i+1}. {r}" for i, r in enumerate(sample))

    system_prompt = """You are an expert qualitative researcher specialising in
thematic analysis of open-ended survey data. Your job is to identify the main
topics present across a set of responses. Return ONLY valid JSON — no preamble,
no markdown fences, no explanation outside the JSON object."""

    user_prompt = f"""Analyse the survey responses below and identify the main topics.

{f'Survey question: "{question_context}"' if question_context else ""}

Guidelines:
- Aim for {min_topics}–{max_topics} distinct topics.
- Topics should be specific enough to be meaningful, broad enough to cover multiple responses.
- Each topic needs a short snake_case id (e.g. "pricing_concerns"), a human-readable label,
  and a one-sentence description of what responses in this topic typically say.
- Human-readable lable and desription should be in russian.
- Topics must be mutually exclusive where possible; a response may still match several.
- Don't analyze empty, meaningless or off-topic responses

Respond with this exact JSON structure:
{{
  "topics": [
    {{
      "id": "snake_case_id",
      "label": "Название темы человеческим языком",
      "description": "Одно предложение, описывающее содержание темы"
    }}
  ],
  "model_notes": "Any caveats, ambiguities, or suggestions for the analyst."
}}

Survey responses:
{numbered}"""

    response = client.messages.create(
        model="claude-sonnet-4-20250514",   # Use Sonnet for discovery — nuance matters
        max_tokens=1500,
        system=system_prompt,
        messages=[{"role": "user", "content": user_prompt}],
    )

    raw = response.content[0].text.strip()

    # Strip accidental markdown fences if present
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    raw = raw.strip()

    result = json.loads(raw)
    return result


# ── Example usage ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    sample_responses = [
        "The price is too high compared to competitors.",
        "Customer support took three days to reply — very frustrating.",
        "Love the product but the mobile app crashes constantly.",
        "Shipping was fast and packaging was excellent.",
        "I couldn't find how to cancel my subscription anywhere.",
        "The quality has dropped since the last update.",
        "Great value for money, I'd recommend it to friends.",
        "The website is confusing and hard to navigate.",
        "Support team was helpful once I got through to them.",
        "Too expensive for what you get.",
        "App needs dark mode badly.",
        "Delivery arrived two weeks late with no update.",
        "The new features are great but there are too many bugs.",
        "Cancellation process should be simpler.",
        "Best product I've bought this year — amazing quality.",
    ]

    print("Running Pass 1 — Topic Discovery...\n")
    result = discover_topics(
        responses=sample_responses,
        question_context="What could we improve about your experience with our product?",
    )

    print("Discovered topics:")
    for topic in result["topics"]:
        print(f"  [{topic['id']}] {topic['label']}: {topic['description']}")

    if result.get("model_notes"):
        print(f"\nModel notes: {result['model_notes']}")

    # Save topics to a file so Pass 2 can load them
    with open("topics.json", "w") as f:
        json.dump(result, f, indent=2)
    print("\nTopics saved to topics.json")
