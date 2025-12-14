import json
from typing import Any, Dict, List, Optional

from openai import OpenAI


class LMStudioClient:
    """
    Simple wrapper for LM Studio's OpenAI-compatible server.
    Ensure LM Studio server is running and exposing /v1.
    """

    def __init__(self, base_url: str = "http://localhost:1234/v1", api_key: str = "lm-studio"):
        self.client = OpenAI(base_url=base_url, api_key=api_key)

    def list_models(self) -> List[Dict[str, Any]]:
        """
        List available models from the LM Studio server.
        """
        resp = self.client.models.list()
        return [model.model_dump() for model in resp.data]

    def chat_json(
        self,
        model: str,
        system_prompt: str,
        user_payload: Dict[str, Any],
        temperature: float = 0.0,
        max_tokens: int = 1024,
        timeout: Optional[float] = 120.0,
    ) -> List[Dict[str, Any]]:
        """
        Sends a JSON game round to the model and expects a JSON list response.
        Returns parsed Python object (list of dicts).
        Raises ValueError if parsing fails.
        """
        user_text = json.dumps(user_payload, ensure_ascii=False)

        resp = self.client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_text},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
        )

        content = resp.choices[0].message.content or ""

        # Expect the model to return raw JSON (a list). Try strict parse first.
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            # Fallback: attempt to extract JSON from fenced blocks if the model added them.
            content_stripped = content.strip()

            # Common pattern: ```json ... ```
            if "```" in content_stripped:
                parts = content_stripped.split("```")
                # Try parsing any block that looks like JSON
                for p in parts:
                    p = p.strip()
                    if not p:
                        continue
                    # Remove optional "json" prefix line
                    if p.lower().startswith("json"):
                        p = p[4:].strip()
                    try:
                        return json.loads(p)
                    except json.JSONDecodeError:
                        continue

            raise ValueError(f"Model did not return valid JSON. Raw output:\n{content}")
