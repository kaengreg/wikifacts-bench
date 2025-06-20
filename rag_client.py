import time
from typing import Optional, Dict
from openai import OpenAI


class RagClient:
    def __init__(self, model_name: str, api_url: str, api_key: Optional[str] = None, timeout: int = 360, max_attempts: int = 3):
        self.client = OpenAI(api_key=api_key, base_url=api_url, timeout=timeout)
        self.model = model_name
        self.timeout = timeout
        self.max_attempts = max_attempts
        self.prompt_template = 'Is the following statement factually correct: "{fact}"? Answer only with "yes", "no", or "i don\'t know".'

    def call_llm(self, fact: str) -> str:
        for attempt in range(1, self.max_attempts + 1):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": (
                            "You are solving a factual verification task.\n"
                            "You will be given a factual statement.\n"
                            "Your task is to verify whether the statement is true, false, or uncertain based on your knowledge.\n"
                            "Respond strictly with one of the following: \"yes\" (if it is true), \"no\" (if it is false), or \"i don't know\" (if you are not sure)."
                        )},
                        {"role": "user", "content": self.prompt_template.format(fact=fact)}
                    ],
                    temperature=0.0,
                    top_p=0.9,
                    timeout=360
                )
                raw = response.choices[0].message.content.strip()
                print(f"LLM raw response (attempt {attempt}):\n{raw}")
                return raw
            except Exception as e:
                print(f"[Fragment LLM Error] attempt {attempt} failed: {e}")
                time.sleep(1)

        return ''
