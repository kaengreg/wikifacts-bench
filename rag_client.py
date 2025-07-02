import time
from typing import Optional, List
from openai import OpenAI
import re
import json


class SimpleRagClient:
    def __init__(self, model_name: str, api_url: str, api_key: Optional[str] = None,
                 timeout: int = 360, max_attempts: int = 3, allow_idk: bool = False, temperature: int = 0.1):
        self.client = OpenAI(api_key=api_key, base_url=api_url, timeout=timeout)
        self.model = model_name
        self.timeout = timeout
        self.max_attempts = max_attempts
        self.allow_idk = allow_idk
        self.temperature = temperature

    def _build_messages(self, system_instruction: str, user_prompt: str, no_think: bool) -> List[dict]:
        if not self.allow_idk:
            system_instruction = re.sub(r",\s*or 'idk' if uncertain\.?", "", system_instruction)
            system_instruction = system_instruction.replace("one of 'yes', 'no', 'idk'", "one of 'yes', 'no'")
            user_prompt = user_prompt.replace(", or 'idk'", "")

        if no_think:
            user_prompt += "/no_think"

        return [
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": user_prompt}
        ]

    def _call(self, messages: List[dict]) -> dict:
        """
        Send messages to the chat API, extract and parse JSON with keys 'answer' and 'reasoning',
        retrying up to self.max_attempts times if parsing fails.
        Returns the parsed JSON dict.
        """
        last_raw = ""
        for attempt in range(1, self.max_attempts + 1):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature,
                    top_p=0.9,
                    timeout=self.timeout
                )
                raw = response.choices[0].message.content.strip()
                last_raw = raw

                match = re.search(r'\{.*\}', raw, flags=re.DOTALL)
                candidate = match.group(0) if match else raw

                try:
                    resp_json = json.loads(candidate)
                    if 'answer' in resp_json and 'reasoning' in resp_json:
                        return resp_json
                except json.JSONDecodeError:
                    pass

                messages.append({
                    "role": "system",
                    "content": (
                        "Your last response was not valid JSON with keys 'answer' and 'reasoning'. "
                        "Please reply strictly with a JSON object containing exactly these two keys."
                    )
                })
                time.sleep(1)
            except Exception:
                time.sleep(1)

        try:
            return json.loads(last_raw)
        except Exception:
            raise ValueError(f"Unable to parse JSON from LLM output after {self.max_attempts} attempts:\n{last_raw}")


class FactOnlyClient(SimpleRagClient):
    def call_llm(self, fact: str, no_think: bool = False) -> str:
        system_msg = (
            "You are solving a factual verification task.\n"
            "You will be given a factual statement.\n"
            "Your task is to verify whether the statement is true, false, or uncertain based on your knowledge.\n"
            "Answer with 'yes' if true, 'no' if false, or 'idk' if uncertain.\n"
            "Respond strictly in JSON format with the following keys:\n"
            "  - 'answer': one of 'yes', 'no', 'idk'\n"
            "  - 'reasoning': your reasoning for the answer\n"
            "Provide the reasoning in the same language as the fact and articles.\n"
            "Do not include any additional text outside the JSON.\n"
            "The 'reasoning' field must be only your explanation in the same language as the input; do not include translations, quoted statements, or any additional markup.\n"
        )
        user_prompt = f'Is the following statement factually correct: "{fact}"?'
        messages = self._build_messages(system_msg, user_prompt, no_think)
        resp_json = self._call(messages)
        response = json.dumps(resp_json, ensure_ascii=False)
        return user_prompt, response


class LinkedAbstractClient(SimpleRagClient):
    def call_llm(self, fact: str, contexts: List[str], no_think: bool = False) -> str:
        abstracts_text = "\n\n".join(contexts)
        system_msg = (
            "You are solving a factual verification task.\n"
            "You will be given a factual statement and abstracts of Wikipedia articles.\n"
            "Use both your general knowledge and the abstracts to determine factual correctness.\n"
            "Answer with 'yes' if true, 'no' if false, or 'idk' if uncertain.\n"
            "Respond strictly in JSON format with the following keys:\n"
            "  - 'answer': one of 'yes', 'no', 'idk'\n"
            "  - 'reasoning': your reasoning for the answer\n"
            "Provide the reasoning in the same language as the fact and abstracts.\n"
            "Do not include any additional text outside the JSON.\n"
            "The 'reasoning' field must be only your explanation in the same language as the input; do not include translations, quoted statements, or any additional markup.\n"
        )
        user_prompt = (
            f'Is the following statement factually correct based on your knowledge and the abstracts below?\n\n'
            f'FACT:\n"{fact}"\n\n'
            f'ABSTRACTS:\n\n{abstracts_text}'
        )
        messages = self._build_messages(system_msg, user_prompt, no_think)
        resp_json = self._call(messages)
        response = json.dumps(resp_json, ensure_ascii=False)
        return user_prompt, response


class RelevantAbstractClient(SimpleRagClient):
    def call_llm(self, fact: str, contexts: List[str], no_think: bool = False) -> str:
        abstracts_text = "\n\n".join(contexts)
        system_msg = (
            "You are solving a factual verification task.\n"
            "You will be given a factual statement and abstracts of Wikipedia articles.\n"
            "Use both your general knowledge and the abstracts to determine factual correctness.\n"
            "Answer with 'yes' if true, 'no' if false, or 'idk' if uncertain.\n"
            "Respond strictly in JSON format with the following keys:\n"
            "  - 'answer': one of 'yes', 'no', 'idk'\n"
            "  - 'reasoning': your reasoning for the answer\n"
            "Provide the reasoning in the same language as the fact and abstracts.\n"
            "Do not include any additional text outside the JSON.\n"
            "The 'reasoning' field must be only your explanation in the same language as the input; do not include translations, quoted statements, or any additional markup.\n"
        )
        user_prompt = (
            f'Is the following statement factually correct based on your knowledge and the abstracts below?\n\n'
            f'FACT:\n"{fact}"\n\n'
            f'RELEVANT ABSTRACTS:\n\n{abstracts_text}'
        )
        messages = self._build_messages(system_msg, user_prompt, no_think)
        resp_json = self._call(messages)
        response = json.dumps(resp_json, ensure_ascii=False)
        return user_prompt, response