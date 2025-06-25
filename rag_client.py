import time
from typing import Optional, List
from openai import OpenAI


class SimpleRagClient:
    def __init__(self, model_name: str, api_url: str, api_key: Optional[str] = None,
                 timeout: int = 360, max_attempts: int = 3, allow_idk: bool = True, temperature: int = 0.1):
        self.client = OpenAI(api_key=api_key, base_url=api_url, timeout=timeout)
        self.model = model_name
        self.timeout = timeout
        self.max_attempts = max_attempts
        self.allow_idk = allow_idk
        self.temperature = temperature

    def _build_messages(self, system_instruction: str, user_prompt: str, no_think: bool) -> List[dict]:
        if not self.allow_idk:
            system_instruction = system_instruction.replace('"yes", "no", or "i don\'t know"', '"yes" or "no"')
            user_prompt = user_prompt.replace('"yes", "no", or "i don\'t know"', '"yes" or "no"')

        if no_think:
            user_prompt += "/no_think"

        return [
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": user_prompt}
        ]

    def _call(self, messages: List[dict]) -> str:
        for attempt in range(1, self.max_attempts + 1):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature,
                    top_p=0.9,
                    timeout=self.timeout
                )
                return response.choices[0].message.content.strip()
            except Exception:
                time.sleep(1)
        return ''


class FactOnlyClient(SimpleRagClient):
    def call_llm(self, fact: str, no_think: bool = True) -> str:
        system_msg = (
            "You are solving a factual verification task.\n"
            "You will be given a factual statement.\n"
            "Your task is to verify whether the statement is true, false, or uncertain based on your knowledge.\n"
            "Respond strictly with one of the following: \"yes\", \"no\", or \"i don't know\".\n"
            "Only English responses are allowed."
        )
        user_prompt = f'Is the following statement factually correct: "{fact}"?\nAnswer only with "yes", "no", or "i don\'t know".'
        messages = self._build_messages(system_msg, user_prompt, no_think)
        response = self._call(messages).strip().lower()
        return user_prompt, response


class LinkedAbstractClient(SimpleRagClient):
    def call_llm(self, fact: str, contexts: List[str], no_think: bool = True) -> str:
        abstracts_text = "\n\n".join(contexts)
        system_msg = (
            "You are solving a factual verification task.\n"
            "You will be given a factual statement and abstracts of Wikipedia articles.\n"
            "Use both your own knowledge and the abstracts to determine whether the statement is factually correct.\n"
            "If the abstracts are unhelpful, you may rely on your own knowledge.\n"
            "Only answer \"I don't know\" if neither the abstracts nor your knowledge provide enough information.\n"
            "Respond strictly with one of: \"yes\", \"no\", or \"i don't know\".\n"
            "Only English responses are allowed."
        )
        user_prompt = (
            f'FACT:\n"{fact}"\n\n'
            'Is this factually correct based on your knowledge and the abstracts below?\n'
            'Answer with "yes", "no", or "i don\'t know".\n\n'
            f'ABSTRACTS:\n\n{abstracts_text}'
        )
        messages = self._build_messages(system_msg, user_prompt, no_think)
        response = self._call(messages).strip().lower()
        return user_prompt, response


class RelevantAbstractClient(SimpleRagClient):
    def call_llm(self, fact: str, contexts: List[str], no_think: bool = True) -> str:
        abstracts_text = "\n\n".join(contexts)
        system_msg = (
            "You are solving a factual verification task.\n"
            "You will be given a factual statement and relevant abstracts of Wikipedia articles.\n"
            "Use both your general knowledge and the abstracts provided to determine whether the statement is factually correct.\n"
            "If the abstracts are unhelpful or unrelated, you may rely on your own knowledge.\n"
            "Only answer \"I don't know\" if neither the abstracts nor your knowledge provide enough information.\n"
            "Try to match phrases or facts from the statement to the abstracts.\n"
            "Respond strictly with one of: \"yes\", \"no\", or \"i don't know\".\n"
            "Only English responses are allowed."
        )
        user_prompt = (
            f'FACT:\n"{fact}"\n\n'
            'Is this factually correct based on your knowledge and the abstracts below?\n'
            'Answer with "yes", "no", or "i don\'t know".\n\n'
            f'ABSTRACTS:\n\n{abstracts_text}'
        )
        messages = self._build_messages(system_msg, user_prompt, no_think)
        response = self._call(messages).strip().lower()
        return user_prompt, response