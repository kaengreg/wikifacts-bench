import time
from typing import Optional, List, Dict
from openai import OpenAI
import re
import json
import os
from deep_translator import GoogleTranslator


class SimpleRagClient:
    def __init__(self, model_name: str, api_url: str, api_key: Optional[str] = None,
                 timeout: int = 360, max_attempts: int = 3, allow_idk: bool = False, temperature: int = 0.1, failed_facts_path: str = 'failed_facts.jsonl',
                 translator: Optional[GoogleTranslator] = None,
                 use_few_shots: bool = False):
        self.client = OpenAI(api_key=api_key, base_url=api_url, timeout=timeout)
        self.model = model_name
        self.timeout = timeout
        self.max_attempts = max_attempts
        self.allow_idk = allow_idk
        self.temperature = temperature
        self.failed_facts = failed_facts_path
        self.translator = translator
        self.use_few_shots = use_few_shots
        self._translations_cache: Dict[str, str] = {}

        failed_dir = os.path.dirname(self.failed_facts)
        if failed_dir and not os.path.exists(failed_dir):
            os.makedirs(failed_dir, exist_ok=True)

        open(self.failed_facts, 'a', encoding='utf-8').close()

    def _translate_cached(self, text: str) -> str:
        if not self.translator:
            return text
        
        if text in self._translations_cache:
            return self._translations_cache[text]
            
        masked_text = re.sub(r"(?i)['\"]answer['\"]", "__ANS__", text)
        masked_text = re.sub(r"(?i)['\"]reasoning['\"]", "__RSP__", masked_text)
        try:
            translated_masked = self.translator.translate(masked_text)
        except Exception:
            translated = text
        else:
            translated = re.sub(r"(?i)__ANS__", '"answer"', translated_masked)
            translated = re.sub(r"(?i)__RSP__", '"reasoning"', translated)

        self._translations_cache[text] = translated
        return translated

    def _strip_idk(self, text: str) -> str:
        """Remove 'idk' option from English templates before translation."""
        text = re.sub(r",\s*or 'idk' if uncertain\.?", "", text)
        text = text.replace("one of 'yes', 'no', 'idk'", "one of 'yes', 'no'")
        text = text.replace(", or 'idk'", "")
        return text
    
    def _strip_think(self, text: str) -> str:
        return re.sub(r"(?is)<think>.*?</think>", "", text)

    def _find_json_object(self, s: str) -> str:
        objs = []
        level = 0
        start = None
        in_string = False
        quote_char = ""
        esc = False
        for i, ch in enumerate(s):
            if in_string:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == quote_char:
                    in_string = False
            else:
                if ch in ('"', "'"):
                    in_string = True
                    quote_char = ch
                elif ch == '{':
                    if level == 0:
                        start = i
                    level += 1
                elif ch == '}':
                    if level > 0:
                        level -= 1
                        if level == 0 and start is not None:
                            objs.append(s[start:i+1])
                            start = None
        return objs

    def _extract_json_dict(self, text: str) -> Optional[dict]:
        if not text:
            return None
        
        text = self._strip_think(text).strip()
        print("THINK BLOCK STRIPPED:", text)

        for pattern in (r"(?is)```json\s*(\{.*?\})\s*```", r"(?is)```\s*(\{.*?\})\s*```"):
            m = re.search(pattern, text)
            if m:
                try:
                    d = json.loads(m.group(1))
                    return d
                except Exception:
                    pass
        candidates = self._find_json_object(text)
        for cand in reversed(candidates):
            try:
                d = json.loads(cand)
                return d
            except Exception:
                continue
        return None

    def _build_messages(self, system_instruction: str, user_prompt: str, no_think: bool, few_shots: Optional[List[dict]] = None) -> List[dict]:
        if no_think:
            user_prompt += "/no_think"

        messages = [{"role": "system", "content": system_instruction}]
        if few_shots:
            messages.extend(few_shots)
        messages.append({"role": "user", "content": user_prompt})
        return messages

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
                raw = response.choices[0].message.content.strip() if response and response.choices else ""
                if raw is None:
                    raw = ""
                raw = raw.strip()

                last_raw = raw 

                parsed = self._extract_json_dict(raw)
                print(parsed)
                if parsed is not None:
                    return parsed 
            
                fallback_content = (
                    "Your last response was not valid JSON with keys 'answer' and 'reasoning'. "
                    "Please reply strictly with a JSON object containing exactly these two keys."
                )
                if self.translator:
                    fallback_content = self._translate_cached(fallback_content)
                messages.append({
                    "role": "system",
                    "content": fallback_content
                })
                time.sleep(1)
            except Exception:
                time.sleep(1)

        return self._extract_json_dict(last_raw) if last_raw else None


class FactOnlyClient(SimpleRagClient):
    def call_llm(self, fact: str, no_think: bool = True) -> str:
        template_sys = (
            "You are solving a factual verification task.\n"
            "You will be given a factual statement or a question that may start with 'Did you know...'.\n"
            "Your task is to verify whether the statement is true, false, or uncertain based on your knowledge.\n"
            "Answer with 'yes' if true, 'no' if false, or 'idk' if uncertain.\n"
            "Respond strictly in JSON format with the following keys:\n"
            "  - 'answer': one of 'yes', 'no', 'idk'\n"
            "  - 'reasoning': your reasoning for the answer\n"
            "Provide the reasoning in the same language as the fact and articles.\n"
            "Do not include any additional text outside the JSON.\n"
            "The 'reasoning' field must be only your explanation in the same language as the input; do not include translations, quoted statements, or any additional markup.\n"
        )
        template_user = 'Is the following statement factually correct: "{}"?'
        if not self.allow_idk:
            template_sys = self._strip_idk(template_sys)
            template_user = self._strip_idk(template_user)
        if self.translator:
            template_sys = self._translate_cached(template_sys)
        system_msg = template_sys

        if self.translator:
            template_user = self._translate_cached(template_user)
        user_prompt = template_user.format(fact)

        few_shot_msgs = None
        if self.use_few_shots:
            ex1_fact = "London is the capital of Great Britain."
            ex2_fact = "Did you know that the Sun revolves around the Earth?"
            ex_user_template = template_user  
            ex1_user = ex_user_template.format(ex1_fact)
            ex2_user = ex_user_template.format(ex2_fact)

            ex1_assistant = json.dumps({"answer": "yes", "reasoning": "London is the capital of the United Kingdom."}, ensure_ascii=False)
            ex2_assistant = json.dumps({"answer": "no", "reasoning": "The Earth orbits the Sun, not the other way around."}, ensure_ascii=False)
            if self.translator:
                ex1_assistant = self._translate_cached(ex1_assistant)
                ex2_assistant = self._translate_cached(ex2_assistant)

            few_shot_msgs = [
                {"role": "user", "content": ex1_user},
                {"role": "assistant", "content": ex1_assistant},
                {"role": "user", "content": ex2_user},
                {"role": "assistant", "content": ex2_assistant},
            ]

        messages = self._build_messages(system_msg, user_prompt, no_think, few_shots=few_shot_msgs)
        resp_json = self._call(messages)

        if resp_json is None:
            with open(self.failed_facts, 'a', encoding='utf-8') as f:
                json.dump({'fact': fact, 'error': 'Unable to parse JSON'}, f, ensure_ascii=False)
                f.write('\n')
            return None, None
        
        response = json.dumps(resp_json, ensure_ascii=False)
        return user_prompt, response


class LinkedAbstractClient(SimpleRagClient):
    def call_llm(self, fact: str, contexts: List[str], no_think: bool = True) -> str:
        abstracts_text = "\n\n".join(contexts)
        template_sys = (
            "You are solving a factual verification task.\n"
            "You will be given a factual statement or a question that may start with 'Did you know...' and abstracts of Wikipedia articles.\n"
            "Use both your general knowledge and the abstracts to determine factual correctness.\n"
            "Answer with 'yes' if true, 'no' if false, or 'idk' if uncertain.\n"
            "Respond strictly in JSON format with the following keys:\n"
            "  - 'answer': one of 'yes', 'no', 'idk'\n"
            "  - 'reasoning': your reasoning for the answer\n"
            "Provide the reasoning in the same language as the fact and abstracts.\n"
            "Do not include any additional text outside the JSON.\n"
            "The 'reasoning' field must be only your explanation in the same language as the input; do not include translations, quoted statements, or any additional markup.\n"
        )
        template_user = (
            'Is the following statement factually correct based on your knowledge and the abstracts below?\n\n'
            'FACT:\n"{}"\n\n'
            'ABSTRACTS:\n\n{}'
        )
        if not self.allow_idk:
            template_sys = self._strip_idk(template_sys)
            template_user = self._strip_idk(template_user)
        if self.translator:
            template_sys = self._translate_cached(template_sys)
        system_msg = template_sys

        if self.translator:
            template_user = self._translate_cached(template_user)
            
        user_prompt = template_user.format(fact, abstracts_text)

        few_shot_msgs = None
        if self.use_few_shots:
            ex1_fact = "London is the capital of Great Britain."
            ex2_fact = "Did you know that the Sun revolves around the Earth?"
            ex_user_template = template_user
            ex1_user = ex_user_template.format(ex1_fact, "")
            ex2_user = ex_user_template.format(ex2_fact, "")

            ex1_assistant = json.dumps({"answer": "yes", "reasoning": "London is the capital of the United Kingdom."}, ensure_ascii=False)
            ex2_assistant = json.dumps({"answer": "no", "reasoning": "The Earth orbits the Sun, not the other way around."}, ensure_ascii=False)
            if self.translator:
                ex1_assistant = self._translate_cached(ex1_assistant)
                ex2_assistant = self._translate_cached(ex2_assistant)

            few_shot_msgs = [
                {"role": "user", "content": ex1_user},
                {"role": "assistant", "content": ex1_assistant},
                {"role": "user", "content": ex2_user},
                {"role": "assistant", "content": ex2_assistant},
            ]

        messages = self._build_messages(system_msg, user_prompt, no_think, few_shots=few_shot_msgs)
        resp_json = self._call(messages)

        if resp_json is None:
            with open(self.failed_facts, 'a', encoding='utf-8') as f:
                json.dump({'fact': fact, 'error': 'Unable to parse JSON'}, f, ensure_ascii=False)
                f.write('\n')
            return None, None
        
        response = json.dumps(resp_json, ensure_ascii=False)
        return user_prompt, response


class RelevantAbstractClient(SimpleRagClient):
    def call_llm(self, fact: str, contexts: List[str], no_think: bool = True) -> str:
        abstracts_text = "\n\n".join(contexts)
        template_sys = (
            "You are solving a factual verification task.\n"
            "You will be given a factual statement or a question that may start with 'Did you know...' and abstracts of Wikipedia articles.\n"
            "Use both your general knowledge and the abstracts to determine factual correctness.\n"
            "Answer with 'yes' if true, 'no' if false, or 'idk' if uncertain.\n"
            "Respond strictly in JSON format with the following keys:\n"
            "  - 'answer': one of 'yes', 'no', 'idk'\n"
            "  - 'reasoning': your reasoning for the answer\n"
            "Provide the reasoning in the same language as the fact and abstracts.\n"
            "Do not include any additional text outside the JSON.\n"
            "The 'reasoning' field must be only your explanation in the same language as the input; do not include translations, quoted statements, or any additional markup.\n"
        )
        template_user = (
            'Is the following statement factually correct based on your knowledge and the abstracts below?\n\n'
            'FACT:\n"{}"\n\n'
            'RELEVANT ABSTRACTS:\n\n{}'
        )
        if not self.allow_idk:
            template_sys = self._strip_idk(template_sys)
            template_user = self._strip_idk(template_user)
            
        if self.translator:
            template_sys = self._translate_cached(template_sys)
        system_msg = template_sys

        if self.translator:
            template_user = self._translate_cached(template_user)
        user_prompt = template_user.format(fact, abstracts_text)

        few_shot_msgs = None
        if self.use_few_shots:

            ex1_fact = "London is the capital of Great Britain."
            ex2_fact = "Did you know that the Sun revolves around the Earth?"
            ex_user_template = template_user
            ex1_user = ex_user_template.format(ex1_fact, "")
            ex2_user = ex_user_template.format(ex2_fact, "")

            ex1_assistant = json.dumps({"answer": "yes", "reasoning": "London is the capital of the United Kingdom."}, ensure_ascii=False)
            ex2_assistant = json.dumps({"answer": "no", "reasoning": "The Earth orbits the Sun, not the other way around."}, ensure_ascii=False)

            if self.translator:
                ex1_assistant = self._translate_cached(ex1_assistant)
                ex2_assistant = self._translate_cached(ex2_assistant)

            few_shot_msgs = [
                {"role": "user", "content": ex1_user},
                {"role": "assistant", "content": ex1_assistant},
                {"role": "user", "content": ex2_user},
                {"role": "assistant", "content": ex2_assistant},
            ]

        messages = self._build_messages(system_msg, user_prompt, no_think, few_shots=few_shot_msgs)
        resp_json = self._call(messages)

        if resp_json is None:
            with open(self.failed_facts, 'a', encoding='utf-8') as f:
                json.dump({'fact': fact, 'error': 'Unable to parse JSON'}, f, ensure_ascii=False)
                f.write('\n')
            return None, None
        
        response = json.dumps(resp_json, ensure_ascii=False)
        return user_prompt, response