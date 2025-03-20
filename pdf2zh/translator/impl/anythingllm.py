import json
import logging

import requests

from pdf2zh.translator.base import BaseTranslator, TranslatorRegistry

logger = logging.getLogger(__name__)


@TranslatorRegistry.register()
class AnythingLLMTranslator(BaseTranslator):
    name = "anythingllm"
    envs = {
        "AnythingLLM_URL": None,
        "AnythingLLM_APIKEY": None,
    }
    CustomPrompt = True

    def __init__(self, lang_out, lang_in, model, envs=None, prompt=None, ignore_cache=False):
        self.set_envs(envs)
        super().__init__(lang_out, lang_in, model, ignore_cache)
        self.api_url = self.envs["AnythingLLM_URL"]
        self.api_key = self.envs["AnythingLLM_APIKEY"]
        self.headers = {
            "accept": "application/json",
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        self.prompttext = prompt

    def do_translate(self, text):
        messages = self.prompt(text, self.prompttext)
        payload = {
            "message": messages,
            "mode": "chat",
            "sessionId": "translation_expert",
        }

        response = requests.post(self.api_url, headers=self.headers, data=json.dumps(payload))
        response.raise_for_status()
        data = response.json()

        if "textResponse" in data:
            return data["textResponse"].strip()
