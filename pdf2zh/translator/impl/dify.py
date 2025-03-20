import json
import logging

import requests

from pdf2zh.translator.base import BaseTranslator, TranslatorRegistry

logger = logging.getLogger(__name__)


@TranslatorRegistry.register()
class DifyTranslator(BaseTranslator):
    name = "dify"
    envs = {
        "DIFY_API_URL": None,  # 填写实际 Dify API 地址
        "DIFY_API_KEY": None,  # 替换为实际 API 密钥
    }

    def __init__(self, lang_out, lang_in, model, envs=None, ignore_cache=False, **kwargs):
        self.set_envs(envs)
        super().__init__(lang_out, lang_in, model, ignore_cache)
        self.api_url = self.envs["DIFY_API_URL"]
        self.api_key = self.envs["DIFY_API_KEY"]

    def do_translate(self, text):
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "inputs": {
                "lang_out": self.lang_out,
                "lang_in": self.lang_in,
                "text": text,
            },
            "response_mode": "blocking",
            "user": "translator-service",
        }

        # 向 Dify 服务器发送请求
        response = requests.post(self.api_url, headers=headers, data=json.dumps(payload))
        response.raise_for_status()
        response_data = response.json()

        # 解析响应
        return response_data.get("data", {}).get("outputs", {}).get("text", [])
