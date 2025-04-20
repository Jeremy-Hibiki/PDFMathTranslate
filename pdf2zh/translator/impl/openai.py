import ast
import json
import logging
import re

import openai
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from pdf2zh.tokenizer import TokenizerManager
from pdf2zh.translator.base import BaseTranslator, TranslatorRegistry

logger = logging.getLogger(__name__)


retry_429 = retry(
    retry=retry_if_exception_type(openai.RateLimitError),
    stop=stop_after_attempt(10),
    wait=wait_exponential(multiplier=1, min=1, max=15),
    before_sleep=lambda retry_state: logger.warning(
        f"RateLimitError, retrying in {retry_state.next_action.sleep} seconds... "
        f"(Attempt {retry_state.attempt_number}/10)"
    ),
)


@TranslatorRegistry.register()
class OpenAITranslator(BaseTranslator):
    # https://github.com/openai/openai-python
    name = "openai"
    envs = {
        "OPENAI_BASE_URL": "https://api.openai.com/v1",
        "OPENAI_API_KEY": None,
        "OPENAI_MODEL": "gpt-4o-mini",
        "OPENAI_EXTRA_KWARGS": "",
    }
    CustomPrompt = True

    def __init__(
        self,
        lang_in,
        lang_out,
        model,
        base_url=None,
        api_key=None,
        envs=None,
        prompt=None,
        ignore_cache=False,
        extra_kwargs=None,
    ):
        self.set_envs(envs)
        if not model:
            model = self.envs["OPENAI_MODEL"]
        super().__init__(lang_in, lang_out, model, ignore_cache)
        env_kwargs = extra_kwargs or self.envs.get("OPENAI_EXTRA_KWARGS", "")
        if env_kwargs:
            try:
                try:
                    client_kwargs = json.loads(env_kwargs)
                except json.JSONDecodeError:
                    client_kwargs = ast.literal_eval(env_kwargs)
            except Exception:
                logger.warning(
                    "Ignoring illegal OPENAI(LIKED)_EXTRA_KWARGS, must be a valid JSON string or Python expression"
                )
                client_kwargs = dict()
            for k in ["model", "messages"]:
                if k in client_kwargs:
                    del client_kwargs[k]
        else:
            client_kwargs = dict()
        client_kwargs.update({"temperature": 0})  # 随机采样可能会打断公式标记
        self.options = client_kwargs
        self.client = openai.OpenAI(
            base_url=base_url or self.envs["OPENAI_BASE_URL"],
            api_key=api_key or self.envs["OPENAI_API_KEY"],
        )
        self.async_client = openai.AsyncOpenAI(
            base_url=base_url or self.envs["OPENAI_BASE_URL"],
            api_key=api_key or self.envs["OPENAI_API_KEY"],
        )
        self.prompttext = prompt
        self.add_cache_impact_parameters("temperature", self.options["temperature"])
        self.add_cache_impact_parameters("prompt", self.prompt("", self.prompttext))
        think_filter_regex = r"^<think>.+?\n*(</think>|\n)*(</think>)\n*"
        self.add_cache_impact_parameters("think_filter_regex", think_filter_regex)
        self.think_filter_regex = re.compile(think_filter_regex, flags=re.DOTALL)

    @retry_429
    def do_translate(self, text) -> str:
        messages = self.prompt(text, self.prompttext)
        try:
            tokens = TokenizerManager.count_tokens(text)
            max_tokens = int(max(20, tokens * 2))  # 短文本 Token 估算容易非常不准
            response = self.client.chat.completions.create(
                model=self.model,
                **self.options,
                max_tokens=max_tokens,
                messages=messages,
            )
        except openai.BadRequestError as e:
            # Maybe the max_tokens is larger than the api accepts
            if re.findall(r"max[_ ]tokens", e.message, re.I):
                response = self.client.chat.completions.create(
                    model=self.model,
                    **self.options,
                    messages=messages,
                )
            else:
                raise
        if not response.choices:
            if hasattr(response, "error"):
                raise ValueError("Error response from Service", response.error)
        if response.choices[0].finish_reason == "length":
            raise ValueError("Response length limit exceeded")
        content = response.choices[0].message.content.strip()
        content = self.think_filter_regex.sub("", content).strip()
        return content

    @retry_429
    async def ado_translate(self, text: str) -> str:
        messages = self.prompt(text, self.prompttext)
        try:
            tokens = TokenizerManager.count_tokens(text)
            max_tokens = int(max(20, tokens * 2))  # 短文本 Token 估算容易非常不准
            response = await self.async_client.chat.completions.create(
                model=self.model,
                **self.options,
                max_tokens=max_tokens,
                messages=messages,
            )
        except openai.BadRequestError as e:
            # Maybe the max_tokens is larger than the api accepts
            if re.findall(r"max[_ ]tokens", e.message, re.I):
                response = await self.async_client.chat.completions.create(
                    model=self.model,
                    **self.options,
                    messages=messages,
                )
            else:
                raise
        if not response.choices:
            if hasattr(response, "error"):
                raise ValueError("Error response from Service", response.error)
        if response.choices[0].finish_reason == "length":
            raise ValueError("Response length limit exceeded")
        content = response.choices[0].message.content.strip()
        content = self.think_filter_regex.sub("", content).strip()
        return content

    def get_formular_placeholder(self, id: int):
        return "{{v" + str(id) + "}}"

    def get_rich_text_left_placeholder(self, id: int):
        return self.get_formular_placeholder(id)

    def get_rich_text_right_placeholder(self, id: int):
        return self.get_formular_placeholder(id + 1)


@TranslatorRegistry.register()
class AzureOpenAITranslator(BaseTranslator):
    name = "azure-openai"
    envs = {
        "AZURE_OPENAI_BASE_URL": None,  # e.g. "https://xxx.openai.azure.com"
        "AZURE_OPENAI_API_KEY": None,
        "AZURE_OPENAI_MODEL": "gpt-4o-mini",
    }
    CustomPrompt = True

    def __init__(
        self,
        lang_in,
        lang_out,
        model,
        base_url=None,
        api_key=None,
        envs=None,
        prompt=None,
        ignore_cache=False,
    ):
        self.set_envs(envs)
        base_url = self.envs["AZURE_OPENAI_BASE_URL"]
        if not model:
            model = self.envs["AZURE_OPENAI_MODEL"]
        super().__init__(lang_in, lang_out, model, ignore_cache)
        self.options = {"temperature": 0}
        self.client = openai.AzureOpenAI(
            azure_endpoint=base_url,
            azure_deployment=model,
            api_version="2024-06-01",
            api_key=api_key,
        )
        self.prompttext = prompt
        self.add_cache_impact_parameters("temperature", self.options["temperature"])
        self.add_cache_impact_parameters("prompt", self.prompt("", self.prompttext))

    def do_translate(self, text) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            **self.options,
            messages=self.prompt(text, self.prompttext),
        )
        return response.choices[0].message.content.strip()


@TranslatorRegistry.register()
class ModelScopeTranslator(OpenAITranslator):
    name = "modelscope"
    envs = {
        "MODELSCOPE_BASE_URL": "https://api-inference.modelscope.cn/v1",
        "MODELSCOPE_API_KEY": None,
        "MODELSCOPE_MODEL": "Qwen/Qwen2.5-32B-Instruct",
    }
    CustomPrompt = True

    def __init__(
        self,
        lang_in,
        lang_out,
        model,
        base_url=None,
        api_key=None,
        envs=None,
        prompt=None,
        ignore_cache=False,
    ):
        self.set_envs(envs)
        base_url = "https://api-inference.modelscope.cn/v1"
        api_key = self.envs["MODELSCOPE_API_KEY"]
        if not model:
            model = self.envs["MODELSCOPE_MODEL"]
        super().__init__(
            lang_in,
            lang_out,
            model,
            base_url=base_url,
            api_key=api_key,
            ignore_cache=ignore_cache,
        )
        self.prompttext = prompt
        self.add_cache_impact_parameters("prompt", self.prompt("", self.prompttext))


@TranslatorRegistry.register()
class ZhipuTranslator(OpenAITranslator):
    # https://bigmodel.cn/dev/api/thirdparty-frame/openai-sdk
    name = "zhipu"
    envs = {
        "ZHIPU_API_KEY": None,
        "ZHIPU_MODEL": "glm-4-flash",
    }
    CustomPrompt = True

    def __init__(self, lang_in, lang_out, model, envs=None, prompt=None, ignore_cache=False):
        self.set_envs(envs)
        base_url = "https://open.bigmodel.cn/api/paas/v4"
        api_key = self.envs["ZHIPU_API_KEY"]
        if not model:
            model = self.envs["ZHIPU_MODEL"]
        super().__init__(
            lang_in,
            lang_out,
            model,
            base_url=base_url,
            api_key=api_key,
            ignore_cache=ignore_cache,
        )
        self.prompttext = prompt
        self.add_cache_impact_parameters("prompt", self.prompt("", self.prompttext))

    def do_translate(self, text) -> str:
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                **self.options,
                messages=self.prompt(text, self.prompttext),
            )
        except openai.BadRequestError as e:
            if json.loads(response.choices[0].message.content.strip())["error"]["code"] == "1301":
                return "IRREPARABLE TRANSLATION ERROR"
            raise e
        return response.choices[0].message.content.strip()


@TranslatorRegistry.register()
class SiliconTranslator(OpenAITranslator):
    # https://docs.siliconflow.cn/quickstart
    name = "silicon"
    envs = {
        "SILICON_API_KEY": None,
        "SILICON_MODEL": "Qwen/Qwen2.5-7B-Instruct",
    }
    CustomPrompt = True

    def __init__(self, lang_in, lang_out, model, envs=None, prompt=None, ignore_cache=False):
        self.set_envs(envs)
        base_url = "https://api.siliconflow.cn/v1"
        api_key = self.envs["SILICON_API_KEY"]
        if not model:
            model = self.envs["SILICON_MODEL"]
        super().__init__(
            lang_in,
            lang_out,
            model,
            base_url=base_url,
            api_key=api_key,
            ignore_cache=ignore_cache,
        )
        self.prompttext = prompt
        self.add_cache_impact_parameters("prompt", self.prompt("", self.prompttext))


@TranslatorRegistry.register()
class GeminiTranslator(OpenAITranslator):
    # https://ai.google.dev/gemini-api/docs/openai
    name = "gemini"
    envs = {
        "GEMINI_API_KEY": None,
        "GEMINI_MODEL": "gemini-1.5-flash",
    }
    CustomPrompt = True

    def __init__(self, lang_in, lang_out, model, envs=None, prompt=None, ignore_cache=False):
        self.set_envs(envs)
        base_url = "https://generativelanguage.googleapis.com/v1beta/openai/"
        api_key = self.envs["GEMINI_API_KEY"]
        if not model:
            model = self.envs["GEMINI_MODEL"]
        super().__init__(
            lang_in,
            lang_out,
            model,
            base_url=base_url,
            api_key=api_key,
            ignore_cache=ignore_cache,
        )
        self.prompttext = prompt
        self.add_cache_impact_parameters("prompt", self.prompt("", self.prompttext))


@TranslatorRegistry.register()
class GrokTranslator(OpenAITranslator):
    # https://docs.x.ai/docs/overview#getting-started
    name = "grok"
    envs = {
        "GORK_API_KEY": None,
        "GORK_MODEL": "grok-2-1212",
    }
    CustomPrompt = True

    def __init__(self, lang_in, lang_out, model, envs=None, prompt=None, ignore_cache=False):
        self.set_envs(envs)
        base_url = "https://api.x.ai/v1"
        api_key = self.envs["GORK_API_KEY"]
        if not model:
            model = self.envs["GORK_MODEL"]
        super().__init__(
            lang_in,
            lang_out,
            model,
            base_url=base_url,
            api_key=api_key,
            ignore_cache=ignore_cache,
        )
        self.prompttext = prompt


@TranslatorRegistry.register()
class GroqTranslator(OpenAITranslator):
    name = "groq"
    envs = {
        "GROQ_API_KEY": None,
        "GROQ_MODEL": "llama-3-3-70b-versatile",
    }
    CustomPrompt = True

    def __init__(self, lang_in, lang_out, model, envs=None, prompt=None, ignore_cache=False):
        self.set_envs(envs)
        base_url = "https://api.groq.com/openai/v1"
        api_key = self.envs["GROQ_API_KEY"]
        if not model:
            model = self.envs["GROQ_MODEL"]
        super().__init__(
            lang_in,
            lang_out,
            model,
            base_url=base_url,
            api_key=api_key,
            ignore_cache=ignore_cache,
        )
        self.prompttext = prompt


@TranslatorRegistry.register()
class DeepseekTranslator(OpenAITranslator):
    name = "deepseek"
    envs = {
        "DEEPSEEK_API_KEY": None,
        "DEEPSEEK_MODEL": "deepseek-chat",
    }
    CustomPrompt = True

    def __init__(self, lang_in, lang_out, model, envs=None, prompt=None, ignore_cache=False):
        self.set_envs(envs)
        base_url = "https://api.deepseek.com/v1"
        api_key = self.envs["DEEPSEEK_API_KEY"]
        if not model:
            model = self.envs["DEEPSEEK_MODEL"]
        super().__init__(
            lang_in,
            lang_out,
            model,
            base_url=base_url,
            api_key=api_key,
            ignore_cache=ignore_cache,
        )
        self.prompttext = prompt


@TranslatorRegistry.register()
class OpenAIlikedTranslator(OpenAITranslator):
    name = "openailiked"
    envs = {
        "OPENAILIKED_BASE_URL": None,
        "OPENAILIKED_API_KEY": None,
        "OPENAILIKED_MODEL": None,
        "OPENAILIKED_EXTRA_KWARGS": "",
    }
    CustomPrompt = True

    def __init__(self, lang_in, lang_out, model, envs=None, prompt=None, ignore_cache=False):
        self.set_envs(envs)
        if self.envs["OPENAILIKED_BASE_URL"]:
            base_url = self.envs["OPENAILIKED_BASE_URL"]
        else:
            raise ValueError("The OPENAILIKED_BASE_URL is missing.")
        if not model:
            if self.envs["OPENAILIKED_MODEL"]:
                model = self.envs["OPENAILIKED_MODEL"]
            else:
                raise ValueError("The OPENAILIKED_MODEL is missing.")
        if self.envs["OPENAILIKED_API_KEY"] is None:
            api_key = "openailiked"
        else:
            api_key = self.envs["OPENAILIKED_API_KEY"]
        super().__init__(
            lang_in,
            lang_out,
            model,
            base_url=base_url,
            api_key=api_key,
            ignore_cache=ignore_cache,
            extra_kwargs=self.envs.get("OPENAILIKED_EXTRA_KWARGS", None),
        )
        self.prompttext = prompt
