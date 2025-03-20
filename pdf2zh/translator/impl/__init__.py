from pdf2zh.translator.impl.anythingllm import AnythingLLMTranslator
from pdf2zh.translator.impl.argos import ArgosTranslator
from pdf2zh.translator.impl.azure import AzureTranslator
from pdf2zh.translator.impl.bing import BingTranslator
from pdf2zh.translator.impl.deepl import DeepLTranslator, DeepLXTranslator
from pdf2zh.translator.impl.dify import DifyTranslator
from pdf2zh.translator.impl.google import GoogleTranslator
from pdf2zh.translator.impl.ollama import OllamaTranslator
from pdf2zh.translator.impl.openai import (
    AzureOpenAITranslator,
    DeepseekTranslator,
    GeminiTranslator,
    GrokTranslator,
    GroqTranslator,
    ModelScopeTranslator,
    OpenAIlikedTranslator,
    OpenAITranslator,
    SiliconTranslator,
    ZhipuTranslator,
)
from pdf2zh.translator.impl.qwen_mt import QwenMtTranslator
from pdf2zh.translator.impl.tencent import TencentTranslator
from pdf2zh.translator.impl.xinference import XinferenceTranslator

__all__ = [
    "AnythingLLMTranslator",
    "ArgosTranslator",
    "AzureOpenAITranslator",
    "AzureTranslator",
    "BingTranslator",
    "DeepLTranslator",
    "DeepLXTranslator",
    "DeepseekTranslator",
    "DifyTranslator",
    "GeminiTranslator",
    "GoogleTranslator",
    "GrokTranslator",
    "GroqTranslator",
    "ModelScopeTranslator",
    "OllamaTranslator",
    "OpenAIlikedTranslator",
    "OpenAITranslator",
    "QwenMtTranslator",
    "SiliconTranslator",
    "TencentTranslator",
    "XinferenceTranslator",
    "ZhipuTranslator",
]
