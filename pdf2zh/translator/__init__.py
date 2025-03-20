"""
翻译器模块，提供不同的翻译器实现和注册机制
"""

from pdf2zh.translator.base import BaseTranslator, TranslatorRegistry
from pdf2zh.translator.impl import *  # noqa: F403

# 导出主要的类和函数
__all__ = [
    "BaseTranslator",
    "TranslatorRegistry",
]
