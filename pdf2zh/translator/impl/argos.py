import logging

from pdf2zh.translator.base import BaseTranslator, TranslatorRegistry

logger = logging.getLogger(__name__)


@TranslatorRegistry.register()
class ArgosTranslator(BaseTranslator):
    name = "argos"

    def __init__(self, lang_in, lang_out, model, ignore_cache=False, **kwargs):
        super().__init__(lang_in, lang_out, model, ignore_cache)
        lang_in = self.lang_map.get(lang_in.lower(), lang_in)
        lang_out = self.lang_map.get(lang_out.lower(), lang_out)
        self.lang_in = lang_in
        self.lang_out = lang_out

        try:
            import argostranslate.package  # type: ignore
            import argostranslate.translate  # type: ignore

            argostranslate.package.update_package_index()
            available_packages = argostranslate.package.get_available_packages()
            try:
                available_package = list(
                    filter(
                        lambda x: x.from_code == self.lang_in and x.to_code == self.lang_out,
                        available_packages,
                    )
                )[0]
            except Exception:
                raise ValueError("lang_in and lang_out pair not supported by Argos Translate.") from None
            download_path = available_package.download()
            argostranslate.package.install_from_path(download_path)
            self.argostranslate = argostranslate
        except ImportError:
            logger.warning("argos-translate is not installed. Please install it if you want to use ArgosTranslator.")
            raise ImportError("argos-translate is not installed") from None

    def do_translate(self, text: str):
        # Translate
        installed_languages = self.argostranslate.translate.get_installed_languages()
        from_lang = list(filter(lambda x: x.code == self.lang_in, installed_languages))[0]
        to_lang = list(filter(lambda x: x.code == self.lang_out, installed_languages))[0]
        translation = from_lang.get_translation(to_lang)
        translated_text = translation.translate(text)
        return translated_text
