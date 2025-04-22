from typing import TYPE_CHECKING

from pdf2zh.translator.base import BaseTranslator, TranslatorRegistry

if TYPE_CHECKING:
    from tencentcloud.tmt.v20180321.models import TextTranslateResponse


@TranslatorRegistry.register()
class TencentTranslator(BaseTranslator):
    # https://github.com/TencentCloud/tencentcloud-sdk-python
    name = "tencent"
    envs = {
        "TENCENTCLOUD_SECRET_ID": None,
        "TENCENTCLOUD_SECRET_KEY": None,
    }

    def __init__(self, lang_in, lang_out, model, envs=None, ignore_cache=False, **kwargs):
        try:
            from tencentcloud.common import credential
            from tencentcloud.tmt.v20180321.models import TextTranslateRequest
            from tencentcloud.tmt.v20180321.tmt_client import TmtClient
        except ImportError:
            raise ImportError("tencentcloud-sdk-python is not installed") from None
        self.set_envs(envs)
        super().__init__(lang_in, lang_out, model, ignore_cache)
        try:
            cred = credential.DefaultCredentialProvider().get_credential()
        except OSError:
            cred = credential.Credential(
                self.envs["TENCENTCLOUD_SECRET_ID"],
                self.envs["TENCENTCLOUD_SECRET_KEY"],
            )
        self.client = TmtClient(cred, "ap-beijing")
        self.req = TextTranslateRequest()
        self.req.Source = self.lang_in
        self.req.Target = self.lang_out
        self.req.ProjectId = 0

    def do_translate(self, text):
        self.req.SourceText = text
        resp: TextTranslateResponse = self.client.TextTranslate(self.req)
        return resp.TargetText
