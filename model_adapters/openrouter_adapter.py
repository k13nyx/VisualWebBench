from openai import OpenAI
from PIL import Image
import base64
import io, re

from model_adapters.openai_adapter import OpenAIAdapter
from utils.constants import *


class OpenRouterAdapter(OpenAIAdapter):
    """OpenAIAdapter for any OpenAI-compatible API (e.g. OpenRouter) that also records tokens and cost."""

    def __init__(self, client: OpenAI, model: str, max_tokens: int = 2048, extra_body=None):
        super().__init__(client, model)
        self.max_tokens = max_tokens
        self.extra_body = {"usage": {"include": True}, **(extra_body or {})}  # ask OpenRouter for cost
        self.usage = []  # one entry per generate() call, in call order

    def generate(self, query: str, image: Image.Image, task_type: str) -> str:
        image_data = io.BytesIO()
        image.save(image_data, format="PNG")
        image_b64 = base64.b64encode(image_data.getvalue()).decode('utf-8')

        image_url = f"data:image/png;base64,{image_b64}"
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": query},
                            {"type": "image_url", "image_url": {"url": image_url}}
                        ],
                    }
                ],
                max_tokens=2048,    # changed from 512
                temperature=0.0
            )

            outputs = response.choices[0].message.content or ""
            usage = response.usage
            self.usage.append({
                "prompt_tokens": usage.prompt_tokens if usage else None,
                "completion_tokens": usage.completion_tokens if usage else None,
                "cost": (usage.model_extra or {}).get("cost") if usage else None,
                "host": (response.model_extra or {}).get("provider"),
            })
        except Exception as e:
            outputs = ""
            self.usage.append({"error": repr(e)})  # recorded, not hidden

        # Some hosts put a reasoning model's thinking inline; remove it before scoring
        outputs = re.sub(r"<think>.*?</think>", "", outputs, flags=re.S).strip()

        # Same answer clean-up as OpenAIAdapter.generate()
        if task_type == CAPTION_TASK:
            pattern = re.compile(r"<meta name=\"description\" content=\"(.*)\">")
            cur_meta = re.findall(pattern, outputs)
            if cur_meta:
                return cur_meta[0]
            else:
                return outputs
        elif task_type == ELEMENT_OCR_TASK:
            if ":" not in outputs:
                return outputs
            outputs = ":".join(outputs.split(":")[1:])
            outputs = outputs.strip().strip('"').strip("'")
            return outputs
        else:
            return outputs