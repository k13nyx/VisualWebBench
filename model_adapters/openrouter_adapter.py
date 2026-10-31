from openai import OpenAI
from PIL import Image
import base64
import io, re

from model_adapters.base_adapter import BaseAdapter
from utils.constants import *


class OpenRouterAdapter(BaseAdapter):
    """OpenAIAdapter for any OpenAI-compatible API (e.g. OpenRouter) that also records tokens and cost."""

    def __init__(
        self, base_url: str, api_key: str, model: str,
        max_tokens: int = 2048, temperature: float = 0.0,
        extra_body=None
    ):
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.model = model
        
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.extra_body = {"usage": {"include": True}, **(extra_body or {})}  # ask OpenRouter for cost

    def generate(self, query: str, image: Image.Image, task_type: str):
        image_data = io.BytesIO()
        image.save(image_data, format="PNG")
        image_b64 = base64.b64encode(image_data.getvalue()).decode('utf-8')

        image_url = f"data:image/png;base64,{image_b64}"

        usage_info = {}
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
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                extra_body=self.extra_body,  # request usage/cost from OpenRouter
            )

            outputs = response.choices[0].message.content or ""
            usage = response.usage
            usage_info.update({
                "prompt_tokens": usage.prompt_tokens if usage else None,
                "completion_tokens": usage.completion_tokens if usage else None,
                "cost_reported": (usage.model_extra or {}).get("cost") if usage else None,
                "host": (response.model_extra or {}).get("provider"),
            })
        except Exception as e:
            outputs = ""
            usage_info["error"] = repr(e)

        raw = outputs
        # Some hosts put a reasoning model's thinking inline; remove it before scoring
        outputs = re.sub(r"<think>.*?</think>", "", outputs, flags=re.S).strip()

        # Same answer clean-up as OpenAIAdapter.generate()
        if task_type == CAPTION_TASK:
            pattern = re.compile(r"<meta name=\"description\" content=\"(.*)\">")
            cur_meta = re.findall(pattern, outputs)
            outputs = cur_meta[0] if cur_meta else outputs
        elif task_type == ELEMENT_OCR_TASK:
            if ":" in outputs:
                outputs = ":".join(outputs.split(":")[1:])
                outputs = outputs.strip().strip('"').strip("'")

        usage_info["empty"] = not outputs.strip()
        usage_info["raw"] = raw
        return outputs, usage_info