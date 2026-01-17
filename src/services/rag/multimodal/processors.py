from __future__ import annotations

import base64
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.logging import get_logger

from .prompts import PROMPTS
from .utils import separate_content


class MultimodalProcessor:
    def __init__(
        self,
        llm_func,
        vision_func=None,
        enable_image: bool = True,
        enable_table: bool = True,
        enable_equation: bool = True,
    ) -> None:
        self.logger = get_logger("RAGMultimodalProcessor")
        self.llm_func = llm_func
        self.vision_func = vision_func or llm_func
        self.enable_image = enable_image
        self.enable_table = enable_table
        self.enable_equation = enable_equation

    async def process_content_list(self, content_list: List[Dict[str, Any]]) -> List[str]:
        text_content, multimodal_items = separate_content(content_list)
        chunks: List[str] = []
        if text_content.strip():
            chunks.append(text_content)

        for item in multimodal_items:
            item_type = item.get("type")
            if item_type == "image" and self.enable_image:
                chunks.append(await self._process_image(item))
            elif item_type == "table" and self.enable_table:
                chunks.append(await self._process_table(item))
            elif item_type == "equation" and self.enable_equation:
                chunks.append(await self._process_equation(item))

        return [chunk for chunk in chunks if chunk.strip()]

    async def _process_image(self, item: Dict[str, Any]) -> str:
        prompt = PROMPTS["vision_prompt"].format(
            entity_name=item.get("image_caption", "image"),
            image_path=item.get("img_path", ""),
            captions=item.get("image_caption", ""),
            footnotes=item.get("image_footnote", ""),
        )
        image_path = item.get("img_path")
        image_b64 = ""
        if image_path:
            try:
                image_b64 = base64.b64encode(Path(image_path).read_bytes()).decode("utf-8")
            except Exception:
                image_b64 = ""
        if image_b64:
            messages = [
                {"role": "system", "content": PROMPTS["IMAGE_ANALYSIS_SYSTEM"]},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{image_b64}"},
                        },
                    ],
                },
            ]
            description = await self.vision_func(prompt="", messages=messages)
        else:
            description = await self.vision_func(prompt, system_prompt=PROMPTS["IMAGE_ANALYSIS_SYSTEM"])
        return PROMPTS["image_chunk"].format(
            image_path=item.get("img_path", ""),
            captions=item.get("image_caption", ""),
            footnotes=item.get("image_footnote", ""),
            enhanced_caption=description,
        )

    async def _process_table(self, item: Dict[str, Any]) -> str:
        prompt = PROMPTS["table_prompt"].format(
            entity_name=item.get("table_caption", "table"),
            table_img_path=item.get("table_img_path", item.get("img_path", "")),
            table_caption=item.get("table_caption", ""),
            table_body=item.get("table_body", ""),
            table_footnote=item.get("table_footnote", ""),
        )
        description = await self.llm_func(prompt, system_prompt=PROMPTS["TABLE_ANALYSIS_SYSTEM"])
        return PROMPTS["table_chunk"].format(
            table_img_path=item.get("table_img_path", item.get("img_path", "")),
            table_caption=item.get("table_caption", ""),
            table_body=item.get("table_body", ""),
            table_footnote=item.get("table_footnote", ""),
            enhanced_caption=description,
        )

    async def _process_equation(self, item: Dict[str, Any]) -> str:
        prompt = PROMPTS["equation_prompt"].format(
            entity_name=item.get("text", "equation"),
            equation_text=item.get("text", ""),
            equation_format=item.get("text_format", "latex"),
        )
        description = await self.llm_func(prompt, system_prompt=PROMPTS["EQUATION_ANALYSIS_SYSTEM"])
        return PROMPTS["equation_chunk"].format(
            equation_text=item.get("text", ""),
            equation_format=item.get("text_format", "latex"),
            enhanced_caption=description,
        )

