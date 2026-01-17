from __future__ import annotations

from typing import Any, Dict, List, Tuple

from src.logging import get_logger


logger = get_logger("RAGMultimodalUtils")


def separate_content(
    content_list: List[Dict[str, Any]],
) -> Tuple[str, List[Dict[str, Any]]]:
    text_parts: List[str] = []
    multimodal_items: List[Dict[str, Any]] = []

    for item in content_list:
        content_type = item.get("type", "text")
        if content_type == "text":
            text = item.get("text", "")
            if text.strip():
                text_parts.append(text)
        else:
            multimodal_items.append(item)

    text_content = "\n\n".join(text_parts)
    logger.info(
        f"Content separation: text_len={len(text_content)}, multimodal_items={len(multimodal_items)}"
    )
    return text_content, multimodal_items

