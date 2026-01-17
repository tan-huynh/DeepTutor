from __future__ import annotations

import base64
import json
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from src.services.rag.types import Document

from .base import BaseParser, ParseResult


class DoclingParser(BaseParser):
    """
    Docling parser for multiple formats.

    Supported formats:
    - Documents: .pdf, .doc, .docx, .ppt, .pptx, .xls, .xlsx
    - Web/Text: .html, .htm, .xhtml, .md, .csv, .json
    - Images: .png, .jpg, .jpeg, .tiff, .tif
    - Audio: .mp3, .wav
    """

    name = "docling"
    supported_extensions = (
        ".pdf",
        ".doc",
        ".docx",
        ".ppt",
        ".pptx",
        ".xls",
        ".xlsx",
        ".html",
        ".htm",
        ".xhtml",
        ".md",
        ".csv",
        ".json",
        ".png",
        ".jpg",
        ".jpeg",
        ".tiff",
        ".tif",
        ".mp3",
        ".wav",
    )

    OFFICE_FORMATS = {".doc", ".docx", ".ppt", ".pptx", ".xls", ".xlsx"}
    HTML_FORMATS = {".html", ".htm", ".xhtml"}

    def parse(self, file_path: str | Path, output_dir: Optional[str] = None) -> ParseResult:
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        content_list = self.parse_document(file_path, output_dir=output_dir)
        text = self._content_list_to_text(content_list)
        document = Document(
            content=text,
            file_path=str(file_path),
            metadata={"file_name": file_path.name},
            content_items=content_list,
        )
        return ParseResult(document=document, content_list=content_list, markdown=text)

    def parse_document(
        self,
        file_path: Union[str, Path],
        output_dir: Optional[str] = None,
        **kwargs,
    ) -> List[Dict[str, Any]]:
        file_path = Path(file_path)
        ext = file_path.suffix.lower()

        if ext == ".pdf":
            return self.parse_pdf(file_path, output_dir=output_dir, **kwargs)
        if ext in self.OFFICE_FORMATS:
            return self.parse_office_doc(file_path, output_dir=output_dir, **kwargs)
        if ext in self.HTML_FORMATS:
            return self.parse_html(file_path, output_dir=output_dir, **kwargs)
        if ext in self.supported_extensions:
            return self.parse_generic(file_path, output_dir=output_dir, **kwargs)
        raise ValueError(f"Unsupported file format: {ext}")

    def parse_pdf(
        self,
        pdf_path: Union[str, Path],
        output_dir: Optional[str] = None,
        **kwargs,
    ) -> List[Dict[str, Any]]:
        pdf_path = Path(pdf_path)
        base_output_dir = Path(output_dir) if output_dir else pdf_path.parent / "docling_output"
        base_output_dir.mkdir(parents=True, exist_ok=True)

        self._run_docling_command(
            input_path=pdf_path,
            output_dir=base_output_dir,
            file_stem=pdf_path.stem,
            **kwargs,
        )
        content_list, _ = self._read_output_files(base_output_dir, pdf_path.stem)
        return content_list

    def parse_office_doc(
        self,
        doc_path: Union[str, Path],
        output_dir: Optional[str] = None,
        **kwargs,
    ) -> List[Dict[str, Any]]:
        doc_path = Path(doc_path)
        base_output_dir = Path(output_dir) if output_dir else doc_path.parent / "docling_output"
        base_output_dir.mkdir(parents=True, exist_ok=True)

        self._run_docling_command(
            input_path=doc_path,
            output_dir=base_output_dir,
            file_stem=doc_path.stem,
            **kwargs,
        )
        content_list, _ = self._read_output_files(base_output_dir, doc_path.stem)
        return content_list

    def parse_html(
        self,
        html_path: Union[str, Path],
        output_dir: Optional[str] = None,
        **kwargs,
    ) -> List[Dict[str, Any]]:
        html_path = Path(html_path)
        base_output_dir = Path(output_dir) if output_dir else html_path.parent / "docling_output"
        base_output_dir.mkdir(parents=True, exist_ok=True)

        self._run_docling_command(
            input_path=html_path,
            output_dir=base_output_dir,
            file_stem=html_path.stem,
            **kwargs,
        )
        content_list, _ = self._read_output_files(base_output_dir, html_path.stem)
        return content_list

    def parse_generic(
        self,
        file_path: Union[str, Path],
        output_dir: Optional[str] = None,
        **kwargs,
    ) -> List[Dict[str, Any]]:
        file_path = Path(file_path)
        base_output_dir = Path(output_dir) if output_dir else file_path.parent / "docling_output"
        base_output_dir.mkdir(parents=True, exist_ok=True)

        self._run_docling_command(
            input_path=file_path,
            output_dir=base_output_dir,
            file_stem=file_path.stem,
            **kwargs,
        )
        content_list, _ = self._read_output_files(base_output_dir, file_path.stem)
        return content_list

    def _run_docling_command(
        self,
        input_path: Union[str, Path],
        output_dir: Union[str, Path],
        file_stem: str,
        **kwargs,
    ) -> None:
        file_output_dir = Path(output_dir) / file_stem / "docling"
        file_output_dir.mkdir(parents=True, exist_ok=True)

        cmd_json = ["docling", "--output", str(file_output_dir), "--to", "json", str(input_path)]
        cmd_md = ["docling", "--output", str(file_output_dir), "--to", "md", str(input_path)]

        try:
            subprocess.run(
                cmd_json,
                capture_output=True,
                text=True,
                check=True,
                encoding="utf-8",
                errors="ignore",
            )
            subprocess.run(
                cmd_md,
                capture_output=True,
                text=True,
                check=True,
                encoding="utf-8",
                errors="ignore",
            )
        except FileNotFoundError as exc:
            raise RuntimeError("docling command not found. Please install docling.") from exc
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(f"Docling failed: {exc.stderr}") from exc

    def _read_output_files(
        self,
        output_dir: Path,
        file_stem: str,
    ) -> Tuple[List[Dict[str, Any]], str]:
        file_subdir = output_dir / file_stem / "docling"
        md_file = file_subdir / f"{file_stem}.md"
        json_file = file_subdir / f"{file_stem}.json"

        markdown = ""
        if md_file.exists():
            markdown = md_file.read_text(encoding="utf-8")

        content_list: List[Dict[str, Any]] = []
        if json_file.exists():
            docling_content = json.loads(json_file.read_text(encoding="utf-8"))
            content_list = self.read_from_block_recursive(
                docling_content["body"],
                "body",
                file_subdir,
                0,
                "0",
                docling_content,
            )

        return content_list, markdown

    def read_from_block_recursive(
        self,
        block: Dict[str, Any],
        block_type: str,
        output_dir: Path,
        cnt: int,
        num: str,
        docling_content: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        content_list: List[Dict[str, Any]] = []
        if not block.get("children"):
            cnt += 1
            content_list.append(self.read_from_block(block, block_type, output_dir, cnt, num))
        else:
            if block_type not in ["groups", "body"]:
                cnt += 1
                content_list.append(
                    self.read_from_block(block, block_type, output_dir, cnt, num)
                )
            members = block["children"]
            for member in members:
                cnt += 1
                member_tag = member["$ref"]
                member_type = member_tag.split("/")[1]
                member_num = member_tag.split("/")[2]
                member_block = docling_content[member_type][int(member_num)]
                content_list.extend(
                    self.read_from_block_recursive(
                        member_block,
                        member_type,
                        output_dir,
                        cnt,
                        member_num,
                        docling_content,
                    )
                )
        return content_list

    def read_from_block(
        self, block: Dict[str, Any], block_type: str, output_dir: Path, cnt: int, num: str
    ) -> Dict[str, Any]:
        if block_type == "texts":
            if block.get("label") == "formula":
                return {
                    "type": "equation",
                    "img_path": "",
                    "text": block.get("orig", ""),
                    "text_format": "unknown",
                    "page_idx": cnt // 10,
                }
            return {
                "type": "text",
                "text": block.get("orig", ""),
                "page_idx": cnt // 10,
            }
        if block_type == "pictures":
            try:
                base64_uri = block["image"]["uri"]
                base64_str = base64_uri.split(",")[1]
                image_dir = output_dir / "images"
                image_dir.mkdir(parents=True, exist_ok=True)
                image_path = image_dir / f"image_{num}.png"
                image_path.write_bytes(base64.b64decode(base64_str))
                return {
                    "type": "image",
                    "img_path": str(image_path.resolve()),
                    "image_caption": block.get("caption", ""),
                    "image_footnote": block.get("footnote", ""),
                    "page_idx": cnt // 10,
                }
            except Exception:
                return {
                    "type": "text",
                    "text": f"[Image processing failed: {block.get('caption', '')}]",
                    "page_idx": cnt // 10,
                }
        try:
            return {
                "type": "table",
                "img_path": "",
                "table_caption": block.get("caption", ""),
                "table_footnote": block.get("footnote", ""),
                "table_body": block.get("data", []),
                "page_idx": cnt // 10,
            }
        except Exception:
            return {
                "type": "text",
                "text": f"[Table processing failed: {block.get('caption', '')}]",
                "page_idx": cnt // 10,
            }

    @staticmethod
    def _content_list_to_text(content_list: List[Dict[str, Any]]) -> str:
        chunks: List[str] = []
        for item in content_list:
            if not isinstance(item, dict):
                continue
            if item.get("type") == "text" and item.get("text"):
                chunks.append(item["text"])
        return "\n".join(chunks)

