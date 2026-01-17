from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from src.services.rag.types import Document

from .base import BaseParser, ParseResult


class MineruParser(BaseParser):
    name = "mineru"
    supported_extensions = (".pdf",)

    def parse(self, file_path: str | Path, output_dir: Optional[str] = None) -> ParseResult:
        file_path = Path(file_path)
        self.logger.info(f"Parsing file: {file_path}")

        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        if file_path.suffix.lower() not in self.supported_extensions:
            raise ValueError(f"Unsupported file type: {file_path.suffix}")

        base_output_dir = Path(output_dir) if output_dir else file_path.parent / "mineru_output"
        base_output_dir.mkdir(parents=True, exist_ok=True)

        try:
            self._run_mineru_command(input_path=file_path, output_dir=base_output_dir)
        except RuntimeError as e:
            self.logger.warning(f"MinerU failed: {e}. Falling back to basic PDF extraction.")
            return self._fallback_parse(file_path)

        content_list, markdown = self._read_output_files(
            output_dir=base_output_dir, file_stem=file_path.stem
        )
        text = self._content_list_to_text(content_list) or markdown
        document = Document(
            content=text,
            file_path=str(file_path),
            metadata={"file_name": file_path.name},
            content_items=content_list,
        )
        return ParseResult(document=document, content_list=content_list, markdown=markdown)

    def _fallback_parse(self, file_path: Path) -> ParseResult:
        """Fallback PDF parsing using PyMuPDF when MinerU is not available."""
        try:
            import fitz  # PyMuPDF
            text_chunks = []
            with fitz.open(file_path) as doc:
                for page in doc:
                    text_chunks.append(page.get_text())
            text = "\n".join(text_chunks)
            self.logger.info(f"Fallback extraction: {len(text)} characters from {file_path.name}")
        except ImportError:
            self.logger.warning("PyMuPDF not installed. Cannot extract PDF text.")
            text = f"[PDF file: {file_path.name} - content extraction failed]"
        except Exception as e:
            self.logger.warning(f"Fallback PDF extraction failed: {e}")
            text = f"[PDF file: {file_path.name} - content extraction failed: {e}]"

        document = Document(
            content=text,
            file_path=str(file_path),
            metadata={"file_name": file_path.name},
        )
        return ParseResult(document=document, content_list=[], markdown=text)

    def _run_mineru_command(
        self,
        input_path: Union[str, Path],
        output_dir: Union[str, Path],
        method: str = "auto",
        lang: Optional[str] = None,
        backend: Optional[str] = None,
        start_page: Optional[int] = None,
        end_page: Optional[int] = None,
        formula: bool = True,
        table: bool = True,
        device: Optional[str] = None,
        source: Optional[str] = None,
        vlm_url: Optional[str] = None,
    ) -> None:
        cmd = [
            "mineru",
            "-p",
            str(input_path),
            "-o",
            str(output_dir),
            "-m",
            method,
        ]
        if backend:
            cmd.extend(["-b", backend])
        if source:
            cmd.extend(["--source", source])
        if lang:
            cmd.extend(["-l", lang])
        if start_page is not None:
            cmd.extend(["-s", str(start_page)])
        if end_page is not None:
            cmd.extend(["-e", str(end_page)])
        if not formula:
            cmd.extend(["-f", "false"])
        if not table:
            cmd.extend(["-t", "false"])
        if device:
            cmd.extend(["-d", device])
        if vlm_url:
            cmd.extend(["-u", vlm_url])

        try:
            self.logger.info(f"Executing mineru: {' '.join(cmd)}")
            subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True,
                encoding="utf-8",
                errors="ignore",
            )
        except FileNotFoundError as exc:
            raise RuntimeError(
                "mineru command not found. Please install MinerU: pip install -U 'mineru[core]'"
            ) from exc
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(f"MinerU failed: {exc.stderr}") from exc

    def _read_output_files(
        self, output_dir: Path, file_stem: str, method: str = "auto"
    ) -> Tuple[List[Dict[str, Any]], str]:
        md_file = output_dir / f"{file_stem}.md"
        json_file = output_dir / f"{file_stem}_content_list.json"
        images_base_dir = output_dir

        file_stem_subdir = output_dir / file_stem
        if file_stem_subdir.is_dir():
            found = False
            for subdir in file_stem_subdir.iterdir():
                if not subdir.is_dir():
                    continue
                candidate_json = subdir / f"{file_stem}_content_list.json"
                if candidate_json.exists():
                    md_file = subdir / f"{file_stem}.md"
                    json_file = candidate_json
                    images_base_dir = subdir
                    found = True
                    break

            if not found:
                md_file = file_stem_subdir / method / f"{file_stem}.md"
                json_file = file_stem_subdir / method / f"{file_stem}_content_list.json"
                images_base_dir = file_stem_subdir / method

        markdown = ""
        if md_file.exists():
            markdown = md_file.read_text(encoding="utf-8")

        content_list: List[Dict[str, Any]] = []
        if json_file.exists():
            content_list = json.loads(json_file.read_text(encoding="utf-8"))
            for item in content_list:
                if not isinstance(item, dict):
                    continue
                for field_name in ["img_path", "table_img_path", "equation_img_path"]:
                    if item.get(field_name):
                        absolute_img_path = (images_base_dir / item[field_name]).resolve()
                        item[field_name] = str(absolute_img_path)

        return content_list, markdown

    @staticmethod
    def _content_list_to_text(content_list: List[Dict[str, Any]]) -> str:
        chunks: List[str] = []
        for item in content_list:
            if not isinstance(item, dict):
                continue
            if item.get("type") == "text" and item.get("text"):
                chunks.append(item["text"])
        return "\n".join(chunks)

