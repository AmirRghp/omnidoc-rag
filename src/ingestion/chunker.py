"""Header-aware and code-safe semantic Markdown chunker for bilingual technical documents."""

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import tiktoken

from src.common.exceptions import DocumentParsingError
from src.common.schemas import DocumentChunkPayload
from src.common.security import normalize_persian_text, strip_directional_overrides

DEFAULT_MAX_TOKENS: int = 512
DEFAULT_OVERLAP_TOKENS: int = 50
DEFAULT_ENCODING_NAME: str = "cl100k_base"
DEFAULT_SECTION_HEADING: str = "# Introduction"
PERSIAN_SCRIPT_RATIO_THRESHOLD: float = 0.25

_HEADER_PATTERN: re.Pattern[str] = re.compile(r"^\s{0,3}(#{1,3})\s+(.+?)\s*$")
_FENCE_PATTERN: re.Pattern[str] = re.compile(r"^\s{0,3}(`{3,}|~{3,})")
_HORIZONTAL_RULE_PATTERN: re.Pattern[str] = re.compile(
    r"^\s{0,3}(?:-\s*){3,}$|^\s{0,3}(?:\*\s*){3,}$|^\s{0,3}(?:_\s*){3,}$"
)
_FRONTMATTER_PATTERN: re.Pattern[str] = re.compile(r"\A---\r?\n([\s\S]*?)\r?\n---\r?\n?")
_FRONTMATTER_TITLE_PATTERN: re.Pattern[str] = re.compile(
    r"^title:\s*['\"]?(.+?)['\"]?\s*$", re.MULTILINE
)
_CODE_BLOCK_STRIP_PATTERN: re.Pattern[str] = re.compile(r"(`{3,}|~{3,})[\s\S]*?\1|`[^`\n]+`")
_PERSIAN_ALPHA_PATTERN: re.Pattern[str] = re.compile(r"[\u0600-\u06FF\uFB50-\uFDFF\uFE70-\uFEFF]")
_LATIN_ALPHA_PATTERN: re.Pattern[str] = re.compile(r"[A-Za-z]")
_FALLBACK_TOKEN_PATTERN: re.Pattern[str] = re.compile(r"\w+|[^\w\s]", re.UNICODE)
_SENTENCE_SPLIT_PATTERN: re.Pattern[str] = re.compile(r"(?<=[.!؟\n])\s+")


@dataclass(frozen=True)
class _AtomicBlock:
    """Internal representation of an indivisible or paragraph-level Markdown block."""

    kind: Literal["prose", "code", "table"]
    text: str


@dataclass
class _Section:
    """Internal representation of a Markdown section under a specific heading."""

    heading: str
    blocks: list[_AtomicBlock]


class MarkdownSemanticChunker:
    """Splits Markdown documents by headers while keeping code blocks and tables intact."""

    def __init__(
        self,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        overlap_tokens: int = DEFAULT_OVERLAP_TOKENS,
        encoding_name: str = DEFAULT_ENCODING_NAME,
    ) -> None:
        if max_tokens <= 0:
            raise ValueError(f"max_tokens must be positive, got {max_tokens}")
        if overlap_tokens < 0:
            raise ValueError(f"overlap_tokens must be non-negative, got {overlap_tokens}")
        self.max_tokens = max_tokens
        self.overlap_tokens = min(overlap_tokens, max(0, max_tokens // 5))
        self.encoding_name = encoding_name
        self._encoding: tiktoken.Encoding | None = self._load_encoding(encoding_name)

    @staticmethod
    def _load_encoding(encoding_name: str) -> tiktoken.Encoding | None:
        """Load tiktoken encoding with graceful offline fallback if network is unavailable."""
        try:
            return tiktoken.get_encoding(encoding_name)
        except Exception:
            return None

    def count_tokens(self, text: str) -> int:
        """Calculate token count for a string using tiktoken."""
        if not text:
            return 0
        if self._encoding is not None:
            return len(self._encoding.encode(text, disallowed_special=()))
        return len(_FALLBACK_TOKEN_PATTERN.findall(text))

    @staticmethod
    def detect_language(text: str) -> Literal["fa", "en"]:
        """Detect primary chunk language ('fa' vs 'en') based on character script ratio."""
        prose_text = _CODE_BLOCK_STRIP_PATTERN.sub(" ", text)
        persian_count = len(_PERSIAN_ALPHA_PATTERN.findall(prose_text))
        latin_count = len(_LATIN_ALPHA_PATTERN.findall(prose_text))
        total_alpha = persian_count + latin_count

        if total_alpha == 0:
            persian_count = len(_PERSIAN_ALPHA_PATTERN.findall(text))
            latin_count = len(_LATIN_ALPHA_PATTERN.findall(text))
            total_alpha = persian_count + latin_count

        if total_alpha == 0:
            return "en"

        persian_ratio = persian_count / total_alpha
        return "fa" if persian_ratio >= PERSIAN_SCRIPT_RATIO_THRESHOLD else "en"

    def chunk_text(
        self,
        text: str,
        source_file: str,
        doc_id: str | None = None,
    ) -> list[DocumentChunkPayload]:
        """Parse and split raw Markdown text into validated DocumentChunkPayload objects."""
        if not isinstance(text, str):
            raise DocumentParsingError(
                message="Document content must be a string",
                detail="Received non-string content during Markdown chunking.",
            )
        if not isinstance(source_file, str) or not source_file.strip():
            raise DocumentParsingError(
                message="source_file must be a non-empty string",
                detail="Cannot chunk document without a valid source_file path.",
            )

        cleaned_text = normalize_persian_text(strip_directional_overrides(text))
        if not cleaned_text.strip():
            raise DocumentParsingError(
                message="Document content is empty",
                detail=f"File '{source_file}' contains no parseable content.",
            )

        clean_source = source_file.strip()
        resolved_doc_id = doc_id.strip() if doc_id and doc_id.strip() else Path(clean_source).stem
        if not resolved_doc_id:
            resolved_doc_id = clean_source

        body_text, initial_heading = self._extract_frontmatter(cleaned_text)
        sections = self._parse_sections(body_text, initial_heading=initial_heading)

        chunks: list[DocumentChunkPayload] = []
        chunk_index = 0

        for section in sections:
            packed_texts = self._pack_section_blocks(section.blocks)
            for chunk_content in packed_texts:
                stripped_content = chunk_content.strip()
                if not stripped_content:
                    continue
                token_count = max(1, self.count_tokens(stripped_content))
                language = self.detect_language(
                    f"{section.heading}\n{stripped_content}"
                    if self._is_pure_code_or_table(stripped_content)
                    else stripped_content
                )
                payload = DocumentChunkPayload(
                    doc_id=resolved_doc_id,
                    chunk_id=chunk_index,
                    content=stripped_content,
                    source_file=clean_source,
                    section_heading=section.heading,
                    language=language,
                    token_count=token_count,
                )
                chunks.append(payload)
                chunk_index += 1

        if not chunks:
            raise DocumentParsingError(
                message="No valid chunks produced from document",
                detail=f"Document '{clean_source}' yielded zero content chunks after parsing.",
            )

        return chunks

    def chunk_file(
        self,
        file_path: Path | str,
        root_dir: Path | str | None = None,
        doc_id: str | None = None,
    ) -> list[DocumentChunkPayload]:
        """Read a Markdown file from disk and split it into DocumentChunkPayload objects."""
        path = Path(file_path)
        if not path.is_file():
            raise DocumentParsingError(
                message=f"Document file not found: {path}",
                detail=f"File '{path}' does not exist or is not a regular file.",
            )

        try:
            raw_text = path.read_text(encoding="utf-8")
        except Exception as exc:
            raise DocumentParsingError(
                message=f"Failed to read document file: {path}",
                detail=str(exc),
            ) from exc

        if root_dir is not None:
            try:
                source_file = path.resolve().relative_to(Path(root_dir).resolve()).as_posix()
            except ValueError:
                source_file = path.name
        else:
            resolved = path.resolve()
            raw_root = Path("data/raw").resolve()
            cwd = Path.cwd().resolve()
            if resolved.is_relative_to(raw_root):
                source_file = resolved.relative_to(raw_root).as_posix()
            elif resolved.is_relative_to(cwd):
                source_file = resolved.relative_to(cwd).as_posix()
            else:
                source_file = path.name

        return self.chunk_text(
            text=raw_text,
            source_file=source_file,
            doc_id=doc_id or path.stem,
        )

    @staticmethod
    def _extract_frontmatter(text: str) -> tuple[str, str]:
        """Strip optional YAML frontmatter and extract title if present."""
        match = _FRONTMATTER_PATTERN.match(text)
        if not match:
            return text, DEFAULT_SECTION_HEADING

        frontmatter_content = match.group(1)
        remaining_text = text[match.end() :]
        title_match = _FRONTMATTER_TITLE_PATTERN.search(frontmatter_content)
        if title_match:
            title = title_match.group(1).strip()
            if title:
                return remaining_text, f"# {title}"

        return remaining_text, DEFAULT_SECTION_HEADING

    @staticmethod
    def _is_table_line(line: str, in_table: bool) -> bool:
        """Determine whether a line belongs to a Markdown table."""
        stripped = line.strip()
        if not stripped:
            return False
        if stripped.startswith("|") and stripped.endswith("|") and len(stripped) > 1:
            return True
        return bool(in_table and "|" in stripped and not stripped.startswith("#"))

    @staticmethod
    def _is_pure_code_or_table(text: str) -> bool:
        """Check if chunk text has no prose outside fenced code blocks."""
        stripped_prose = _CODE_BLOCK_STRIP_PATTERN.sub("", text).strip()
        return len(stripped_prose) == 0

    def _parse_sections(self, text: str, initial_heading: str) -> list[_Section]:
        """Parse Markdown text into sections by '#', '##', '###' outside code blocks."""
        sections: list[_Section] = []
        current_section = _Section(heading=initial_heading, blocks=[])

        in_code_block = False
        fence_char = ""
        fence_len = 0

        code_lines: list[str] = []
        table_lines: list[str] = []
        prose_lines: list[str] = []

        def flush_prose() -> None:
            if prose_lines:
                content = "\n".join(prose_lines).strip()
                if content:
                    current_section.blocks.append(_AtomicBlock(kind="prose", text=content))
                prose_lines.clear()

        def flush_table() -> None:
            if table_lines:
                content = "\n".join(table_lines).strip()
                if content:
                    current_section.blocks.append(_AtomicBlock(kind="table", text=content))
                table_lines.clear()

        def flush_code() -> None:
            if code_lines:
                content = "\n".join(code_lines).strip()
                if content:
                    current_section.blocks.append(_AtomicBlock(kind="code", text=content))
                code_lines.clear()

        for line in text.splitlines():
            if in_code_block:
                code_lines.append(line)
                stripped = line.strip()
                if (
                    stripped.startswith(fence_char * fence_len)
                    and not stripped.lstrip(fence_char).strip()
                ):
                    in_code_block = False
                    fence_char = ""
                    fence_len = 0
                    flush_code()
                continue

            fence_match = _FENCE_PATTERN.match(line)
            if fence_match:
                flush_prose()
                flush_table()
                marker = fence_match.group(1)
                in_code_block = True
                fence_char = marker[0]
                fence_len = len(marker)
                code_lines.append(line)
                continue

            header_match = _HEADER_PATTERN.match(line)
            if header_match:
                flush_prose()
                flush_table()
                hashes, heading_text = header_match.groups()
                normalized_heading = f"{hashes} {heading_text.strip()}"
                if current_section.blocks:
                    sections.append(current_section)
                current_section = _Section(heading=normalized_heading, blocks=[])
                continue

            if _HORIZONTAL_RULE_PATTERN.match(line):
                flush_prose()
                flush_table()
                continue

            if not line.strip():
                flush_prose()
                flush_table()
                continue

            if self._is_table_line(line, in_table=bool(table_lines)):
                flush_prose()
                table_lines.append(line)
                continue

            flush_table()
            prose_lines.append(line)

        flush_code()
        flush_prose()
        flush_table()

        if current_section.blocks:
            sections.append(current_section)
        elif not sections and current_section.heading != DEFAULT_SECTION_HEADING:
            sections.append(
                _Section(
                    heading=current_section.heading,
                    blocks=[_AtomicBlock(kind="prose", text=current_section.heading)],
                )
            )

        return sections

    def _pack_section_blocks(self, blocks: list[_AtomicBlock]) -> list[str]:
        """Combine atomic blocks within a section up to max_tokens without breaking code/tables."""
        expanded_blocks: list[_AtomicBlock] = []
        for block in blocks:
            if block.kind == "prose" and self.count_tokens(block.text) > self.max_tokens:
                for piece in self._split_oversized_prose(block.text):
                    expanded_blocks.append(_AtomicBlock(kind="prose", text=piece))
            else:
                expanded_blocks.append(block)

        chunks: list[str] = []
        current_parts: list[str] = []
        current_tokens = 0

        for block in expanded_blocks:
            block_tokens = self.count_tokens(block.text)
            separator_tokens = 1 if current_parts else 0

            if current_parts and (
                current_tokens + separator_tokens + block_tokens > self.max_tokens
            ):
                chunks.append("\n\n".join(current_parts))
                current_parts = [block.text]
                current_tokens = block_tokens
            else:
                current_parts.append(block.text)
                current_tokens += separator_tokens + block_tokens

        if current_parts:
            chunks.append("\n\n".join(current_parts))

        return chunks

    def _split_oversized_prose(self, text: str) -> list[str]:
        """Split a single prose paragraph exceeding max_tokens along sentence or word boundaries."""
        sentences = [s.strip() for s in _SENTENCE_SPLIT_PATTERN.split(text) if s.strip()]
        if len(sentences) <= 1:
            return self._split_by_words(text)

        pieces: list[str] = []
        current_sentences: list[str] = []
        current_tokens = 0

        for sentence in sentences:
            sentence_tokens = self.count_tokens(sentence)
            if sentence_tokens > self.max_tokens:
                if current_sentences:
                    pieces.append(" ".join(current_sentences))
                    current_sentences = []
                    current_tokens = 0
                pieces.extend(self._split_by_words(sentence))
                continue

            if current_sentences and (current_tokens + sentence_tokens > self.max_tokens):
                pieces.append(" ".join(current_sentences))
                overlap_list: list[str] = []
                overlap_toks = 0
                if self.overlap_tokens > 0:
                    for prev_sent in reversed(current_sentences):
                        prev_toks = self.count_tokens(prev_sent)
                        if (
                            overlap_toks + prev_toks <= self.overlap_tokens
                            and prev_toks + sentence_tokens <= self.max_tokens
                        ):
                            overlap_list.insert(0, prev_sent)
                            overlap_toks += prev_toks
                        else:
                            break
                current_sentences = [*overlap_list, sentence]
                current_tokens = self.count_tokens(" ".join(current_sentences))
            else:
                current_sentences.append(sentence)
                current_tokens += sentence_tokens

        if current_sentences:
            pieces.append(" ".join(current_sentences))

        return pieces

    def _split_by_words(self, text: str) -> list[str]:
        """Split oversized sentence by whitespace words to respect max_tokens."""
        words = text.split()
        pieces: list[str] = []
        current_words: list[str] = []

        for word in words:
            candidate = " ".join([*current_words, word])
            if current_words and self.count_tokens(candidate) > self.max_tokens:
                pieces.append(" ".join(current_words))
                current_words = [word]
            else:
                current_words.append(word)

        if current_words:
            pieces.append(" ".join(current_words))

        return pieces
