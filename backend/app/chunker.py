"""Code chunker for RepoPilot's indexing pipeline.

Splits raw file content into overlapping token-bounded chunks suitable for
embedding and semantic search. Preserves source attribution (file path, line
numbers) on every chunk.

Strategy:
- Uses tiktoken (cl100k_base encoding) to count tokens accurately
- Default chunk size: 400 tokens
- Default overlap: 50 tokens (preserves context across chunk boundaries)
- Markdown files are split on headings (##) for better semantic coherence
"""

from dataclasses import dataclass

import tiktoken

# Tiktoken encoding used for token counting.
# cl100k_base is the encoding used by text-embedding-3-small and GPT-4.
# It gives accurate token counts even when using sentence-transformers for embedding.
_ENCODING = tiktoken.get_encoding("cl100k_base")

# Map file extensions to human-readable language names stored in the DB
EXTENSION_TO_LANGUAGE: dict[str, str] = {
    ".py": "python",
    ".js": "javascript",
    ".ts": "typescript",
    ".jsx": "javascript",
    ".tsx": "typescript",
    ".java": "java",
    ".go": "go",
    ".rb": "ruby",
    ".rs": "rust",
    ".cpp": "cpp",
    ".c": "c",
    ".h": "c",
    ".cs": "csharp",
    ".php": "php",
    ".swift": "swift",
    ".kt": "kotlin",
    ".md": "markdown",
    ".yml": "yaml",
    ".yaml": "yaml",
    ".toml": "toml",
    ".html": "html",
    ".css": "css",
    ".sh": "shell",
    ".bash": "shell",
}


@dataclass
class Chunk:
    """A single code or text chunk ready for embedding.

    Attributes:
        file_path:  Relative path of the source file in the repository.
        start_line: First line number of this chunk (1-indexed).
        end_line:   Last line number of this chunk (1-indexed).
        content:    Raw source text of this chunk.
        language:   Human-readable language name (e.g., 'python', 'typescript').
    """
    file_path: str
    start_line: int
    end_line: int
    content: str
    language: str | None


def _detect_language(file_path: str) -> str | None:
    """Detect the programming language from the file extension.

    Args:
        file_path: Relative path of the file (e.g., 'src/utils/parser.py').

    Returns:
        Human-readable language string or None if extension is not recognized.
    """
    ext = "." + file_path.rsplit(".", 1)[-1].lower() if "." in file_path else ""
    return EXTENSION_TO_LANGUAGE.get(ext)


def _count_tokens(text: str) -> int:
    """Count the number of tokens in a text string using cl100k_base encoding.

    Args:
        text: Any string of text.

    Returns:
        Integer token count.
    """
    return len(_ENCODING.encode(text, disallowed_special=()))


def _chunk_by_tokens(
    lines: list[str],
    file_path: str,
    language: str | None,
    chunk_size: int,
    overlap: int,
) -> list[Chunk]:
    """Split a list of lines into overlapping token-bounded chunks.

    Algorithm:
    1. Walk lines one by one, accumulating them into a window.
    2. When the token count exceeds chunk_size, emit the current window as a chunk.
    3. Roll back 'overlap' tokens worth of lines to create the next window.

    Args:
        lines:      Lines of the source file (preserving newlines).
        file_path:  Relative file path for metadata.
        language:   Detected language name.
        chunk_size: Maximum tokens per chunk.
        overlap:    Number of overlap tokens between consecutive chunks.

    Returns:
        List of Chunk objects.
    """
    chunks: list[Chunk] = []

    current_lines: list[str] = []
    current_tokens: int = 0
    start_line: int = 1  # 1-indexed

    for i, line in enumerate(lines, start=1):
        line_tokens = _count_tokens(line)
        current_lines.append(line)
        current_tokens += line_tokens

        if current_tokens >= chunk_size:
            content = "".join(current_lines).rstrip()
            if content:
                chunks.append(Chunk(
                    file_path=file_path,
                    start_line=start_line,
                    end_line=i,
                    content=content,
                    language=language,
                ))

            # Roll back `overlap` tokens worth of lines for the next window
            overlap_lines: list[str] = []
            overlap_tokens: int = 0
            for line_back in reversed(current_lines):
                t = _count_tokens(line_back)
                if overlap_tokens + t > overlap:
                    break
                overlap_lines.insert(0, line_back)
                overlap_tokens += t

            current_lines = overlap_lines
            current_tokens = overlap_tokens
            # Start line of the new window = (i + 1) - len(overlap_lines)
            start_line = i + 1 - len(overlap_lines)

    # Emit the final chunk (whatever is left over)
    content = "".join(current_lines).rstrip()
    if content:
        chunks.append(Chunk(
            file_path=file_path,
            start_line=start_line,
            end_line=len(lines),
            content=content,
            language=language,
        ))

    return chunks


def _chunk_markdown(
    content: str,
    file_path: str,
    chunk_size: int,
    overlap: int,
) -> list[Chunk]:
    """Split a Markdown file on heading boundaries (## / ###).

    Each section between headings is treated as a semantic unit. Sections that
    are still too large are further split by tokens.

    Args:
        content:    Full markdown text.
        file_path:  Relative file path for metadata.
        chunk_size: Maximum tokens per chunk.
        overlap:    Overlap tokens between chunks.

    Returns:
        List of Chunk objects.
    """
    lines = content.splitlines(keepends=True)
    sections: list[list[str]] = []
    current_section: list[str] = []
    section_start: list[int] = []  # start line index for each section
    current_start = 1

    for i, line in enumerate(lines, start=1):
        stripped = line.lstrip()
        is_heading = stripped.startswith("## ") or stripped.startswith("### ")
        if is_heading and current_section:
            sections.append(current_section)
            section_start.append(current_start)
            current_section = [line]
            current_start = i
        else:
            current_section.append(line)

    if current_section:
        sections.append(current_section)
        section_start.append(current_start)

    chunks: list[Chunk] = []
    language = "markdown"

    for section_lines, s_start in zip(sections, section_start):
        section_text = "".join(section_lines)
        if _count_tokens(section_text) <= chunk_size:
            # Small section — emit as a single chunk
            content_stripped = section_text.rstrip()
            if content_stripped:
                chunks.append(Chunk(
                    file_path=file_path,
                    start_line=s_start,
                    end_line=s_start + len(section_lines) - 1,
                    content=content_stripped,
                    language=language,
                ))
        else:
            # Large section — further split by tokens
            chunks.extend(
                _chunk_by_tokens(section_lines, file_path, language, chunk_size, overlap)
            )

    return chunks


def chunk_file(
    file_path: str,
    content: str,
    chunk_size: int = 400,
    overlap: int = 50,
) -> list[Chunk]:
    """Split a file's content into searchable chunks.

    Dispatches to the appropriate strategy:
    - Markdown files → heading-based splitting
    - All other files → token-based overlapping window

    Args:
        file_path:  Relative path of the file in the repository.
        content:    Full text content of the file.
        chunk_size: Maximum number of tokens per chunk (default: 400).
        overlap:    Overlap tokens between consecutive chunks (default: 50).

    Returns:
        List of Chunk objects. Returns an empty list for empty files.
    """
    if not content or not content.strip():
        return []

    ext = "." + file_path.rsplit(".", 1)[-1].lower() if "." in file_path else ""
    language = _detect_language(file_path)

    if ext == ".md":
        return _chunk_markdown(content, file_path, chunk_size, overlap)

    lines = content.splitlines(keepends=True)
    return _chunk_by_tokens(lines, file_path, language, chunk_size, overlap)
