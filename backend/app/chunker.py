"""Code chunker for RepoPilot's indexing pipeline.

Splits raw file content into overlapping token-bounded chunks suitable for
embedding and semantic search. Preserves source attribution (file path, line
numbers) on every chunk.

Strategy:
- Uses tiktoken (cl100k_base encoding) to count tokens accurately
- Default chunk size: 400 tokens (configurable via CHUNK_SIZE_TOKENS env var)
- Default overlap: 50 tokens (configurable via CHUNK_OVERLAP_TOKENS env var)
- Markdown files are split on headings (##) for better semantic coherence
"""

import logging
import os
from dataclasses import dataclass

import tiktoken

logger = logging.getLogger(__name__)

# Tiktoken encoding used for token counting.
# cl100k_base is the encoding used by text-embedding-3-small and GPT-4.
# It gives accurate token counts even when using sentence-transformers for embedding.
_ENCODING = tiktoken.get_encoding("cl100k_base")

# Chunk size and overlap are read from the environment at module load time,
# matching the values defined in .env (CHUNK_SIZE_TOKENS / CHUNK_OVERLAP_TOKENS).
# Hardcoded fallbacks (400 / 50) are used when the env vars are absent.
DEFAULT_CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE_TOKENS", "200"))
DEFAULT_CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP_TOKENS", "40"))

# Chunking strategy: 'ast' for syntax-tree aware splitting, 'tokens' for sliding window.
DEFAULT_CHUNK_STRATEGY: str = os.getenv("CHUNK_STRATEGY", "ast").lower()

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
        is_heading = stripped.startswith(("## ", "### "))
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

    for section_lines, s_start in zip(sections, section_start, strict=False):
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


_PARSER_CACHE: dict = {}


def _get_tree_sitter_parser(language: str | None):
    """Retrieve or initialize a tree-sitter Parser for the given language.

    Returns:
        A tree_sitter.Parser instance if supported and available, else None.
    """
    if not language:
        return None

    if language in _PARSER_CACHE:
        return _PARSER_CACHE[language]

    parser = None
    try:
        if language == "python":
            import tree_sitter
            import tree_sitter_python

            py_lang = tree_sitter.Language(tree_sitter_python.language())
            parser = tree_sitter.Parser(py_lang)
    except Exception as exc:
        logger.debug("Tree-sitter parser initialization failed for %s: %s", language, exc)
        parser = None

    _PARSER_CACHE[language] = parser
    return parser


def chunk_file_ast(
    file_path: str,
    content: str,
    language: str | None,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[Chunk]:
    """Split source code into AST-aware chunks (classes, functions, methods).

    Uses Tree-sitter to identify syntactic boundaries. Complete functions and
    classes that fit within `chunk_size` are preserved as coherent units.
    Large definitions are broken down into methods or token-bounded segments.

    Args:
        file_path:  Relative file path for metadata.
        content:    Raw source text.
        language:   Detected language name.
        chunk_size: Maximum tokens per chunk.
        overlap:    Overlap tokens when falling back to sliding window.

    Returns:
        List of Chunk objects.
    """
    lines = content.splitlines(keepends=True)
    parser = _get_tree_sitter_parser(language)

    if parser is None:
        # Fall back to token-based chunking for unsupported languages
        return _chunk_by_tokens(lines, file_path, language, chunk_size, overlap)

    try:
        tree = parser.parse(content.encode("utf-8"))
    except Exception as exc:
        logger.warning("Tree-sitter parse failed for %s, falling back to tokens: %s", file_path, exc)
        return _chunk_by_tokens(lines, file_path, language, chunk_size, overlap)

    chunks: list[Chunk] = []
    pending_lines: list[str] = []
    pending_start: int = 1

    def flush_pending(end_line_idx: int) -> None:
        """Flush accumulated non-definition lines (imports, preamble, comments)."""
        nonlocal pending_lines, pending_start
        if not pending_lines:
            return

        text_block = "".join(pending_lines).rstrip()
        if text_block:
            if _count_tokens(text_block) <= chunk_size:
                chunks.append(
                    Chunk(
                        file_path=file_path,
                        start_line=pending_start,
                        end_line=end_line_idx,
                        content=text_block,
                        language=language,
                    )
                )
            else:
                chunks.extend(
                    _chunk_by_tokens(pending_lines, file_path, language, chunk_size, overlap)
                )
        pending_lines = []

    def process_node(node, is_top_level: bool = True) -> None:
        nonlocal pending_start, pending_lines
        node_type = node.type

        # Unwrap decorated definition (e.g. @decorator\ndef foo(): ...)
        target_node = node
        if node_type == "decorated_definition":
            for child in node.children:
                if child.type in ("function_definition", "class_definition"):
                    target_node = child
                    break

        is_definition = target_node.type in ("function_definition", "class_definition")

        if not is_definition:
            if not pending_lines:
                pending_start = node.start_point.row + 1
            node_lines = lines[node.start_point.row : node.end_point.row + 1]
            pending_lines.extend(node_lines)
            return

        # It is a function or class definition. First flush any accumulated preamble.
        if pending_lines:
            flush_pending(node.start_point.row)

        start_line = node.start_point.row + 1
        end_line = node.end_point.row + 1
        node_lines = lines[node.start_point.row : node.end_point.row + 1]
        node_text = "".join(node_lines).rstrip()

        tokens = _count_tokens(node_text)

        if tokens <= chunk_size:
            # Whole definition fits in one chunk
            if node_text:
                chunks.append(
                    Chunk(
                        file_path=file_path,
                        start_line=start_line,
                        end_line=end_line,
                        content=node_text,
                        language=language,
                    )
                )
        else:
            # Definition is too large. If it's a class with body methods, decompose it.
            if target_node.type == "class_definition":
                body_block = None
                for child in target_node.children:
                    if child.type == "block":
                        body_block = child
                        break

                if body_block and body_block.children:
                    # Collect class header (up to the first method)
                    first_method = None
                    for child in body_block.children:
                        c_type = child.type
                        if c_type == "decorated_definition":
                            for sc in child.children:
                                if sc.type == "function_definition":
                                    first_method = child
                                    break
                            if first_method:
                                break
                        elif c_type == "function_definition":
                            first_method = child
                            break

                    header_end_row = (first_method.start_point.row - 1) if first_method else body_block.start_point.row
                    if header_end_row >= node.start_point.row:
                        header_lines = lines[node.start_point.row : header_end_row + 1]
                        header_text = "".join(header_lines).rstrip()
                        if header_text:
                            chunks.append(
                                Chunk(
                                    file_path=file_path,
                                    start_line=start_line,
                                    end_line=header_end_row + 1,
                                    content=header_text,
                                    language=language,
                                )
                            )

                    # Now emit each method individually
                    for child in body_block.children:
                        c_type = child.type
                        is_method = c_type == "function_definition" or (
                            c_type == "decorated_definition"
                            and any(sc.type == "function_definition" for sc in child.children)
                        )
                        if is_method:
                            m_start = child.start_point.row + 1
                            m_end = child.end_point.row + 1
                            m_lines = lines[child.start_point.row : child.end_point.row + 1]
                            m_text = "".join(m_lines).rstrip()
                            if _count_tokens(m_text) <= chunk_size:
                                chunks.append(
                                    Chunk(
                                        file_path=file_path,
                                        start_line=m_start,
                                        end_line=m_end,
                                        content=m_text,
                                        language=language,
                                    )
                                )
                            else:
                                chunks.extend(
                                    _chunk_by_tokens(m_lines, file_path, language, chunk_size, overlap)
                                )
                    return

            # For functions or classes without granular methods, fall back to sliding token window
            chunks.extend(
                _chunk_by_tokens(node_lines, file_path, language, chunk_size, overlap)
            )

    root = tree.root_node
    for child in root.children:
        process_node(child, is_top_level=True)

    # Flush any remaining lines at the end of the file
    flush_pending(len(lines))

    # If AST produced no chunks for non-empty code, fall back to token chunking
    if not chunks:
        return _chunk_by_tokens(lines, file_path, language, chunk_size, overlap)

    return chunks


def chunk_file(
    file_path: str,
    content: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
    strategy: str | None = None,
) -> list[Chunk]:
    """Split a file's content into searchable chunks.

    Dispatches to the appropriate strategy:
    - Markdown files → heading-based splitting
    - Code files with strategy='ast' (default) → syntax-tree aware chunking
    - Code files with strategy='tokens' → sliding token window

    Args:
        file_path:  Relative path of the file in the repository.
        content:    Full text content of the file.
        chunk_size: Maximum number of tokens per chunk (default: CHUNK_SIZE_TOKENS env var or 200).
        overlap:    Overlap tokens between consecutive chunks (default: CHUNK_OVERLAP_TOKENS env var or 40).
        strategy:   'ast' for syntax-tree aware, 'tokens' for sliding window.
                    Defaults to CHUNK_STRATEGY env var (or 'ast').

    Returns:
        List of Chunk objects. Returns an empty list for empty files.
    """
    if not content or not content.strip():
        return []

    language = _detect_language(file_path)

    # Use a simple suffix check to avoid re-extracting the extension
    if file_path.endswith(".md"):
        return _chunk_markdown(content, file_path, chunk_size, overlap)

    active_strategy = (strategy or DEFAULT_CHUNK_STRATEGY).lower()
    if active_strategy == "ast":
        return chunk_file_ast(file_path, content, language, chunk_size, overlap)

    lines = content.splitlines(keepends=True)
    return _chunk_by_tokens(lines, file_path, language, chunk_size, overlap)
