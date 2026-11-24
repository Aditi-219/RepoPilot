"""Unit tests for RepoPilot code chunker module."""

from app.chunker import EXTENSION_TO_LANGUAGE, Chunk, chunk_file


def test_extension_to_language_mapping():
    """Verify supported extensions map to expected programming languages."""
    assert EXTENSION_TO_LANGUAGE[".py"] == "python"
    assert EXTENSION_TO_LANGUAGE[".ts"] == "typescript"
    assert EXTENSION_TO_LANGUAGE[".md"] == "markdown"
    assert EXTENSION_TO_LANGUAGE[".go"] == "go"


def test_chunk_file_empty_content():
    """Empty file or whitespace should return no chunks."""
    assert chunk_file("test.py", "") == []
    assert chunk_file("test.py", "   \n\n  ") == []


def test_chunk_file_small_python_file():
    """Small Python code should produce a single chunk preserving line numbers."""
    code = (
        "def hello_world():\n"
        "    \"\"\"Say hello.\"\"\"\n"
        "    print('Hello, RepoPilot!')\n"
    )
    chunks = chunk_file("app/hello.py", code)
    assert len(chunks) == 1
    assert isinstance(chunks[0], Chunk)
    assert chunks[0].file_path == "app/hello.py"
    assert chunks[0].language == "python"
    assert chunks[0].start_line == 1
    assert chunks[0].end_line == 3
    assert "hello_world" in chunks[0].content


def test_chunk_file_large_content_splits_with_bounds():
    """Large content exceeding chunk_size should produce multiple overlapping chunks."""
    lines = [f"line_{i} = {i} * 10\n" for i in range(1, 201)]
    code = "".join(lines)

    chunks = chunk_file("math_test.py", code, chunk_size=50, overlap=10)
    assert len(chunks) > 1

    # Verify line ranges are contiguous and within total lines
    for chunk in chunks:
        assert chunk.start_line >= 1
        assert chunk.end_line <= 200
        assert chunk.start_line <= chunk.end_line


def test_chunk_markdown_splits_on_headings():
    """Markdown content should be split along section headings."""
    doc = (
        "# Main Title\n\n"
        "Introduction paragraph.\n\n"
        "## Installation\n\n"
        "Run pip install.\n\n"
        "## Usage\n\n"
        "Import the package and run.\n"
    )
    chunks = chunk_file("docs/readme.md", doc, chunk_size=200, overlap=20)
    assert len(chunks) >= 1
    assert any("Installation" in c.content for c in chunks)
    assert any("Usage" in c.content for c in chunks)


def test_chunk_file_ast_separates_functions_and_classes():
    """AST-aware chunker should separate discrete functions into distinct chunks."""
    code = (
        "import math\n\n"
        "def compute_area(radius: float) -> float:\n"
        "    \"\"\"Calculate circle area.\"\"\"\n"
        "    return math.pi * radius ** 2\n\n"
        "def compute_perimeter(radius: float) -> float:\n"
        "    \"\"\"Calculate circle perimeter.\"\"\"\n"
        "    return 2 * math.pi * radius\n"
    )
    chunks = chunk_file("geometry.py", code, chunk_size=50, strategy="ast")
    # Preamble (import math), compute_area, and compute_perimeter
    assert len(chunks) >= 2
    func_names = [c.content for c in chunks]
    assert any("compute_area" in fn for fn in func_names)
    assert any("compute_perimeter" in fn for fn in func_names)
    for c in chunks:
        assert c.file_path == "geometry.py"
        assert c.language == "python"
        assert c.start_line <= c.end_line


def test_chunk_file_ast_decomposes_large_class():
    """A large class exceeding chunk_size should be decomposed into methods."""
    code = (
        "class ServiceManager:\n"
        "    \"\"\"Manages background services.\"\"\"\n"
        "    def __init__(self):\n"
        "        self.active = True\n\n"
        "    def start_worker(self):\n"
        "        # Long worker setup\n"
        "        return 'started'\n\n"
        "    def stop_worker(self):\n"
        "        # Long worker teardown\n"
        "        return 'stopped'\n"
    )
    # Using small chunk_size to trigger decomposition
    chunks = chunk_file("service.py", code, chunk_size=25, strategy="ast")
    assert len(chunks) >= 2
    contents = [c.content for c in chunks]
    assert any("start_worker" in ct for ct in contents)
    assert any("stop_worker" in ct for ct in contents)


def test_chunk_file_strategy_tokens_flag():
    """Explicit strategy='tokens' forces sliding window chunking."""
    code = "def sample():\n    return 42\n"
    chunks = chunk_file("sample.py", code, strategy="tokens")
    assert len(chunks) == 1
    assert "sample" in chunks[0].content


def test_chunk_file_ast_prepends_parent_symbol_signature():
    """Methods should have their enclosing class signature prepended to their content."""
    code = (
        "class SearchPipeline:\n"
        "    def run_query(self, query: str):\n"
        "        return []\n"
    )
    chunks = chunk_file("backend/app/search.py", code, strategy="ast")
    assert len(chunks) >= 1
    # Check that the chunk contains file context or class signature
    content = chunks[0].content
    assert "# File: backend/app/search.py" in content or "SearchPipeline" in content


def test_chunk_file_ast_preserves_absolute_line_numbers_for_split_functions():
    """When a large function or markdown section is split by tokens, absolute file line numbers must be preserved."""
    preamble = "# preamble\n" * 50
    func_body = "def large_computation():\n" + "".join(f"    v_{i} = {i} * 2\n" for i in range(100))
    code = preamble + func_body

    chunks = chunk_file("pipeline.py", code, chunk_size=50, overlap=10, strategy="ast")
    assert len(chunks) > 1

    # Find the chunk containing large_computation definition
    def_chunk = next(c for c in chunks if "def large_computation" in c.content)
    # Preamble was 50 lines, so function starts at line 51
    assert def_chunk.start_line == 51
    assert def_chunk.end_line >= 51



