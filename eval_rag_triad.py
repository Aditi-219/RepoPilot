"""Root convenience entry point for eval_rag_triad.

Delegates directly to evals.scripts.eval_rag_triad.
Run with:
    python eval_rag_triad.py --help
"""

import os
import sys

REPO_ROOT = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(REPO_ROOT, "evals", "scripts"))

from eval_rag_triad import main

if __name__ == "__main__":
    main()
