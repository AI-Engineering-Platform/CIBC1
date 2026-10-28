import logging
import os
import random
from pathlib import Path

from aieng.syn_data.text import (
    LLMSettings,
    OpenAICompatibleClient,
    load_implementation_dotenv,
    use_repo_root,
)
from rich.console import Console
from rich.table import Table


repo_root = use_repo_root()
load_implementation_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s", force=True)
console = Console(width=100)
print(f"Running from repository root: {repo_root}")

SEED = 42
random.seed(SEED)

SOURCE_DOCS_DIR = repo_root / "implementations/qa_text_generation/data/documents"
DESIGN_DIR = repo_root / "implementations/RAG_QA_design"
DATA_DIR = DESIGN_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

CHUNKS_PATH = DATA_DIR / "chunks.jsonl"
NAIVE_QA_PATH = DATA_DIR / "naive_qa.jsonl"
NAIVE_RESULTS_PATH = DATA_DIR / "naive_results.jsonl"

console.print(f"Source documents: {SOURCE_DOCS_DIR}")
console.print(f"Output data dir:  {DATA_DIR}")

CANDIDATE_MODELS = [
    os.getenv("TEACHER_MODEL", "gpt-4o"),
    os.getenv("JUDGE_MODEL", "gemini-3.5-flash"),
    "gpt-4o-mini",
]
CANDIDATE_MODELS = list(dict.fromkeys(CANDIDATE_MODELS))


def check_model_access(model: str) -> tuple[bool, str]:
    """Send a tiny request and report whether the proxy allows the model (reasoning models need token headroom)."""
    client = OpenAICompatibleClient(LLMSettings.from_env(model_var="__UNSET__", default_model=model))
    try:
        reply = client.complete("Reply with the single word: ok", temperature=0.0, max_tokens=256)
    except Exception as exc:  # noqa: BLE001
        return False, str(exc)[:80]
    return True, reply.strip()[:40]


table = Table(show_header=True, header_style="bold magenta")
table.add_column("model")
table.add_column("allowed")
table.add_column("reply / error")

access = {}
for model in CANDIDATE_MODELS:
    ok, detail = check_model_access(model)
    access[model] = ok
    table.add_row(model, "yes" if ok else "NO", detail)

import re
from collections import Counter

from aieng.syn_data.text import list_domain_documents
from aieng.syn_data.text.documents import load_document_text


specs = {spec.doc_id: spec for spec in list_domain_documents("finance")}
raw_texts = {doc_id: load_document_text(Path(spec.local_path)) for doc_id, spec in specs.items()}

table = Table(show_header=True, header_style="bold magenta")
for column in ("doc_id", "role", "chars", "words", "lines"):
    table.add_column(column)
for doc_id, text in raw_texts.items():
    table.add_row(
        doc_id,
        str(specs[doc_id].role),
        str(len(text)),
        str(len(text.split())),
        str(text.count("\n") + 1),
    )
console.print(table)

PAGE_NUMBER_RE = re.compile(r"^\s*(?:page\s+)?\d+(?:\s+(?:of|/)\s+\d+)?\s*$", re.IGNORECASE)
LIST_ITEM_RE = re.compile(r"^(?:[a-z]\.\s|-\s|\d+\.\s)")  # "a. ", "- ", "1. "
SENTENCE_END = (".", "?", "!", ":", ";", '"', "\u201d", ")")
MAX_HEADER_LEN = 60  # a short line repeated 3+ times is treated as a header or footer


def clean_text(text: str) -> tuple[str, dict[str, int]]:
    """Join hard-wrapped lines, keep list items separate, and drop page numbers and repeated headers/footers."""
    lines = [line.strip() for line in text.split("\n")]
    counts = Counter(line for line in lines if line)
    repeated = {line for line, n in counts.items() if n >= 3 and len(line) <= MAX_HEADER_LEN}

    lengths = sorted(len(line) for line in lines if line)
    wrap_width = lengths[int(len(lengths) * 0.9)]  # the typical full-line width of the source

    stats = {"page_numbers_removed": 0, "headers_footers_removed": 0, "lines_joined": 0}
    blocks: list[str] = []
    for block in re.split(r"\n\s*\n", text):
        logical: list[str] = []
        current = ""
        last_len = 0
        for line in (raw.strip() for raw in block.split("\n")):
            if not line:
                continue
            if PAGE_NUMBER_RE.match(line):
                stats["page_numbers_removed"] += 1
                continue
            if line in repeated:
                stats["headers_footers_removed"] += 1
                continue
            if not current:
                current, last_len = line, len(line)
                continue
            starts_new_line = (
                LIST_ITEM_RE.match(line) is not None
                or (current.endswith(SENTENCE_END) and last_len < 0.85 * wrap_width)
                or (last_len < 0.6 * wrap_width and line[0].isupper())  # short line, then a capital: heading
            )
            if starts_new_line:
                logical.append(current)
                current = line
            else:
                current = f"{current} {line}"
                stats["lines_joined"] += 1
            last_len = len(line)
        if current:
            logical.append(current)
        if logical:
            blocks.append("\n".join(logical))
    return "\n\n".join(blocks), stats

clean_texts: dict[str, str] = {}
table = Table(show_header=True, header_style="bold magenta")
for column in ("doc_id", "chars before", "chars after", "lines joined", "page numbers", "headers/footers"):
    table.add_column(column)

for doc_id, text in raw_texts.items():
    cleaned, stats = clean_text(text)
    clean_texts[doc_id] = cleaned
    table.add_row(
        doc_id,
        str(len(text)),
        str(len(cleaned)),
        str(stats["lines_joined"]),
        str(stats["page_numbers_removed"]),
        str(stats["headers_footers_removed"]),
    )
console.print(table)

sample_doc = "cfpb_credit_card_agreement"
marker = "12. LIABILITY"
start = raw_texts[sample_doc].index(marker)
console.print("[bold]Before (section 12):[/bold]")
console.print(raw_texts[sample_doc][start : start + 450])
start = clean_texts[sample_doc].index(marker)
console.print("\n[bold]After (section 12):[/bold]")
console.print(clean_texts[sample_doc][start : start + 450])
