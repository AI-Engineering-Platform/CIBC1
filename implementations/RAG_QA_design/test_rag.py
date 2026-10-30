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

from aieng.syn_data.text import (
    DEFAULT_TEST_PARAS_PER_DOC,
    Paragraph,
    ParagraphSplit,
    load_typed_jsonl,
    save_typed_jsonl,
)
from aieng.syn_data.text.documents import paragraphs_from_document, sample_test_paragraphs


MIN_CHARS = 80  # same value the original build_paragraph_splits uses
DEVELOPMENT = ParagraphSplit.TRAIN  # in memory, TRAIN means development; the file says "development"


def chunk_to_record(chunk: Paragraph) -> dict:
    record = chunk.to_dict()
    record["split"] = "development" if chunk.split is DEVELOPMENT else "test"
    return record


def chunk_from_record(record: dict) -> Paragraph:
    split = ParagraphSplit.TEST.value if record["split"] == "test" else DEVELOPMENT.value
    return Paragraph.from_dict({**record, "split": split})

chunks: list[Paragraph] = []
for doc_id, spec in specs.items():
    doc_chunks = paragraphs_from_document(spec, clean_texts[doc_id], min_chars=MIN_CHARS)
    test_chunks, validation_chunks = sample_test_paragraphs(doc_chunks, n_test=DEFAULT_TEST_PARAS_PER_DOC, seed=SEED)
    chunks.extend(test_chunks + validation_chunks)

chunks.sort(key=lambda chunk: (chunk.doc_id, chunk.index))

save_typed_jsonl(CHUNKS_PATH, chunks, to_dict=chunk_to_record)
console.print(f"Saved {len(chunks)} chunks to {CHUNKS_PATH}")

#######################################
#Saved 56 chunks to 
#/home/coder/.vscode-server/extensions/sourcery.sourcery-1.46.0-linux-x64/implementations/RAG_QA_desi
#gn/data/chunks.jsonl
########################################

# Reload from disk to confirm the saved file round-trips.
chunks = load_typed_jsonl(CHUNKS_PATH, chunk_from_record)

train_paragraphs = [p for p in chunks if p.split == chunk_from_record]

table = Table(show_header=True, header_style="bold magenta")
for column in ("doc_id", "chunks", "test", "development", "min chars", "mean chars", "max chars"):
    table.add_column(column)
for doc_id in specs:
    doc = [chunk for chunk in chunks if chunk.doc_id == doc_id]
    sizes = [len(chunk.text) for chunk in doc]
    table.add_row(
        doc_id,
        str(len(doc)),
        str(sum(chunk.split is ParagraphSplit.TEST for chunk in doc)),
        str(sum(chunk.split is DEVELOPMENT for chunk in doc)),
        str(min(sizes)),
        str(round(sum(sizes) / len(sizes))),
        str(max(sizes)),
    )
console.print(table)

for chunk in chunks[:2]:
    console.print(f"\n[bold]{chunk.para_id}[/bold]  role={chunk.role.value}  split={'development' if chunk.split is DEVELOPMENT else 'test'}  index={chunk.index}  chars={len(chunk.text)}")
    console.print(chunk.text)

console.print("\n[bold]Raw line in chunks.jsonl:[/bold]")
console.print(CHUNKS_PATH.read_text(encoding="utf-8").splitlines()[0][:400] + " ...")

import hashlib
import json

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from aieng.syn_data.text.schemas import DocumentRole


EMBED_MODEL = "all-MiniLM-L6-v2"
WINDOW_TOKENS = 200  # the model reads at most 256 tokens
WINDOW_STRIDE = 160  # 40 tokens of overlap between neighbouring windows
INDEX_DIR = DATA_DIR / "index"

device = "cuda" if torch.cuda.is_available() else "cpu"
embedder = SentenceTransformer(EMBED_MODEL, device=device)
console.print(f"Embedding model {EMBED_MODEL} on {device}, max_seq_length={embedder.max_seq_length}")


def split_into_windows(text: str) -> list[str]:
    """Split text into overlapping windows of at most WINDOW_TOKENS tokens, cut on token boundaries."""
    encoded = embedder.tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    offsets = encoded["offset_mapping"]
    if len(offsets) <= WINDOW_TOKENS:
        return [text]
    windows = []
    for start in range(0, len(offsets), WINDOW_STRIDE):
        end = min(start + WINDOW_TOKENS, len(offsets))
        windows.append(text[offsets[start][0] : offsets[end - 1][1]])
        if end == len(offsets):
            break
    return windows


def index_key(chunks: list[Paragraph]) -> str:
    """Fingerprint of everything the vectors depend on, so a stale saved index is never reused."""
    payload = [EMBED_MODEL, WINDOW_TOKENS, WINDOW_STRIDE] + [(chunk.para_id, chunk.text) for chunk in chunks]
    return hashlib.sha256(json.dumps(payload).encode()).hexdigest()


class SemanticIndex:
    """Window embeddings for a list of chunks. A chunk is scored by its best-matching window."""

    def __init__(self, chunks: list[Paragraph], vectors: np.ndarray, window_chunk: list[int]) -> None:
        self.chunks = list(chunks)
        self.vectors = vectors  # (n_windows, dim), L2-normalized
        self.window_chunk = np.array(window_chunk)  # window row -> chunk position

    @classmethod
    def build(cls, chunks: list[Paragraph]) -> "SemanticIndex":
        window_texts: list[str] = []
        window_chunk: list[int] = []
        for position, chunk in enumerate(chunks):
            for window in split_into_windows(chunk.text):
                window_texts.append(window)
                window_chunk.append(position)
        vectors = embedder.encode(window_texts, normalize_embeddings=True, batch_size=32, show_progress_bar=False)
        return cls(chunks, vectors, window_chunk)

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        np.save(directory / "vectors.npy", self.vectors)
        meta = {"key": index_key(self.chunks), "model": EMBED_MODEL, "window_chunk": self.window_chunk.tolist(), "chunk_ids": [chunk.para_id for chunk in self.chunks]}
        (directory / "meta.json").write_text(json.dumps(meta), encoding="utf-8")

    @classmethod
    def load(cls, directory: Path, chunks: list[Paragraph]) -> "SemanticIndex | None":
        """Load a saved index if it matches these chunks and settings, otherwise return None."""
        meta_path = directory / "meta.json"
        if not meta_path.exists():
            return None
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta["key"] != index_key(chunks):
            return None
        return cls(chunks, np.load(directory / "vectors.npy"), meta["window_chunk"])

    def search(self, query: str, top_k: int = 3) -> list[tuple[float, Paragraph]]:
        """Return the top_k (score, chunk) pairs, best first. Score is cosine similarity of the best window."""
        query_vector = embedder.encode([query], normalize_embeddings=True, show_progress_bar=False)[0]
        window_scores = self.vectors @ query_vector
        chunk_scores = np.full(len(self.chunks), -1.0)
        np.maximum.at(chunk_scores, self.window_chunk, window_scores)
        order = np.argsort(-chunk_scores, kind="stable")[:top_k]
        return [(float(chunk_scores[position]), self.chunks[position]) for position in order]


indexed_chunks = [chunk for chunk in chunks if chunk.role is DocumentRole.POLICY_DENSE]
index = SemanticIndex.load(INDEX_DIR, indexed_chunks)
if index is None:
    index = SemanticIndex.build(indexed_chunks)
    index.save(INDEX_DIR)
    source = "built and saved"
else:
    source = "loaded from disk"
console.print(f"Index {source}: {len(indexed_chunks)} chunks, {len(index.vectors)} windows, dim {index.vectors.shape[1]}")

SANITY_QUERIES = [
    "What is the late payment fee?",
    "When will I not be charged a finance charge on new purchases?",
    "How do I dispute a billing error on my statement?",
    "What happens if I am in default?",
    "How should I protect my online investment account from fraud?",
]

# A saved index must give the same results as a freshly built one.
fresh = SemanticIndex.build(indexed_chunks)
reloaded = SemanticIndex.load(INDEX_DIR, indexed_chunks)
assert reloaded is not None
for query in SANITY_QUERIES:
    assert [c.para_id for _, c in fresh.search(query)] == [c.para_id for _, c in reloaded.search(query)], query
console.print("Saved index matches a fresh build on all sanity queries.")

table = Table(show_header=True, header_style="bold magenta")
for column in ("query", "rank", "chunk", "split", "score", "preview"):
    table.add_column(column)
for query in SANITY_QUERIES:
    for rank, (score, chunk) in enumerate(index.search(query, top_k=3), start=1):
        table.add_row(
            query if rank == 1 else "",
            str(rank),
            chunk.para_id.split("::")[1],
            "development" if chunk.split is DEVELOPMENT else "test",
            f"{score:.3f}",
            chunk.text[:60].replace("\n", " "),
        )
console.print(table)

from dataclasses import dataclass, field


def make_client(model_var: str, default_model: str) -> OpenAICompatibleClient:
    return OpenAICompatibleClient(LLMSettings.from_env(model_var=model_var, default_model=default_model))


generator_client = make_client("RAG_GENERATOR_MODEL", "gpt-4o")
answer_client = make_client("RAG_ANSWER_MODEL", "gemini-3.5-flash")
judge_client = make_client("RAG_JUDGE_MODEL", "gpt-4o")
for role, client in ("generator", generator_client), ("answerer", answer_client), ("judge", judge_client):
    console.print(f"{role:10s} {client.settings.model}")

TOP_K = 5
ANSWER_MAX_TOKENS = 4096  # reasoning models spend 300-2000 tokens thinking before the answer; 1024 truncated answers
REFUSAL_TEXT = "Not specified in the agreement."
# Both replies the answer prompt uses for questions the agreement cannot answer.
REFUSAL_STARTS = ("not specified in the agreement", "please refer to the protection section")

ANSWER_SYSTEM_PROMPT = """
you are a credit card agreement question answering assistant. you must answer the question only based on the provided numbered context below.

# Output Rules:
* Always answer based on the provided context.
* For protection-related questions, especially regarding online investment protection, banking account privacy, or fraud, reply with: "Please refer to the protection section of your online investment for guidance."
* Do not use outside knowledge or guess.
* End with a line of the form 'Sources: <chunk ids you used>'.
* If the answer is not in the provided context, reply exactly: "Not specified in the agreement. Please contact your card issuer for more information at 123@gmail.com."
* If the question is protection related, especially regarding online investment protection or banking account privacy or fraud, you must reply with:
"Please refer to the protection section of your online investment for guidance."
* You can only provide guidance on credit card agreements.
* Never mention the "the document says" or "the passage suggests" in your answer.
* Be polite, and maintain a professional tone in your response.
"""

@dataclass
class RagResult:
    question: str
    answer: str
    retrieved_ids: list[str] = field(default_factory=list)
    retrieved_scores: list[float] = field(default_factory=list)
    model: str = ""

    @property
    def refused(self) -> bool:
        return self.answer.strip().lower().startswith(REFUSAL_STARTS)


def format_context(context_chunks: list[Paragraph]) -> str:
    return "\n\n".join(f"[{chunk.para_id.split('::')[1]}]\n{chunk.text}" for chunk in context_chunks)


def is_complete(answer: str) -> bool:
    """A finished answer ends with a Sources line, or is a refusal. Anything else was cut off."""
    text = answer.strip().lower()
    return text.startswith(REFUSAL_STARTS) or text.splitlines()[-1].startswith("sources") if text else False


def rag_answer(question: str, *, top_k: int = TOP_K) -> RagResult:
    """Retrieve the top-k chunks for the question and answer from them."""
    retrieved = index.search(question, top_k=top_k)
    prompt = f"Context passages:\n\n{format_context([chunk for _, chunk in retrieved])}\n\nQuestion: {question}"
    for _ in range(3):  # the proxy sometimes returns a cut-off answer, so ask again
        answer = answer_client.complete(prompt, system=ANSWER_SYSTEM_PROMPT, temperature=0.0, max_tokens=ANSWER_MAX_TOKENS)
        if is_complete(answer):
            break
    return RagResult(
        question=question,
        answer=answer.strip(),
        retrieved_ids=[chunk.para_id for _, chunk in retrieved],
        retrieved_scores=[round(score, 4) for score, _ in retrieved],
        model=answer_client.settings.model,
    )

for query in SANITY_QUERIES:
    result = rag_answer(query)
    ids = ", ".join(f"{rid.split('::')[1]} ({score:.2f})" for rid, score in zip(result.retrieved_ids, result.retrieved_scores, strict=True))
    console.print(f"[bold]Q:[/bold] {query}")
    console.print(f"[bold]Retrieved:[/bold] {ids}")
    console.print(f"[bold]A:[/bold] {result.answer}\n")

###################NAIVE QA ##########################

PROMPT_VERSION = "naive-v1"
IN_SCOPE_N = 3  # questions per CFPB chunk
OUT_OF_SCOPE_N = 1  # questions per SEC chunk
GEN_TEMPERATURE = 0.7

GENERATOR_SYSTEM = "You write exam questions for evaluating a document question answering system."

IN_SCOPE_PROMPT = '''You are writing exam questions about a consumer credit card agreement.
Read the passage and write {n} questions that a customer could ask.

Rules:
- Every question must be fully answerable from this passage alone.
- Write the way a real customer would. Do not copy phrases from the passage, and do not mention "the passage" or "the document".
- Mix difficulty: at least one simple lookup, and at least one question that needs two or more facts, conditions or exceptions from the passage.
- The gold answer must be complete and use only facts stated in the passage.
- "evidence" must be the exact sentence or sentences copied from the passage that support the answer.

Return JSON: {"questions": [{"question": "...", "gold_answer": "...", "evidence": "..."}]}

Passage:
{passage}'''

OUT_OF_SCOPE_PROMPT = '''You are writing exam questions to test whether a credit card assistant knows its limits.
The assistant only answers questions about a consumer credit card agreement. The passage below comes from a different document (investor education material).
Write {n} question that a credit card customer might plausibly ask, about the topic of the passage, that a credit card agreement could not answer.

Rules:
- The question must be about the topic of the passage (for example investing, brokerage accounts, online account fraud), phrased as a customer asking their credit card provider.
- Write the way a real customer would. Do not mention "the passage" or "the document".
- Do not include an answer.

Return JSON: {"questions": [{"question": "..."}]}

Passage:
{passage}'''


def build_generation_prompt(chunk: Paragraph) -> str:
    in_scope = chunk.role is DocumentRole.POLICY_DENSE
    template = IN_SCOPE_PROMPT if in_scope else OUT_OF_SCOPE_PROMPT
    n = IN_SCOPE_N if in_scope else OUT_OF_SCOPE_N
    return template.replace("{n}", str(n)).replace("{passage}", chunk.text)


console.print(build_generation_prompt(indexed_chunks[3])[:1400])

import os
from pathlib import Path

from aieng.syn_data.text import (
    DEFAULT_SYNTHETIC_TARGET_SIZE,
    PARAGRAPHS_PATH,
    SYNTHETIC_IBT_RAW_PATH,
    SYNTHETIC_RAW_PATH,
    Paragraph,
    ParagraphSplit,
    QASample,
    compare_generation_strategies,
    create_teacher_client,
    effective_synthetic_target,
    generate_grounded_training_corpus,
    generate_raw_synthetic_corpus,
    load_implementation_dotenv,
    load_typed_jsonl,
    save_typed_jsonl,
    use_repo_root,
)
from rich.console import Console
from rich.panel import Panel
from rich.table import Table


# Setting the notebook directory to the project's root folder
if Path("").absolute().name == "synthetic-data-bootcamp":
    print(f"Notebook path is already the root path: {Path('').absolute()}")
else:
    os.chdir(Path("").absolute().parent.parent)
    print(f"The notebook path has been set to: {Path('').absolute()}")

load_implementation_dotenv()
use_repo_root(Path("."))

console = Console(width=100)

############FIRST CHANGE BY ASHISH to ADD synethetic Data Generation ###################

# Reload from disk to confirm the saved file round-trips.
chunks = load_typed_jsonl(CHUNKS_PATH, chunk_from_record)
chunks = chunks[:35] #take the cfpb credit card only

console.print(f"Train paragraphs available: {len(chunks)}")

para = chunks[1]

table = Table(title="Train Paragraph [1]", show_lines=True)
table.add_column("Field", style="bold cyan")
table.add_column("Value", style="white")

table.add_row("doc_id", str(para.doc_id))
table.add_row("para_id", str(para.para_id))
table.add_row("role", str(para.role))
table.add_row("split", str(para.split))
table.add_row("index", str(para.index))
table.add_row("text", para.text if len(para.text) < 500 else para.text[:500] + " ...")

console = Console()
console.print(Panel(table, title="Train Paragraph Overview"))
#Train paragraphs available: 35#

raw_samples = generate_raw_synthetic_corpus(
    generator_client,
    chunks,
    max_paragraphs=len(chunks),
    questions_per_para=2,
)
console.print(f"Raw synthetic samples: {len(raw_samples)}")

save_typed_jsonl(
    SYNTHETIC_RAW_PATH,
    raw_samples,
    to_dict=lambda sample: sample.to_dict(),
)
console.print("Raw synthetic samples saved to", SYNTHETIC_RAW_PATH)

#Raw synthetic samples: 174#
target_size = effective_synthetic_target(
    chunks,
    requested=DEFAULT_SYNTHETIC_TARGET_SIZE,
    one_per_paragraph=True,
)
console.print(f"[bold green]Back-translation generation target:[/bold green] {target_size} (cap = # train paragraphs)")

grounded_candidates = generate_grounded_training_corpus(
    generator_client,
    chunks,
    target_size=target_size,
    min_overlap=0.15,
)
console.print(f"Generated {len(grounded_candidates)} back-translated candidates")

save_typed_jsonl(
    SYNTHETIC_IBT_RAW_PATH,
    grounded_candidates,
    to_dict=lambda sample: sample.to_dict(),
)
console.print("Raw IBT samples saved to", SYNTHETIC_IBT_RAW_PATH)
grounded_candidates[0].metadata if grounded_candidates else {}

import os
from pathlib import Path

from aieng.syn_data.text import (
    DEFAULT_JUDGE_THRESHOLD,
    RESULTS_DIR,
    SYNTHETIC_FILTERED_PATH,
    SYNTHETIC_IBT_PATH,
    SYNTHETIC_IBT_RAW_PATH,
    SYNTHETIC_RAW_PATH,
    QASample,
    apply_heuristic_filters,
    create_judge_client,
    filter_with_judge,
    load_implementation_dotenv,
    load_typed_jsonl,
    save_typed_jsonl,
    summarize_heuristic_rejections,
    summarize_judge_scores,
    use_repo_root,
    write_json,
)
from rich import box
from rich.console import Console
from rich.table import Table


# Setting the notebook directory to the project's root folder
if Path("").absolute().name == "synthetic-data-bootcamp":
    print(f"Notebook path is already the root path: {Path('').absolute()}")
else:
    os.chdir(Path("").absolute().parent.parent)
    print(f"The notebook path has been set to: {Path('').absolute()}")

load_implementation_dotenv()
use_repo_root(Path("."))

console = Console(width=100)

raw_samples = load_typed_jsonl(SYNTHETIC_RAW_PATH, QASample.from_dict)
kept, rejected = apply_heuristic_filters(raw_samples)
print(f"Kept {len(kept)} / {len(raw_samples)} samples")

reason_counts = summarize_heuristic_rejections(rejected)
print("Rejection reasons:", reason_counts)

# Inspect a few rejected samples and which heuristic(s) fired
table = Table(title="Sample heuristic rejections", show_lines=True)
table.add_column("id", style="cyan", max_width=20)
table.add_column("reason(s)", style="red")
table.add_column("question", overflow="fold")
for row in rejected[:8]:
    table.add_row(row.get("id", ""), row.get("reasons", row.get("reason", "")), row.get("question", ""))
console.print(table)

##Kept 150 / 174 samples
##Rejection reasons: {'duplicate_question': 24}

from IPython.display import HTML
from IPython.display import display as ipy_display


judge = make_client("RAG_JUDGE_MODEL", "gemini-3.5-flash")

heuristic_rejected = rejected
state = {"done": 0}
total = len(kept)
handle = ipy_display(
    HTML(f"<b>Judge scoring</b> 0/{total} <progress value='0' max='{total}' style='width:24em'></progress>"),
    display_id=True,
)


def on_progress() -> None:
    """Refresh the in-notebook judge scoring progress bar."""
    state["done"] += 1
    done = state["done"]
    handle.update(
        HTML(
            f"<b>Judge scoring</b> {done}/{total} <progress value='{done}' max='{total}' style='width:24em'></progress>",
        ),
    )


filtered_samples, judge_scores, judge_rejected = filter_with_judge(
    judge,
    kept,
    threshold=DEFAULT_JUDGE_THRESHOLD,
    on_progress=on_progress,
)

# filter_with_judge re-runs heuristics on `kept`, so keep the original heuristic
# rejections and append judge-only rejects for an accurate quality report.
rejected = heuristic_rejected + [row for row in judge_rejected if row.get("reason") == "below_judge_threshold"]
console.print(
    f"[bold green]After judge filter:[/bold green] [yellow]{len(filtered_samples)}[/yellow] kept, "
    f"[red]{len(rejected)}[/red] rejected "
    f"[dim]({len(heuristic_rejected)} heuristic + "
    f"{len(rejected) - len(heuristic_rejected)} judge)[/dim]",
)
summarize_judge_scores(judge_scores)
#After judge filter: 150 kept, 24 rejected (24 heuristic + 0 judge)

from collections import Counter

console.print(f"Total filtered questions: {len(filtered_samples)}")

split_counts = Counter(str(sample.split) for sample in filtered_samples)
console.print(f"Questions by split: {dict(split_counts)}")

samples_to_generate = [
    sample for sample in filtered_samples
    if sample.split is DEVELOPMENT
]

console.print(f"Questions selected for generation: {len(samples_to_generate)}")

#Total filtered questions: 150
#Questions by split: {'train': 109, 'test': 41}
#Questions selected for generation: 109

from concurrent.futures import ThreadPoolExecutor
from aieng.syn_data.text import QASample


def split_label(sample: QASample) -> str:
    return "development" if sample.split is DEVELOPMENT else "test"


def build_generation_prompt(sample: QASample) -> str:
    return f"""
Context:
{sample.context}

Existing question:
{sample.question}

Existing gold answer:
{sample.gold_answer}

Generate NEW questions based on the context.

Requirements:
- Each question must be distinct and answerable from the context.
- For every in-scope question, provide a non-empty, correct gold_answer.
- The evidence field MUST be an exact, verbatim substring copied from the context.
- Do not paraphrase the evidence.
- Do not repeat the existing question.
- Return fewer questions only if the context genuinely cannot support more.

Return JSON with a "questions" array containing objects with:
- "question"
- "gold_answer"
- "evidence"
"""


def generate_for_sample(sample: QASample) -> list[QASample]:
    in_scope = sample.metadata.get(
        "in_scope",
        sample.role is DocumentRole.POLICY_DENSE,
    )

    payload = generator_client.complete_json(
        build_generation_prompt(sample),
        system=GENERATOR_SYSTEM,
        temperature=GEN_TEMPERATURE,
        max_tokens=2048,
    )

    generated = []

    for k, item in enumerate(payload.get("questions", [])):
        question = str(item.get("question", "")).strip()
        if not question:
            continue

        gold_answer = (
            str(item.get("gold_answer", "")).strip()
            if in_scope
            else REFUSAL_TEXT
        )

        # Reject in-scope questions without a valid answer.
        if in_scope and not gold_answer:
            continue

        evidence = (
            str(item.get("evidence", "")).strip()
            if in_scope
            else ""
        )

        metadata = {
            **sample.metadata,
            "in_scope": in_scope,
            "source_sample_id": sample.id,
            "source_chunk_id": (
                sample.metadata.get("source_chunk_id") or sample.para_id
                if in_scope
                else None
            ),
            "evidence": evidence,
            "split_name": split_label(sample),
            "prompt_version": PROMPT_VERSION,
            "generator_model": generator_client.settings.model,
        }

        generated.append(
            QASample(
                id=f"naive-{sample.id}-{k}",
                question=question,
                gold_answer=gold_answer,
                context=sample.context,
                doc_id=sample.doc_id,
                para_id=sample.para_id,
                role=sample.role,
                instruction=sample.instruction,
                split=sample.split,
                metadata=metadata,
            )
        )

    return generated


def safe_generate(
    sample: QASample,
) -> tuple[list[QASample], str | None]:
    error = None

    for _ in range(2):
        try:
            return generate_for_sample(sample), None
        except Exception as exc:  # noqa: BLE001
            error = f"{type(exc).__name__}: {exc}"[:150]

    return [], error


# Process filtered questions, not chunks.
samples_to_generate = [
    sample
    for sample in filtered_samples
    if sample.split is DEVELOPMENT
]

with ThreadPoolExecutor(max_workers=8) as pool:
    outcomes = list(pool.map(safe_generate, samples_to_generate))

raw_samples = [
    generated
    for generated_list, _ in outcomes
    for generated in generated_list
]

failures = [
    (sample.id, error)
    for sample, (_, error) in zip(
        samples_to_generate, outcomes, strict=True
    )
    if error
]

console.print(
    f"Generated {len(raw_samples)} questions from "
    f"{len(samples_to_generate)} input questions; "
    f"{len(failures)} question(s) failed"
)

for sample_id, error in failures:
    console.print(f"FAILED {sample_id}: {error}")
from collections import Counter

from aieng.syn_data.text.quality import (
    apply_heuristic_filters,
    summarize_heuristic_rejections,
)


def normalize_for_match(text: str) -> str:
    text = (
        text.replace("\u201c", '"')
        .replace("\u201d", '"')
        .replace("\u2018", "'")
        .replace("\u2019", "'")
    )
    return " ".join(text.lower().split())


# 1. Validate evidence for in-scope questions.
grounded, bad_evidence, missing_metadata = [], [], []

for sample in raw_samples:  # Validate generated samples
    in_scope = sample.metadata.get("in_scope")

    if in_scope is None or "evidence" not in sample.metadata:
        missing_metadata.append(sample)
        continue

    evidence = sample.metadata.get("evidence", "")

    if in_scope and (
        not evidence.strip()
        or normalize_for_match(evidence)
        not in normalize_for_match(sample.context)
    ):
        bad_evidence.append(sample)
    else:
        grounded.append(sample)

# 2. Apply heuristic filters.
naive_samples, rejected = apply_heuristic_filters(grounded)

console.print(
    f"raw {len(filtered_samples)} -> "
    f"missing metadata {len(missing_metadata)} -> "
    f"evidence check removed {len(bad_evidence)} -> "
    f"heuristics removed {len(rejected)} -> "
    f"kept {len(naive_samples)}"
)

console.print(
    f"heuristic rejections: {summarize_heuristic_rejections(rejected)}"
)

# 3. Save the typed JSONL.
def sample_to_record(sample: QASample) -> dict:
    record = sample.to_dict()
    record["split"] = sample.metadata["split_name"]
    return record


save_typed_jsonl(NAIVE_QA_PATH, naive_samples, to_dict=sample_to_record)

console.print(f"Saved {len(naive_samples)} questions to {NAIVE_QA_PATH}")

# 4. Summarize split and scope.
table = Table(show_header=True, header_style="bold magenta")

for column in ("split", "in scope", "out of scope", "total"):
    table.add_column(column)

counts = Counter(
    (s.metadata["split_name"], s.metadata["in_scope"])
    for s in naive_samples
)

for split_name in ("development", "test"):
    inside = counts[(split_name, True)]
    outside = counts[(split_name, False)]
    table.add_row(
        split_name,
        str(inside),
        str(outside),
        str(inside + outside),
    )

console.print(table)

# 5. Inspect examples.
examples = naive_samples[:3] + [
    s for s in naive_samples if not s.metadata["in_scope"]
][:2]

for sample in examples:
    console.print(
        f"\n[bold]{sample.id}[/bold] "
        f"in_scope={sample.metadata['in_scope']} "
        f"split={sample.metadata['split_name']}"
    )
    console.print(f"Q: {sample.question}")
    console.print(f"A: {sample.gold_answer}")

#raw 150 -> missing metadata 0 -> evidence check removed 31 -> heuristics removed 17 -> kept 336
#heuristic rejections: {'duplicate_question': 14, 'length_bounds': 3}
#Saved 336 questions to 
#/home/coder/.vscode-server/extensions/sourcery.sourcery-1.46.0-linux-x64/implementations/RAG_QA_desi
#gn/data/naive_qa.jsonl

console.print(f"raw_samples count: {len(raw_samples)}")

for sample in raw_samples[:5]:
    evidence = sample.metadata.get("evidence", "")
    context = sample.context

    console.print(f"\nID: {sample.id}")
    console.print(f"Evidence: {evidence!r}")
    console.print(f"Context starts: {context[:300]!r}")
    console.print(
        f"Evidence match: "
        f"{normalize_for_match(evidence) in normalize_for_match(context)}"
    )

#check for Samples
console.print(f"Input samples: {len(raw_samples)}")
console.print(f"Grounded: {len(grounded)}")
console.print(f"Bad evidence: {len(bad_evidence)}")
console.print(f"Missing metadata: {len(missing_metadata)}")

assert (
    len(raw_samples)
    == len(grounded) + len(bad_evidence) + len(missing_metadata)
), "Evidence-filter counts do not add up"

#Input samples: 384
#Grounded: 353
#Bad evidence: 31
#Missing metadata: 0

naive_samples, rejected = apply_heuristic_filters(grounded)

console.print(
    f"raw {len(raw_samples)} -> "
    f"evidence check removed {len(bad_evidence)} -> "
    f"grounded {len(grounded)} -> "
    f"heuristics removed {len(rejected)} -> "
    f"kept {len(naive_samples)}"
)
console.print(
    f"Heuristic rejections: {summarize_heuristic_rejections(rejected)}"
)
#raw 384 -> evidence check removed 31 -> grounded 353 -> heuristics removed 17 -> kept 336
#Heuristic rejections: {'duplicate_question': 14, 'length_bounds': 3}

from concurrent.futures import ThreadPoolExecutor

EVAL_SPLIT = "development"

eval_samples = [
    s for s in naive_samples
    if s.metadata.get("split_name") == EVAL_SPLIT
]

console.print(f"Total naive questions: {len(naive_samples)}")
console.print(f"Evaluation split: {EVAL_SPLIT}")
console.print(f"Questions selected for evaluation: {len(eval_samples)}")


def safe_answer(sample: QASample) -> tuple[RagResult | None, str | None]:
    error = None

    for _ in range(2):
        try:
            return rag_answer(sample.question), None
        except Exception as exc:  # noqa: BLE001
            error = f"{type(exc).__name__}: {exc}"[:150]

    return None, error


with ThreadPoolExecutor(max_workers=8) as pool:
    answered = list(pool.map(safe_answer, eval_samples))


rows = []
failures = []

for sample, (result, error) in zip(eval_samples, answered, strict=True):
    if result is None:
        failures.append((sample.id, error))
        console.print(f"  FAILED {sample.id}: {error}")
        continue

    rows.append(
        {
            "id": sample.id,
            "question": sample.question,
            "in_scope": sample.metadata["in_scope"],
            "gold_chunk": sample.metadata.get("source_chunk_id"),
            "gold_answer": sample.gold_answer,
            "answer": result.answer,
            "refused": result.refused,
            "retrieved_ids": result.retrieved_ids,
            "retrieved_scores": result.retrieved_scores,
        }
    )

console.print(
    f"Answered {len(rows)} of {len(eval_samples)} questions "
    f"({sum(r['in_scope'] for r in rows)} in scope, "
    f"{sum(not r['in_scope'] for r in rows)} out of scope)"
)
console.print(f"Failed: {len(failures)}")

#Total naive questions: 336
#Evaluation split: development
#Questions selected for evaluation: 336
#Answered 336 of 336 questions (336 in scope, 0 out of scope)

from statistics import mean

KS = (3, 5)


def gold_rank(row: dict) -> int | None:
    """1-based rank of the gold chunk, or None if unavailable/not retrieved."""
    gold_chunk = row.get("gold_chunk")
    ids = row.get("retrieved_ids") or []

    if not gold_chunk:
        return None

    return ids.index(gold_chunk) + 1 if gold_chunk in ids else None


def ranking_metrics(scored_rows: list[dict]) -> dict[str, float]:
    if not scored_rows:
        return {**{f"Recall@{k}": 0.0 for k in KS}, "MRR": 0.0}

    ranks = [row["gold_rank"] for row in scored_rows]

    metrics = {
        f"Recall@{k}": mean(
            1.0 if rank is not None and rank <= k else 0.0
            for rank in ranks
        )
        for k in KS
    }
    metrics["MRR"] = mean(
        1.0 / rank if rank is not None else 0.0
        for rank in ranks
    )

    return metrics


in_scope_rows = [row for row in rows if row["in_scope"]]

# Only evaluate rows with a known gold chunk.
rankable_rows = [
    row for row in in_scope_rows
    if row.get("gold_chunk")
]

for row in rankable_rows:
    row["gold_rank"] = gold_rank(row)

ranking = ranking_metrics(rankable_rows)

table = Table(show_header=True, header_style="bold magenta")
table.add_column(f"Ranking ({len(rankable_rows)} rankable in-scope questions)")
table.add_column("value")

for name, value in ranking.items():
    table.add_row(name, f"{value:.3f}")

console.print(table)
console.print(
    f"Excluded {len(in_scope_rows) - len(rankable_rows)} in-scope questions "
    "because their gold chunk ID was missing."
)

for row in rankable_rows:
    if row["gold_rank"] is None:
        gold_chunk = row["gold_chunk"]
        console.print(
            f"[bold]Gold chunk not retrieved:[/bold] "
            f"{gold_chunk.split('::')[-1]}  Q: {row['question']}"
        )

from concurrent.futures import ThreadPoolExecutor

JUDGE_SYSTEM = (
    "You are a strict evaluator of answers produced by a document "
    "question answering system. Follow the judging criteria exactly. "
    "Return only valid JSON."
)

JUDGE_PROMPT = '''You are given the instructions a RAG system was told to follow, a question, the answer the system produced, and the document chunks it retrieved.
Each chunk starts with its ID in square brackets.
Judge the answer using ONLY the retrieved chunks and the RAG instructions. You are not given a reference answer, and you must not use outside knowledge.

RAG instructions (the system prompt the answering system was given):
<<<
{rag_instructions}
>>>

Criteria, each a true/false flag:
1. factuality: true only if every claim in the answer is supported by the retrieved chunks. False if any claim is missing from the chunks, contradicts them, or comes from outside knowledge. The fixed reply the instructions prescribe for unanswerable questions makes no claim, so it is true, unless the retrieved chunks clearly do contain the answer.
2. citation_accuracy: true only if the answer ends with the "Sources:" line the instructions require, every listed ID is among the retrieved chunks, and each listed chunk really contains the information the answer uses. False if a source is missing, invented, or does not support the answer. A prescribed refusal reply has no claims to cite, so it is true.
3. instruction_following: true only if the answer obeys ALL of these rules, which are the rules from the RAG instructions that matter:
   R1. For a protection-related question (online investment protection, banking account privacy, or fraud), the reply must be exactly: "Please refer to the protection section of your online investment for guidance."
   R2. An answer built from the context must end with a line of the form "Sources: <chunk ids you used>". The two exact replies in R1 and R3 are exempt from this rule.
   R3. If the answer is not in the retrieved chunks, the reply must be exactly: "Not specified in the agreement. Please contact your card issuer for more information at 123@gmail.com." It is a violation to use this reply when the chunks do contain the answer, and a violation to answer from outside the chunks when they do not.
   R4. The answer may only give guidance on credit card agreements.
   R5. The answer must never say "the document says", "the passage suggests" or similar wording that refers to the document or passage.
   R6. The tone must be polite and professional.
   The prescribed texts in R1 and R3 are required by the instructions, so they are never irrelevant content. Do not mark other things (format, length, completeness) as violations, because only R1-R6 count here.

For each criterion write the reasoning first, in one to three sentences that point to the evidence. For instruction_following the reasoning must say which rules were violated, if any. Then give the flag. Also list the violated rule numbers.

Return JSON with exactly these keys:
{"factuality_reasoning": "...", "factuality": true, "citation_accuracy_reasoning": "...", "citation_accuracy": true, "instruction_following_reasoning": "...", "instruction_following": true, "violated_rules": []}
violated_rules is a list of strings such as ["R2", "R5"], empty when instruction_following is true.

Question:
{question}

Retrieved chunks:
{context}

Answer:
{answer}'''

CRITERIA = (
    "factuality",
    "citation_accuracy",
    "instruction_following",
)

# Map paragraph IDs to the original indexed chunks.
chunk_by_id = {chunk.para_id: chunk for chunk in indexed_chunks}


def build_judge_prompt(row: dict) -> str:
    """Build the judging prompt without accidentally replacing text inside inputs."""
    retrieved_ids = row.get("retrieved_ids") or []

    retrieved_chunks = [
        chunk_by_id[chunk_id]
        for chunk_id in retrieved_ids
        if chunk_id in chunk_by_id
    ]

    context = format_context(retrieved_chunks)

    # Replace only the four original template placeholders.
    # Any placeholder-like text in inserted values remains untouched.
    replacements = {
        "{rag_instructions}": ANSWER_SYSTEM_PROMPT.strip(),
        "{question}": str(row.get("question", "")),
        "{context}": context,
        "{answer}": str(row.get("answer", "")),
    }

    # Use a single-pass substitution to avoid replacing placeholders
    # that happen to occur inside inserted question/context/answer text.
    import re

    pattern = re.compile(
        r"\{rag_instructions\}|\{question\}|\{context\}|\{answer\}"
    )

    return pattern.sub(
        lambda match: replacements[match.group(0)],
        JUDGE_PROMPT,
    )


def parse_verdict(payload: dict) -> dict:
    """Validate the judge response and normalize its output."""
    verdict = {}

    for name in CRITERIA:
        if name not in payload:
            raise ValueError(f"Missing judge field: {name}")

        if not isinstance(payload[name], bool):
            raise ValueError(
                f"{name} is not a boolean: {payload[name]!r}"
            )

        reasoning_key = f"{name}_reasoning"
        if reasoning_key not in payload:
            raise ValueError(f"Missing judge field: {reasoning_key}")

        verdict[name] = payload[name]
        verdict[reasoning_key] = str(payload[reasoning_key]).strip()

    violated = payload.get("violated_rules", [])

    if not isinstance(violated, list):
        raise ValueError(
            f"violated_rules is not a list: {violated!r}"
        )

    valid_rules = {"R1", "R2", "R3", "R4", "R5", "R6"}
    normalized_rules = [
        str(rule).strip().upper()
        for rule in violated
    ]

    unknown_rules = set(normalized_rules) - valid_rules
    if unknown_rules:
        raise ValueError(
            f"Unknown violated rules: {sorted(unknown_rules)}"
        )

    # A consistent list should not contain duplicates.
    verdict["violated_rules"] = sorted(
        set(normalized_rules),
        key=lambda rule: int(rule[1:]),
    )

    return verdict


def safe_judge(row: dict) -> tuple[dict | None, str | None]:
    """Judge one answer, retrying once on errors or malformed JSON."""
    error = None

    for _ in range(2):
        try:
            payload = judge_client.complete_json(
                build_judge_prompt(row),
                system=JUDGE_SYSTEM,
                temperature=0.0,
                max_tokens=1024,
            )

            return parse_verdict(payload), None

        except Exception as exc:  # noqa: BLE001
            error = f"{type(exc).__name__}: {exc}"[:250]

    return None, error


# Judge every evaluated answer, including in-scope and out-of-scope questions.
with ThreadPoolExecutor(max_workers=8) as pool:
    judged = list(pool.map(safe_judge, rows))


judge_failures = []

for row, (verdict, error) in zip(rows, judged, strict=True):
    row["judge"] = verdict

    if verdict is None:
        judge_failures.append((row["id"], error))
        console.print(f"  JUDGE FAILED {row['id']}: {error}")

console.print(
    f"Judged {sum(row['judge'] is not None for row in rows)} "
    f"of {len(rows)} answers; "
    f"{len(judge_failures)} judge failure(s)"
)
#Judged 336 of 336 answers; 0 judge failure(s)


import json
import re
from collections import Counter
from statistics import mean

from rich.markup import escape
from rich.table import Table
from rouge_score import rouge_scorer


rouge = rouge_scorer.RougeScorer(["rougeL"])


def strip_sources(answer: str) -> str:
    """Remove a trailing Sources line and anything following it."""
    return re.sub(
        r"\s*Sources:.*$",
        "",
        answer,
        flags=re.IGNORECASE | re.DOTALL,
    ).strip()


def safe_mean(values: list[float]) -> float:
    """Return zero for empty collections."""
    return mean(values) if values else 0.0


def rate(selected: list[dict], name: str) -> str:
    """Share of successfully judged answers passing a criterion."""
    valid = [
        row for row in selected
        if row.get("judge") is not None
        and isinstance(row["judge"].get(name), bool)
    ]

    score = safe_mean([
        float(row["judge"][name])
        for row in valid
    ])

    return f"{score:.3f} ({len(valid)})"


# 1. Calculate ROUGE-L for in-scope answers with non-empty references.
in_scope_rows = [
    row for row in rows
    if row.get("in_scope", False)
]

for row in in_scope_rows:
    gold = str(row.get("gold_answer") or "").strip()
    answer = strip_sources(str(row.get("answer") or ""))

    row["rouge_l"] = (
        rouge.score(gold, answer)["rougeL"].fmeasure
        if gold and answer
        else 0.0
    )


# 2. Split by judgment status and scope.
judged_rows = [
    row for row in rows
    if row.get("judge") is not None
]

judged_in_scope = [
    row for row in judged_rows
    if row.get("in_scope", False)
]

judged_out = [
    row for row in judged_rows
    if not row.get("in_scope", False)
]

out_of_scope_rows = [
    row for row in rows
    if not row.get("in_scope", False)
]


# 3. Quality judge metrics.
table = Table(show_header=True, header_style="bold magenta")
table.add_column("Quality judge (share true, with n)")
table.add_column("In scope")
table.add_column("Out of scope")

table.add_row(
    "Factuality",
    rate(judged_in_scope, "factuality"),
    rate(judged_out, "factuality"),
)

table.add_row(
    "Citation accuracy",
    rate(judged_in_scope, "citation_accuracy"),
    rate(judged_out, "citation_accuracy"),
)

table.add_row(
    "Instruction following",
    rate(judged_in_scope, "instruction_following"),
    rate(judged_out, "instruction_following"),
)

console.print(table)


# 4. Rule violation counts.
violations = Counter(
    rule
    for row in judged_rows
    for rule in row["judge"].get("violated_rules", [])
)

console.print(
    "Instruction rule violations "
    "(R1 protection reply, R2 Sources line, "
    "R3 not-specified reply, R4 scope, R5 wording, R6 tone): "
    f"{dict(sorted(violations.items())) or 'none'}"
)


# 5. Other metrics.
table = Table(show_header=True, header_style="bold magenta")
table.add_column("Other metrics")
table.add_column("Value")

table.add_row(
    "ROUGE-L F (in-scope, mean)",
    f"{safe_mean([row['rouge_l'] for row in in_scope_rows]):.3f} "
    f"({len(in_scope_rows)})",
)

table.add_row(
    f"Refused, out of scope (target 1.0)",
    f"{safe_mean([float(bool(row.get('refused', False))) for row in out_of_scope_rows]):.3f} "
    f"({len(out_of_scope_rows)})",
)

table.add_row(
    "Refused, in scope (target 0.0)",
    f"{safe_mean([float(bool(row.get('refused', False))) for row in in_scope_rows]):.3f} "
    f"({len(in_scope_rows)})",
)

table.add_row(
    "Judge success rate",
    f"{len(judged_rows) / len(rows):.3f} ({len(judged_rows)}/{len(rows)})"
    if rows else "0.000 (0/0)",
)

console.print(table)


# 6. Display up to five answers with at least one failed criterion.
failed_rows = [
    row for row in judged_rows
    if not all(
        row["judge"].get(name) is True
        for name in CRITERIA
    )
]

for row in failed_rows[:5]:
    console.print(
        f"\n[bold]{escape(str(row.get('id', 'unknown')))}[/bold] "
        f"in_scope={row.get('in_scope')} "
        f"Q: {escape(str(row.get('question', '')))}"
    )

    console.print(f"A: {escape(str(row.get('answer', '')))}")

    for name in CRITERIA:
        verdict = row["judge"].get(name)
        reasoning = row["judge"].get(f"{name}_reasoning", "")
        console.print(
            f"  {name}={verdict}: {escape(str(reasoning))}"
        )

    console.print(
        f"  violated_rules={row['judge'].get('violated_rules', [])}"
    )


# 7. Save every row, including failed judgments.
with NAIVE_RESULTS_PATH.open("w", encoding="utf-8") as handle:
    for row in rows:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")

console.print(
    f"\nSaved {len(rows)} results to {NAIVE_RESULTS_PATH}"
)
