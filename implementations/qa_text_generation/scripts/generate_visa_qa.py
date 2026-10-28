"""Generate an unsplit synthetic QA corpus from a Visa credit card agreement.

This is the script equivalent of notebook 02 for a single PDF. Every extracted
paragraph is used for generation; no train/test holdout is created.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Any

import pymupdf
from aieng.syn_data.text import (
    DocumentRole,
    DocumentSpec,
    Paragraph,
    ParagraphSplit,
    effective_synthetic_target,
    generate_grounded_training_corpus,
    generate_raw_synthetic_corpus,
    load_implementation_dotenv,
    paragraphs_from_document,
    save_typed_jsonl,
    use_repo_root,
)
from openai import OpenAI


LOGGER = logging.getLogger(__name__)
DEFAULT_PDF = Path("implementations/qa_text_generation/data/documents/Visa_Credit_Card_Agreement.pdf")
DEFAULT_OUTPUT_DIR = Path("implementations/qa_text_generation/data/visa_synthetic")
DEFAULT_MIN_CHARS = 80
DEFAULT_QUESTIONS_PER_PARAGRAPH = 2
DEFAULT_MIN_OVERLAP = 0.15
DEFAULT_OPENAI_BASE_URL = "https://api.openai.com/v1"


class VisaOpenAIClient:
    """OpenAI SDK adapter matching the text-generation client's interface."""

    def __init__(
        self,
        *,
        model: str,
        api_key: str,
        base_url: str = DEFAULT_OPENAI_BASE_URL,
    ) -> None:
        self.model = model
        self.client = OpenAI(api_key=api_key, base_url=base_url)

    def complete(
        self,
        prompt: str,
        *,
        system: str | None = None,
        temperature: float = 0.2,
        max_tokens: int = 1024,
        response_format: dict[str, Any] | None = None,
    ) -> str:
        """Complete a prompt using the direct OpenAI chat-completions API."""
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format=response_format,
        )
        content = response.choices[0].message.content
        if content is None:
            raise ValueError(f"OpenAI model {self.model} returned empty content.")
        return content

    def complete_json(
        self,
        prompt: str,
        *,
        system: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 512,
    ) -> dict[str, Any]:
        """Request and parse a JSON object from the direct OpenAI API."""
        raw = self.complete(
            prompt,
            system=system,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format={"type": "json_object"},
        )
        payload = json.loads(raw)
        if not isinstance(payload, dict):
            raise ValueError(f"Expected a JSON object, got {type(payload).__name__}.")
        return payload


def create_visa_teacher_client() -> VisaOpenAIClient:
    """Create the direct OpenAI teacher client used by the PDF workflow."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY must be set before generating Visa QA.")
    return VisaOpenAIClient(
        api_key=api_key,
        model=os.getenv("TEACHER_MODEL", "gpt-4o"),
        base_url=os.getenv("VISA_OPENAI_BASE_URL", DEFAULT_OPENAI_BASE_URL),
    )


def extract_pdf_text(pdf_path: Path) -> str:
    """Extract page text from a PDF while retaining page boundaries."""
    if not pdf_path.exists():
        raise FileNotFoundError(f"Visa agreement PDF not found: {pdf_path}")

    with pymupdf.open(pdf_path) as document:
        pages = [page.get_text("text").strip() for page in document]
    text = "\n\n".join(page for page in pages if page)
    if not text.strip():
        raise ValueError(f"Visa agreement PDF contains no extractable text: {pdf_path}")
    return text


def build_unsplit_paragraphs(
    pdf_path: Path,
    *,
    min_chars: int = DEFAULT_MIN_CHARS,
) -> list[Paragraph]:
    """Extract and chunk the agreement without assigning a train/test holdout."""
    spec = DocumentSpec(
        doc_id="visa_credit_card_agreement",
        title="Visa Credit Card Agreement",
        role=DocumentRole.POLICY_DENSE,
        domain="finance",
        local_path=str(pdf_path),
    )
    paragraphs = paragraphs_from_document(
        spec,
        extract_pdf_text(pdf_path),
        min_chars=min_chars,
    )
    return [
        Paragraph(
            doc_id=paragraph.doc_id,
            para_id=paragraph.para_id,
            text=paragraph.text,
            role=paragraph.role,
            split=ParagraphSplit.ALL,
            index=paragraph.index,
        )
        for paragraph in paragraphs
    ]


def generate_visa_qa(
    pdf_path: Path,
    output_dir: Path,
    *,
    max_paragraphs: int | None = None,
    questions_per_paragraph: int = DEFAULT_QUESTIONS_PER_PARAGRAPH,
    target_size: int | None = None,
    min_chars: int = DEFAULT_MIN_CHARS,
    min_overlap: float = DEFAULT_MIN_OVERLAP,
) -> dict[str, Any]:
    """Generate notebook-02-style QA streams and a combined overall corpus."""
    paragraphs = build_unsplit_paragraphs(pdf_path, min_chars=min_chars)
    selected = paragraphs if max_paragraphs is None else paragraphs[:max_paragraphs]
    if not selected:
        raise ValueError("The Visa agreement produced no paragraphs for generation.")

    teacher = create_visa_teacher_client()
    raw_samples = generate_raw_synthetic_corpus(
        teacher,
        selected,
        max_paragraphs=len(selected),
        questions_per_para=questions_per_paragraph,
    )
    ibt_target = (
        target_size
        if target_size is not None
        else effective_synthetic_target(selected, one_per_paragraph=True)
    )
    ibt_samples = generate_grounded_training_corpus(
        teacher,
        selected,
        target_size=ibt_target,
        min_overlap=min_overlap,
    )
    all_samples = raw_samples + ibt_samples

    output_dir.mkdir(parents=True, exist_ok=True)
    save_typed_jsonl(
        output_dir / "visa_qa_paragraphs.jsonl",
        selected,
        to_dict=lambda paragraph: paragraph.to_dict(),
    )
    save_typed_jsonl(
        output_dir / "visa_qa_raw.jsonl",
        raw_samples,
        to_dict=lambda sample: sample.to_dict(),
    )
    save_typed_jsonl(
        output_dir / "visa_qa_back_translation.jsonl",
        ibt_samples,
        to_dict=lambda sample: sample.to_dict(),
    )
    save_typed_jsonl(
        output_dir / "visa_qa_all.jsonl",
        all_samples,
        to_dict=lambda sample: sample.to_dict(),
    )

    summary = {
        "source": str(pdf_path),
        "paragraphs": len(selected),
        "split": "all",
        "raw_count": len(raw_samples),
        "back_translation_count": len(ibt_samples),
        "overall_count": len(all_samples),
        "outputs": {
            "paragraphs": str(output_dir / "visa_qa_paragraphs.jsonl"),
            "raw": str(output_dir / "visa_qa_raw.jsonl"),
            "back_translation": str(output_dir / "visa_qa_back_translation.jsonl"),
            "all": str(output_dir / "visa_qa_all.jsonl"),
        },
    }
    (output_dir / "visa_qa_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def parse_args() -> argparse.Namespace:
    """Parse command-line options for the generation script."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdf", type=Path, default=DEFAULT_PDF)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-paragraphs", type=int)
    parser.add_argument(
        "--questions-per-paragraph",
        type=int,
        default=DEFAULT_QUESTIONS_PER_PARAGRAPH,
    )
    parser.add_argument("--target-size", type=int)
    parser.add_argument("--min-chars", type=int, default=DEFAULT_MIN_CHARS)
    parser.add_argument("--min-overlap", type=float, default=DEFAULT_MIN_OVERLAP)
    return parser.parse_args()


def main() -> None:
    """Run Visa agreement QA generation from the command line."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args()
    use_repo_root(Path(__file__).resolve().parents[3])
    load_implementation_dotenv()
    summary = generate_visa_qa(
        args.pdf,
        args.output_dir,
        max_paragraphs=args.max_paragraphs,
        questions_per_paragraph=args.questions_per_paragraph,
        target_size=args.target_size,
        min_chars=args.min_chars,
        min_overlap=args.min_overlap,
    )
    LOGGER.info("Generated %d overall QA samples.", summary["overall_count"])


if __name__ == "__main__":
    main()
