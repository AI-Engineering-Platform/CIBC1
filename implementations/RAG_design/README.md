# Synthetic Q&A for Evaluating and Improving a Document RAG Agent

## 1. Project description

Build and evaluate a Retrieval-Augmented Generation (RAG) agent that retrieves and
extracts relevant information from a credit card agreement in response to user
questions.

**Business context.** Organizations building a document-based RAG system for the
first time usually have no ground-truth dataset. Without one they cannot reliably
assess answer quality, identify weaknesses, or refine the system.

**Technical challenges.**
1. Evaluating and improving a RAG system when no historical question-answer pairs exist.
2. Avoiding evaluation bias. Generating questions and answers with an LLM from the
   same source context can inflate accuracy and RAG metrics, giving an overly
   optimistic picture of system performance.

**Data.** CFPB sample credit card agreement (Vector pre-approved data list).
Known formatting issues: headers, footers and broken bullet points.

## 2. Goal and claim

Show that **mode-controlled, verified synthetic Q&A** gives a more realistic
evaluation of a RAG agent than naive synthetic Q&A, and gives the findings needed to
improve the agent.

> Naive synthetic Q&A overstates RAG quality. Controlled synthetic Q&A exposes real
> weaknesses, and fixing them improves the agent on questions it never saw.

## 3. Components

The design builds on `implementations/qa_text_generation` and the
`aieng.syn_data.text` package. SynBench (`agent_benchmark_generation`) contributes
ideas only, not code.

| Component | Source |
|---|---|
| Document prep and chunking | **Reuse:** `build_paragraph_splits`, `documents.py`. **Add:** cleanup of headers, footers and broken bullets. |
| Naive generator (control) | **Reuse:** `zero_shot_generate` |
| Controlled generator | **Reuse:** `topic_controlled_generate`, `FailureMode`. **Add:** multi-chunk and out-of-boundary modes, persona styles. |
| Filter | **Reuse:** `apply_heuristic_filters`, `filter_with_judge`. **Add:** support check and mode check. |
| RAG agent | **Reuse:** `retrieve_paragraphs` (TF-IDF) as the baseline retriever. **Add:** answering from retrieved chunks. |
| Judge | **Reuse:** `judge.py` |
| Human reference set | **New:** 20-30 handwritten questions, including out-of-boundary |
| Metrics | **Add:** Recall@k, MRR, ROUGE-L, BLEU, required-fact and refusal scoring |

## 4. Question modes

| Mode | What it tests |
|---|---|
| Plain factual | Basic retrieval and answering |
| Format | An explicit output format is requested (list, JSON, clause citation) |
| Vocabulary | Defined terms such as APR, grace period, minimum payment |
| Multi-constraint | Several rules or conditions in one question |
| Multi-chunk | The answer needs two or more chunks |
| Out of domain | Unrelated question, e.g. investment advice. Expect a refusal. |
| In domain, not in document | Looks relevant but is not stated. Expect "not specified". |

**Persona styles** (rushed, anxious, casual/paraphrasing) vary question wording so
questions do not copy the passage's phrasing. They plug into the generation prompt
through `extra_instruction`, and the style is stored in `QASample.metadata`.
Out-of-boundary questions make up about 15-25% of every split.

## 5. Data splits

Split **by chunk**, so no paragraph appears in two splits.

| Split | Use |
|---|---|
| Validation chunks | Diagnose failures and tune the agent |
| Test chunks | Frozen. Run once at the end. |
| Human set | Reality check. Never tuned on. |

## 6. Pipeline

1. **Prepare.** Clean and chunk the document, assign chunk IDs, split into
   validation and test.
2. **Naive baseline.** Generate naive Q&A. Run the baseline RAG (TF-IDF, top_k=3,
   fixed prompt, temperature 0). Record all metrics.
3. **Controlled Q&A.** Generate across modes and personas, then filter.
4. **Diagnose on validation.** Run the baseline RAG and report scores by mode. Label
   each failure:
   - **Retrieval:** the source chunk is not in the top k.
   - **Generation:** the chunk is present but the answer is wrong.
   - **Refusal:** it answered when it should refuse, or the reverse.
5. **Improve.** Apply one fix at a time and re-check on validation: chunk size and
   overlap, embedding or hybrid retrieval, top_k and reranking, the answer prompt
   with a refusal rule.
6. **Final experiment.** Run baseline and tuned RAG on the test set and the human set.

## 7. Filtering

Filtering removes bad synthetic Q&A before it is used for scoring.

1. **Heuristics:** empty items, length bounds, duplicate questions, answer inside the
   question, leaked prompt text.
2. **Support check:** the gold answer must be supported by the source chunk. For
   unanswerable items, no chunk may support an answer.
3. **LLM judge:** correctness, coherence, usefulness and faithfulness, scored 1-5.
   Keep items with an average of at least 3.5.
4. **Mode check:** the question really matches its mode (explicit format request,
   truly multi-part, truly unanswerable). Run it after the persona is applied.

## 8. Evaluation

| Category | Metrics |
|---|---|
| Retrieval | Recall@k (k = 1, 3, 5), MRR against the known source chunk |
| Overlap | ROUGE-L, BLEU against the gold answer |
| Quality | LLM judge: correctness and faithfulness to the retrieved context (1-5) |
| Required facts | Share of must-have facts or phrases present in the answer |
| Refusal | Accuracy on out-of-boundary items, counting both over-answering and over-refusing |
| Realism | Question-passage lexical overlap (also by persona), topic and type diversity, score gap to the human set |

ROUGE-L and BLEU only count word overlap and penalise correct answers that are
worded differently, so the judge and required-fact score cover that gap.

## 9. Success criteria

| Question | Evidence |
|---|---|
| Is naive evaluation inflated? | Baseline scores are clearly higher on the naive set than on the human set. |
| Is the controlled set more realistic? | Its scores are closer to the human set, and agent variants rank in the same order on both. |
| Does it improve the agent? | Tuned RAG beats baseline on the test and human sets. |

## 10. Controls and reference points

- **No-retrieval floor:** the same LLM with no context.
- **Oracle ceiling:** the gold chunk given directly (what notebook 01 of
  `qa_text_generation` does).
- Fixed seeds, temperature 0, equal sample sizes for naive and controlled sets,
  confidence intervals.
- Use different model families for question generation, answering and judging. The
  Vector proxy key only permits some models (e.g. `gpt-4o` works,
  others return `model_access_denied`), so check access first.

## 11. Mapping to the repo

| Need | Source |
|---|---|
| Chunking and splits | `documents.py`, `build_paragraph_splits` |
| Naive generator | `zero_shot_generate` |
| Controlled generator | `topic_controlled_generate`, `FailureMode`, `FAILURE_MODE_GUIDANCE` |
| Filter | `quality.py`, `filter_with_judge` |
| Retriever | `retrieve_paragraphs` (rebuilds its TF-IDF index on every call, so cache it) |
| Judge | `judge.py`, `score_predictions`, `JudgeScore` |
| Storage | `save_typed_jsonl`, `load_typed_jsonl` |

**Small changes needed**
- Add new values to `FailureMode` (`schemas.py`) and guidance text to
  `FAILURE_MODE_GUIDANCE` (`config.py`).
- A wrapper that rotates modes and personas instead of the role-based assignment in
  `generate_test_qa_batch`.
- A multi-chunk prompt that joins two paragraphs.

**New code**
- RAG answering step (fill the context from `retrieve_paragraphs`, not from
  `sample.context`).
- Recall@k, MRR, ROUGE-L, BLEU (add `rouge-score` or `sacrebleu` as a dependency),
  required-fact and refusal scoring.
- Support check and mode check prompts.
- Header, footer and bullet cleanup.
- Lexical-overlap realism metric.
- The human reference set.

Suggested location: a thin package `aieng/syn_data/rag/` (`metrics.py`,
`rag_agent.py`, `verify.py`) imported by the notebooks.

## 12. Ideas borrowed from SynBench (concepts only)

- Verify generated items before keeping them.
- Stratified sampling across modes x personas x distinct chunks.
- Persona styles for realistic wording.
- Hand-verified seed examples (the human set can also seed few-shot prompts).
- Config-driven generation.
- Outcome-style scoring (the required-fact metric).

## 13. Deliverables

Five notebooks in this folder:

1. Prepare and naive baseline
2. Controlled generation and filtering
3. Validation diagnosis
4. Tuning
5. Final test and human set

The final result is one table comparing naive vs controlled vs human sets, and
baseline vs tuned agent.

## 14. Out of scope

Red teaming (prompt-injection and leakage probes). Mention as future work.






**1. Modes control question generation.**
They tell the generator what kind of question to write (format, multi-chunk, out of domain and so on). They also label each question, so you can report scores per mode later.

**2. The mode check at filter time (a different job).**
The filter step asks "does this question really match its mode?" For example, a "format" question that is really plain recall gets dropped or relabelled. That is a quality check on the *data*. It uses the same kind of LLM, but it is not the evaluation judge.

**3. The evaluation judge never looks at modes.**
It scores the RAG agent's answer: correctness against the gold answer, and faithfulness to the retrieved chunks. The mode is only used afterwards to group the scores, for example "correctness on multi-chunk questions".

**4. Retrieval metrics are separate and need no judge.**
Recall@k and MRR compare the retrieved chunks with the known source chunk. For multi-chunk questions, you check that all source chunks were found. For out-of-boundary questions there is no source chunk, so retrieval is skipped and refusal accuracy is used instead.

| Stage | Who decides | Uses modes? |
|---|---|---|
| Generate Q&A | Generator LLM | Yes, it writes to the mode |
| Filter Q&A | Heuristics and a judge | Yes, it checks the mode matches |
| Evaluate the agent | Judge, ROUGE/BLEU, Recall@k | Only for grouping the results |

So the LLM judge appears in two places, the filter and the evaluation, with different prompts and different jobs. That is probably what caused the confusion.