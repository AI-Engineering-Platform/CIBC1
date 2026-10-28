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

## 3. How to read this document

The pipeline has seven steps. Each step says what we do, whether it is **done** (built in
`00-RAG_naive.ipynb`) or **next**, and which existing code can be reused.

Reusable code lives in the `aieng.syn_data.text` package, referred to below as `text/`:
`aieng-synthetic-data/aieng/syn_data/text/`. The reference project is
`implementations/qa_text_generation`. SynBench (`agent_benchmark_generation`) contributes
ideas only, not code.

## 4. Pipeline

### Step 1. Document preparation and chunking (done)

Goal: turn the raw agreement into clean chunks with stable IDs.

- Clean the text with our own `clean_text`: join lines that continue the previous line,
  and drop headers, footers and page numbers.
- Split the cleaned text on blank lines into chunks (`min_chars=80`).
- Hold out 10 chunks per document as test chunks (random, seed 42). The rest are
  development chunks. Only development chunks are used until the final test.
- The CFPB agreement is the in-scope document. The SEC investor bulletin is the
  out-of-scope document, used only to write out-of-scope questions.

Reusable code:
- `text/documents.py`: `load_document_text`, `paragraphs_from_document`, `sample_test_paragraphs`
- `text/schemas.py`: `Paragraph`, `ParagraphSplit`
- `text/pipeline.py`: `build_paragraph_splits`

Known limits of the cleaning are in `challenge.md`.

### Step 2. Indexing and retrieval (done)

Goal: find the chunks that answer a question.

- Embed chunks with `all-MiniLM-L6-v2` (sentence-transformers).
- Each chunk is covered by 200-token windows with a stride of 160. A chunk is scored by
  its best window.
- Retrieve the top 5 chunks. The index is cached in `data/index/`.

This replaces the TF-IDF retriever `retrieve_paragraphs` in `text/rag.py`, which we do
not use. Later tuning can try hybrid retrieval or reranking.

### Step 3. RAG agent (done)

Goal: answer a question from the retrieved chunks.

- Model: `gemini-3.5-flash`, with `ANSWER_SYSTEM_PROMPT` in the notebook.
- The prompt rules are: answer only from the context, end with `Sources: <chunk ids>`,
  reply with a fixed sentence when the answer is not in the context, reply with a fixed
  sentence for protection, fraud or privacy questions, only discuss credit card
  agreements, never say "the document says", and stay polite.
- The answer is retried when it is incomplete (no `Sources:` line and no refusal).

### Step 4. Naive Q&A (done)

Goal: the simple question set that the controlled set will be compared against.

- A model (`gpt-4o`, temperature 0.7) gets one chunk and writes 3 questions with
  gold answers. Out-of-scope questions are written from the SEC chunks.
- Items whose evidence is not in the chunk are dropped.
- Output: about 80 questions, about 13% out of scope.

The prompt is new and simple on purpose. `zero_shot_generate` in `text/generation.py` is
the closest existing function, but we do not call it.

### Step 5. Evaluation (done)

Goal: score the RAG agent on a question set.

Ranking, with no judge needed:
- Recall@3, Recall@5 and MRR against the known source chunk, on in-scope questions.

Overlap:
- ROUGE-L F against the gold answer, after removing the `Sources:` line.

Refusal:
- Share of out-of-scope questions that were refused (want 1.0), and share of in-scope
  questions that were refused (want 0.0).

LLM judge (`gpt-4o`, temperature 0):
- Inputs: the question, the RAG answer, the retrieved chunks and the RAG instructions.
  It never sees the gold answer.
- Outputs, each with a written reasoning:
  - factuality: the answer is grounded in the retrieved chunks
  - citation accuracy: the chunk IDs in `Sources:` are the ones actually used
  - instruction following: the answer obeys the rules R1 to R6 below, and the judge
    lists any rule that was broken in `violated_rules`
- Rules: R1 protection reply, R2 `Sources:` line, R3 "Not specified" reply (a
  violation if the chunks do contain the answer), R4 credit card scope only, R5 no
  "the document says", R6 polite tone.

Results are saved to `data/naive_results.jsonl` and to two Excel files in
`data/exports/` (answers next to gold answers, and the judge output).

This judge is our own prompt. The existing `text/judge.py` helpers
(`absolute_response_judge_prompt`, `pairwise_response_judge_prompt`) score an answer
against a reference and do not look at retrieved chunks or citations, so we do not use
them. Their structure (JSON output, retry, `parse_judge_score`) is still a good model.

### Step 6. Controlled Q&A (next)

Goal: questions that are harder than the naive set, each labelled with the mode it was
written for, so scores can be reported per mode.

**Existing code to refer to**

All in `aieng-synthetic-data/aieng/syn_data/text/`.

- `generation.py`: how Q&A is written with modes and topics.
  - `_base_generation_prompt` builds the prompt: "Generate one challenging
    question-answer pair about this passage. The answer must be grounded in the passage
    only. Return JSON with keys: question, gold_answer." If a mode is given it adds
    "Target failure mode: <mode>" and the guidance text for that mode. If a topic is
    given it adds "Focus topic: <topic>. The question must be specifically about this
    topic and answerable from the passage alone."
  - `extract_topics` asks the model for up to 5 short topics that the chunk explicitly
    supports.
  - `topic_controlled_generate` writes one question per topic, and stores the topic in
    `QASample.metadata`.
  - `_ask_json` sends the prompt and parses the JSON reply.
- `schemas.py`: `FailureMode` (the list of modes) and `QASample` (question, gold answer,
  chunk id, mode, metadata).
- `config.py`: `FAILURE_MODE_GUIDANCE` (the text added to the prompt for each mode) and
  `DOCUMENT_ROLE_BY_FAILURE` (which document gets which modes).
- `pipeline.py`: `failure_modes_for_paragraph` and `generate_test_qa_batch` pick modes
  for a chunk and rotate them across its topics, one question per topic.

The existing code is used in `implementations/qa_text_generation/01_baseline_evaluation.ipynb`
(section 3) and explained there.

**Modes and topics in the existing code**

Topics: for each chunk, the model lists up to 5 topics that the chunk explicitly
supports (for example grace period, late fee, APR). One question is then written per
topic, so a long chunk is covered by several different questions instead of one.

Modes: four, defined in `FailureMode`. Each one is meant to expose a different mistake
the answering model can make.

- format non-compliance: the question asks for a set output structure such as JSON, a
  numbered list or a clause citation. It exposes answers that ignore the format (extra
  preamble, missing list or citation).
- domain vocabulary drift: the question uses the exact terms of the passage (APR, grace
  period, fiduciary). It exposes answers that paraphrase or swap defined terms.
- refusal calibration: a mix of in-scope and out-of-scope questions, where the gold
  answer refuses when needed. It exposes answering what should be refused, and refusing
  what should be answered.
- multi-constraint collapse: one question that stacks several rules or exceptions. It
  exposes answers that drop one of the conditions.

Which mode is used where: the dense policy document (CFPB) gets format, vocabulary and
multi-constraint. The scope-boundary document (SEC bulletin) gets refusal and format.
Modes are set only for the test questions. The training questions are written with no
mode.

**Suggestion for our project**

The existing modes were chosen for a model that follows instructions poorly. A RAG
agent also fails at retrieval, so we can think of our own modes and topics. A starting
set, to adjust as we see where the agent fails:

- plain factual: one fact from one chunk, basic retrieval and answering (the naive set
  is mostly this)
- vocabulary: defined terms such as APR, grace period, minimum payment
- multi-constraint: several conditions in one question
- format: asks for a list or a clause citation (the agent already has to end with a
  `Sources:` line)
- multi-chunk: the answer needs two or more chunks, which tests retrieval
- out of domain: unrelated to credit cards, expect the scope refusal
- in domain but not in the document: looks relevant but is not stated, expect "Not
  specified in the agreement..."
- protection, fraud or privacy: expect the fixed protection reply (rule R1)

For topics we can reuse `extract_topics` as it is (this generates topic based on the given chunk, up to 5 topics), or fix a topic list for the credit
card agreement (fees, interest, payments, billing disputes, default, privacy).

To implement: add the new values to `FailureMode` and their guidance text to
`FAILURE_MODE_GUIDANCE`, write a wrapper that rotates modes across chunks instead of
the document-role rule in `generate_test_qa_batch`, and write a multi-chunk prompt that
joins two chunks. Aim for about 15 to 25% out-of-scope questions.

### Step 7. Filtering and validation of Q&A (next)

Goal: remove bad generated Q&A before it is used to score the RAG agent.

**Existing code to refer to**

- `quality.py`: cheap rule-based checks, no LLM. `apply_heuristic_filters` runs
  `heuristic_rejection_reasons` on every item and returns the kept items and a rejection
  record with the reasons. `summarize_heuristic_rejections` counts the reasons.
- `pipeline.py`: `filter_with_judge` runs the heuristics, then asks an LLM judge to
  score each remaining item, and keeps those at or above the threshold.
- `judge.py`: `judge_synthetic_sample` sends one Q&A pair to the judge and
  `parse_judge_score` reads the result.
- `prompts.py`: `SYNTHETIC_QA_JUDGE_SYSTEM_PROMPT` and `synthetic_qa_quality_prompt`, the
  judge prompt.
- `config.py`: `DEFAULT_JUDGE_THRESHOLD` (3.5).
- `rag.py`: `grounding_overlap_score`, the share of answer words that appear in the
  passage. It is used by the back-translation path to drop answers that are not
  grounded.
- `implementations/qa_text_generation/03_quality_filtering.ipynb` shows the flow.

**What the existing checks do**

Rule-based checks (`quality.py`), in this order:
- missing required fields (question, answer, document id, chunk id)
- length bounds (question 12 to 500 characters, answer 8 to 8000)
- prompt leakage (the generation instruction or "return JSON" shows up in the question or
  answer)
- answer in the question (the gold answer is copied into the question)
- duplicate question (after lowercasing and removing punctuation)

LLM judge (`prompts.py`): it sees the source chunk, the question and the gold answer, and
gives a 1 to 5 score for four things, plus a short reasoning. The scores are averaged
and an item is kept at 3.5 or more.
- correctness: is the gold answer accurate for the question, given the chunk
- coherence: are the question and answer clear and well formed
- instruction following: is the question a natural, useful question
- factual plausibility: is the answer grounded in the chunk

Important: the existing judge does not look at the mode. It does not check that a
"format" question really asks for a format, or that a "refusal" question really cannot
be answered. In the existing project the mode is only a label, and it is used after
inference to group the scores. It also does not look at retrieved chunks, because the
question is judged against its own source chunk only.

This step is only to make sure generated QAs quality is actually high, diverse, and the golden asnwers are actually correct for the downstream task.
**How each mode is caught in the existing project**

The mode is checked when the answering model is scored, not when the question is
written. This table lists what each mode exposes (from the failure-mode table in
`01_baseline_evaluation.ipynb`). The "where it is caught" column is our reading of the
judge criteria: the notebook states it only for hallucination.

| Mode | What goes wrong | Where it is caught |
| --- | --- | --- |
| Format non-compliance | Extra preamble, missing JSON, list or citation | Judge instruction following |
| Domain vocabulary drift | Defined terms are paraphrased or swapped | Judge correctness |
| Refusal calibration | Answers an out-of-scope question, or refuses an in-scope one | Judge correctness, gold answer is a refusal |
| Multi-constraint collapse | One clause of a stacked question is dropped | Judge correctness |
| Hallucination (not its own mode) | Fluent facts the chunk does not support | Judge correctness and factual plausibility |

**Suggestion for our project**

Keep the existing rule-based checks and the judge as the base (`apply_heuristic_filters`,
`filter_with_judge`). Because we write questions by mode, we should also check the data
itself, using prompts built like `synthetic_qa_quality_prompt` (a prompt, a JSON answer,
a parse function, temperature 0). Starting criteria:

- support check: the gold answer is supported by the source chunk. For multi-chunk
  items, all the chunks are needed.
- unanswerable check: for "not in the document" items, no chunk gives an answer. For
  out-of-domain items, the question really is unrelated to credit cards.
- mode check: the question really matches its mode (an explicit format request, truly
  several conditions, truly needs two chunks).
- gold answer follows our answer rules: the gold answer for a refusal uses the fixed
  refusal sentence, and the gold answer for a normal question is not already a refusal.
- no near-duplicates across modes, and a balanced count per mode.

Other validation criteria can be added later if the data shows other problems, for
example question clarity (can it be understood without the chunk) or whether a question
is answerable by general knowledge alone.

The filter judge is not the evaluation judge of Step 5. The filter judge checks the Q&A
data. The evaluation judge checks the RAG agent's answers. They use different prompts.

### Step 8. Diagnose, improve, final test (next)

1. Diagnose on the validation chunks. Run the RAG agent on the controlled set and report
   scores per mode. Label each failure as:
   - retrieval: the source chunk is not in the top 5
   - generation: the chunk is present but the answer is wrong
   - refusal: it answered when it should refuse, or the reverse
2. Improve one thing at a time and re-check on validation: chunk size and overlap,
   embedding or hybrid retrieval, top_k and reranking, the answer prompt.
3. Final experiment. Run the baseline and the tuned RAG once on the frozen test chunks.

## 5. Data splits

Split by chunk, so no chunk appears in two splits.
- Development (validation) chunks: used to diagnose failures and tune the agent.
- Test chunks: frozen, run once at the end.


## 6. Success criteria

- Is naive evaluation inflated? Baseline scores are clearly higher on the naive set than
  on the controlled set.
- Does the controlled set find real weaknesses? Scores differ clearly by mode, and the
  failures can be labelled as retrieval, generation or refusal.
- Does it improve the agent? The tuned RAG beats the baseline on the frozen test chunks.

## 7. Controls and good practice

- Fixed seeds, temperature 0 for answering and judging, equal sample sizes for the naive
  and controlled sets, confidence intervals.
- Use different model families for question generation, answering and judging. Our setup
  is `gpt-4o` (generator), `gemini-3.5-flash` (answerer) and `gpt-4o` (judge). The proxy
  key only allows `gpt-4o`, `gemini-3.5-flash` and `gemini-2.5-flash`.
- ROUGE-L only counts word overlap and penalises correct answers that are worded
  differently, so read it together with the judge.
- Question generation uses temperature 0.7, so each run writes a new naive set. Save the
  set before comparing runs.

## 8. Remaining metrics

Not built yet: BLEU, required-fact score (share of must-have facts present in the
answer), refusal accuracy per mode, and diversity checks (topics and question types).

Storage helpers for typed JSONL are in `text/io.py` (`save_typed_jsonl`,
`load_typed_jsonl`).

## 9. Deliverables

Notebooks in this folder:
1. `00-RAG_naive.ipynb`: prepare, index, naive RAG, naive Q&A, evaluation (done)
2. Controlled generation and filtering
3. Validation diagnosis
4. Tuning
5. Final test

The final result is one comparison of the naive and controlled sets, and of the
baseline and tuned agent.

## 10. Ideas borrowed from SynBench (concepts only)

- Verify generated items before keeping them.
- Stratified sampling across modes and distinct chunks.
- Config-driven generation.
- Outcome-style scoring (the required-fact metric).

## 11. Out of scope

Red teaming (prompt-injection and leakage probes). Mention as future work.
