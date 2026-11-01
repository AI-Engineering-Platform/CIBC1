# Takeaway: Naive and Controlled Synthetic Q&A

## Problem Statement

A RAG project often starts without historical user questions or trusted reference answers. A tempting baseline is to ask a language model to generate questions and gold answers directly from the same document chunks used by the RAG system. That can produce an evaluation set that is too closely aligned with its source material and may overstate system quality, while offering little guidance about specific failure modes.

Our hypothesis was that naive synthetic Q&A could make the RAG agent look artificially strong, while more challenging, mode-controlled Q&A would expose weaknesses in retrieval, answer generation, refusal behavior, and instruction following. The work below tests that hypothesis and builds a reusable way to evaluate prompt and RAG behavior before production.

## 1. Naive Q&A Generation

The naive generator was run over all 56 saved chunks. It generated 10 candidate questions per chunk: 560 candidates total. CFPB chunks produced in-scope questions with model-written gold answers; SEC chunks produced out-of-scope questions with a fixed refusal gold answer.

A source-evidence check removed 55 candidates whose cited evidence was not found in the source chunk. The heuristic filter removed no additional records, leaving 505 questions in the combined `data/naive_qa.jsonl`. 

This is the simple baseline: questions are grounded in one source chunk, without explicit controlled failure-mode targets.

## 2. Controlled Synthetic Q&A

The controlled generator creates topic- and mode-targeted questions, including format non-compliance, messy customer wording, multi-constraint collapse, multi-chunk retrieval, out-of-domain, near-miss, protection, complex structured, and pressure cases. The heuristic stage checks record structure and consistency; the LLM verifier checks correctness, support, completeness, answerability, mode compliance, naturalness, and applicable special criteria. Fixed replies are checked for out-of-scope questions.

**Modes and generation counts.** CFPB chunks supply in-scope questions and the out-of-domain hard negatives. SEC investor-bulletin chunks supply out-of-scope questions. Counts below are generated raw candidates and the number that remained in the verified file:

| Source document | Mode | Raw generated | Verified |
|---|---|---:|---:|
| CFPB agreement | `plain_factual` | 113 | 112 |
| CFPB agreement | `format_non_compliance` | 116 | 113 |
| CFPB agreement | `messy_questions` | 117 | 117 |
| CFPB agreement | `multi_constraint_collapse` | 77 | 77 |
| CFPB agreement | `multi_chunk` | 65 | 61 |
| CFPB agreement | `out_of_domain` | 74 | 74 |
| **CFPB total** | | **562** | **554** |
| SEC bulletin | `plain_factual` | 16 | 8 |
| SEC bulletin | `near_miss` | 24 | 24 |
| SEC bulletin | `protection` | 20 | 20 |
| SEC bulletin | `complex_structured` | 22 | 22 |
| SEC bulletin | `pressure` | 15 | 15 |
| **SEC total** | | **97** | **89** |
| **All modes** | | **659** | **643** |

The gold-answer policy for out-of-scope cases is explicit in the generator:

- CFPB `out_of_domain`: exactly **“Not specified in the agreement. Please contact your card issuer for more information at 123@gmail.com.”**
- SEC `near_miss`, `complex_structured`, and `pressure`: the same exact “Not specified” reply.
- SEC `protection`: exactly **“Please refer to the protection section of your online investment for guidance.”**
- SEC `plain_factual`: one of those two fixed replies, selected from the question wording by the protection-keyword rule.
- Other CFPB in-scope modes: a model-written gold answer grounded in the source chunk(s), with the requested structure for `format_non_compliance` and an integrated answer for `multi_chunk`.

Current saved pipeline counts:

| Stage | Questions |
|---|---:|
| Generated raw candidates | 659 |
| Passed structural filtering | 646 |
| Passed LLM verification | 643 |
| Rejected by heuristic filtering | 13 |
| Rejected by LLM verification | 3 |

After verification, the 643 accepted QA records were shuffled at the question level with a fixed seed and split 70/30: 450 development and 193 test. This is not a chunk-grouped split, so questions sourced from the same chunk can appear in both splits.

## 3. Random-200 Comparison

Step 8 samples 200 questions without replacement from each set using seed 123, then runs the same RAG answerer and LLM judge. The naive set is sampled from the combined naive file; the controlled set is sampled from verified development records.

- Naive: 200 sampled from the 505-row combined naive file: 115 in-scope and 85 out-of-scope questions.
- Controlled: 200 sampled from the 450 verified development records: 152 in-scope and 48 out-of-scope questions.
- RAG answerer: `gemini-3.5-flash`, retrieving the top 5 CFPB chunks.
- LLM judge: `gpt-4o`. All 200 answers in each set were judged; no judge calls failed.

The Step 8 evaluation judge checks these instruction-following rules. The current prompt defines **eight** rules (R1-R8), including the requested-format rule R8:

| Rule | Requirement |
|---|---|
| R1 | Protection-related questions (investment protection, account privacy, or fraud) get the exact prescribed protection reply. |
| R2 | Answers based on retrieved context end with a `Sources:` line containing the chunk IDs used; prescribed refusals are exempt. |
| R3 | If the answer is unavailable in retrieved chunks, use the exact “Not specified in the agreement” reply; do not use it when the chunks contain the answer or answer from outside context. |
| R4 | Give guidance on credit-card agreements only. |
| R5 | Do not refer to “the document”, “the passage”, or similar source wording. |
| R6 | Maintain a polite, professional tone. |
| R7 | Do not hallucinate or answer an unanswerable or highly unreadable question. |
| R8 | Follow any specific output format requested by the user. |

| Metric | Naive | Controlled development | Controlled minus naive |
|---|---:|---:|---:|
| Recall@3 (in-scope) | 0.913 | 0.954 | +0.041 |
| Recall@5 (in-scope) | 0.948 | 0.987 | +0.039 |
| MRR (in-scope) | 0.818 | 0.879 | +0.061 |
| Reply correctness (all) | 0.745 | 0.910 | +0.165 |
| ROUGE-L (in-scope) | 0.475 | 0.590 | +0.115 |
| LLM answer correctness (all) | 0.810 | 0.885 | +0.075 |
| LLM factuality (in-scope) | 0.965 | 0.921 | -0.044 |
| LLM citation accuracy (in-scope) | 0.983 | 0.928 | -0.055 |
| LLM instruction following (all) | 0.805 | 0.885 | +0.080 |

The most frequent judge-flagged instruction violation was R3 (incorrect use of the prescribed not-specified reply): 36 naive versus 21 controlled. R1 protection-reply violations were 6 versus 3; R2 Sources-line violations were 2 versus 6. In the controlled sample, `multi_chunk` was a notable weak spot (15 questions; answer correctness 0.667 and citation accuracy 0.733).

**Which controlled modes failed instruction following?** The judge marked 23 of 200 controlled answers false for instruction following, tied with answer correctness for the largest false count among the four judge criteria. The mode breakdown shows where those failures clustered:

| Controlled mode | Instruction-following failures / questions | Failure rate |
|---|---:|---:|
| `sec/complex_structured` | 4 / 6 | 66.7% |
| `sec/plain_factual` | 2 / 4 | 50.0% |
| `cfpb/multi_chunk` | 5 / 15 | 33.3% |
| `sec/pressure` | 1 / 4 | 25.0% |
| `sec/near_miss` | 1 / 5 | 20.0% |
| `cfpb/multi_constraint_collapse` | 4 / 32 | 12.5% |
| `cfpb/messy_questions` | 4 / 33 | 12.1% |
| `cfpb/format_non_compliance` | 2 / 38 | 5.3% |

This mode-level view makes the controlled set useful for diagnosis, not just scoring: it points to likely stress points such as following complex structured requests and combining evidence across chunks, which can guide targeted prompt or retrieval changes and focused regression tests. Interpret rates alongside their denominators; the high percentages for SEC modes come from very small samples, while `multi_chunk` has the highest absolute failure count. The naive set has no mode labels, so it cannot provide the same failure attribution.

The comparison is exploratory, not a controlled causal experiment. The samples have different in-scope/out-of-scope proportions (115/85 naive versus 152/48 controlled), different question modes, and different chunk-split provenance. Retrieval, ROUGE-L, factuality, and citation metrics use only the in-scope subset; reply correctness and all other judge metrics use all 200 questions. Each result is one random sample, with no confidence intervals.

## Conclusion

On this 200-question sample, controlled Q&A scored higher on all reported retrieval and answer-quality metrics except LLM-judged factuality and citation accuracy, where naive Q&A remained higher. The largest aggregate differences were reply correctness (+0.165) and ROUGE-L (+0.115) in favor of controlled Q&A. Its instruction-following score was also higher (+0.080), with fewer R3 violations. This supports the value of targeted questions for surfacing behavior and improving diagnostic coverage, but it does not show that controlled generation is universally superior: the controlled sample's factuality and citation scores were lower, and multi-chunk questions remained difficult.

The strongest takeaway is the value of explicit failure-mode coverage. A mode-labeled, verified dataset can show where an agent is weak, such as format compliance, multi-constraint handling, retrieval across chunks, refusal calibration, or protection/privacy behavior. That is particularly useful for governance and policy-sensitive domains: it connects observed failures to concrete prompt and system requirements and supports targeted remediation before production.

Treat the current 200-row comparison as a useful diagnostic, not proof that one dataset is universally superior. The unequal scope proportions can affect aggregate comparisons, particularly reply correctness and instruction following. For a stronger estimate, repeat sampling across seeds, balance scope proportions, and report results by mode and scope. To test improvements, change one RAG or prompt component at a time and evaluate against held-out questions, preferably with source chunks kept disjoint between development and test.
