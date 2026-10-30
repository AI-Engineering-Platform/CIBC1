# Cleaning and Chunking Strategy

This note explains, in plain words, how we prepare the documents for the naive RAG and what can go wrong.
The code is in `00-RAG_naive.ipynb` (Steps 2 and 3).

## The documents

| Document | Role | Used for |
|---|---|---|
| CFPB credit card agreement | in scope | the RAG answers questions from it |
| SEC investor bulletin | out of scope | questions it inspires should be refused |

## 1. Cleaning

**The problem.** The CFPB file is hard-wrapped: a line break every ~120 characters, even in the middle of a sentence. If we leave it, the search and the question generator see broken sentences.

**What we do.** We go through each block of text line by line and decide whether the next line continues the sentence or starts something new.

1. We measure the usual full line width of the file (about 120 characters for the CFPB file).
2. A line is **joined** to the next one when it runs to the margin and stops mid-sentence.
3. A line **stays separate** when:
   - the next line starts a list item (`a.`, `1.`, `-`),
   - it ends a sentence (`. ? ! : ;`) well before the margin,
   - it is a short line followed by a capitalised line (a heading).
4. We remove page numbers (`3`, `Page 3 of 10`) and short lines that repeat 3 or more times (headers and footers). Neither document had any, so nothing was removed.

**Result.** About 225 wrapped lines in the CFPB file were joined, and 2 in the SEC file.

### Limitations of cleaning

- It uses layout rules, not meaning. It can be wrong when a layout rule and a sentence disagree.
- A wrapped line that happens to end in a full stop right at the margin may stay split.
- A genuinely short line that ends mid-sentence may be joined to the next one.
- Tables, columns, footnotes and bullet styles we have not seen are not handled specially.
- The 90th-percentile line width is a guess of the "full width". A different document may need different thresholds.
- Page numbers and headers are only removed by pattern, and the check was done on these two files only.
- The older `qa_text_generation` notebooks do not clean the text, so chunks here are not word-for-word the same as theirs.

## 2. Chunking

**What we do.**

1. **Split on blank lines.** Each paragraph (a block of text between blank lines) is one chunk. In the CFPB agreement this is roughly one numbered clause per chunk.
2. **Drop short pieces.** Anything under 80 characters (mostly headings) is dropped.
3. **Pick test chunks.** For each document, 10 chunks are chosen at random as the **test** set, and the rest are the **development** set. The seed is 42, so the choice is the same on every run.
4. **Save.** Chunks go to `data/chunks.jsonl` with an ID such as `cfpb_credit_card_agreement::p0005`, the document, its role and its split.

| Document | Development | Test | Total |
|---|---|---|---|
| CFPB (in scope) | 25 | 10 | 35 |
| SEC (out of scope) | 11 | 10 | 21 |

For now we generate questions and evaluate on development chunks only. Test chunks are kept for later.

## 3. How the search handles long chunks

The embedding model (`all-MiniLM-L6-v2`) reads at most 256 tokens. Five CFPB chunks are longer.

- Each chunk is cut into overlapping windows of 200 tokens (stride 160) only for the search index.
- Every window is embedded, and a chunk's score is the score of its best window.
- The search still returns whole chunks, so the answerer sees complete clauses.

## Limitations of chunking

- **Very uneven sizes.** CFPB chunks run from 99 to 4,376 characters. Large clauses cover many topics, so a question can match a chunk that contains the answer only in one corner.
- **No overlap between chunks.** If an answer needs two neighbouring paragraphs, it is split across chunks.
- **Structure-dependent.** It works because the CFPB file uses blank lines between clauses. A document without that structure would become one giant chunk.
- **Dropped short pieces.** Headings are removed, so their wording is lost as context.
- **Random test pick.** The 10 test chunks per document are chosen without looking at size or topic. The test set could miss important clauses (for example fees or defaults) or over-represent short ones.
- **Small data.** With only 25 development CFPB chunks, results are noisy, and one chunk changes the numbers noticeably.
- **Missing information.** Some answers are not in the document at all. For example, the late payment fee amount is on a separate Disclosure, so the RAG correctly says "Not specified in the agreement."
- **Windows are only for search.** Text beyond the best window of a long chunk does not help the ranking.

## Ideas to try later

- Fixed-size chunks with overlap, or sentence-based chunks, then compare retrieval metrics.
- Splitting very long clauses at sentence boundaries.
- A stratified test split (by position or by size) instead of a purely random one.
