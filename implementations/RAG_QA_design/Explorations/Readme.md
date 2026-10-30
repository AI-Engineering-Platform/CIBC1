**More Details on Exploration**

1. **12_RAG_naive_vs_controlled (Yujia).ipynb**

    Naive Q&A vs controlled, mode-targeted Q&A
    This notebook extends 1_RAG_naive.ipynb with a matched comparison:

    Naive Q&A (control): one model, one chunk, and one plain prompt generate each question and answer.
    Controlled Q&A: one model uses mode-specific instructions and answer contracts for plain factual, format, vocabulary, multi-constraint, multi-chunk, out-of-scope, and not-specified questions.
    The experiment writes JSONL records and reports mode-level metrics. It evaluates a stratified, evenly spaced sample from every available document rather than selecting random chunks from one document. Set RUN_GENERATION = True to call        the configured model; the default is False so the analysis can be rerun from saved records without incurring API costs.

2. **1_Visa_Naive_vs_Controlled_QA_Experiments.ipynb**

    Experiments: Naive vs Controlled Synthetic QA
    Run from implementations/qa_text_generation. Uses existing chunk CSV if present; otherwise generates page-aware chunks from the PDF. No experimental results are pre-filled. API usage incurs costs. The same chunk sample, model and             question budget are used for both conditions.

3. **1_Visa_PDF_Synthetic_QA_Generation (yujia).ipynb**
   
    Synthetic QA Generation from a Single PDF
    Input: Visa_Credit_Card_Agreement.pdf (6 pages)
    Output: grounded synthetic question–answer pairs in JSONL and CSV, plus validation and summary files.

    This notebook extracts page-aware text, chunks it, uses an OpenAI model to generate diverse QA pairs with verbatim evidence quotes, validates the evidence against the extracted text, deduplicates, and exports results. It also generates       explicitly unanswerable questions (with null answers) to evaluate hallucination resistance. Model-generated questions and answers should be reviewed before use as a benchmark.

    **Important: The agreement refers to a separate Account Opening Disclosure for actual APRs and many fee amounts. The notebook instructs the model not to invent those missing values.**
