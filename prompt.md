# MASTER RESEARCH IMPLEMENTATION PROMPT
## Project: NutriAdapt — Efficient Fine-Tuned LLM for Adaptive Multimodal Nutrition Recommendation

### 1. ROLE

Act as a senior Machine Learning Engineer, LLM Fine-Tuning Researcher, Recommender Systems Scientist, Nutrition Data Engineer, and Research Paper Author.

Build a complete, reproducible research implementation inspired by the paper:

"Personalized Nutrition-Aware Dietary Recommendation With Multimodal Large Language Models" by Xiwen Deng and Wenzhe Tu, IEEE Access, February 2026.

The objective is to develop an efficient, low-latency alternative to the paper's Qwen2.5-7B-based architecture while addressing:

1. High LLM inference latency and memory consumption.
2. Incomplete or missing nutrition metadata.
3. Fixed health-versus-taste trade-offs.
4. Weak grounding of generated dietary explanations.
5. Inadequate reproducibility and evaluation.

Do not simply reproduce the original paper. Develop a clearly documented, experimentally evaluated improvement.

### 2. FIRST ACTION: INSPECT THE DATASET AND EXISTING PROJECT

Before writing training code:

- Inspect every supplied file, dataset, schema, README, and existing source file.
- Identify the file formats, column names, data types, dataset sizes, duplicate records, missing values, and licensing restrictions.
- Identify whether the dataset contains recipes, ingredient lists, nutrition values, food images, user profiles, interaction histories, timestamps, ratings, or preference labels.
- Determine which tasks are actually supported by the available data.
- Create a dataset audit report with missing-value percentages and a data-quality summary.
- Detect data leakage, duplicate recipes, inconsistent serving sizes, and invalid nutrient values.
- Do not assume the dataset contains fields that are absent.
- If user-interaction data is missing, explicitly state that sequential recommendation cannot be fully supervised from that dataset alone.
- If images are missing, disable the image branch rather than fabricating image inputs.
- If clinical labels are missing, do not invent clinical ground truth.
- Ask for missing essential files only after inspecting everything already supplied.

Deliverable: `reports/dataset_audit.md`.

### 3. MODEL SELECTION AND BENCHMARKING

Use the following candidate models:

Primary candidate:
- Qwen/Qwen3.5-4B

Comparison candidate:
- microsoft/Phi-4-mini-instruct

Optional multimodal comparison:
- google/gemma-3-4b-it

First verify each model's current model card, license, supported input modalities, Transformers compatibility, fine-tuning support, and hardware requirements.

Do not assume all models have identical architectures or support identical training implementations.

Use a configurable model registry so that candidates can be switched through configuration without changing the application logic.

Evaluate at least:
A. Base pretrained/instruction-tuned model.
B. Fine-tuned model.
C. Fine-tuned, quantized model where supported.
D. A non-LLM structured baseline.

Use identical held-out evaluation examples and equivalent output requirements wherever technically appropriate.

Choose the winning model based on nutrition-specific task quality, factual reliability, inference latency, peak memory usage, and training cost—not parameter count alone.

### 4. SEPARATE THE RECOMMENDER FROM THE LLM

Implement independent modules.

A. Food representation:
- Pretrained text encoder.
- Optional pretrained image encoder.
- Structured nutrition feature encoder.
- Trainable projection layer to combine available modalities.
- Explicit handling of unavailable modalities.

B. User preference and sequential recommendation:
- Implement a popularity baseline.
- Implement content-based recommendation.
- Implement SASRec if sufficient chronological user-item interaction data exists.
- Train the recommender using a leakage-free chronological split.
- Implement candidate retrieval and ranking separately from text generation.

C. Nutrition engine:
- Calculate nutrient totals from verified ingredient quantities and serving sizes when possible.
- Use source-based nutrition values before ML estimates.
- Train a nutrient prediction model only on valid labeled data.
- Track each nutrient's source, confidence, and estimation method.
- Distinguish measured, database-derived, calculated, and predicted values.
- Do not treat an uncertain nutrient prediction as a verified fact.

D. Adaptive preference-health optimization:
- Implement a fixed-weight baseline.
- Implement a rule-based dynamic weighting baseline.
- Optionally train a context-dependent weighting model when sufficient data exists.
- Use user preferences, explicit health goals, remaining daily targets, meal context, and prior acceptance feedback.
- Apply allergy and hard safety constraints before ranking.
- Never allow taste preference to override a hard safety constraint.

E. LLM explanation service:
- Receive the selected meal, verified nutrient facts, user preferences, relevant goals, and explicit comparisons.
- Generate a concise explanation of nutrient improvement and preference retention.
- Never allow the LLM to alter the recommender's ranking score.
- Never ask the LLM to invent missing nutrient values.
- Validate structured outputs against a JSON schema.
- Reject or regenerate malformed or unsupported explanations.

### 5. FINE-TUNING STRATEGY

Do not train a foundation model from scratch.

Begin with parameter-efficient supervised fine-tuning using LoRA or QLoRA, selecting a supported implementation appropriate to each candidate model and the available GPU.

Use the instruction-tuned model as the default starting point. Compare it with a base-model-plus-supervised-fine-tuning configuration only if resources permit.

Prepare supervised examples for these tasks:

Task 1: Nutrition extraction
Input: Recipe, ingredient quantities, serving size, and trusted source data.
Output: Valid structured JSON containing supported nutrient facts, units, sources, and uncertainty indicators.

Task 2: Nutritional comparison
Input: Two meals with verified nutrition data and a user's explicitly supplied goals.
Output: Structured differences between the meals.

Task 3: Personalized explanation
Input: Selected recommendation, preference evidence, nutrient differences, and relevant goals.
Output: An explanation that identifies nutrient changes and retained preferences.

Task 4: Evidence-grounded dietary question answering
Input: A question and a retrieved trusted nutrition reference.
Output: A concise answer supported by the supplied evidence, or an explicit statement that the evidence is insufficient.

Do not use arbitrary generated text as ground truth.

Create or obtain training labels from reliable sources, validated deterministic transformations, or documented human annotation.

Store every training example in a consistent JSONL schema, for example:

{"messages":[{"role":"system","content":"Use only supplied evidence. Do not invent nutrition facts."},{"role":"user","content":"...recipe, verified nutrients, goals..."},{"role":"assistant","content":"...validated JSON output..."}]}

Split data by source, recipe, and user where appropriate to prevent leakage. Keep the test set isolated from training-data generation and prompt development.

Start with conservative LoRA settings and a small pilot run. Monitor training loss, validation loss, gradient stability, peak GPU memory, and elapsed training time.

Use early stopping and save the best validation checkpoint. Do not assume that lower training loss means higher real-world accuracy.

Save:
- Base model identifier and revision.
- Tokenizer and chat-template configuration.
- Adapter weights and configuration.
- Training configuration.
- Dataset version and split manifest.
- Random seeds.
- Environment and package versions.
- Training and evaluation logs.

### 6. MISSING NUTRITION METADATA EXPERIMENT

Simulate missingness in the training/evaluation data at controlled rates: 10%, 20%, 30%, 40%, and 50%.

Compare:
1. Complete-data reference baseline.
2. No imputation.
3. Simple statistical imputation.
4. Ingredient-based nutrient calculation.
5. A supervised ML nutrient estimator.
6. The proposed confidence-aware fallback pipeline.

Where feasible, distinguish random missingness from systematically missing fields.

Evaluate each nutrient separately.

Regression metrics:
- MAE.
- RMSE.
- R-squared where meaningful.
- Error by nutrient and missingness rate.

If nutrient values are represented as classes, also report appropriate classification metrics.

Never train on the test targets or use imputed test labels as ground truth.

### 7. DYNAMIC HEALTH-VERSUS-TASTE EXPERIMENT

Compare:
A. Fixed health weight of 0.3.
B. Fixed health weight of 0.5.
C. Fixed health weight of 0.7.
D. Rule-based adaptive weighting.
E. Learned adaptive weighting, if sufficient training data exists.

Document the exact scoring formula, normalization method, constraints, and optimization procedure.

Use a development set to choose hyperparameters and reserve the test set for final evaluation.

Test distinct contexts, including high-priority nutritional goals, ordinary preference-led meals, and changes in remaining daily targets.

Do not claim that synthetic context simulations prove improved real-world health outcomes.

### 8. RESEARCH EVALUATION

Use suitable evaluation metrics for each task.

Recommendation:
- NDCG@5 and NDCG@10.
- Recall@5 and Recall@10.
- Hit Rate@10.
- Precision@5 and Precision@10.
- MRR.
- Catalog coverage and diversity where appropriate.

Nutrition prediction:
- MAE, RMSE, and nutrient-specific error analysis.

Nutritional suitability:
- Guideline Compliance Rate.
- Nutrient constraint violation rate.
- Nutrient target deviation.
- Health Score Improvement, only when the score definition and data justify it.

LLM output:
- JSON schema validity.
- Nutrient factual consistency.
- Evidence support rate.
- Unsupported-claim rate.
- Task completion rate.
- Human evaluation of explanation relevance and clarity.

Efficiency:
- P50, P95, and P99 latency.
- Time to first token where relevant.
- End-to-end recommendation latency.
- Explanation latency measured separately.
- Peak RAM and GPU memory.
- Throughput.
- Model and adapter size.
- Training time and inference cost.

Report classification accuracy, precision, recall, F1, ROC-AUC, and a confusion matrix only for clearly defined classification tasks such as predicting whether a user will accept a meal.

Do not use classification accuracy as the primary metric for ranked meal recommendations.

### 9. EXPERIMENTAL CONTROLS

Use:
- Chronological user-level splits for sequential recommendation.
- Source- or recipe-grouped splits for food-content generalization.
- Identical held-out examples for fair model comparisons.
- Multiple random seeds when computationally feasible.
- Confidence intervals and suitable paired statistical tests.
- A predeclared primary metric and evaluation protocol.
- A clear record of exclusions and failed experiments.

Avoid tuning on the test set.

Do not claim a percentage improvement unless it is calculated from actual baseline and proposed-model results.

### 10. REQUIRED ABLATION STUDY

Evaluate:
- Full system.
- Without dynamic weighting.
- Without nutrient features.
- Without the image branch, if images exist.
- Without the sequential recommender.
- Without alignment loss, if implemented.
- Without LLM fine-tuning.
- Without nutrition imputation.
- Quantized versus non-quantized inference.

Explain which components improve recommendation quality, nutritional suitability, factual reliability, and efficiency.

### 11. LOW-LATENCY DEPLOYMENT

Use:
- Python and PyTorch for model training.
- Hugging Face Transformers and PEFT for supported LLM fine-tuning.
- FastAPI for the backend.
- PostgreSQL for structured data.
- FAISS or an appropriate vector database for retrieval.
- React or Next.js for the frontend.

Use caching, bounded candidate sets, asynchronous explanation generation where appropriate, and a smaller quantized inference configuration where supported.

Keep recommendation ranking independent of LLM generation so a slow explanation request does not block the initial recommendation.

Benchmark on clearly specified hardware. Report whether timings are measured locally, on a GPU server, or through a hosted API.

### 12. REQUIRED PROJECT STRUCTURE

Create a maintainable structure similar to:

data/
  raw/
  processed/
  splits/

configs/
  models.yaml
  training.yaml
  evaluation.yaml

src/
  data/
  nutrition/
  embeddings/
  recommender/
  adaptive_weights/
  llm/
  evaluation/
  api/

scripts/
  audit_dataset.py
  prepare_data.py
  train_nutrition_model.py
  train_recommender.py
  prepare_sft_data.py
  train_lora.py
  evaluate_models.py
  benchmark_latency.py
  run_ablations.py

tests/
  unit/
  integration/
  data_quality/

reports/
  dataset_audit.md
  experiment_results.md
  ablation_study.md
  latency_benchmark.md

paper/
  manuscript.md
  tables/
  figures/

Provide reproducible setup instructions, configuration files, dependency versions, tests, and commands for every training and evaluation stage.

Do not generate placeholder functions for core training logic and claim the project is complete.

### 13. RESEARCH PAPER GENERATION

After all experiments have actually run, create a manuscript with:

1. Abstract.
2. Introduction.
3. Related Work.
4. Research Questions and Contributions.
5. Dataset and Preprocessing.
6. Proposed Architecture.
7. Training Objectives and Fine-Tuning Procedure.
8. Experimental Setup.
9. Results and Baseline Comparison.
10. Ablation Study.
11. Nutrition Estimation Robustness.
12. Latency and Memory Analysis.
13. Error Analysis.
14. Limitations, Ethics, and Safety.
15. Conclusion.
16. References.

Include tables and figures generated from saved experiment outputs.

Clearly distinguish:
- Results reported by the original paper.
- Results produced by this implementation.
- Results obtained from synthetic data.
- Hypotheses that remain untested.

Do not invent metrics, citations, statistical significance, datasets, human-study results, or clinical benefits.

If experiments are incomplete, label the manuscript as a draft and explicitly mark unavailable results.

Reproduce the original paper's comparisons only when its data, splits, implementation details, and evaluation protocol are sufficiently available. Otherwise describe the comparison as an adapted baseline, not an exact reproduction.

### 14. EXECUTION ORDER

Proceed in this order:

Phase 1: Inspect the dataset and project.
Phase 2: Produce the audit report and finalize research questions.
Phase 3: Build a deterministic nutrition-aware baseline.
Phase 4: Train and evaluate the sequential recommender.
Phase 5: Benchmark candidate LLMs before fine-tuning.
Phase 6: Prepare and validate supervised fine-tuning data.
Phase 7: Run a small LoRA pilot.
Phase 8: Select the best checkpoint using validation metrics.
Phase 9: Run final test-set evaluation.
Phase 10: Perform missing-data, adaptive-weighting, and ablation experiments.
Phase 11: Benchmark latency, memory, and cost.
Phase 12: Integrate the model into the application.
Phase 13: Generate the research manuscript from verified results.

At each phase, report the files created, commands executed, actual metrics, failures, and next steps. Ask for approval before expensive GPU jobs or large-scale experiments.

Start now with Phase 1. Do not begin full fine-tuning until the dataset audit, baseline plan, and hardware requirements have been documented.