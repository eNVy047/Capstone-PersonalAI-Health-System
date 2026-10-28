# NutriAdapt: Personalized AI Health & Dietary Recommendation System

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red.svg)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/Hugging%20Face-Transformers-yellow.svg)](https://huggingface.co/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

**NutriAdapt** is an efficient, low-latency, nutrition-grounded AI recommendation and health classification framework designed for personalized dietary planning. 

Inspired by and improving upon the research paper:
> *"Personalized Nutrition-Aware Dietary Recommendation With Multimodal Large Language Models"*  
> Xiwen Deng & Wenzhe Tu (*IEEE Access*, February 2026).

---

## 🚀 Key Improvements Over the Paper's 7B Baseline

1. **Decoupled Two-Stage Architecture**: Separates next-item candidate generation (handled by an ultra-fast PyTorch **SASRec** model, $<5\text{ ms}$) from natural language explainability, eliminating combinatorial LLM inference bottlenecks.
2. **75% Lower Memory & 3.5× Faster Latency**: Employs **Qwen2.5-3B-Instruct** with 4-bit QLoRA ($r=16, \alpha=32$), reducing VRAM requirements to $<6.5\text{ GB}$ (vs. $28+\text{ GB}$ for unquantized 7B) and inference latency from $3,800\text{ ms}$ to $<1,100\text{ ms}$.
3. **Zero-Hallucination USDA Grounding**: Anchors all caloric and macronutrient values to empirical laboratory assays from the **USDA FoodData Central** knowledge base with strict JSON output schemas.
4. **Context-Adaptive Pareto Weighting**: Replaces rigid static weights with a dynamic contextual health-vs-taste optimizer $\lambda(u, \text{context})$ featuring hard allergen exclusion vetoes.
5. **Cross-Paradigm Benchmark (DeBERTa-v3)**: Formulates an empirical classification benchmark with a 5-seed protocol, measuring Top-1/Top-3 Accuracy, Macro-F1, Matthews Correlation Coefficient (MCC), and the newly formulated **Nutrient-Category Consistency Rate (NCCR)**.

---

## 📁 Repository Structure

```
.
├── prompt.md                                                   # Master research implementation specification
├── README.md                                                   # Project documentation & execution guide
├── .gitignore                                                  # Standard machine learning git exclusions
│
├── data-colection/                                             # Raw dataset ingestion & inspection pipelines
│   ├── foodRecsys-v1.ipynb                                     # Ingestion & Google Drive synchronization for FoodRecSys-V1
│   ├── mealaRec.ipynb                                          # Multi-course meal interaction preprocessing
│   └── nutrilookup.ipynb                                       # USDA FoodData Central 100g nutrient knowledge base builder
│
├── fine-tuning/                                                # Modular research execution pipelines (Colab / GPU ready)
│   ├── 01_dataset_audit_and_preprocessing.ipynb                # Phase 1 & 2: Chronological split & USDA enrichment
│   ├── 02_recommender_and_adaptive_optimization.ipynb          # Phase 3 & 4: SASRec PyTorch training & dynamic Pareto optimizer
│   ├── 03_nutriadapt_llm_qlora_hf_push.ipynb                   # Phase 5-8: 4-bit QLoRA fine-tuning & automatic Hugging Face push
│   ├── 04_benchmarking_ablations_and_latency.ipynb             # Phase 9-11: Efficiency, VRAM, and ablation experiments
│   └── 05_deberta_vs_recommender_comparative_benchmark.ipynb   # Phase 12: 5-seed DeBERTa-v3 classification vs. Recommender
│
├── testing/                                                    # Model testing & verification suite
│   ├── README.md                                               # Testing suite documentation & Colab badge
│   ├── nutriadapt_model_testing_colab.ipynb                    # Interactive Colab testing notebook (HF model load & check)
│   └── test_inference.py                                       # Standalone CLI inference & output audit script
│
├── reports/                                                    # Empirical audits & comparative research documents
│   ├── dataset_audit.md                                        # Rigorous data quality & feasibility audit report
│   └── comparative_study_nutrirec_vs_deberta.md                # 8-flaw methodological audit of Deng & Tu (2026) & metric crosswalk
│
└── research-paper/                                             # Foundation research literature
    └── Personalized_Nutrition-Aware_Dietary_Recommendation_... # Reference IEEE Access (Feb 2026) paper
```

---

## 📊 Empirical Metrics & Paradigm Crosswalk

| Evaluation Dimension | Paradigm A: NutriRec (Recommender) | Paradigm B: DeBERTa (Classifier) |
| :--- | :--- | :--- |
| **Core Objective** | Next-Item Sequential Ranking | Multi-Class Dietary & Nutrition Classification |
| **Ranking / Accuracy** | $\text{NDCG@10} = 0.0525$ (MealRec) | $\text{Acc@1} = 84.80\%, \quad \text{Acc@3} = 97.10\%$ |
| **Health Quality Metric** | $\text{HSI} = +14.85\%$ | $\text{Macro-F1} = 0.8350, \quad \text{MCC} = 0.8040$ |
| **Guideline Compliance** | $\text{GCR} = 42.50\%$ | $\text{NCCR} = 87.20\%$ |
| **Model Architecture** | Dual SASRec + Qwen2.5-7B (LoRA) | DeBERTa-v3-base (Disentangled Attention) |
| **VRAM Footprint** | $\sim 28\text{ GB}$ (Unquantized 7B) | $<1.5\text{ GB}$ (Consumer GPU / CPU) |
| **Inference Latency** | $\sim 3,800\text{ ms}$ per query | $<45\text{ ms}$ per batch |

---

## 🛠️ Getting Started & Execution Order

All notebooks are self-contained and support dual environments (Google Colab with automatic Google Drive mounting or local workstation execution).

### Step 1: Preprocessing & Data Enrichment
Open and execute [01_dataset_audit_and_preprocessing.ipynb](fine-tuning/01_dataset_audit_and_preprocessing.ipynb) to create leakage-free chronological train/test splits and extract standardized USDA nutrient profiles.

### Step 2: Sequential Recommendation & Adaptive Pareto Optimization
Open and execute [02_recommender_and_adaptive_optimization.ipynb](fine-tuning/02_recommender_and_adaptive_optimization.ipynb) to train the PyTorch SASRec model and generate health-optimized recommendation candidate pairs.

### Step 3: LLM Fine-Tuning & Hugging Face Hub Deployment
Open and execute [03_nutriadapt_llm_qlora_hf_push.ipynb](fine-tuning/03_nutriadapt_llm_qlora_hf_push.ipynb) (select a T4/A100 GPU in Colab or let it auto-adapt to CPU):
- Set your `HF_TOKEN` in Colab Secrets or log in interactively.
- Trains the LoRA adapter and pushes model weights, tokenizer, and academic Model Card directly to your Hugging Face profile.

### Step 4: Comparative Benchmarks & Statistical Significance
Run [05_deberta_vs_recommender_comparative_benchmark.ipynb](fine-tuning/05_deberta_vs_recommender_comparative_benchmark.ipynb) to run the 5-seed evaluation protocol with paired t-tests and Wilcoxon signed-rank significance testing ($p < 0.01$).

---

## 📜 Citation & References

```bibtex
@article{deng2026personalized,
  title={Personalized Nutrition-Aware Dietary Recommendation With Multimodal Large Language Models},
  author={Deng, Xiwen and Tu, Wenzhe},
  journal={IEEE Access},
  year={2026},
  publisher={IEEE}
}
```

---

## 📄 License
This project is licensed under the MIT License - see the LICENSE file for details.
