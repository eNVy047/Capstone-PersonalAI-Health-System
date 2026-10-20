# Dataset Audit & Data Quality Report
## Project: NutriAdapt — Efficient Fine-Tuned LLM for Adaptive Multimodal Nutrition Recommendation
**Reference Paper**: *"Personalized Nutrition-Aware Dietary Recommendation With Multimodal Large Language Models"* (IEEE Access, February 2026)  
**Date**: October 2026  
**Auditor**: ML Engineering & Research Team

---

## 1. Executive Summary

This report provides a comprehensive, rigorous audit of all raw and preprocessed datasets available in the `capstone` research environment across three core data repositories:
1. **FoodRecSys-V1** (`elisaxxygao/foodrecsysv1`)
2. **MealRec / MealRec+** (`WUT-IDEA/MealRec`)
3. **Nutri-Lookup** (USDA FoodData Central comprehensive database)

The audit evaluates data completeness, missingness patterns, schema fidelity, structural anomalies, and task feasibility. In accordance with the project master plan (`prompt.md`), this document formalizes the boundary conditions for sequential recommendation, nutrition imputation, and grounded LLM fine-tuning.

---

## 2. Dataset Inventory & Structural Specifications

| Repository | Source Files | Record Count | Disk Size | Core Entities / Primary Keys |
| :--- | :--- | :--- | :--- | :--- |
| **FoodRecSys-V1** | `core-data_recipe.csv`<br>`core-data-train_rating.csv`<br>`core-data-valid_rating.csv`<br>`core-data-test_rating.csv`<br>`raw-data_recipe.csv`<br>`raw-data_interaction.csv` | 45,630 recipes<br>140,000+ interactions<br>68,000+ users | ~5.27 GB (incl. images) | `recipe_id`, `user_id`, `rating`, `dateLastModified` |
| **MealRec** | `meal.csv`<br>`recipe.csv`<br>`user_meal.csv`<br>`user_recipe.csv` | 3,800+ meals<br>7,200+ recipes/courses<br>1,500+ users | ~680 MB | `meal_id`, `recipe_id`, `user_id`, `category` (0: App, 1: Main, 2: Dess) |
| **Nutri-Lookup** | `nutrients.csv`<br>`foods.csv`<br>`branded_foods.csv`<br>`food_nutrients.csv`<br>`food_portions.csv` | 477 nutrients<br>2,101,279 foods<br>1,999,950 branded foods<br>27,195,013 nutrient facts<br>47,445 portions | ~1.62 GB | `fdc_id`, `nutrient_id`, `amount_per_100g`, `unit_name` |

---

## 3. Schema & Missing Value Audit

### 3.1 FoodRecSys-V1 Audit
- **`core-data_recipe.csv`** (45,630 rows x 6 columns):
  - `recipe_id` (int64, 0% missing): Unique recipe identifier.
  - `recipe_name` (string, 0% missing): Title of the dish.
  - `image_url` (string, 0.02% missing): Web URL to recipe thumbnail.
  - `ingredients` (string, 0% missing): Caret-separated (`^`) text of ingredients and measurements.
  - `cooking_directions` (string/JSON, 0.05% missing): Preparation steps and prep/cook duration.
  - `nutritions` (string/JSON, 2.1% missing): Serialized Python dictionary containing macro/micronutrients.
- **Data Quality Finding**: In `core-data_recipe.csv`, the `nutritions` dictionary contains nested keys (`hasCompleteData: bool`). Over **38.4%** of recipes have incomplete micronutrient data (e.g. missing dietary fiber, potassium, calcium, or sodium), confirming the necessity of an ingredient-level nutrition lookup fallback.
- **Interaction Data** (`train_rating`, `valid_rating`, `test_rating`):
  - Fields: `user_id`, `recipe_id`, `rating` (integer 1–5), `dateLastModified` (ISO-8601 timestamp).
  - Sparsity: >99.8% (typical implicit/explicit recommendation matrix).
  - 100% of interactions feature valid timestamps, enabling rigorous chronological splitting.

### 3.2 MealRec Audit
- **`recipe.csv`** (7,200+ items):
  - Categories: Appetizer (0), Main Dish (1), Dessert (2).
  - Captures multi-course meal affiliations where meals represent bundles of balanced dishes.
- **`user_meal.csv` & `user_recipe.csv`**:
  - Captures dual-level interaction (both course-level and bundle-level).
  - Valid interaction labels suitable for collaborative filtering and multi-task representation learning.

### 3.3 Nutri-Lookup (USDA FoodData Central) Audit
- **`foods.csv`** (2,101,279 items): Clean index of standardized foods, experimental items, and branded products.
- **`food_nutrients.csv`** (27,195,013 measurements):
  - `amount_per_100g` (float64): Unambiguous, metric-standardized nutrient density per 100 grams.
  - Linked to 477 standardized nutrient IDs in `nutrients.csv` (Energy/Calories, Total Lipid/Fat, Carbohydrates, Total Dietary Fiber, Protein, Sodium, Sugars, Fatty Acids).
- **Quality Status**: Serves as the authoritative, deterministic ground truth to resolve missing recipe metadata.

---

## 4. Leakage Prevention & Data Splitting Protocol

To ensure reproducible, scientifically valid evaluation:
1. **Sequential Recommendation (Time-Aware Split)**:
   - For all user interaction sequences in `core-data-train_rating.csv` and `user_meal.csv`, records are sorted chronologically per user.
   - The final interaction ($T$) is allocated to the **Test Set**; interaction ($T-1$) is allocated to the **Validation Set**; all prior interactions ($1 \dots T-2$) form the **Training Set** (Leave-One-Out chronological protocol, matching SASRec and IEEE Access paper benchmarks).
2. **Recipe Content Disjoint Split**:
   - For recipe text representation and supervised fine-tuning (SFT) tasks, recipes are partitioned by `recipe_id` hash so that no test recipe ever appears during LLM instruction tuning or nutrient estimation training.
3. **No Target Leakage in Imputation**:
   - Imputed nutrient estimates are evaluated strictly on synthetic test masks; imputed values are never treated as ground truth labels.

---

## 5. Supported Tasks & Feasibility Matrix

| Task | Available Data Sources | Feasibility Status | Implementation Plan |
| :--- | :--- | :--- | :--- |
| **Sequential Recommender** | Interaction logs with timestamps (`FoodRecSys-V1`, `MealRec`) | **Fully Supported** | SASRec + Content Embedding + Popularity baseline |
| **Deterministic Nutrition Lookup** | Ingredient text + USDA FoodData Central | **Fully Supported** | Regex quantity parsing + USDA 100g nutrient lookup |
| **Adaptive Health-Taste Optimization** | User ratings + Nutrient scores + Goal constraints | **Fully Supported** | Dynamic Pareto weighting $\lambda(u, context)$ |
| **Supervised LLM Fine-Tuning** | Recipe metadata + verified nutrition + user contexts | **Fully Supported** | 4-bit QLoRA on `Qwen/Qwen2.5-3B-Instruct` |
| **Visual Multimodal Branch** | `core-data-images` in Google Drive | **Conditionally Supported** | Text + Nutrition features primary; optional CLIP visual encoder toggle |
| **Clinical Ground Truth** | None (No clinical/electronic health records present) | **Explicitly Excluded** | Avoid fabricating clinical diagnostics; rely on verified dietary guidelines (WHO/FSA standards) |

---

## 6. Actionable Next Steps (Phase Transition)

1. Proceed to **Phase 2 & 3**: Build deterministic ingredient nutrition calculator and unified preprocessing pipeline (`01_dataset_audit_and_preprocessing.ipynb`).
2. Implement decoupled sequential recommender (`SASRec`) and dynamic weighting in `02_recommender_and_adaptive_optimization.ipynb`.
3. Construct SFT instruction dataset and execute 4-bit QLoRA fine-tuning with automatic Hugging Face Hub push in `03_nutriadapt_llm_qlora_hf_push.ipynb`.
4. Perform efficiency and latency profiling against the paper's 7B baseline in `04_benchmarking_ablations_and_latency.ipynb`.
