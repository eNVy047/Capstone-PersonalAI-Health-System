# 🧪 NutriAdapt Model Testing & Verification Suite

This folder provides a testing framework to load, evaluate, and audit the fine-tuned NutriAdapt model from Hugging Face Hub:

- **Hugging Face Model Hub**: [`narayah90/nutriadapt-qwen-3b-nutrition`](https://huggingface.co/narayah90/nutriadapt-qwen-3b-nutrition)
- **Base Architecture**: `Qwen/Qwen2.5-3B-Instruct`
- **Tuning Method**: 4-bit QLoRA (PEFT Adapter)

---

## 🚀 1. Run in Google Colab (Recommended)

You can run the interactive testing notebooks directly in Google Colab with free T4 GPU support:

- **`test_inference.ipynb` (Direct notebook conversion of `test_inference.py`)**:  
  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/eNVy047/Capstone-PersonalAI-Health-System/blob/main/testing/test_inference.ipynb)

- **`nutriadapt_model_testing_colab.ipynb` (Full Testing & Verification Suite)**:  
  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/eNVy047/Capstone-PersonalAI-Health-System/blob/main/testing/nutriadapt_model_testing_colab.ipynb)

### Features in the Notebooks:
1. **Auto Hardware Detection**: Detects NVIDIA GPU (T4 / A100) with 4-bit NF4 quantization or falls back cleanly to CPU mode.
2. **Hugging Face Import**: Automatically fetches base weights and attaches the `narayah90/nutriadapt-qwen-3b-nutrition` PEFT adapter.
3. **Structured Task Tests**:
   - **Task 1: Nutrition Extraction** (Ingredients -> Standardized nutrient JSON).
   - **Task 2: Goal-Grounded Recommendation** (Nutritional facts + goal -> Structured justification).
   - **Task 3: Comparative Evaluation** (Two-meal dietary trade-off analysis).
4. **Interactive Playground**: Easily input your own custom meal, calories, protein, and health goal, and check the generated output in real time.
5. **Automated Verification**:
   - **JSON Schema Validation**: Validates that keys and structure conform strictly to NutriAdapt specifications.
   - **Zero-Hallucination Numeric Grounding**: Audits whether every number in the justification originates from input prompt facts or defined goal thresholds.

---

## 💻 2. Run Locally or on a Server (`test_inference.py`)

To test inference via the command line:

```bash
# Install dependencies
pip install transformers peft bitsandbytes accelerate jsonschema torch

# Run default goal-grounded explanation test
python testing/test_inference.py --task explain

# Test nutrition extraction task
python testing/test_inference.py --task extract --meal "Mediterranean Salad"

# Test two-meal comparison task
python testing/test_inference.py --task compare

# Test custom inputs
python testing/test_inference.py \
  --meal "Grilled Chicken & Quinoa" \
  --goal "muscle_gain" \
  --calories 480 \
  --protein 44 \
  --fiber 6 \
  --sodium 390
```

---

## 📋 Evaluation Criteria & Output Checking

| Check | Description | Target |
|---|---|---|
| **JSON Syntax** | Raw model completion parsed as JSON without markdown parsing errors | 100% Valid JSON |
| **Schema Compliance** | Output fields match required schema (`recommendation`, `target_goal`, `nutritional_justification`, `criteria_met`, `factual_compliance`) | `✓ PASSED` |
| **Numeric Grounding** | Every numeric figure in the generated explanation must be present in the user prompt or predefined goal thresholds | Zero Hallucinations |
