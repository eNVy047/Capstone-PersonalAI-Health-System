#!/usr/bin/env python3
"""
NutriAdapt Model Testing Script
Loads the fine-tuned NutriAdapt LoRA adapter from Hugging Face Hub (narayah90/nutriadapt-qwen-3b-nutrition)
and verifies structured inference, JSON schema compliance, and numeric grounding.
"""

import argparse
import json
import os
import re
import sys

SYSTEM_PROMPT = (
    "You are NutriAdapt, an expert dietary reasoning assistant. Provide factual, evidence-grounded "
    "nutrition advice adhering strictly to the nutrient data given in the prompt. "
    "Never fabricate nutrient values. Return valid JSON when requested."
)

GOAL_RULES = {
    "weight_loss": "low-calorie (<=500 kcal) and high-fiber (>=4g)",
    "heart_healthy": "low-sodium (<500mg)",
    "muscle_gain": "high-protein (>30g)",
}
GOAL_THRESHOLDS = {"500", "4", "30"}

SCHEMAS = {
    "explain": {
        "type": "object",
        "required": ["recommendation", "target_goal", "nutritional_justification", "criteria_met", "factual_compliance"],
        "properties": {
            "recommendation": {"type": "string"},
            "target_goal": {"type": "string"},
            "nutritional_justification": {"type": "string"},
            "criteria_met": {"type": "array", "items": {"type": "string"}},
            "factual_compliance": {"type": "boolean"},
        },
    },
    "extract": {
        "type": "object",
        "required": ["meal", "calories", "macronutrients", "micronutrients"],
        "properties": {
            "meal": {"type": "string"},
            "calories": {"type": "number"},
            "macronutrients": {
                "type": "object",
                "required": ["protein_g", "fat_g", "carbs_g"],
            },
            "micronutrients": {
                "type": "object",
                "required": ["sodium_mg", "fiber_g"],
            },
        },
    },
    "compare": {
        "type": "object",
        "required": ["meal_a", "meal_b", "lower_calorie", "higher_protein", "higher_fiber", "lower_sodium"],
    }
}


def extract_json(raw_text: str):
    """Strip markdown code block fences and extract JSON object."""
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw_text.strip())
    match = re.search(r"(\{.*\})", cleaned, re.DOTALL)
    if match:
        cleaned = match.group(1)
    return json.loads(cleaned)


def extract_numbers(text: str):
    """Extract set of numeric tokens from text for grounding check."""
    return {n.rstrip("0").rstrip(".") if "." in n else n for n in re.findall(r"\d+(?:\.\d+)?", str(text))}


def load_model(base_model_id: str, adapter_repo_id: str, force_cpu: bool = False):
    """Load base model and attach PEFT LoRA adapter."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import PeftModel

    device = "cpu" if force_cpu or not torch.cuda.is_available() else "cuda"
    print(f"Device mode: {device.upper()}")
    print(f"Loading Tokenizer from: {adapter_repo_id}")
    
    tokenizer = AutoTokenizer.from_pretrained(adapter_repo_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading Base Model:    {base_model_id}")
    if device == "cuda":
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True
        )
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_id,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.float16,
            trust_remote_code=True
        )
    else:
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_id,
            device_map="cpu",
            torch_dtype=torch.float32,
            trust_remote_code=True
        )

    print(f"Attaching LoRA Adapter: {adapter_repo_id}")
    model = PeftModel.from_pretrained(base_model, adapter_repo_id)
    model.eval()
    print("✓ Model successfully initialized.")
    return model, tokenizer, device


def run_inference(model, tokenizer, user_prompt: str, schema_type: str = None, max_new_tokens: int = 256):
    """Run generation and check structured output."""
    import torch
    from jsonschema import validate, ValidationError

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt}
    ]
    prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id
        )

    response = tokenizer.decode(output_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()

    parsed_json = None
    validation_status = "Skipped"
    try:
        parsed_json = extract_json(response)
        if schema_type and schema_type in SCHEMAS:
            validate(parsed_json, SCHEMAS[schema_type])
            validation_status = "✓ PASSED (Valid schema)"
        else:
            validation_status = "✓ PASSED (Valid JSON)"
    except Exception as e:
        validation_status = f"✗ FAILED ({str(e)})"

    grounding_status = "N/A"
    if parsed_json and schema_type == "explain":
        input_nums = extract_numbers(user_prompt) | GOAL_THRESHOLDS
        justification_text = str(parsed_json.get("nutritional_justification", "")) + " " + " ".join(str(c) for c in parsed_json.get("criteria_met", []))
        claimed_nums = extract_numbers(justification_text)
        invented = claimed_nums - input_nums
        if not invented:
            grounding_status = "✓ ZERO HALLUCINATIONS (All numbers grounded in input)"
        else:
            grounding_status = f"⚠️ Unverified numbers found: {sorted(invented)}"

    return {
        "raw_response": response,
        "parsed_json": parsed_json,
        "validation_status": validation_status,
        "grounding_status": grounding_status
    }


def main():
    parser = argparse.ArgumentParser(description="Test NutriAdapt Hugging Face Model")
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-3B-Instruct", help="Base model Hugging Face ID")
    parser.add_argument("--adapter-repo", default="narayah90/nutriadapt-qwen-3b-nutrition", help="PEFT LoRA adapter repo ID")
    parser.add_argument("--cpu", action="store_true", help="Force CPU inference")
    parser.add_argument("--task", choices=["explain", "extract", "compare", "custom"], default="explain", help="Task to test")
    parser.add_argument("--meal", default="Grilled Salmon Rice Bowl", help="Meal name for custom/explain test")
    parser.add_argument("--goal", choices=["muscle_gain", "weight_loss", "heart_healthy"], default="muscle_gain", help="Target dietary goal")
    parser.add_argument("--calories", type=float, default=450.0, help="Calories")
    parser.add_argument("--protein", type=float, default=38.0, help="Protein in grams")
    parser.add_argument("--fiber", type=float, default=6.0, help="Fiber in grams")
    parser.add_argument("--sodium", type=float, default=420.0, help="Sodium in mg")

    args = parser.parse_args()

    model, tokenizer, device = load_model(args.base_model, args.adapter_repo, force_cpu=args.cpu)

    if args.task in ["explain", "custom"]:
        prompt = (
            f"Explain why '{args.meal}' is recommended for a user with the goal '{args.goal}'.\n"
            f"Nutritional Facts: {args.calories:.0f} kcal, {args.protein:.0f}g protein, {args.fiber:.0f}g fiber, {args.sodium:.0f}mg sodium."
        )
        schema = "explain"
    elif args.task == "extract":
        prompt = (
            f"Extract the standardized nutritional breakdown for the following meal:\n"
            f"Meal: {args.meal}\n"
            f"Ingredients: salmon^brown rice^broccoli^olive oil"
        )
        schema = "extract"
    elif args.task == "compare":
        prompt = (
            "Compare the following two meals.\n"
            "Meal A: Grilled Salmon Bowl (450 kcal, 38g protein, 6g fiber, 420mg sodium)\n"
            "Meal B: Mediterranean Salad (320 kcal, 12g protein, 7g fiber, 480mg sodium)"
        )
        schema = "compare"

    print("\n" + "=" * 60)
    print("INPUT PROMPT:")
    print("=" * 60)
    print(prompt)

    print("\n" + "=" * 60)
    print("RUNNING INFERENCE...")
    print("=" * 60)
    result = run_inference(model, tokenizer, prompt, schema_type=schema)

    print("\nRAW OUTPUT:")
    print(result["raw_response"])

    print("\n" + "=" * 60)
    print("VALIDATION & AUDIT:")
    print("=" * 60)
    print(f"Schema Validation : {result['validation_status']}")
    print(f"Numeric Grounding : {result['grounding_status']}")
    if result["parsed_json"]:
        print("\nPARSED JSON:")
        print(json.dumps(result["parsed_json"], indent=2))


if __name__ == "__main__":
    main()
