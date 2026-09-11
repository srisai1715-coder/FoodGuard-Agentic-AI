"""
FoodGuard AI - Sample Execution Traces Inspector (scratch/sample_execution_traces.py)

Extracts and displays self-validation traces for:
- 3 Manager cases
- 3 Nutrition cases
- 3 Safety cases
- 3 Comparison cases
- 3 RAG cases

For EACH sampled case shows:
EXPECTED -> ACTUAL FOODGUARD OUTPUT -> EXTRACTED PREDICTION -> METRIC RESULT
Proves that benchmark scores are calculated from REAL FoodGuard system outputs.
"""

import json
import os
import sys

# Ensure UTF-8 output on Windows console
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except AttributeError:
        pass

def display_sampled_traces(report_path="evaluation/results/latest_report.json"):
    if not os.path.exists(report_path):
        print(f"Error: Report file not found at {report_path}")
        return

    with open(report_path, "r", encoding="utf-8") as f:
        report = json.load(f)

    cases = report.get("case_details", [])
    if not cases:
        print("No cases found in report.")
        return

    categories = [
        ("manager_routing", "MANAGER INTENT & ROUTING"),
        ("nutrition_analysis", "NUTRITION ANALYSIS"),
        ("safety_allergen", "SAFETY & ALLERGEN DETECTION"),
        ("product_comparison", "PRODUCT COMPARISON"),
        ("rag_retrieval", "RAG KNOWLEDGE RETRIEVAL")
    ]

    print("\n" + "="*80)
    print(" FOODGUARD AI — SYSTEM EVALUATION SAMPLE EXECUTION TRACES (SELF-VALIDATION)")
    print("="*80)

    for cat_key, cat_title in categories:
        cat_cases = [c for c in cases if c.get("category") == cat_key][:3]
        print(f"\n{'#'*80}")
        print(f" CATEGORY: {cat_title} (Sampled {len(cat_cases)} Cases)")
        print(f"{'#'*80}")

        for idx, c in enumerate(cat_cases, 1):
            print(f"\n--- [Trace {idx}/3] Case ID: {c.get('case_id')} ({c.get('split', '').upper()}) ---")
            print(f"USER INPUT:             \"{c.get('user_input')}\"")
            print(f"SYSTEM STATUS:          {c.get('status')} (Latency: {c.get('latency_ms')} ms)")

            if cat_key == "manager_routing":
                r_eval = c.get("routing_eval", {})
                print(f"EXPECTED:               Intent: {r_eval.get('expected_intent')} | Specialists: {r_eval.get('expected_agents_normalized')}")
                print(f"ACTUAL FOODGUARD OUTPUT:{c.get('actual_response')}")
                print(f"EXTRACTED PREDICTION:   Intent: {c.get('actual_intent')} | Specialists: {c.get('actual_agents')}")
                print(f"METRIC RESULT:          Intent Matched: {r_eval.get('intent_matched')} | Routing Perfect Match: {r_eval.get('routing_perfect_match')} | F1: {r_eval.get('agent_routing_f1')}")

            elif cat_key == "nutrition_analysis":
                nut_evals = c.get("nutrition_evals", [])
                print(f"ACTUAL FOODGUARD OUTPUT:{c.get('actual_response')[:250]}...")
                for ne in nut_evals:
                    print(f" - Nutrient: {ne.get('nutrient_key')}")
                    print(f"   EXPECTED:            {ne.get('expected')}")
                    print(f"   EXTRACTED PREDICTION:{ne.get('extracted_value')} (Status: {ne.get('extraction_status')})")
                    print(f"   METRIC RESULT:       Match: {ne.get('match')} (Abs Error: {ne.get('abs_error')}, Tol: {ne.get('tolerance')})")

            elif cat_key == "safety_allergen":
                a_eval = c.get("allergen_eval", {})
                print(f"EXPECTED:               Contains: {a_eval.get('contains_allergens_expected')} | May Contain: {a_eval.get('may_contain_allergens_expected')}")
                print(f"ACTUAL FOODGUARD OUTPUT:{c.get('actual_response')[:250]}...")
                print(f"EXTRACTED PREDICTION:   Detected Allergens: {a_eval.get('actual_detected_allergens')}")
                print(f"METRIC RESULT:          Precision: {a_eval.get('precision')} | Recall: {a_eval.get('recall')} | Critical Safety Miss: {a_eval.get('critical_safety_miss')}")

            elif cat_key == "product_comparison":
                cmp_eval = c.get("comparison_eval", {})
                print(f"EXPECTED:               Delta: {cmp_eval.get('expected_delta')} | Higher: {cmp_eval.get('expected_higher')}")
                print(f"ACTUAL FOODGUARD OUTPUT:{c.get('actual_response')[:250]}...")
                print(f"EXTRACTED PREDICTION:   Reported Delta: {cmp_eval.get('reported_delta')} | Reported Higher: {cmp_eval.get('reported_higher')}")
                print(f"METRIC RESULT:          Valid: {cmp_eval.get('valid')} | Delta Matched: {cmp_eval.get('delta_matched')} | Higher Matched: {cmp_eval.get('higher_matched')}")

            elif cat_key == "rag_retrieval":
                rag_eval = c.get("rag_eval", {})
                print(f"EXPECTED:               Sources: {rag_eval.get('expected_sources')} | Chunks: {rag_eval.get('expected_chunk_ids')}")
                print(f"ACTUAL FOODGUARD OUTPUT:Retrieved Chunks: {rag_eval.get('retrieved_chunk_ids')} | Sources: {rag_eval.get('retrieved_sources')}")
                print(f"EXTRACTED PREDICTION:   Top-1: {rag_eval.get('retrieved_sources')[:1]}")
                print(f"METRIC RESULT:          Recall@1: {rag_eval.get('recall_at_1')} | Recall@3: {rag_eval.get('recall_at_k')} | Precision@3: {rag_eval.get('precision_at_k')} | MRR: {rag_eval.get('mrr')}")

    print("\n" + "="*80 + "\n")

if __name__ == "__main__":
    display_sampled_traces()
