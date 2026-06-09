TIGER_DICT = {
    "instruction": "instruction",
    "response": "output",
    "source": "source",
    "ds_name": "tiger"
}
METAMATH_DICT = {
    "instruction": "query",
    "response": "response",
    "source": "type",
    "ds_name": "metamath"
}
MOL_DICT = {
    "instruction": "instruction",
    "input": "input",
    "response": "output",
    "source": "source",
    "ds_name": "MOL"
}
LAWINSTRUCT_DICT = {
    "instruction": "instruction",
    "input": "prompt",         # LawInstruct stores the passage/input in 'prompt'
    "response": "answer",
    # >>> DECISION: source grouping for domain weighting <<<
    # Change to "task_type" for coarser grouping (~6-8 types like
    # TEXT_CLASSIFICATION, QUESTION_ANSWERING, etc.)
    # or keep "dataset_name" for fine-grained grouping (~58 sub-datasets
    # like LexGLUE, CUAD, ContractNLI, etc.)
    "source": "task_type",
    "ds_name": "lawinstruct"
}

# ── LegalBench eval (curated subset of nguha/legalbench) ──
# >>> SOURCE GROUPING: per-task (default) vs. task-type <<<
# Each task gets its own source ID for fine-grained per-task metrics.
# To switch to coarser task-type grouping (~6 categories), set
# LEGALBENCH_SOURCE_MODE = "task_type".
LEGALBENCH_SOURCE_MODE = "per_task"  # or "task_type"
LEGALBENCH_TASK_TYPE_MAP = {
    # prefix -> type (used when LEGALBENCH_SOURCE_MODE="task_type")
    "contract_nli_": "rule_application",
    "learned_hands_": "issue_spotting",
    "unfair_tos": "interpretation",
    "consumer_contracts_qa": "interpretation",
    "contract_qa": "interpretation",
    "supply_chain_disclosure_": "issue_spotting",
    "corporate_lobbying": "rule_application",
    "proa": "rule_application",
    "definition_extraction": "rule_recall",
}
LEGALBENCH_TASKS = [
    # ── Issue Spotting (6 tasks, binary) ──
    "learned_hands_business",
    "learned_hands_crime",
    "learned_hands_employment",
    "learned_hands_housing",
    "learned_hands_torts",
    "corporate_lobbying",
    # ── Interpretation (6 tasks, trimmed from 12 for balance) ──
    "contract_nli_confidentiality_of_agreement",
    "contract_nli_limited_use",
    "contract_nli_sharing_with_third-parties",
    "consumer_contracts_qa",
    "unfair_tos",
    "supply_chain_disclosure_best_practice_verification",
    # ── Rule Recall (4 tasks) ──
    "citation_prediction_classification",
    "international_citizenship_questions",
    "nys_judicial_ethics",
    "textualism_tool_dictionaries",
    # ── Conclusion (6 tasks) ──
    "abercrombie",
    "hearsay",
    "personal_jurisdiction",
    "diversity_1",
    "telemarketing_sales_rule",
    "ucc_v_common_law",
    # ── Rhetoric (7 tasks) ──
    "definition_extraction",
    "proa",
    "definition_classification",
    "overruling",
    "function_of_decision_section",
    "oral_argument_question_purpose",
    "canada_tax_court_outcomes",
]

# Task categories from legalbench/tasks.py — balanced ~6 tasks per category.
LEGALBENCH_TASK_CATEGORIES = {
    "issue_spotting": [
        "learned_hands_business", "learned_hands_crime", "learned_hands_employment",
        "learned_hands_housing", "learned_hands_torts", "corporate_lobbying",
    ],
    "interpretation": [
        "contract_nli_confidentiality_of_agreement", "contract_nli_limited_use",
        "contract_nli_sharing_with_third-parties",
        "consumer_contracts_qa", "unfair_tos",
        "supply_chain_disclosure_best_practice_verification",
    ],
    "rule_recall": [
        "citation_prediction_classification", "international_citizenship_questions",
        "nys_judicial_ethics", "textualism_tool_dictionaries",
    ],
    "conclusion": [
        "abercrombie", "hearsay", "personal_jurisdiction",
        "diversity_1", "telemarketing_sales_rule", "ucc_v_common_law",
    ],
    "rhetoric": [
        "definition_extraction", "proa", "definition_classification",
        "overruling", "function_of_decision_section",
        "oral_argument_question_purpose", "canada_tax_court_outcomes",
    ],
}

# Valid answer labels per task (used to constrain model output).
# Mirrors the "Reply with either: ..." suffix from legalbench's claude_prompt.txt.
# Set to None for free-form / special-metric tasks.
LEGALBENCH_TASK_LABELS = {
    # issue_spotting
    "learned_hands_business": "Yes, No",
    "learned_hands_crime": "Yes, No",
    "learned_hands_employment": "Yes, No",
    "learned_hands_housing": "Yes, No",
    "learned_hands_torts": "Yes, No",
    "corporate_lobbying": "Yes, No",
    # interpretation
    "contract_nli_confidentiality_of_agreement": "Yes, No",
    "contract_nli_limited_use": "Yes, No",
    "contract_nli_sharing_with_third-parties": "Yes, No",
    "consumer_contracts_qa": "Yes, No",
    "unfair_tos": "Arbitration, Unilateral change, Content removal, Jurisdiction, Choice of law, Limitation of liability, Unilateral termination, Contract by using, Other",
    "supply_chain_disclosure_best_practice_verification": "Yes, No",
    # rule_recall
    "citation_prediction_classification": "Yes, No",
    "international_citizenship_questions": "Yes, No",
    "nys_judicial_ethics": "Yes, No",
    "textualism_tool_dictionaries": "Yes, No",
    # conclusion
    "abercrombie": "arbitrary, descriptive, fanciful, generic, suggestive",
    "hearsay": "Yes, No",
    "personal_jurisdiction": "Yes, No",
    "diversity_1": "Yes, No",
    "telemarketing_sales_rule": "Yes, No",
    "ucc_v_common_law": "UCC, Common Law",
    # rhetoric
    "definition_extraction": None,  # free-form
    "proa": "Yes, No",
    "definition_classification": "Yes, No",
    "overruling": "Yes, No",
    "function_of_decision_section": "Analysis, Conclusion, Decree, Facts, Issue, Procedural History, Rule",
    "oral_argument_question_purpose": "Background, Clarification, Communicate, Criticism, Humor, Implications, Support",
    "canada_tax_court_outcomes": "allowed, dismissed, other",
}
