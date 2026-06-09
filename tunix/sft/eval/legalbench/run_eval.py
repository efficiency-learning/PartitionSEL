import os
import re
import json
import datasets
from tqdm import tqdm

import tunix.sft.eval.chat_templates as chat_templates
from tunix.sft.eval.legalbench.evaluation import evaluate as legalbench_evaluate
from tunix.sft.eval.eval_setup import eval_setup
from tunix.examples.data.ift.configs import (
    LEGALBENCH_TASKS,
    LEGALBENCH_TASK_CATEGORIES,
    LEGALBENCH_TASK_LABELS,
)


# Prompt templates copied from legalbench/tasks/{task}/base_prompt.txt
LEGALBENCH_PROMPTS_DIR = os.path.join(os.path.dirname(__file__), "prompts")


def _constrain_prompt(filled_prompt, task_name):
    """Append an output-format constraint to the prompt.

    Set to None in LEGALBENCH_TASK_LABELS to skip (e.g. free-form tasks).
    To switch strategy (e.g. post-hoc extraction), replace this function.
    """
    labels = LEGALBENCH_TASK_LABELS.get(task_name)
    if labels is None:
        return filled_prompt
    return f"{filled_prompt} Reply with only: {labels}\n"


def _extract_answer(raw_pred, task_name):
    """Extract the answer label from a (possibly verbose) model prediction.

    For tasks with known labels, searches for the first matching label in the
    output.  Falls back to first-line extraction if no label matches.
    To switch strategy, replace this function.
    """
    text = raw_pred.strip()
    labels_str = LEGALBENCH_TASK_LABELS.get(task_name)
    if labels_str is None:
        # Free-form: just take first line
        return text.split("\n")[0].strip()
    # Check for each valid label (longest first to prefer e.g. "Limitation of liability" over "Other")
    labels = [l.strip() for l in labels_str.split(",")]
    labels_sorted = sorted(labels, key=len, reverse=True)
    text_lower = text.lower()
    for label in labels_sorted:
        if label.lower() in text_lower:
            return label
    # No label found — return first line as fallback
    return text.split("\n")[0].strip()


# ---------------------------------------------------------------------------
# Evaluation: delegates to the official legalbench evaluation.py
# ---------------------------------------------------------------------------

def _evaluate_task(task, generations, answers):
    """Score a single task using the official legalbench metric."""
    return legalbench_evaluate(task, generations, answers)


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

def _load_prompt_template(task_name, max_shots=None):
    """Load the prompt template for a task, optionally trimming few-shot examples."""
    path = os.path.join(LEGALBENCH_PROMPTS_DIR, f"{task_name}.txt")
    with open(path) as f:
        template = f.read()
    if max_shots is not None:
        template = _trim_shots(template, max_shots)
    return template


def _trim_shots(template, max_shots):
    """Keep only the first `max_shots` few-shot examples in a template.

    Templates have the structure:
      <instruction paragraph>
      \n
      <Field>: <text>\n<Label>: <answer>\n\n   (repeated N times)
      <Field>: {{placeholder}}\n<Label>:

    We split on `Label: <answer>` boundaries and keep the first max_shots.
    """
    # Split into: [text_before, "Label: answer\n", text, "Label: answer\n", ..., final_query]
    parts = re.split(r"(Label: \S+.*?\n)", template)
    if len(parts) < 3:
        return template  # no few-shot examples found

    n_examples = (len(parts) - 1) // 2
    if max_shots >= n_examples:
        return template  # nothing to trim

    # parts[0] contains the instruction + first example's field text.
    # Separate instruction (before first blank line) from first example content.
    first_blank = parts[0].find("\n\n")
    if first_blank == -1:
        instruction = ""
        first_example_field = parts[0]
    else:
        instruction = parts[0][:first_blank + 2]  # include the blank line
        first_example_field = parts[0][first_blank + 2:]

    test_query = parts[-1]  # "\n<Field>: {{placeholder}}\n<Label>:"

    if max_shots == 0:
        return instruction + test_query.lstrip("\n")

    kept = [instruction, first_example_field, parts[1]]  # instruction + first example
    for i in range(1, max_shots):
        kept.append(parts[2 * i])      # field text
        kept.append(parts[2 * i + 1])  # "Label: answer\n"
    kept.append(test_query)
    return "".join(kept)


def _fill_template(template, row):
    """Fill {{field}} placeholders in the template with row values."""
    prompt = str(template)
    for k, v in row.items():
        prompt = prompt.replace("{{" + k + "}}", str(v))
    return prompt


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_eval(pipeline, model, mesh, tokenizer, eval_split="test",
             tasks=None, cap_per_task=500, batch_size=4, max_shots=2,
             summarize_fn=None):
  if tasks is None:
    tasks = LEGALBENCH_TASKS

  with eval_setup(pipeline, model, mesh, tokenizer,
                  max_prompt_tokens=3072, max_generation_steps=128,
                  cache_size=4096, batch_size=batch_size,
                  eval_split=eval_split, sub_dir="legalbench") as (inf2, tok, out_dir):

    all_res = []
    for task_name in tasks:
      try:
        ds = datasets.load_dataset("nguha/legalbench", task_name,
                                    split=eval_split, trust_remote_code=True)
      except Exception as e:
        print(f"Warning: skipping LegalBench task {task_name}: {e}")
        continue

      if cap_per_task is not None and len(ds) > cap_per_task:
        ds = ds.shuffle(seed=42).select(range(cap_per_task))

      # Load the official prompt template (with few-shot examples baked in)
      prompt_template = _load_prompt_template(task_name, max_shots=max_shots)

      print(f"\nTESTING legalbench/{task_name} ({len(ds)} examples)")
      out_path = os.path.join(out_dir, f"{task_name}.jsonl")
      all_preds, all_gts = [], []

      with open(out_path, "w") as f:
        for start in tqdm(range(0, len(ds), batch_size), desc=task_name):
          batch_rows = [ds[i] for i in range(start, min(start + batch_size, len(ds)))]
          ground_truths = [row["answer"] for row in batch_rows]

          # Build prompts using official template
          filled = [_fill_template(prompt_template, row) for row in batch_rows]
          constrained = [_constrain_prompt(p, task_name) for p in filled]
          prompts = [
            tok.apply_chat_template(
              [{"role": "user", "content": p}],
              tokenize=False, add_generation_prompt=True)
            for p in constrained
          ]
          # Truncate prompts that exceed max_prompt_tokens (keep the tail
          # so the test query + generation prompt are preserved).
          max_tok = 3072
          truncated = []
          for p in prompts:
            ids = tok.encode(p)
            if len(ids) > max_tok:
              ids = ids[-max_tok:]
            truncated.append(tok.decode(ids))
          prompts = truncated

          preds = inf2(prompts)

          for row, gt, pred in zip(batch_rows, ground_truths, preds):
            pred_clean = _extract_answer(pred, task_name)
            all_preds.append(pred_clean)
            all_gts.append(gt)

            record = {
              "prompt": filled[batch_rows.index(row)][-300:],
              "correct": gt,
              "pred": pred_clean,
              "task": task_name,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

      # Score using the official metric
      score = _evaluate_task(task_name, all_preds, all_gts)
      print(f"  {task_name}: {score:.4f} ({len(all_gts)} examples)")
      all_res.append({"dataset": f"legalbench/{task_name}", "accuracy": score, "shots": 0})

  # --- Per-category and overall macro-averages ---
  task2score = {r["dataset"].split("/", 1)[1]: r["accuracy"] for r in all_res}
  print("\n===== LegalBench Summary =====")
  cat_avgs = {}
  for cat, cat_tasks in LEGALBENCH_TASK_CATEGORIES.items():
      scores = [task2score[t] for t in cat_tasks if t in task2score]
      if scores:
          avg = sum(scores) / len(scores)
          cat_avgs[cat] = avg
          print(f"  {cat}: {avg:.4f}  ({len(scores)} tasks)")
          all_res.append({"dataset": f"legalbench_cat/{cat}", "accuracy": avg, "shots": 0})
  if task2score:
      overall = sum(task2score.values()) / len(task2score)
      print(f"  overall: {overall:.4f}  ({len(task2score)} tasks)")
      all_res.append({"dataset": "legalbench/overall", "accuracy": overall, "shots": 0})
  print("==============================\n")

  if summarize_fn is not None:
    summarize_fn(all_res, f"{out_dir}/results.csv")
  return all_res
