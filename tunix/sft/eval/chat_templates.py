import torch
from jinja2 import Environment, TemplateSyntaxError



qwen2_5_template = r'''
{%- if tools %}
    {{- '<|im_start|>system\n' }}
    {%- if messages[0]['role'] == 'system' %}
        {{- messages[0]['content'] }}
    {%- else %}
        {{- 'You are a helpful assistant.' }}
    {%- endif %}
    {{- "\n\n# Tools\n\nYou may call one or more functions to assist with the user query.\n\nYou are provided with function signatures within <tools></tools> XML tags:\n<tools>" }}
    {%- for tool in tools %}
        {{- "\n" }}
        {{- tool | tojson }}
    {%- endfor %}
    {{- "\n</tools>\n\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n{\"name\": <function-name>, \"arguments\": <args-json-object>}\n</tool_call><|im_end|>\n" }}
{%- else %}
    {%- if messages[0]['role'] == 'system' %}
        {{- '<|im_start|>system\n' + messages[0]['content'] + '<|im_end|>\n' }}
    {%- else %}
        {{- '<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n' }}
    {%- endif %}
{%- endif %}
{%- for message in messages %}
    {%- if (message.role == "user") or (message.role == "system" and not loop.first) %}
        {{- '<|im_start|>' + message.role + '\n' + message.content + '<|im_end|>\n' }}
    {%- elif (message.role == "assistant" and not message.tool_calls) %}
        {{- '<|im_start|>assistant\n' -}}
        {%- generation -%}
        {{- message.content -}}
        {%- endgeneration -%}
        {{- '<|im_end|>\n' -}}
    {%- elif message.role == "assistant" %}
        {{- '<|im_start|>' + message.role }}
        {%- if message.content %}
            {{- '\n' -}}
            {%- generation -%}
            {{- message.content -}}
            {%- endgeneration -%}
        {%- endif %}
        {%- for tool_call in message.tool_calls %}
            {%- if tool_call.function is defined %}
                {%- set tool_call = tool_call.function %}
            {%- endif %}
            {{- '\n<tool_call>\n{"name": "' }}
            {{- tool_call.name }}
            {{- '", "arguments": ' }}
            {{- tool_call.arguments | tojson }}
            {{- '}\n</tool_call>' }}
        {%- endfor %}
        {{- '<|im_end|>\n' }}
    {%- elif message.role == "tool" %}
        {%- if (loop.index0 == 0) or (messages[loop.index0 - 1].role != "tool") %}
            {{- '<|im_start|>user' }}
        {%- endif %}
        {{- '\n<tool_response>\n' }}
        {{- message.content }}
        {{- '\n</tool_response>' }}
        {%- if loop.last or (messages[loop.index0 + 1].role != "tool") %}
            {{- '<|im_end|>\n' }}
        {%- endif %}
    {%- endif %}
{%- endfor %}
{%- if add_generation_prompt %}
    {{- '<|im_start|>assistant\n' }}
{%- endif %}
'''

def prompt_alpaca(demonstrations: list, ques: str, resp: str = None):
  raise
  sys = (
      "Below is an instruction that describes a task. "
      "Write a response that appropriately completes the request.\n"
  )

  icl = ""
  # if demos is empty there are not ICL exs
  for q, a in demonstrations:
      icl += '\n' + '### Instruction:\n{query}\n\n### Response: {response}\n'.format(query=q, response=a)

  if resp is None:
    prefix = '\n' + '### Instruction:\n{query}\n\n### Response: '.format(query=ques)
  else:
    prefix = '\n' + '### Instruction:\n{query}\n\n### Response: {resp}'.format(query=ques, resp=resp)

  return sys + icl + prefix

  
def prompt_alpaca_input(demonstrations: list, inst: str, resp:str = None, inp: str = None, extra_instruction=None):
  sys = (
      "Below is an instruction that describes a task. "
      "Write a response that appropriately completes the request."
  )

  icl = ""
  '''
  \n\n### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n
  '''
  def linearize(i, inp, r, extra=False):
    ret = f'\n\n### Instruction:\n{i}'
    if inp is not None and len(inp):
        ret += f'\n\n### Input:\n{inp}'

    if extra_instruction is not None and extra:
        ret += extra_instruction

    if r is not None:
        ret += f'\n\n### Response:\n{r}'
    else:
        ret += f'\n\n### Response:\n'
    return ret

  # if demos is empty there are no ICL exs
  if len(demonstrations):
    if len(demonstrations[0]) == 3:
      for q, i, a in demonstrations:
        icl += '\n\n' + linearize(q, i, a, extra=False)
    else:
      for q, a in demonstrations:
        icl += '\n\n' + linearize(q, None, a, extra=False)

  prefix = linearize(inst, inp, resp, extra=True)

  return sys + icl + prefix



def promptify_alpaca(demonstrations: list, batch, extra_instruction=None):
    insts, inps = batch["instruction"], batch.get("input")
    if inps is None:
        inps = [None for _ in range(len(insts))]
    # NOTE: demonstrations are shared across questions
    fn = prompt_alpaca_input
    inputs_strs = [fn(demonstrations, inst, resp=None, inp=inp, extra_instruction=extra_instruction) for inst, inp in zip(insts, inps)]
    return inputs_strs

def encode_sft_chat(user_text: str, assistant_text: str, tokenizer, max_seq_length: int, print_ex=False):
    messages = [
        {"role": "user", "content": user_text},
        {"role": "assistant", "content": assistant_text},
    ]

    enc = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        truncation=True,
        max_length=max_seq_length,
        add_generation_prompt=False,
        return_assistant_tokens_mask=True,
    )

    # Flatten to 1D to line up with input_ids.
    mask = torch.as_tensor(enc["assistant_masks"]).flatten()

    input_ids = enc["input_ids"].flatten()
    attention_mask = enc.get("attention_mask", torch.ones_like(input_ids)).flatten()

    labels = input_ids.clone()
    labels[mask == 0] = -100

    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": attention_mask,
    }

# def tokenize_prompt_alpaca(
#     tokenizer,
#     demonstrations,
#     ques,
#     max_seq_length,
#     resp=None,
#     mask_value=-100,
#     add_special_tokens=True,
#     return_text=False,
# ): pass
# def encode_sft_chat(user_text: str, assistant_text: str, tokenizer, max_seq_length: int, print_ex=False):
#     messages = [
#         {"role": "user", "content": user_text},
#         {"role": "assistant", "content": assistant_text},
#     ]
def tokenize_chat_template(
  tokenizer, example, 
  max_seq_length: int, add_generation_prompt: bool,
  return_text=False,
  mask_input=True,
):
    """
    Expects example["messages"] = [{"role": "...", "content": "..."}, ...]
    Produces input_ids/attention_mask and labels that train ONLY on assistant tokens.
    """
    messages = example["messages"]
    if not messages:
        raise ValueError("messages field is empty")

    # IMPORTANT: for SFT, do NOT add a generation prompt stub.
    # We want the assistant content (from dataset) to be in the sequence.
    kwargs = dict(
        return_tensors="pt",
        truncation=True,
        max_length=max_seq_length,
        add_generation_prompt=add_generation_prompt,
    )

    try:
        enc = tokenizer.apply_chat_template(
            messages,
            return_assistant_tokens_mask=True,
            tokenize=True,
            return_dict=True,
            **kwargs,
        )
        assistant_mask = torch.Tensor(enc["assistant_masks"]).flatten()

    except TypeError:
        raise

    if assistant_mask is None: raise

    input_ids = enc["input_ids"] # shape: (1, seq_len)
    input_ids = input_ids.flatten()
    attention_mask = enc.get("attention_mask", torch.ones_like(input_ids)).flatten()
    labels = input_ids.clone()

    if mask_input:
        labels[assistant_mask == 0] = -100
    else:
        labels = input_ids


    ret = {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": attention_mask,
    }

    if return_text:
      text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            truncation=True,
            max_length=max_seq_length,
            add_generation_prompt=add_generation_prompt,
        )
      return ret, text

    return ret

def tokenize_prompt_alpaca(
    tokenizer,
    demonstrations,
    ques,
    max_seq_length,
    resp=None,
    inp=None,
    mask_value=-100,
    add_special_tokens=True,
    return_text=False,
    mask_input=True,
):
  def tok(txt):
    enc = tokenizer(
      txt,
      add_special_tokens=add_special_tokens,
      return_tensors="pt",
      truncation=True,
      max_length=max_seq_length,
    )
    ids = enc["input_ids"].flatten()
    att = enc["attention_mask"].flatten()
    return ids, att

  if resp is None:
    raise
    prompt = prompt_alpaca_input(demonstrations, ques, resp=None, inp=inp)
    ids, att = tok(prompt)
    labels = torch.full_like(ids, mask_value)
    return {
      "input_ids": ids,
      "labels": labels,
      "attention_mask": att,
    }

  full_prompt = prompt_alpaca_input(demonstrations, ques, resp=resp, inp=inp)
  prefix_prompt = prompt_alpaca_input(demonstrations, ques, resp="", inp=inp)

  ids_full, att = tok(full_prompt)
  ids_prefix, _ = tok(prefix_prompt)

  assert ids_full.ndim == 1
  assert ids_prefix.ndim == 1
  assert ids_full.size(0) > 0
  assert ids_prefix.size(0) > 0

  prefix_len = ids_prefix.size(0)

  assert prefix_len <= ids_full.size(0)
  # assert prefix_len < max_seq_length    # otherwise answer completely truncated

  # prefix consistency: after truncation, prefix must match full
  # assert torch.equal(ids_full[:prefix_len], ids_prefix), "prefix mismatch"

  if mask_input:
    labels = torch.full_like(ids_full, mask_value)
    labels[prefix_len:] = ids_full[prefix_len:]

  else:
    labels = ids_full.clone()
      
  ret = {
    "input_ids": ids_full,
    "labels": labels,
    "attention_mask": att,
  }
  if return_text: return ret, full_prompt
  return ret


def print_chat_template(tokenizer):
    print("=== Tokenizer ===")
    print("name_or_path:", getattr(tokenizer, "name_or_path", None))
    print("class:", type(tokenizer).__name__)

    print("\n=== Special tokens ===")
    print("bos_token:", repr(tokenizer.bos_token), "id:", tokenizer.bos_token_id)
    print("eos_token:", repr(tokenizer.eos_token), "id:", tokenizer.eos_token_id)
    print("pad_token:", repr(tokenizer.pad_token), "id:", tokenizer.pad_token_id)

    tmpl = getattr(tokenizer, "chat_template", None)

    print("\n=== chat_template present? ===")
    print(tmpl is not None)

    if tmpl is None:
        return

    print("\n=== chat_template contains generation tags? ===")
    print("has {% generation %}:", "{% generation %}" in tmpl)
    print("has {% endgeneration %}:", "{% endgeneration %}" in tmpl)

    print("\n=== chat_template (verbatim) ===")
    print(tmpl)
# tmpl = qwen2_5_template
# try:
#     Environment().from_string(tmpl)
#     print("✅ Template parses OK")
# except TemplateSyntaxError as e:
#     print("❌ TemplateSyntaxError")
#     print("message:", e.message)
#     print("line:", e.lineno)

#     lines = tmpl.splitlines()
#     start = max(e.lineno - 5, 0)
#     end = min(e.lineno + 4, len(lines))
#     for i in range(start, end):
#         print(f"{i+1:04d}: {lines[i]}")
#     raisee