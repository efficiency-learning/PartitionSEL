import contextlib
from functools import partial
from typing import List, Union

import numpy as np
import torch
from datasets import load_dataset


@contextlib.contextmanager
def temp_seed(seed):
    state = np.random.get_state()
    np.random.seed(seed)
    torch.manual_seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)

def get_training_dataset(train_files: List[str], 
    tokenizer, max_seq_length, encode_func="encode_sft_alpaca", sample_percentage=1.0, seed=0):
    """ get training dataset with a specified seed """

    raw_datasets = load_raw_dataset(
        train_files, sample_percentage=sample_percentage, seed=seed)
    lm_datasets = encode_data(
        raw_datasets, tokenizer, max_seq_length, func_name=encode_func)
    return lm_datasets


def load_raw_dataset(train_files: Union[List[str], str], sample_size=None, sample_percentage=1.0, seed=0):
    """ load raw dataset """
    if isinstance(train_files, str):
        train_files = [train_files]
    processed_datasets = load_dataset(
        "json",
        data_files=train_files,
    )["train"]
    if sample_size is None:
        sample_size = int(len(processed_datasets) * sample_percentage)

    if sample_size == len(processed_datasets):
        return processed_datasets  # not shuffle

    with temp_seed(seed):
        index = np.random.permutation(len(processed_datasets))[:sample_size]

    sampled_dataset = processed_datasets.select(index)

    return sampled_dataset

def encode_datav2(raw_datasets, tokenizer, max_seq_length, mask_input,
                    processing_num_workers=50, verbose=False, overwrite_cache=False):
    # if already encoded, return
    if "input_ids" in raw_datasets.features:
        return raw_datasets

    encode_function = partial(
            encode_sft_alpaca,
            tokenizer=tokenizer,
            max_seq_length=max_seq_length,
            mask_input=mask_input,
            verbose=verbose,
        )
    
    # To speed up this part, we use multiprocessing.
    lm_datasets = raw_datasets.map(
        encode_function,
        batched=False,
        load_from_cache_file=False,
        with_indices=True,
        num_proc=processing_num_workers,
        desc="Tokenizing and reformatting instruction data",
    )
    lm_datasets.set_format(type="pt")
    return lm_datasets
    
def encode_data(raw_datasets, tokenizer, max_seq_length, func_name, processing_num_workers=10, overwrite_cache=False):
    """ encode data with the specified tokenizer and the chat format. """
    # if already encoded, return
    if "input_ids" in raw_datasets.features:
        return raw_datasets
    encode_function = get_encode_function(
        raw_datasets, tokenizer, max_seq_length, func_name)
    
    # To speed up this part, we use multiprocessing.
    lm_datasets = raw_datasets.map(
        encode_function,
        batched=False,
        load_from_cache_file=False,
        with_indices=True,
        num_proc=processing_num_workers,
        desc="Tokenizing and reformatting instruction data",
    )
    lm_datasets.set_format(type="pt")
    return lm_datasets

def get_encode_function(raw_datasets, tokenizer, max_seq_length, func):
    """ get encode function based on the dataset. """
    if "prompt" in raw_datasets.column_names and "completion" in raw_datasets.column_names:
        encode_function = partial(
            encode_with_prompt_completion_format,
            tokenizer=tokenizer,
            max_seq_length=max_seq_length,
        )
    elif "messages" in raw_datasets.column_names:
        if func == "encode_with_chat_template":
            encode_func = encode_with_chat_template
        elif func == "encode_sft_alpaca":
            encode_func = encode_sft_alpaca
        elif func == "encode_with_messages_format":
            encode_func = encode_with_messages_format
        else:
            encode_func = encode_with_messages_format_with_llama2_chat
        encode_function = partial(
            encode_func,
            tokenizer=tokenizer,
            max_seq_length=max_seq_length,
        )
    else:
        raise ValueError(
            "You need to have either 'prompt'&'completion' or 'messages' in your column names.")
    return encode_function


def encode_with_prompt_completion_format(example, tokenizer, max_seq_length):
    '''
    Original implementation of the function: https://github.com/allenai/open-instruct/blob/9ebcb582cfc243a6dab75b4302fa432784db26c2/open_instruct/finetune.py#L238

    Here we assume each example has 'prompt' and 'completion' fields.
    We concatenate prompt and completion and tokenize them together because otherwise prompt will be padded/trancated 
    and it doesn't make sense to follow directly with the completion.
    '''
    # if prompt doesn't end with space and completion doesn't start with space, add space
    if not example['prompt'].endswith((' ', '\n', '\t')) and not example['completion'].startswith((' ', '\n', '\t')):
        example_text = example['prompt'] + ' ' + example['completion']
    else:
        example_text = example['prompt'] + example['completion']
    example_text = example_text + tokenizer.eos_token
    tokenized_example = tokenizer(
        example_text, return_tensors='pt', max_length=max_seq_length, truncation=True)
    input_ids = tokenized_example.input_ids
    labels = input_ids.clone()
    tokenized_prompt = tokenizer(
        example['prompt'], return_tensors='pt', max_length=max_seq_length, truncation=True)
    # mask the prompt part for avoiding loss
    labels[:, :tokenized_prompt.input_ids.shape[1]] = -100
    attention_mask = torch.ones_like(input_ids)
    return {
        'input_ids': input_ids.flatten(),
        'labels': labels.flatten(),
        'attention_mask': attention_mask.flatten(),
    }

def encode_with_chat_template(example, idx, tokenizer, max_seq_length: int):
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
        add_generation_prompt=False,
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

    input_ids = enc["input_ids"] # shape: (1, seq_len)
    input_ids = input_ids.flatten()
    attention_mask = enc.get("attention_mask", torch.ones_like(input_ids)).flatten()

    labels = input_ids.clone()

    if assistant_mask is None:
        raise RuntimeError(
            "Your transformers/tokenizer doesn't provide an assistant token mask via "
            "apply_chat_template(..., return_assistant_tokens_mask=True). "
            "Upgrade transformers, or you'll need a fallback masking method."
        )

    # assistant_mask is True/1 for assistant tokens; mask everything else out of loss.
    labels[assistant_mask == 0] = -100

    ret = {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": attention_mask,
    }
    if "source" in example.keys():
        ret['source'] = example["source"]

    return ret


from tunix.sft.eval import chat_templates
#TODO: check how colm processes tiger
def encode_sft_alpaca(example, idx, tokenizer, max_seq_length: int, mask_input: bool, verbose=False):
    msg =  example["messages"]
    if len(msg) == 2:
        ques = example["messages"][0]["content"]
        resp = example["messages"][1]["content"]
        inp = None
    elif len(msg) == 3:
        ques = example["messages"][0]["content"]
        inp = example["messages"][1]["content"]
        resp = example["messages"][2]["content"]
    else:
        raise NotImplementedError
        
    ret, text = chat_templates.tokenize_prompt_alpaca(tokenizer, [], ques, 
        max_seq_length, resp=resp, inp=inp, mask_value=-100, return_text=True, mask_input=mask_input)

    ret = {k: v.flatten() for k,v in ret.items()}
    if verbose and idx < 2:
      print("train_ds##################")
      print(ret)
      print("text", text)
      print("train_ds##################")


    if "source" in example.keys():
        ret['source'] = example["source"]

    return ret



def encode_with_messages_format(example, tokenizer, max_seq_length):
    '''
    Original implementation of the function: https://github.com/allenai/open-instruct/blob/9ebcb582cfc243a6dab75b4302fa432784db26c2/open_instruct/finetune.py#L264C1-L322C1

    Here we assume each example has a 'messages' field Each message is a dict with 'role' and 'content' fields.
    We concatenate all messages with the roles as delimiters and tokenize them together.
    '''
    messages = example['messages']
    if len(messages) == 0:
        raise ValueError('messages field is empty.')

    example_text = concat_messages(messages, tokenizer)
    tokenized_example = tokenizer(
        example_text, return_tensors='pt', max_length=max_seq_length, truncation=True)
    input_ids = tokenized_example.input_ids
    labels = input_ids.clone()

    # mask the non-assistant part for avoiding loss
    for message_idx, message in enumerate(messages):
        if message["role"] != "assistant":
            if message_idx == 0:
                message_start_idx = 0
            else:
                message_start_idx = tokenizer(
                    concat_messages(messages[:message_idx], tokenizer), return_tensors='pt', max_length=max_seq_length, truncation=True
                ).input_ids.shape[1]
            if message_idx < len(messages) - 1 and messages[message_idx+1]["role"] == "assistant":
                # here we also ignore the role of the assistant
                messages_so_far = concat_messages(
                    messages[:message_idx+1], tokenizer) + "<|assistant|>\n"
            else:
                messages_so_far = concat_messages(
                    messages[:message_idx+1], tokenizer)
            message_end_idx = tokenizer(
                messages_so_far,
                return_tensors='pt',
                max_length=max_seq_length,
                truncation=True
            ).input_ids.shape[1]
            labels[:, message_start_idx:message_end_idx] = -100

            if message_end_idx >= max_seq_length:
                break

    attention_mask = torch.ones_like(input_ids)
    return {
        'input_ids': input_ids.flatten(),
        'labels': labels.flatten(),
        'attention_mask': attention_mask.flatten(),
    }


def concat_messages(messages, tokenizer):
    message_text = ""
    for message in messages:
        if message["role"] == "system":
            message_text += "<|system|>\n" + message["content"].strip() + "\n"
        elif message["role"] == "user":
            message_text += "<|user|>\n" + message["content"].strip() + "\n"
        elif message["role"] == "assistant":
            message_text += "<|assistant|>\n" + \
                message["content"].strip() + tokenizer.eos_token + "\n"
        else:
            raise ValueError("Invalid role: {}".format(message["role"]))
    return message_text


def encode_with_messages_format_with_llama2_chat(example, tokenizer, max_seq_length):
    '''
    Here we assume each example has a 'messages' field Each message is a dict with 'role' and 'content' fields.
    We concatenate all messages with the roles as delimiters and tokenize them together.
    '''
    messages = example['messages']
    if len(messages) == 0:
        raise ValueError('messages field is empty.')

    def _concat_messages(messages, ):
        B_INST, E_INST = "[INST]", "[/INST]"
        bos = "<s>"
        eos = "</s>"
        formatted_text = ""
        for message in messages:
            if message["role"] == "user":
                formatted_text += bos + \
                    f"{B_INST} {(message['content']).strip()} {E_INST}"
            elif message["role"] == "assistant":
                formatted_text += f" {(message['content'])} " + eos
            else:
                raise ValueError(
                    "Llama2 chat template only supports 'system', 'user' and 'assistant' roles. Invalid role: {}.".format(
                        message["role"])
                )
        formatted_text = formatted_text[len(bos):]
        return formatted_text

    example_text = _concat_messages(messages).strip()
    print(example_text)
    tokenized_example = tokenizer(
        example_text, return_tensors='pt', max_length=max_seq_length, truncation=True)
    input_ids = tokenized_example.input_ids
    labels = input_ids.clone()

    # mask the non-assistant part for avoiding loss
    for message_idx, message in enumerate(messages):
        if message["role"] != "assistant":
            if message_idx == 0:
                message_start_idx = 0
            else:
                message_start_idx = tokenizer(
                    _concat_messages(messages[:message_idx]), return_tensors='pt', max_length=max_seq_length, truncation=True
                ).input_ids.shape[1]
            if messages[message_idx+1]["role"] == "assistant":
                messages_so_far = _concat_messages(messages[:message_idx+1])
            message_end_idx = tokenizer(
                messages_so_far,
                return_tensors='pt',
                max_length=max_seq_length,
                truncation=True
            ).input_ids.shape[1]
            labels[:, message_start_idx:message_end_idx] = -100

            if message_end_idx >= max_seq_length:
                break

    attention_mask = torch.ones_like(input_ids)
    return {
        'input_ids': input_ids.flatten(),
        'labels': labels.flatten(),
        'attention_mask': attention_mask.flatten(),
    }
