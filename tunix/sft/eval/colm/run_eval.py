import json
from tqdm import tqdm
import tunix.sft.eval.colm.eval_utils as eval_utils
from tunix.sft.eval.colm.prompt_utils import *
import tunix.sft.eval.chat_templates as chat_templates
from tunix.sft.eval.colm.data_loader import BatchDatasetLoader

'''
train -> chat: done, alp: done
chat:
val ->
    during train -> [{user, asst}]. done
    during eval -> [{}, {}, {}]. labels not reqd. TODO here. 

TODO: to check answer parsing code.

<|im_start|>system
  You are a helpful assistant.
<|im_end|>
######EX1 <|im_start|>user
  If $
      rac{x-1}{3}=k$ and $k=3$, what is the value of $x$ ?
  Answer Choices: (A) 2 (B) 4 (C) 9 (D) 10
<|im_end|>

######<|im_start|>assistant
  If k = 3, then x - 1 = 3 * 3, therfore, x - 1 = 9 and x = 10. The answer is D
<|im_end|>

#####QUES<|im_start|>user
  If $
      rac{x-1}{3}=k$ and $k=3$, what is the value of $x$ ?
  Answer Choices: (A) 2 (B) 4 (C) 9 (D) 10
<|im_end|>
<|im_start|>assistant
#####RESP: If k = 3, then x - 1 = 3 * 3, therfore, x - 1 = 9 and x = 10. The answer is D
<|im_end|>

<|im_start|>user
  If $
      rac{x-1}{3}=k$ and $k=3$, what is the value of $x$ ?
  Answer Choices: (A) 2 (B) 4 (C) 9 (D) 10
<|im_end|>
<|im_start|>assistant
  If k = 3, then x - 1 = 3 * 3, therfore, x - 1 = 9 and x = 10. The answer is D
<|im_end|>
'''

# def convert_to_chatformat(demos, ques):
#   ret = {
#     "messages": [],
#   }
#   for msg in demos:
#     ret["messages"].append({"role": "user", "content": msg[0]})
#     ret["messages"].append({"role": "assistant", "content": msg[1]})

#   ret["messages"].append({"role": "user", "content": ques})

#   return ret

# def promptify_chat(tokenizer, demonstrations, questions):
#     # NOTE: demonstrations are shared across questions
#     inputs_strs = []
#     for ques in questions:
#       _, text = chat_templates.tokenize_chat_template(
#           tokenizer, convert_to_chatformat(demonstrations, ques), 
#           1024, add_generation_prompt=True,
#           return_text=True,
#         )
#       inputs_strs.append(text)
      
#     return inputs_strs
MMLU_SUFFIX = (
    "\n\n"
    "First, think through the problem step by step.\n"
    "Then, on a new line at the very end, write the final answer in the form:\n"
    # "The answer is A.\n"
    # "where A is one of: A, B, C, or D.\n"
    "Do not include the option text on that final answer line."
)

def _run_question_answer(inference_fn, tokenizer, max_prompt_length, dataset, shots, stem_flan_type, 
                         questions: list, groundtruths: list, verbose=False, collect_rerun: bool = False, ):
    icl_ds = dataset
    if "mmlu" in dataset:
        icl_ds = "mmlu_mathematics"
    if "numglue" in dataset:
        icl_ds = "numglue"
    used_examples = get_examples(icl_ds, shots, stem_flan_type)
    # extra_inst = MMLU_SUFFIX if "mmlu" in dataset else None
    extra_inst = None
    prompts = chat_templates.promptify_alpaca(used_examples, 
                                              {"instruction": questions}, extra_instruction=extra_inst)

    dummy = " "
    mask = [len(tokenizer.encode(p)) > max_prompt_length for p in prompts]
    print(f"Skipping examples: {sum(mask)}/{len(mask)}")
    fixed = [dummy if m else p for p, m in zip(prompts, mask)]

    outputs = inference_fn(fixed)
    outputs = ["" if m else o for m, o in zip(mask, outputs)]

    for i in range(3):
      if not verbose: break
      print(f"START##########Prompt {i}#################")
      print(f">>>prompt {prompts[i]}")
      print(f">>>out {outputs[i]}")
      print(f"END##########Prompt {i}#################")

    # We need to collect the values and possibly the rerun questions;
    returned_value = []
    rerun_questions = []
    rerun_groundtruths = []
    for output, prompt, question, groundtruth in zip(outputs, prompts, questions, groundtruths):
      # print("prompt", prompt)
      # print("@@@@@@@@@@@@@@@@@@@@@@@@")
      # print("output", output)
      # raise
      
      # The reason for this split is that the model also tends to generate NEW
      # examples. But for eval, we dont care
      # for these new barfed up examples, but only the answer.
      # Apparently, the model may believe that the few-shot demonstrations
      # are a signal for the intent to generate even more examples.
      # Thus we add this is a safety check. 
      output = output.split("### Instruction")[0]
      # output = output.split("<|im_end|>")[0]
      
      answer_trigger = ('####', 'The answer is')
      if 'print(' in output:
          tmp = eval_utils.execute_with_timeout(output)
          tmp = 'The answer is' + ' ' + tmp
          answer = eval_utils.answer_clean(dataset, answer_trigger, tmp)
      else:
          answer = eval_utils.answer_clean(dataset, answer_trigger, output)

      if answer == "" and collect_rerun:
          rerun_questions.append(eval_utils.remove_flan_tag(question, stem_flan_type))
          # print('Adding back', rerun_questions[-1])
          rerun_groundtruths.append(groundtruth)
          continue

      returned_value.append((question, output, answer, groundtruth))

    if collect_rerun:
        assert len(returned_value) + len(rerun_questions) == len(questions) == len(groundtruths)
        return returned_value, rerun_questions, rerun_groundtruths
    else:
        return returned_value


def run_eval(inference_fn, tokenizer, max_prompt_length, out_file, data_root, dataset, shots, 
             batch_size, stem_flan_type, cot_backup=True, debug=True, split="all", dev_frac=0.2):
    from functools import partial
    correct, wrong = 0, 0
    file_handle = open(out_file, 'w')
    RUN_FUNCTION = partial(_run_question_answer,inference_fn, tokenizer,
                           max_prompt_length, dataset, shots, stem_flan_type)
    i=-1
    ques_num = 0
    for questions, groundtruths in tqdm(BatchDatasetLoader(dataset, data_root, batch_size, split=split, dev_frac=dev_frac)):
        i += 1
        verbose = i == 0
        # First pass to use PoT
        processed_questions = eval_utils.process_question_with_flan_tag(questions, stem_flan_type)

        if stem_flan_type == 'pot_prompt' and cot_backup:
            # if there is hybrid decoding, we try pot fist and then cot
            returned_values, rerun_questions, rerun_groundtruths = RUN_FUNCTION(processed_questions, 
                                                                                groundtruths, verbose=verbose,
                                                                                 collect_rerun=True)
            if rerun_questions:
                # if things are not working well
                processed_questions = eval_utils.process_question_with_flan_tag(rerun_questions, "")
                tmp = RUN_FUNCTION(processed_questions, rerun_groundtruths, verbose=verbose,
                                    collect_rerun=False)
                returned_values += tmp
        else:
            # only cot_prompt or pot_prompt, then we don't need to rerun
            returned_values = RUN_FUNCTION(processed_questions, groundtruths, verbose=verbose,
                                collect_rerun=False)

        for question, output, answer, groundtruth in returned_values:
            ques_num += 1
            # print(question, '#', answer, '#', groundtruth)
            if dataset == 'math':
                assert len(groundtruth) == 2, groundtruth
                groundtruth_str, groundtruth_num = groundtruth
                if eval_utils.compare_both_string_and_number_format(answer, groundtruth_str, groundtruth_num):
                    correct += 1
                else:
                    wrong += 1
            else:
                if answer == groundtruth:
                    correct += 1
                else:
                    wrong += 1

            if debug:
                _acc = correct / (correct + wrong)
                print(f"{dataset}:({ques_num})(ans/gnd):{answer}/{groundtruth} # acc:{_acc} ")

            example = {
                'question': question,
                'correct': groundtruth,
                'solution': output,
                'pred': answer,
                'task': dataset
            }

            file_handle.write(json.dumps(example) + '\n')
        print('finished one epoch')


    final_acc = correct / (correct + wrong)
    print('final accuracy: ', final_acc)
    file_handle.close()

    import os
    import pandas as pd
    # write the final accuracy to a csv file
    filename = out_file.replace('.jsonl', '.csv')
    filename = filename.replace(f'{dataset}_{shots}shots_', '')
    if os.path.exists(filename):
        df = pd.read_csv(filename)
        df = pd.concat([df, pd.DataFrame({'dataset': dataset, 'accuracy': final_acc, 'shots': shots}, index=[0])], ignore_index=True)
    else:
        df = pd.DataFrame({'dataset': [dataset], 'accuracy': [final_acc], 'shots': [shots]})
    df.to_csv(filename, index=False)
    
    return {'dataset': dataset, 'accuracy': final_acc, 'shots': shots}