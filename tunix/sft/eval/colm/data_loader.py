import json
import random
from statistics import mean
from tunix.sft.eval.colm.eval_utils import delete_extra_zero

def _split_data(questions, answers, split, dev_frac=0.2, seed=42):
    """Deterministically split data into dev/test. Returns the requested portion."""
    n = len(questions)
    indices = list(range(n))
    rng = random.Random(seed)
    rng.shuffle(indices)
    k = max(1, int(n * dev_frac))
    if split == "dev":
        chosen = sorted(indices[:k])
    else:
        chosen = sorted(indices[k:])
    return [questions[i] for i in chosen], [answers[i] for i in chosen]

def data_reader(dataset: str, data_root: str, split: str = "all", dev_frac: float = 0.2):
    questions = []
    answers = []
    decoder = json.JSONDecoder()
    DATA_ROOT = data_root
    if dataset == "aqua":
        with open(f'{DATA_ROOT}/AQuA/AQuA.json') as f:
            lines = f.readlines()
            for line in lines:
                json_res = decoder.raw_decode(line)[0]
                choice = "(" + "(".join(json_res["options"])
                choice = choice.replace("(", " (").replace(")", ") ")
                choice = "Answer Choices:" + choice
                questions.append(json_res["question"].strip() + "\n" + choice)
                answers.append(json_res["correct"])
    elif dataset == 'math':
        with open(f'{DATA_ROOT}/math/MATH.json', 'r') as f:
            loaded = json.load(f)
        for d in loaded:
            questions.append(d['question'])
            answers.append(d['answer'])
    elif dataset == "gsm8k":
        with open(f'{DATA_ROOT}/gsm8k/gsm8k.jsonl') as f:
            lines = f.readlines()
            for line in lines:
                json_res = decoder.raw_decode(line)[0]
                questions.append(json_res["question"].strip())
                answers.append(delete_extra_zero(json_res["answer"].split("#### ")[-1].replace(",", "")))
    elif dataset == "svamp":
        with open(f'{DATA_ROOT}/SVAMP/SVAMP.json') as f:
            json_data = json.load(f)
            for line in json_data:
                q = line["Body"].strip() + " " + line["Question"].strip()
                a = str(line["Answer"])
                if a[-2:] == ".0":
                    a = a[:-2]
                questions.append(q)
                answers.append(delete_extra_zero(a))
    elif 'mmlu' in dataset:
        with open(f'{DATA_ROOT}/mmlu/{dataset.split("_")[1]}.json') as f:
            json_data = json.load(f)
            for line in json_data:
                options = f'(A) {line["choices"][0]} (B) {line["choices"][1]} (C) {line["choices"][2]} (D) {line["choices"][3]}'
                q = line["question"] + '\n' + 'Answer Choices: ' + options
                a = ['A', 'B', 'C', 'D'][line['answer']]
                questions.append(q)
                answers.append(a)
    elif 'numglue' in dataset:
        with open(f'{DATA_ROOT}/numglue/{dataset}.json') as f:
            json_data = json.load(f)
            for line in json_data:
                assert isinstance(line['question'], str) and isinstance(line['question'], str), line
                questions.append(line['question'])
                answers.append(str(line['answer']))
    elif dataset in ['simuleq', 'deepmind', 'sat']:
        with open(f'{DATA_ROOT}/{dataset}/{dataset}.json') as f:
            json_data = json.load(f)
            for line in json_data:
                assert isinstance(line['question'], str) and isinstance(line['question'], str), line
                questions.append(line['question'])
                answers.append(str(line['answer']))
    else:
        raise ValueError(f"dataset is not properly defined ...{dataset}")

    q_len_list = []
    for q in questions:
        q_len_list.append(len(q.split(" ")))
    q_len_mean = mean(q_len_list)

    if split in ("dev", "test"):
        questions, answers = _split_data(questions, answers, split, dev_frac=dev_frac)

    q_len_list = [len(q.split(" ")) for q in questions]
    q_len_mean = mean(q_len_list)

    print("dataset : {} (split={})".format(dataset, split))
    print("data size : {}".format(len(answers)))
    print("average num of words for each sample : {}".format(q_len_mean))

    return questions, answers

class BatchDatasetLoader:
    def __init__(self, dataset: str, data_root: str, batch_size: int, split: str = "all", dev_frac: float = 0.2):
        self.inputs, self.outputs = data_reader(dataset, data_root, split=split, dev_frac=dev_frac)
        self.index = 0
        self.batch_size = batch_size
        self.length = len(self.inputs)
        print(self.length, self.batch_size)

    def __len__(self):
        if self.batch_size == -1:
            return 1
        else:
            return self.length // self.batch_size

    def __getitem__(self, index):
        if self.batch_size == -1:
            if index >= self.__len__():
                raise StopIteration
            else:
                return self.inputs, self.outputs
        else:
            if self.length % self.batch_size == 0:
                if index >= self.__len__():
                    raise StopIteration
                else:
                    tmp_inputs, tmp_outputs = [], []
                    for i in range(index * self.batch_size, min((index + 1) * self.batch_size, self.length)):
                        tmp_inputs.append(self.inputs[i])
                        tmp_outputs.append(self.outputs[i])
                    return tmp_inputs, tmp_outputs
            else:
                if index > self.__len__():
                    raise StopIteration
                else:
                    tmp_inputs, tmp_outputs = [], []
                    for i in range(index * self.batch_size, min((index + 1) * self.batch_size, self.length)):
                        tmp_inputs.append(self.inputs[i])
                        tmp_outputs.append(self.outputs[i])
                    return tmp_inputs, tmp_outputs