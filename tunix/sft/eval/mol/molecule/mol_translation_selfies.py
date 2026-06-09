'''
Code from https://github.com/blender-nlp/MolT5

```bibtex
@article{edwards2022translation,
  title={Translation between Molecules and Natural Language},
  author={Edwards, Carl and Lai, Tuan and Ros, Kevin and Honke, Garrett and Ji, Heng},
  journal={arXiv preprint arXiv:2204.11817},
  year={2022}
}
```
'''


import pickle
import argparse
import json

import os.path as osp

import numpy as np

from nltk.translate.bleu_score import corpus_bleu

from Levenshtein import distance as lev

from rdkit import Chem

from rdkit import RDLogger

RDLogger.DisableLog('rdApp.*')

def evaluate(input_fp, verbose=False):
    outputs = []

    with open(input_fp, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)

            gt_self = row['ground truth']
            ot_self = row['output']
            gt_smi = row['ground smiles']
            ot_smi = row['output_smiles']

            outputs.append((gt_self, ot_self, gt_smi, ot_smi))

    if not outputs:
        return {"bleu": 0.0, "em": 0.0, "edit-dist": 0.0, "validity": 0.0}

    bleu_self_scores = []
    bleu_smi_scores = []

    references_self = []
    hypotheses_self = []
    
    references_smi = []
    hypotheses_smi = []
    bad_mols = 0


    for i, (gt_self, ot_self, gt_smi, ot_smi) in enumerate(outputs):

        try:
            gt_self_tokens = [c for c in gt_self]
            out_self_tokens = [c for c in ot_self]

            references_self.append([gt_self_tokens])
            hypotheses_self.append(out_self_tokens)
            
            gt_smi_tokens = [c for c in gt_smi]
            ot_smi_tokens = [c for c in ot_smi]

            references_smi.append([gt_smi_tokens])
            hypotheses_smi.append(ot_smi_tokens)
        except Exception as e:
            # print(e)
            bad_mols += 1
            print("failed at ", i)
            # raise e
        

    # BLEU score
    bleu_score_self = corpus_bleu(references_self, hypotheses_self)
    if verbose: print('SELFIES BLEU score:', bleu_score_self)

    references_self = []
    hypotheses_self = []
    
    references_smi = []
    hypotheses_smi = []

    levs_self = []
    levs_smi = []

    num_exact = 0

    for i, (gt_self, ot_self, gt_smi, ot_smi) in enumerate(outputs):

        hypotheses_self.append(ot_self)
        references_self.append(gt_self)

        hypotheses_smi.append(ot_smi)
        references_smi.append(gt_smi)
        
        try:
            m_out = Chem.MolFromSmiles(ot_smi)
            m_gt = Chem.MolFromSmiles(gt_smi)

            if Chem.MolToInchi(m_out) == Chem.MolToInchi(m_gt): num_exact += 1
            #if gt == out: num_exact += 1 #old version that didn't standardize strings
        
            levs_self.append(lev(ot_self, gt_self))
            levs_smi.append(lev(ot_smi, gt_smi))

        except:
            bad_mols += 1

    # Exact matching score
    exact_match_score = num_exact/len(outputs)
    if verbose:
        print('Exact Match:')
        print(exact_match_score)

    # Levenshtein score
    actual_len = len(outputs)
    lev_local = np.mean(levs_smi) if levs_smi else 0.0
    lev_global = np.sum(levs_smi)/actual_len
    if verbose:
        print('SMILES Levenshtein:')
        print(lev_local, lev_global)
        
    validity_score = 1 - bad_mols/len(outputs)
    if verbose:
        print('validity:', validity_score)
        print("bad_mols", bad_mols)

    return {"bleu": bleu_score_self, 
            "em": exact_match_score, "edit-dist": lev_global, "validity": validity_score }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', type=str, default='caption2smiles_example.txt', help='path where test generations are saved')
    args = parser.parse_args()
    evaluate(args.input_file, verbose=True)