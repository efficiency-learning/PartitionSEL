import pandas as pd
import selfies as sf
from rdkit import Chem
from rdkit import RDLogger     
from sklearn.metrics import mean_absolute_error
from tabulate import tabulate

RDLogger.DisableLog('rdApp.*')

def sf_encode(selfies):
    try:
        smiles = sf.decoder(selfies)
        return smiles
    except Exception:
        return None

def convert_to_canonical_smiles(smiles):
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is not None:
        canonical_smiles = Chem.MolToSmiles(molecule, isomericSmiles=False, canonical=True)
        return canonical_smiles
    else:
        return None
    
def metrics(input_path, output_path, task):
    data = pd.read_json(input_path, lines=True)

    # data = pd.read_table(input_path, sep='\t', on_bad_lines='skip')
    all_data = data.shape[0]
    data["description"] = "desc"
    curr = data.shape[0]
    
    if task == 'property_pred':
        data['output'] = data['output'].astype(str).str.extract(r'(-?\d+\.?\d*)\s*</s>')

        data.dropna(axis=0, how='any', inplace=True)
        data['output'] = pd.to_numeric(data['output'])
        data['ground_truth'] = pd.to_numeric(data['ground_truth'])

        data.dropna(axis=0, how='any', inplace=True)
        
        mae = mean_absolute_error(data['ground_truth'], data['output'])
        print(mae)

    elif task == 'mol_gen':
        data.dropna(axis=0, how='any', inplace=True)
        data['output'] = data['output'].apply(lambda x: x.rsplit(']', 1)[0] + ']' if isinstance(x, str) else x)
        data['output_smiles'] = data['output'].map(sf_encode)
        data.dropna(axis=0, how='any', inplace=True)
        data['output_smiles'] = data['output_smiles'].map(convert_to_canonical_smiles)
        data.dropna(axis=0, how='any', inplace=True)
        data['ground truth'] = data['ground_truth']
        data['ground smiles'] = data['ground_truth'].map(sf_encode)
        data['ground smiles'] = data['ground smiles'].map(convert_to_canonical_smiles)
        data.dropna(axis=0, how='any', inplace=True)
        
    else:
        data.dropna(axis=0, how='any', inplace=True)

        data['SELFIES'] = data['description']
        data['SMILES_org'] = data['description'].map(sf_encode)
        data['SMILES'] = data['SMILES_org'].map(convert_to_canonical_smiles)

        data['output'] = data['output'].apply(lambda x: x.rsplit('.', 1)[0] + '.' if isinstance(x, str) else x)
        # data['output'] = data['output'].str.rsplit('.', 1).str[0] + '.'
        data['ground truth'] = data['ground_truth']
        data.dropna(axis=0, how='any', inplace=True)
        data = data[['SMILES', 'SELFIES', 'ground truth', 'output']]

    data.to_json(output_path, orient="records", lines=True, force_ascii=False)

    # print("orig/dropped", curr, curr-data.shape[0])

'''
description_guided_molecule_design: mol
forward_reaction_prediction: mol
# molecular_description_generation: text
# property_prediction: num 
reagent_prediction: mol
retrosynthesis: mol
'''

import csv

def run_eval(outdir):
    tasks = [
        "description_guided_molecule_design",
        "forward_reaction_prediction",
        "reagent_prediction",
        "retrosynthesis",
    ]

    rows = [] 

    for task in tasks:
        input_path = f"{outdir}/{task}.jsonl"
        output_path = f"{outdir}/pre_{task}.txt"
        metrics(input_path, output_path, "mol_gen")

        from tunix.sft.eval.mol.molecule import mol_translation_selfies as mol
        res = mol.evaluate(output_path, verbose=False)
        row = {"task": task, **res}
        print(row)
        rows.append(row)

    csv_path = f"{outdir}/all_results.csv"
    fieldnames = rows[0].keys()
    avg = {k : sum([rows[i][k] for i in range(len(tasks))])/len(tasks) for k in fieldnames if k != "task"}
    rows.append({"task": "Average", **avg})

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved to {csv_path}")
    print(tabulate(rows, headers="keys", floatfmt=".4f"))

if __name__ == "__main__":
    root = "/home/aiscuser/prayas/temp/tunix/examples/sft/mtnt/results"
    ckpt = "joint_alp_mol_qwen3b_jan16_joint-val2-bs16-2-grad1/2048"
    out_dir = f"{root}/{ckpt}/MOL"
    run_eval(out_dir)
