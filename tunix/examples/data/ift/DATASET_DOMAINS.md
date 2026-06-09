# Dataset Domains & Sizes

## MetaMathQA (`meta-math/MetaMathQA`)

Source column: `type`  
Total samples: **395,000**  

| Domain | Samples | % of Total | Base Dataset | Augmentation Strategy |
|--------|--------:|:----------:|:------------:|:---------------------:|
| `GSM_AnsAug` | 80,000 | 20.3% | GSM8K | Answer augmentation |
| `GSM_FOBAR` | 40,000 | 10.1% | GSM8K | Fill-in-the-blank reverse |
| `GSM_Rephrased` | 80,000 | 20.3% | GSM8K | Rephrased questions |
| `GSM_SV` | 40,000 | 10.1% | GSM8K | Substituted values |
| `MATH_AnsAug` | 75,000 | 19.0% | MATH | Answer augmentation |
| `MATH_FOBAR` | 15,000 | 3.8% | MATH | Fill-in-the-blank reverse |
| `MATH_Rephrased` | 50,000 | 12.7% | MATH | Rephrased questions |
| `MATH_SV` | 15,000 | 3.8% | MATH | Substituted values |

**Imbalance ratio (max/min): 5.3x** (80k vs 15k)

### Domain size summary
- GSM-based: 240,000 (60.8%)
- MATH-based: 155,000 (39.2%)

After `train_test_split(test_size=10000)`: 385,000 train samples.

---

## Mol-Instructions (`zjunlp/Mol-Instructions`)

Config: `Molecule-oriented Instructions`  
Source column: manually added from split keys  
Total samples: **678,771**  

| Domain | Samples | % of Total | After holdout (−1000) |
|--------|--------:|:----------:|----------------------:|
| `description_guided_molecule_design` | 298,319 | 43.9% | 297,319 |
| `forward_reaction_prediction` | 125,384 | 18.5% | 124,384 |
| `reagent_prediction` | 125,384 | 18.5% | 124,384 |
| `retrosynthesis` | 129,684 | 19.1% | 128,684 |

**Imbalance ratio (max/min): 2.38x** (298k vs 125k)

After `train_test_split(test_size=1000)` per domain: **674,771** train samples.

### Notes
- `description_guided_molecule_design` is the clear majority domain (~44%)
- The other three domains are roughly equal (~18-19% each)
- Holdout size is configurable via `task_config.config.test_holdout_per_domain`
- `train_on_input = True` (following [upstream finetune script](https://github.com/zjunlp/Mol-Instructions/blob/main/demo/finetune.py#L49C9-L49C24))
