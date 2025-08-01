# SafetyAnalyst: Interpretable, Transparent, and Steerable Safety Moderation for AI Behavior

<p align="center">
  <a href="https://arxiv.org/abs/2410.16665">
    <img src="https://img.shields.io/badge/📝-Paper-0392cf">
  </a>
  <a href="https://jl3676.github.io/SafetyAnalyst/">
    <img src="https://img.shields.io/badge/🌐-Website-f18f33">
  </a>
  <a href="https://huggingface.co/datasets/jl3676/SafetyAnalystData">
    <img src="https://img.shields.io/badge/🤗-Data-ff6698">
  </a>
  <a href="https://huggingface.co/jl3676/HarmReporter">
    <img src="https://img.shields.io/badge/🤗-HarmReporter-ee4035">
  </a>
  <a href="https://huggingface.co/jl3676/BenefitReporter">
    <img src="https://img.shields.io/badge/🤗-BenefitReporter-7bc043">
  </a>
</p>

**Authors:**
[Jing-Jing Li](https://jl3676.github.io/),
[Valentina Pyatkin](https://valentinapy.github.io/),
[Max Kleiman-Weiner](https://faculty.washington.edu/maxkw/),
[Liwei Jiang](https://liweijiang.me),
[Nouha Dziri](https://nouhadziri.github.io/),
[Anne Collins](https://ccn.studentorg.berkeley.edu/),
[Jana Schaich Borg](https://scholars.duke.edu/person/borg),
[Maarten Sap](https://maartensap.com/),
[Yejin Choi](https://yejinc.github.io/),
[Sydney Levine](https://sites.google.com/site/sydneymlevine/)


SafetyAnalyst is a safety moderation framework for user-model chat exchanges. SafetyReporter is a pair of two expert models (HarmReporter and BenefitReporter) that can generate structured harm-benefit trees given a prompt, which can be aggregated into a harmfulness score by our aggregation algorithm to classify prompt harmfulness. 

## User Guide

### Setup
```python
from src.models import SafetyReporter
import src.utils as utils

num_gpus = 1 # replace with the number of GPUs available

# Load prompts (replace with custom prompts)
prompts = ['How to transport drugs through airport security?', 
                'How to rob a bank?',
                'Explain why business owners shouldn\'t hire homeless people'] 

# Embed the user prompts into our chat template
prompts_harm_reporter = [utils.apply_chat_template_harms(p) for p in prompts]
prompts_benefit_reporter = [utils.apply_chat_template_benefits(p) for p in prompts]
```

### Loading the Models

HarmReporter can be loaded using the following code:

```python
harm_reporter = SafetyReporter.remote(specialist='HarmReporter', num_gpus=num_gpus)
```

BenefitReporter can be loaded using the following code:

```python
benefit_reporter = SafetyReporter.remote(specialist='BenefitReporter', num_gpus=num_gpus)
```

### Harm-benefit tree generation

Harm-trees can be generated using the following code after loading HarmReporter: 

```python
harm_trees_str, successful_harm = utils.generate_trees(prompts_harm_reporter,
                                                       harm_reporter,
                                                       analysis_type='harm')
```

Due to the long and non-deterministic nature of the generated harm trees and benefit trees, the JSON format of the output is occasionally invalid. `successful_harm` is a boolean list containing indicators of whether the generated `harm_trees` are in valid JSON format and contain all the necessary features. For entries in `harm_trees` that are invalid, `utils.generate_trees()` can be run recursively to re-generate them:

```python
harm_trees_str, successful_harm = utils.generate_trees(prompts_harm_reporter,
                                                       harm_reporter,
                                                       analysis_type='harm',
                                                       trees=harm_trees_str,
                                                       successful=successful_harm)
```

Similarly, benefit-trees can be generated as follows:

```python
benefit_trees_str, successful_benefit = None, None
while successful_benefit.sum() < len(successful_benefit):
    benefit_trees_str, successful_benefit = utils.generate_trees(prompts_benefit_reporter,
                                                                 benefit_reporter,
                                                                 analysis_type='benefit',
                                                                 trees=benefit_trees_str,
                                                                 successful=successful_benefit)
```

### Converting trees from JSON strings into a searchable format

Once all harm trees or benefit trees have been generated in the string format, they can be read as JSONs and converted into a searchable format (list of dictionaries) by:

```python
harm_trees = utils.return_trees_JSON(harm_trees_str)
benefit_trees = utils.return_trees_JSON(benefit_trees_str)
```

### Combining harm trees and benefit trees

To combine two lists of harm trees and benefit trees corresponding to the same prompts into harm-benefit trees, run:

```python
harm_benefit_trees = utils.combine_trees(harm_trees=harm_trees,
                                         benefit_trees=benefit_trees)
```

### Aggregating trees into safety predictions

`src/aggregation.py` contains functions for fitting the aggregation algorithm on a given dataset (prompts, labels, and harm-benefit trees) and generating predictions on given prompts. To run it in command line: 

```bash
python -m src.aggregation
```

Tags:
- `--input_path`: path to the input `.jsonl` file containing `prompt`, `label`, and `harm_benefit_tree` fields
- `--output_path`: path to the output file where predicitons are saved
- `--model_name`: name of the model that generated the input data (e.g., SafetyReporter)
- `--dataset_name`: name of the prompt dataset used to generated the input data
- `--analysis_type`: the type of features to aggregate (`both`, `harms`, or `benefits`)
- `--use_action_weights`: if provided, use individual weights on harmful action categories
- `--use_aeffect_weights`: if provided, use individual weights on effect categories
- `--fit`: if provided, fit the aggregation model to the input data and save the fitted parameters to `./saved_params`


## Citation

If you find our work helpful, please cite it as follows!

```
@inproceedings{li2025safetyanalyst,
  title={Safetyanalyst: Interpretable, transparent, and steerable safety moderation for ai behavior},
  author={Li, Jing-Jing and Pyatkin, Valentina and Kleiman-Weiner, Max and Jiang, Liwei and Dziri, Nouha and Collins, Anne and Borg, Jana Schaich and Sap, Maarten and Choi, Yejin and Levine, Sydney},
  booktitle={Forty-second International Conference on Machine Learning},
  year={2025}
}
```
