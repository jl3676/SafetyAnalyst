# SafetyAnalyst: Interpretable, transparent, and steerable LLM safety moderation

<p align="center">
  <a href="https://arxiv.org/abs/2410.16665">
    <img src="https://img.shields.io/badge/📝-Paper-blue">
  </a>
  <a href="https://huggingface.co/datasets/jl3676/SafetyAnalystData">
    <img src="https://img.shields.io/badge/🤗-Data-orange">
  </a>
  <a href="https://huggingface.co/jl3676/HarmReporter">
    <img src="https://img.shields.io/badge/🤗-HarmReporter-green">
  </a>
  <a href="https://huggingface.co/jl3676/BenefitReporter">
    <img src="https://img.shields.io/badge/🤗-BenefitReporter-green">
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
[Yejin Choi](https://homes.cs.washington.edu/~yejin/),
[Sydney Levine](https://sites.google.com/site/sydneymlevine/)


SafetyAnalyst is a safety moderation framework for user-model chat exchanges. SafetyReporter is a collection of two expert models (HarmReporter and BenefitReporter) that can generate structured harm-benefit trees given a prompt, which can be aggregated into a harmfulness score by our aggregation algorithm to classify prompt harmfulness. 

## User Guide

### Loading the Models

HarmReporter can be loaded using the following code:

```python
from vllm import LLM
from transformers import AutoTokenizer

num_gpus = 4 # Replace with the number of GPUs available

harm_reporter = LLM(model='jl3676/HarmReporter',
                    dtype='auto',
                    trust_remote_code=True,
                    tokenizer_mode="auto",
                    tensor_parallel_size=num_gpus)
tokenizer_harm = AutoTokenizer.from_pretrained('jl3676/HarmReporter')
```

BenefitReporter can be loaded using the following code:

```python
benefit_reporter = LLM(model='jl3676/BenefitReporter',
                    dtype='auto',
                    trust_remote_code=True,
                    tokenizer_mode="auto",
                    tensor_parallel_size=num_gpus)
tokenizer_benefit = AutoTokenizer.from_pretrained('jl3676/BenefitReporter')
```

### Harm-benefit tree generation

Harm-trees can be generated using the following code after loading HarmReporter: 

```python
prompts = ['How to transport drugs through airport security?', 'How to rob a bank?'] # Replace with prompts
formatted_prompts = [tokenizer_harm.apply_chat_template(p, tokenize=False) for p in prompts]
sampling_params = SamplingParams(
            max_tokens=19000,
            temperature=1.0,
            top_p=1.0
        )
outputs = harm_reporter.generate(prompts=formatted_prompts, sampling_params=sampling_params)
harm_trees = [it.outputs[0].text for it in outputs]
```

Similarly, benefit-trees can be generated as follows:

```python
formatted_prompts = [tokenizer_benefit.apply_chat_template(p, tokenize=False) for p in prompts]
outputs = benefit_reporter.generate(prompts=formatted_prompts, 
                                    sampling_params=sampling_params)
benefit_trees = [it.outputs[0].text for it in outputs]
```

### Convert generated data into JSON format

Due to the long and non-deterministic nature of the generated harm trees and benefit trees, the JSON format of the output is occasionally invalid. To check the validity of the JSON format of a given harm or benefit tree (TODO)

```python
import json

harm_trees_json = [json.loads(harm_tree) for harm_tree in harm_trees]
benefit_trees_json = [json.loads(harm_tree) for benefit_tree in harm_trees]
```

### Combining trees

To combine two lists of harm trees and benefit trees corresponding to the same prompts into harm-benefit trees, run:

```python
harm_benefit_trees_json = [harm_tree + benefit_tree for harm_tree, benefit_tree in zip(harm_trees_json, benefit_trees_json)]
```

### Aggregating harm-benefit trees 
TODO

## Citation

If you find our work helpful, please cite it as follows!

```
@misc{li2024safetyanalystinterpretabletransparentsteerable,
      title={SafetyAnalyst: Interpretable, transparent, and steerable LLM safety moderation}, 
      author={Jing-Jing Li and Valentina Pyatkin and Max Kleiman-Weiner and Liwei Jiang and Nouha Dziri and Anne G. E. Collins and Jana Schaich Borg and Maarten Sap and Yejin Choi and Sydney Levine},
      year={2024},
      eprint={2410.16665},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2410.16665}, 
}
```