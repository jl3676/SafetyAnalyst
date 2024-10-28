from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
import ray

@ray.remote
class SafetyReporter():
    def __init__(self, 
                 specialist: str = 'HarmReporter',
                 num_gpus: int = 1):
        self.model = LLM(model=f'jl3676/{specialist}',
                         dtype='auto',
                         trust_remote_code=True,
                         tokenizer_mode="auto",
                         tensor_parallel_size=num_gpus)
        self.tokenizer = AutoTokenizer.from_pretrained(f'jl3676/{specialist}')

    def batched_generate(self, 
                         prompts: list[str]):
        formatted_prompts = [self.tokenizer.apply_chat_template(p, tokenize=False) for p in prompts]
        print(f"""Example prompt: {formatted_prompts[0]}""")
        sampling_params = SamplingParams(
            max_tokens=19000,
            temperature=1.0,
            top_p=1.0
        )
        outputs = self.model.generate(prompts=formatted_prompts, 
                                      sampling_params=sampling_params, 
                                      use_tqdm=True)
        outputs = [output.outputs[0].text for output in outputs]
        return outputs
    