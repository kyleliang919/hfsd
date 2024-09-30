# HFSD: Huggingface Compatible Speculative Decoding
A simple patch for speculative decoding, supporting llama3.2 vision instruct and llama3.2 text model.

# Motivation:

```python
target_model_name = "meta-llama/Llama-3.2-90B-Vision-Instruct"
target = MllamaForConditionalGeneration.from_pretrained(target_model_name, torch_dtype=torch.bfloat16, device_map="auto")

drafter_model_name = "meta-llama/Llama-3.2-11B-Vision-Instruct"
drafter = MllamaForConditionalGeneration.from_pretrained(drafter_model_name, torch_dtype=torch.bfloat16, device_map="auto")
```

This HF native implementation will OOM even on 4 H100 GPUs.

```python
target.generate(**inputs, max_new_tokens=30, use_cache = True, assistant_model = drafter)
```

but our implementation will not, 

```python
import hf_speculative_decoding
processor = AutoProcessor.from_pretrained(target_model_name)
target.speculative_generate(
                inputs,
                drafter,
                gamma=5,
                max_new_tokens=30,
                tokenizer = processor.tokenizer,
                use_cache = True
            )
```

# Installation
Very simple and straightforward
```bash
pip install git+https://github.com/kyleliang919/Speculative-Decoding.git
pip install -e .
```

# Usage
Check [example.py](./example.py) and [example_vision.py](./example_vision.py) for usage examples.

# Reference
<a id="1">[1]</a> Leviathan, Y., Kalman, M. &amp; Matias, Y.. (2023). Fast Inference from Transformers via Speculative Decoding. <i>Proceedings of the 40th International Conference on Machine Learning</i>, in <i>Proceedings of Machine Learning Research</i> 202:19274-19286 Available from https://proceedings.mlr.press/v202/leviathan23a.html.

<a id="2">[2]</a> Chen, C., Borgeaud, S., Irving, G., Lespiau, J. B., Sifre, L., & Jumper, J. (2023). Accelerating large language model decoding with speculative sampling. arXiv preprint arXiv:2302.01318. 

<a id="3">[3]</a> https://github.com/romsto/Speculative-Decoding/tree/main 
