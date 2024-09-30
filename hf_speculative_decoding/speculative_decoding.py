import torch
from torch.nn import Module
from .logits_processor import LogitsProcessor, GreedyProcessor
from transformers.cache_utils import DynamicCache
from .caching import prune_cache
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from transformers.models.mllama.modeling_mllama import MllamaForConditionalGeneration
from transformers import AutoModelForCausalLM, AutoTokenizer

def max_fn(x: torch.Tensor) -> torch.Tensor:
    """
    Max function.
        x: input tensor.
    Returns:
        tensor norm(max(0, x)).
    """
    x_max = torch.where(x > 0, x, torch.zeros_like(x))
    x_max_sum = torch.sum(x_max, dim=-1, keepdim=True)
    return x_max / x_max_sum

@torch.no_grad()
def speculative_generate(
        self,
        inputs: Optional[torch.Tensor],
        drafter: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        gamma: int = 5,
        eos_token_id: Union[int, List[int]] = None, 
        pad_token_id: Union[int, List[int]] = None,
        max_new_tokens: int = 10,
        logits_processor: LogitsProcessor = None,
        use_cache: bool = False,
        **kwargs,
    ):
    
    eos_token_id = eos_token_id if eos_token_id else tokenizer.eos_token_id
    if pad_token_id:
        pad_token_id = pad_token_id
    else:
        pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id else tokenizer.eos_token_id

    logits_processor = logits_processor if logits_processor else GreedyProcessor(temperature=1)

    if type(self) is MllamaForConditionalGeneration:
        return handle_mllama_speculative_generate(
                    self,
                    inputs,
                    drafter = drafter,
                    logits_processor = logits_processor,
                    gamma = gamma,
                    max_new_tokens = max_new_tokens,
                    eos_tokens_id = eos_token_id,
                    pad_token_id = pad_token_id,
                    tokenizer = tokenizer,
                    use_cache = use_cache
                )
    else:
        return handle_speculative_generate( # or speculative_generate_encoder_decoder for encoder-decoder models
                self,
                inputs,
                drafter = drafter,
                logits_processor=logits_processor,
                gamma=gamma,
                max_new_tokens=max_new_tokens,
                eos_tokens_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id if pad_token_id is None else tokenizer.eos_token_id,
                tokenizer = tokenizer,
                use_cache = use_cache
            )


@torch.no_grad()
def handle_speculative_generate(
    target: Module,
    inputs: Optional[torch.Tensor],
    drafter: Module,
    tokenizer,
    gamma: int = 5,
    logits_processor: LogitsProcessor = GreedyProcessor(temperature=1),
    max_new_tokens: int = 40,
    eos_tokens_id: int | List[int] = 1,
    pad_token_id: int = 0,
    use_cache: bool = False,
    skip_sample_adjustment: bool = False,
) -> Tuple[List[int], float]:
    """
    Generate text sequence using the speculative decoding algorithm.
    Implementation of Speculative Decoding. (https://arxiv.org/pdf/2211.17192.pdf)
    
    Args:
        target (Module): target model.  
        inputs (List[int]): input sequence of batch size 1.
        drafter (Module): drafter model.
        tokenizer: tokenizer.
        gamma (int): number of drafts generated by the drafter at each step.
        logits_processor (LogitsProcessor): logits processor for sampling.
        max_new_tokens (int): maximum length of the generated sequence.
        eos_tokens_id (int or List[int]): end token id (could be multiple).
        pad_token_id (int): pad token id.
        use_cache (bool): whether to use cache.
        skip_sample_adjustment (bool): whether to skip the sample adjustment step when some drafts are discarded.
    
    Returns:
        List[int]: generated sequence.
        float: acceptance rate (number of accepted drafts divided by the number of total drafts).
        
    Note: This generation methods only works for decoder-only models.
    Note bis: The drafter and target models should output the same logits shape.
    Note ter: NgramModels are currently not supported.
    """
    
    drafter_cache, target_cache = None, None

    list_tokens_id = eos_tokens_id if isinstance(eos_tokens_id, list) else [eos_tokens_id]
    stop_tokens = torch.tensor(list_tokens_id, dtype=torch.long, device=target.device).unsqueeze(1)
    
    drafts_accepted, drafts_speculated = .0, .0
    
    vocabulary_size = target.config.vocab_size    
        
    # prepare input tensor
    prompt_len = len(inputs.input_ids[0])
    max_seq_length = target.config.max_position_embeddings if hasattr(target.config, 'max_position_embeddings') else (target.config.max_context_length if hasattr(target.config, 'max_context_length') else 1024)
    total_len = min(max_seq_length, prompt_len + max_new_tokens)
    input_ids = torch.full((1, total_len), pad_token_id, dtype=torch.long, device=target.device)
    input_ids[0, :prompt_len] = torch.tensor(inputs.input_ids[0], dtype=torch.long, device=target.device)
    
    current_position = prompt_len
    
    # run the target model before the speculative algorithm. Allows to prefill the kvcache and get a first token.
    Mp = target(
        input_ids=input_ids[..., :current_position],
        past_key_values=target_cache,
        use_cache=use_cache,
    )
    target_cache = Mp.past_key_values
    p_p = logits_processor(Mp.logits[..., -1, :])
    t = logits_processor.sample(p_p)
    input_ids[0, current_position] = t
    current_position += 1
    
    if torch.isin(t, stop_tokens):
        return input_ids[0, prompt_len:current_position].tolist(), 0
    
    while current_position < total_len:
        corrected_gamma = min(gamma, total_len - current_position - 1)
        q = torch.zeros((1, corrected_gamma, vocabulary_size), device=target.device)
        
        input_ids = input_ids.to(drafter.device)
        
        # generate gamma drafts
        for k in range(corrected_gamma):
            Mq = drafter(
                input_ids=input_ids[..., :current_position + k],
                past_key_values=drafter_cache,
                use_cache=use_cache,
            )
            drafter_cache = Mq.past_key_values
            
            draft_logits = Mq.logits[..., -1, :]
            draft_probs = logits_processor(draft_logits)
            q[0, k] = draft_probs.to(target.device)
            xi = logits_processor.sample(draft_probs)
            input_ids[0, current_position + k] = xi
        drafts_speculated += corrected_gamma
        input_ids = input_ids.to(target.device)
        
        # run target model on drafts and get logits of the previous tokens plus one more token
        Mp = target(
            input_ids=input_ids[..., :current_position + corrected_gamma],
            past_key_values=target_cache,
            use_cache=use_cache,
        )
        target_cache = Mp.past_key_values
        target_logits = Mp.logits[..., current_position - 1:current_position + corrected_gamma - 1, :] # [1, corrected_gamma, vocab_size]
        p = logits_processor(target_logits) # [1, gamma, vocab_size]
        
        # compute the last accepted draft position (rejection sampling)
        r = torch.rand(corrected_gamma, device=target.device)
        fractions = p / q
        n = corrected_gamma
        for i in range(corrected_gamma):
            if r[i] > fractions[0, i, input_ids[0, current_position + i]]:
                n = i
                break
        
        drafts_accepted += n
        
        # check if the end token is in the drafts
        stop_locations = torch.nonzero(torch.eq(input_ids[..., current_position:current_position + n], stop_tokens))
        if stop_locations.shape[0] > 0:
            stop_location = stop_locations[0, 1].item()
            return input_ids[0, prompt_len:current_position + stop_location + 1].tolist(), drafts_accepted / drafts_speculated

        # adjust the distribution from Mp
        if n == corrected_gamma:
            p_p = Mp.logits[..., current_position + corrected_gamma - 1, :]
            p_p = logits_processor(p_p)
        else:
            # prune the cache
            if use_cache:
                drafter_cache = prune_cache(drafter_cache, corrected_gamma - n)
                target_cache = prune_cache(target_cache, corrected_gamma - n + 1)
            
            if not skip_sample_adjustment:
                p_p = max_fn(p[..., n, :] - q[0, n, :])
            else:
                p_p = p[..., n, :]
        x = logits_processor.sample(p_p)
            
        input_ids[0, current_position + n:current_position + corrected_gamma] = pad_token_id
        input_ids[0, current_position + n] = x
        
        
        current_position += n + 1
        
        if torch.isin(x, stop_tokens):
            return input_ids[0, prompt_len:current_position].tolist(), drafts_accepted / drafts_speculated
    
    return input_ids[0, prompt_len:].tolist(), drafts_accepted / drafts_speculated


@torch.no_grad()
def handle_mllama_speculative_generate(
    target: Module,
    inputs: List[int],
    drafter: Module,
    tokenizer,
    gamma: int = 5,
    logits_processor: LogitsProcessor = GreedyProcessor(temperature=1),
    max_new_tokens: int = 40,
    eos_tokens_id: int | List[int] = 1,
    pad_token_id: int = 0,
    use_cache: bool = False,
    skip_sample_adjustment: bool = False,
) -> Tuple[List[int], float]:
    """
    Generate text sequence using the speculative decoding algorithm.
    Implementation of Speculative Decoding. (https://arxiv.org/pdf/2211.17192.pdf)
    
    Args:
        target (Module): target model.
        inputs (List[int]): input sequence of batch size 1.
        drafter (Module): drafter model.
        tokenizer: tokenizer.
        gamma (int): number of drafts generated by the drafter at each step.
        logits_processor (LogitsProcessor): logits processor for sampling.
        max_new_tokens (int): maximum length of the generated sequence.
        eos_tokens_id (int or List[int]): end token id (could be multiple).
        pad_token_id (int): pad token id.
        use_cache (bool): whether to use cache.
        skip_sample_adjustment (bool): whether to skip the sample adjustment step when some drafts are discarded.
    
    Returns:
        List[int]: generated sequence.
        float: acceptance rate (number of accepted drafts divided by the number of total drafts).
        
    Note: This generation methods only works for decoder-only models.
    Note bis: The drafter and target models should output the same logits shape.
    Note ter: NgramModels are currently not supported.
    """
    
    drafter_cache, target_cache = None, None

    list_tokens_id = eos_tokens_id if isinstance(eos_tokens_id, list) else [eos_tokens_id]
    stop_tokens = torch.tensor(list_tokens_id, dtype=torch.long, device=target.device).unsqueeze(1)
    
    drafts_accepted, drafts_speculated = .0, .0
    
    vocabulary_size = target.config.text_config.vocab_size

    # prepare input tensor
    prompt_len = len(inputs.input_ids[0])
    max_seq_length = target.config.text_config.max_position_embeddings if hasattr(target.config.text_config, 'max_position_embeddings') else (target.config.text_config.max_context_length if hasattr(target.config.text_config, 'max_context_length') else 1024)
    total_len = min(max_seq_length, prompt_len + max_new_tokens)
    input_ids = torch.full((1, total_len), pad_token_id, dtype=torch.long, device=target.device)
    input_ids[0, :prompt_len] = torch.tensor(inputs.input_ids[0], dtype=torch.long, device=target.device)
    
    current_position = prompt_len
    cross_attention_mask = inputs.cross_attention_mask

    # run the target model before the speculative algorithm. Allows to prefill the kvcache and get a first token.
    if current_position - prompt_len > 0:
        cross_attention_mask = torch.cat([inputs.cross_attention_mask, torch.ones_like(cross_attention_mask[:,-1:,:,:]).expand(-1, current_position - prompt_len,-1, -1)],dim = 1)
    else:
        cross_attention_mask = inputs.cross_attention_mask
    Mp = target(
        input_ids=input_ids[..., :current_position],
        pixel_values = inputs.pixel_values,
        aspect_ratio_mask = inputs.aspect_ratio_mask,
        aspect_ratio_ids = inputs.aspect_ratio_ids,
        attention_mask = None,
        cross_attention_mask = cross_attention_mask,
        past_key_values=target_cache,
        use_cache=use_cache,
    )
    target_cache = Mp.past_key_values
    p_p = logits_processor(Mp.logits[..., -1, :])
    t = logits_processor.sample(p_p)
    input_ids[:, current_position] = t
    current_position += 1
    
    if torch.isin(t, stop_tokens):
        return input_ids[0, prompt_len:current_position].tolist(), 0
    
    while current_position < total_len:
        corrected_gamma = min(gamma, total_len - current_position - 1)
        q = torch.zeros((1, corrected_gamma, vocabulary_size), device=target.device)
        
        input_ids = input_ids.to(drafter.device)
        
        # generate gamma drafts
        for k in range(corrected_gamma):
            cross_attention_mask = torch.cat([inputs.cross_attention_mask, torch.ones_like(cross_attention_mask[:,-1:,:,:]).expand(-1, current_position - prompt_len + k,-1, -1)], dim = 1)
            Mq = drafter(
                input_ids=input_ids[..., :current_position + k],
                pixel_values = inputs.pixel_values,
                aspect_ratio_mask = inputs.aspect_ratio_mask,
                aspect_ratio_ids = inputs.aspect_ratio_ids,
                attention_mask = None,
                cross_attention_mask = cross_attention_mask,
                past_key_values=drafter_cache,
                use_cache=use_cache,
            )
            drafter_cache = Mq.past_key_values
            
            draft_logits = Mq.logits[..., -1, :]
            draft_probs = logits_processor(draft_logits)
            q[:, k] = draft_probs.to(target.device)
            xi = logits_processor.sample(draft_probs)
            input_ids[:, current_position + k] = xi
        drafts_speculated += corrected_gamma
        input_ids = input_ids.to(target.device)
        
        # run target model on drafts and get logits of the previous tokens plus one more token
        cross_attention_mask = torch.cat([inputs.cross_attention_mask, torch.ones_like(cross_attention_mask[:,-1:,:,:]).expand(-1, current_position - prompt_len + corrected_gamma,-1, -1)], dim = 1)
        Mp = target(
            input_ids=input_ids[..., :current_position + corrected_gamma],
            pixel_values = inputs.pixel_values,
            aspect_ratio_mask = inputs.aspect_ratio_mask,
            aspect_ratio_ids = inputs.aspect_ratio_ids,
            attention_mask = None,
            cross_attention_mask = cross_attention_mask,
            past_key_values=target_cache,
            use_cache=use_cache,
        )
        target_cache = Mp.past_key_values
        target_logits = Mp.logits[..., current_position - 1:current_position + corrected_gamma - 1, :] # [1, corrected_gamma, vocab_size]
        p = logits_processor(target_logits) # [1, gamma, vocab_size]
        
        # compute the last accepted draft position (rejection sampling)
        r = torch.rand(corrected_gamma, device=target.device)
        fractions = p / q
        n = corrected_gamma
        for i in range(corrected_gamma):
            if r[i] > fractions[0, i, input_ids[0, current_position + i]]:
                n = i
                break
        
        drafts_accepted += n
        
        # check if the end token is in the drafts
        stop_locations = torch.nonzero(torch.eq(input_ids[..., current_position:current_position + n], stop_tokens))
        if stop_locations.shape[0] > 0:
            stop_location = stop_locations[0, 1].item()
            return input_ids[0, prompt_len:current_position + stop_location + 1].tolist(), drafts_accepted / drafts_speculated

        # adjust the distribution from Mp
        if n == corrected_gamma:
            p_p = Mp.logits[..., current_position + corrected_gamma - 1, :]
            p_p = logits_processor(p_p)
        else:
            # prune the cache
            if use_cache:
                drafter_cache = prune_cache(drafter_cache, corrected_gamma - n)
                target_cache = prune_cache(target_cache, corrected_gamma - n + 1)
            
            if not skip_sample_adjustment:
                p_p = max_fn(p[..., n, :] - q[0, n, :])
            else:
                p_p = p[..., n, :]
        x = logits_processor.sample(p_p)
            
        input_ids[:, current_position + n:current_position + corrected_gamma] = pad_token_id
        input_ids[:, current_position + n] = x
        
            
        current_position += n + 1
        
        if torch.isin(x, stop_tokens):
            return input_ids[0, prompt_len:current_position].tolist(), drafts_accepted / drafts_speculated
    
    return input_ids[0, prompt_len:].tolist(), drafts_accepted + 1 / drafts_speculated + 1
