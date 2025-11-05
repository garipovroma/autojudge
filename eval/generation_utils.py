import torch
import itertools
import typing as tp
import transformers
import os

import sys

sys.path.append('.')

from lm_eval_utils import stop_sequences_criteria, GSM8K_STOP_SEQUENCES, MATH_STOP_SEQUENCES

"""
draft_token_draft_hidden (DD), draft_token_target_hidden (DT), target_token_draft_hidden (TD), target_token_target_hidden (TT)
"""

def run_spec_dec_top_k(batch_input_ids, target_model, draft_model, tokenizer, K, args):
    if args.bench_name == 'GSM8K':
        stopping_criteria = stop_sequences_criteria(tokenizer, GSM8K_STOP_SEQUENCES, batch_input_ids.shape[1], 1)
    elif args.bench_name == 'MMLU_PRO':
        stopping_criteria = transformers.StopStringCriteria(tokenizer=tokenizer, stop_strings=["Question:"])
        stopping_criteria = transformers.StoppingCriteriaList([stopping_criteria])
    elif args.bench_name == 'MATH500':
        stopping_criteria = stop_sequences_criteria(tokenizer, MATH_STOP_SEQUENCES, batch_input_ids.shape[1], 1)
    else:
        stopping_criteria = None

    current_gen = batch_input_ids
    accepts = []
    mismatches = []
    draft_calls, target_calls = 0, 0
    eos_position = None
    accepts_levi = []
    mismatch_found = False
    window_idx = 0
    eos_count = 0
    device = torch.device('cuda:0')
    for _ in itertools.count():
        draft_hiddens = []
        # drafting tokens
        draft_cache = transformers.DynamicCache()
        draft_batch = dict(
                input_ids=current_gen,
                attention_mask=torch.ones_like(current_gen),
        )
        for pos_in_window in range(args.window_size + 1):
            
            with torch.no_grad():
                draft_last_hidden_states = draft_model.model.forward(
                                    **draft_batch,
                                    output_hidden_states=False,
                                    use_cache=True,
                                    past_key_values=draft_cache,
                                    return_dict=True,
                                ).last_hidden_state
                argmax_token = draft_model.lm_head(draft_last_hidden_states[:, -1, :]).argmax(dim=-1)[None].to(device)
                del draft_last_hidden_states

            if pos_in_window != args.window_size:
                if argmax_token.item() == tokenizer.eos_token_id:
                    eos_position = current_gen.shape[1]
                current_gen = torch.cat((current_gen.to(device), argmax_token), dim=-1)
                draft_batch = dict(
                        input_ids=argmax_token.to(draft_model.device),
                        attention_mask=torch.ones_like(current_gen),
                )

            draft_calls += 1
        del draft_cache


        # verification
        target_input_ids = current_gen
        target_batch = dict(
            input_ids=target_input_ids,
            attention_mask=torch.ones_like(target_input_ids)
        )

        with torch.no_grad():
            target_logits = target_model.forward(
                **target_batch,
                output_hidden_states=False,
                use_cache=False,
                return_dict=True,
            ).logits
            target_argmax_tokens_ = target_logits.argmax(dim=-1)
            target_topk_tokens = target_logits.cpu().topk(K, -1).indices[:, -args.window_size:, :]
            del target_logits


        target_calls += 1

        target_argmax_tokens = torch.cat(
            (torch.tensor([tokenizer.bos_token_id], device=target_argmax_tokens_.device)[None], target_argmax_tokens_),
            dim=-1)[:, :-1]

        target_window_tokens = target_argmax_tokens[:, -args.window_size:]
        draft_window_tokens = current_gen[:, -args.window_size:]

        match_mask = (target_window_tokens == draft_window_tokens).flatten().to(torch.int32).cpu()

        draft_window_tokens = draft_window_tokens.to(device)
        target_topk_tokens = target_topk_tokens.to(device)

        topk_mask = (draft_window_tokens.cpu().unsqueeze(-1) == target_topk_tokens.cpu()).any(-1).cpu()
        match_and_imp_mask = (match_mask | topk_mask)

        n_accepted_levi = match_mask.cumprod(dim=-1).sum().item()
        n_accepted = match_and_imp_mask.cumprod(dim=-1).sum().item()

        if n_accepted != args.window_size:
            important_token_pos = current_gen.shape[1] - args.window_size + n_accepted
            drafted_token_is_not_in_topk = not topk_mask[0, n_accepted]

            if drafted_token_is_not_in_topk:
                important_target_token = target_window_tokens[:, n_accepted][None]
                current_gen[:, -args.window_size + n_accepted] = important_target_token
                n_accepted = n_accepted + 1

        first_not_included_token = current_gen.shape[1] - (args.window_size - n_accepted)
        n_tokens_with_full_last_window = current_gen.shape[1]
        current_gen = current_gen[:, :first_not_included_token]

        mismatch_window_positions = (1 - match_mask).nonzero().flatten().tolist()
        if mismatch_window_positions:
            first_mismatch_position = mismatch_window_positions[0] + n_tokens_with_full_last_window - args.window_size
            mismatches.append(first_mismatch_position)

        accepts.append(n_accepted)
        accepts_levi.append(n_accepted_levi)
        if tokenizer.eos_token_id in current_gen[:, batch_input_ids.shape[1]:]:
            eos_count += 1
        n_generated_tokens = current_gen.shape[1] - batch_input_ids.shape[1]

        if eos_count > 0:
            eos_position = batch_input_ids.shape[1] + torch.nonzero(current_gen[:, batch_input_ids.shape[1]:] == tokenizer.eos_token_id)[0, 1].item()
            current_gen = current_gen[:, :eos_position + 1]
            break

        if n_generated_tokens >= args.max_new_tokens:
            current_gen = current_gen[:, :batch_input_ids.shape[1] + args.max_new_tokens]
            break

        window_idx += 1

        if stopping_criteria is not None and stopping_criteria(current_gen, scores=None):
            break

    return dict(
        accepts=accepts,
        accepts_levi=accepts_levi,
        mismatches=mismatches,
        draft_calls=draft_calls,
        target_calls=target_calls,
        current_gen=current_gen,
    )



def run_spec_dec(batch_input_ids, target_model, draft_model, tokenizer, scaler, head, args):
    if args.bench_name == 'GSM8K':
        stopping_criteria = stop_sequences_criteria(tokenizer, GSM8K_STOP_SEQUENCES, batch_input_ids.shape[1], 1)
    elif args.bench_name == 'MMLU_PRO':
        stopping_criteria = transformers.StopStringCriteria(tokenizer=tokenizer, stop_strings=["Question:"])
        stopping_criteria = transformers.StoppingCriteriaList([stopping_criteria])
    elif args.bench_name == 'MATH500':
        stopping_criteria = stop_sequences_criteria(tokenizer, MATH_STOP_SEQUENCES, batch_input_ids.shape[1], 1)
    else:
        stopping_criteria = None

    setup = args.setup

    current_gen = batch_input_ids
    accepts = []
    mismatches = []
    draft_calls, target_calls = 0, 0
    eos_position = None
    accepts_levi = []
    mismatch_found = False
    window_idx = 0
    eos_count = 0
    # device = target_model.device
    device = torch.device('cuda:0')
    for _ in itertools.count():
        draft_hiddens = []
        # drafting tokens
        draft_cache = transformers.DynamicCache()
        draft_batch = dict(
                input_ids=current_gen,
                attention_mask=torch.ones_like(current_gen),
        )
        for pos_in_window in range(args.window_size + 1):
            with torch.no_grad():
                draft_last_hidden_states = draft_model.model.forward(
                                    **draft_batch,
                                    output_hidden_states=False,
                                    use_cache=True,
                                    past_key_values=draft_cache,
                                    return_dict=True,
                                ).last_hidden_state
                argmax_token = draft_model.lm_head(draft_last_hidden_states[:, -1, :]).argmax(dim=-1)[None].to(device)
                

            if pos_in_window != args.window_size:
                if argmax_token.item() == tokenizer.eos_token_id:
                    eos_position = current_gen.shape[1]
                # current_gen = current_gen.to(device)
                current_gen = torch.cat((current_gen.to(device), argmax_token), dim=-1)

            draft_hiddens.append(draft_last_hidden_states[0, -1, :][None, ...].cpu())
            del draft_last_hidden_states
            draft_calls += 1
            draft_batch = dict(
                input_ids=argmax_token.to(draft_model.device),
                attention_mask=torch.ones_like(current_gen),
            )
        del draft_cache

        draft_hiddens_ = torch.cat(draft_hiddens[:-1], dim=0)
        draft_hiddens = torch.cat(draft_hiddens[1:], dim=0)

        # verification
        target_input_ids = current_gen
        target_batch = dict(
            input_ids=target_input_ids,
            attention_mask=torch.ones_like(target_input_ids)
        )

        with torch.no_grad():
            target_outputs = target_model.model.forward(
                **target_batch,
                output_hidden_states=False,
                return_dict=True,
            )
            logits = target_model.lm_head(target_outputs.last_hidden_state).to(device)

        target_calls += 1

        target_argmax_tokens_ = logits.argmax(dim=-1)
        target_argmax_tokens = torch.cat(
            (torch.tensor([tokenizer.bos_token_id], device=target_argmax_tokens_.device)[None], target_argmax_tokens_),
            dim=-1)[:, :-1]
        target_hiddens = target_outputs.last_hidden_state[0, -args.window_size:].cpu()
        del target_outputs, logits

        if setup == "DD-DT":
            hiddens = torch.cat((draft_hiddens.cpu(), target_hiddens), dim=-1).cpu().to(torch.float32).numpy()
        elif setup == "DT":
            hiddens = target_hiddens.cpu().to(torch.float32).numpy()

        hiddens = scaler.transform(hiddens)
        p_head = torch.tensor(head.predict_proba(hiddens)[:, 1])

        target_window_tokens = target_argmax_tokens[:, -args.window_size:]
        draft_window_tokens = current_gen[:, -args.window_size:]

        target_window_tokens = target_window_tokens.to(device)
        draft_window_tokens = draft_window_tokens.to(device)

        match_mask = (target_window_tokens == draft_window_tokens).flatten().to(torch.int32)
        head_mask = (p_head < args.head_threshold)

        head_mask = head_mask.to(device)
        match_and_imp_mask = (match_mask | head_mask)

        n_accepted_levi = match_mask.cumprod(dim=-1).sum().item()
        n_accepted = match_and_imp_mask.cumprod(dim=-1).sum().item()


        if n_accepted != args.window_size:
            important_token_pos = current_gen.shape[1] - args.window_size + n_accepted
            is_first_rejected_important = (p_head[n_accepted] >= args.head_threshold)
            if is_first_rejected_important:
                important_target_token = target_window_tokens[:, n_accepted][None]
                current_gen[:, -args.window_size + n_accepted] = important_target_token
                n_accepted = n_accepted + 1

        first_not_included_token = current_gen.shape[1] - (args.window_size - n_accepted)
        n_tokens_with_full_last_window = current_gen.shape[1]
        current_gen = current_gen[:, :first_not_included_token]

        mismatch_window_positions = (1 - match_mask).nonzero().flatten().tolist()
        if mismatch_window_positions:
            first_mismatch_position = mismatch_window_positions[0] + n_tokens_with_full_last_window - args.window_size
            mismatches.append(first_mismatch_position)

        accepts.append(n_accepted)
        accepts_levi.append(n_accepted_levi)
        if tokenizer.eos_token_id in current_gen[:, batch_input_ids.shape[1]:]:
            eos_count += 1
        n_generated_tokens = current_gen.shape[1] - batch_input_ids.shape[1]

        if eos_count > 0:
            eos_position = batch_input_ids.shape[1] + torch.nonzero(current_gen[:, batch_input_ids.shape[1]:] == tokenizer.eos_token_id)[0, 1].item()
            current_gen = current_gen[:, :eos_position + 1]
            break

        if n_generated_tokens >= args.max_new_tokens:
            current_gen = current_gen[:, :batch_input_ids.shape[1] + args.max_new_tokens]
            break

        window_idx += 1

        if stopping_criteria is not None and stopping_criteria(current_gen, scores=None):
            break

    return dict(
        accepts=accepts,
        accepts_levi=accepts_levi,
        mismatches=mismatches,
        draft_calls=draft_calls,
        target_calls=target_calls,
        current_gen=current_gen,
    )