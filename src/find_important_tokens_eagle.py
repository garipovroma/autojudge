import os
import sys
from typing import Optional

parent_dir = os.path.dirname(os.path.dirname(__file__))
sys.path.append(parent_dir)

import argparse
import time
import json
import yaml
import logging
import warnings
from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import transformers
from eagle.model.ea_model import (EaModel, prepare_logits_processor, initialize_past_key_values, reset_tree_mode,
                                  initialize_tree, tree_decoding, evaluate_posterior, update_inference_inputs)

from lm_eval_utils import GSM8KParser, GSM8KEvaluator
from prompts import GSM8KPrompts, llama_assistant_turn_end

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class EaModelForAutoJudge(EaModel):

    # TODO: this hard-codes GSM8K
    parser = GSM8KParser()
    evaluator = GSM8KEvaluator()
    target_prob_threshold = 0.01  # a draft token cannot be accepted if its probability is less than this
    length_max_rate = 1.5
    #/TODO

    @classmethod
    def from_pretrained_with_tied_ref_model(cls, *args, **kwargs):
        print("ACHTUNG: Model will be loaded twice! This can be optimized.", file=sys.stderr)
        # ACHTUNG: this may not work for larger models. If so, init second model instance via deepcopy with memo?
        main_model = cls.from_pretrained(*args, **kwargs)
        ref_model = super().from_pretrained(*args, **kwargs)
        # tie weights, keep separate cache / buffers
        named_original_params: dict[str, torch.Tensor] = dict(main_model.named_parameters())
        for name, param in ref_model.named_parameters():
            param.data = named_original_params[name].data
        named_original_buffers: dict[str, torch.Tensor] = dict(main_model.named_buffers())
        for name, buffer in ref_model.named_buffers():
            buffer.data = buffer.data.to(named_original_buffers[name].device)
        del named_original_params, named_original_buffers
        torch.cuda.empty_cache()  # remove duplicate model weights.
        main_model.ref_model = ref_model
        return main_model

    @torch.no_grad()
    def check_if_mismatch_is_important(
        self, prompt_ids: torch.Tensor, accepted_prefix: torch.Tensor, target_response: torch.Tensor,
        mismatch_draft_token: int, mismatch_target_token: int, target_logp_for_mismatching_token: torch.Tensor,
        generation_kwargs: dict,
    ) -> bool:
        if target_logp_for_mismatching_token[..., mismatch_draft_token].exp().item() < self.target_prob_threshold:
            logger.info(" ".join((
                "LogP too low: ...",
                (self.tokenizer.batch_decode(accepted_prefix)[0][-80:] + ' | ' +
                 self.tokenizer.decode(mismatch_draft_token)).replace('\n', '\\n').replace('\t', '\\t'),
                 f"[logp={target_logp_for_mismatching_token[..., mismatch_draft_token].exp().item():.5f}]",
            )))
            return True  # very unlikely draft token, forbid replacing it

        target_tokens_generated = target_response.shape[1] - prompt_ids.shape[1]
        prefix_tokens_generated = (accepted_prefix.shape[1] + 1) - prompt_ids.shape[1]
        alternative_response = self.ref_model.eagenerate(
            input_ids=torch.cat([accepted_prefix, torch.as_tensor(
                [[mismatch_draft_token]], device=accepted_prefix.device)], dim=1),
            **dict(generation_kwargs,
                   max_new_tokens=max(
                       1, int(target_tokens_generated * self.length_max_rate) - prefix_tokens_generated))
        )
        if self.tokenizer.eos_token_id in alternative_response[0, accepted_prefix.shape[1]:]:
            eos_index = alternative_response[0].tolist().index(self.tokenizer.eos_token_id, accepted_prefix.shape[1])
            alternative_response = alternative_response[:, :eos_index + 1]
        
        target_answer, alt_answer = self.parser([*self.tokenizer.batch_decode(target_response),
                                                 *self.tokenizer.batch_decode(alternative_response)])
        alternative_response_str = self.tokenizer.decode(alternative_response[0])
        answers_match = self.evaluator(generations=[alternative_response_str], references=[target_answer]) == 1
        logger.info("\n\n\n" + "=" * 80)
        logger.info("PREFIX: " + self.tokenizer.batch_decode(accepted_prefix)[0])
        logger.info(f"[mismatch_target_token={self.tokenizer.decode(mismatch_target_token)}]")
        logger.info(f"[mismatch_draft_token={self.tokenizer.decode(mismatch_draft_token)}]")
        logger.info("SUFFIX: " + self.tokenizer.decode(alternative_response[0, accepted_prefix.shape[1]:]))
        return not answers_match

    @torch.no_grad()
    def find_important_tokens_greedy(
            self, *, input_ids: torch.Tensor, max_new_tokens: int, max_length: Optional[int] = None,
    ):
        assert self.ref_model is not None, f"initialize {self.__class__.__name__}.from_pretrained_with_tied_ref_model"
        assert input_ids.ndim == 2
        assert input_ids.shape[0] == 1, "Only support batch size 1 for now!!"
        assert not self.base_model.generation_config.do_sample, "tell your model not to do_sample! (please)"
        is_llama3 = any(x in self.base_model_name_or_path.lower() for x in ("llama-3", "llama3"))
        stop_token_id = None
        if is_llama3:
            warnings.warn("Model recognized as Llama 3.x")
            assert self.base_model.config.model_type == "llama"
            assert self.base_model.config.rope_scaling["rope_type"] == "llama3"
            stop_token_id = self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        else:
            warnings.warn("Model recognized as something other than Llama 3.x")
            assert input_ids.shape[0] == 1, "Only support batch size 1 for now!!"
            assert self.base_model.config.rope_scaling["rope_type"] != "llama3"
        logits_processor = None  # greedy inference
        max_length = input_ids.shape[1] + max_new_tokens if max_length is None else max_length

        changed_token_indices = []

        generation_kwargs = dict(
            max_length=max_length, max_new_tokens=max_new_tokens,
            temperature=0, top_k=self.ea_layer.top_k, top_p=0, log=False, is_llama3=is_llama3
        )
        target_response = self.ref_model.eagenerate(input_ids, **generation_kwargs)

        # Avoid modifying the input_ids in-place. /* Yes, this is a comment from EAGLE source code */
        padding = (torch.zeros(1, 1, dtype=torch.long) - 1).to(input_ids.device)
        input_ids = input_ids.clone()
        self.ea_layer.reset_kv()

        # Initialize the past key and value states
        if hasattr(self, "past_key_values"):
            past_key_values = self.past_key_values  # yes, it's a kriffin' property that will eat off memory...
            past_key_values_data = self.past_key_values_data
            current_length_data = self.current_length_data
            # Reset the past key and value states
            current_length_data.zero_()
        else:
            (
                past_key_values,
                past_key_values_data,
                current_length_data,
            ) = initialize_past_key_values(self.base_model, max_length=max_length)
            self.past_key_values = past_key_values
            self.past_key_values_data = past_key_values_data
            self.current_length_data = current_length_data

        prompt_ids = input_ids.clone()
        input_len = input_ids.shape[1]
        reset_tree_mode(self)
        # prefill
        draft_tokens, retrieve_indices, tree_mask, tree_position_ids, logits, hidden_state, sample_token = initialize_tree(
            input_ids, self, past_key_values, logits_processor
        )
        new_token = 0
        max_length = max_length - self.ea_layer.total_tokens - 10
        for _step in range(max_length):
            self.base_model.model.tree_mask = tree_mask
            draft_tokens = draft_tokens.to(input_ids.device)
            # Target model forward, get logits
            logits, hidden_state_new, outputs = tree_decoding(
                self,
                draft_tokens,
                past_key_values,
                tree_position_ids,
                input_ids,
                retrieve_indices,
            )
            draft_tokens = torch.cat((draft_tokens, padding), dim=1)
            candidates = draft_tokens[0, retrieve_indices]
            # verification
            best_candidate, accept_length, sample_p = evaluate_posterior(
                logits, candidates, logits_processor
            )

            # <begin AutoJudge important token mining>
            # [jh] Note: we can *very* easily get rid of separate hidden states step - just keep 'hidden_states_new'
            assert best_candidate.item() == 0
            target_argmax_next_token_ids = logits.argmax(-1)  # [1, window_size]
            accepted_token_mask = candidates[..., 1:] == target_argmax_next_token_ids[..., :-1]
            assert accepted_token_mask.cumprod(-1).sum(-1).item() == accept_length

            while _step > 0 and accept_length + 1 < candidates.shape[1]:  # repeat for all mismatches in current window (if any)
                assert torch.all(accepted_token_mask[..., :accept_length]) and not accepted_token_mask[:, accept_length]
                accepted_prefix = torch.cat([input_ids, candidates[:, :accept_length + 1]], dim=1)
                mismatch_draft_token = candidates[..., accept_length + 1].item()
                mismatch_target_token = target_argmax_next_token_ids[..., accept_length].item()
                is_important = self.check_if_mismatch_is_important(
                    prompt_ids=prompt_ids,
                    accepted_prefix=accepted_prefix,
                    mismatch_draft_token=mismatch_draft_token,
                    mismatch_target_token=mismatch_target_token,
                    target_response=target_response,
                    generation_kwargs=generation_kwargs,
                    target_logp_for_mismatching_token=logits[0, accept_length, :].log_softmax(dim=-1)
                )
                changed_token_indices.append(dict(
                    mismatch_index=accepted_prefix.shape[1],
                    is_important=is_important,
                    mismatch_target_token=mismatch_target_token,
                    mismatch_draft_token=mismatch_draft_token
                ))
                if is_important:
                    logger.info("IMPORTANT")
                    break  # the token is important; keep it from target model; do not consider subsequent tokens
                else:
                    logger.info("NOT IMPORTANT")
                    # the mismatching draft token does not affect the answer quality; keep it, accept subsequent tokens
                    old_accept_length = accept_length
                    accepted_token_mask[:, accept_length] = True  # mark this token as accepted
                    accept_length = accepted_token_mask.cumprod(-1).sum(-1).item()  # accept any subsequent matches
                    sample_p = logits[best_candidate, accept_length]  # for sampling the final token
                    logger.info(f"accept_length {old_accept_length} => {accept_length}", )
                    del old_accept_length
            # </end AutoJudge important token mining>

            # Adjusting the input sequence, draft model forward
            input_ids, draft_tokens, retrieve_indices, tree_mask, tree_position_ids, new_token, hidden_state, sample_token = update_inference_inputs(
                input_ids,
                candidates,
                best_candidate,
                accept_length,
                retrieve_indices,
                logits_processor,
                new_token,
                past_key_values_data,
                current_length_data,
                self,
                hidden_state_new,
                sample_p
            )

            assert torch.all(tree_position_ids == torch.arange(
                len(tree_position_ids), device=tree_position_ids.device)
                             ), "This code only supports total_tokens == depth + 1 (single trunk)"

            if is_llama3:
                if stop_token_id in input_ids[0, input_len:].tolist():
                    break

            if self.tokenizer.eos_token_id in input_ids[0, input_len:].tolist():
                break
            if new_token > max_new_tokens:
                break
            if input_ids.shape[1] > max_length:
                break

        actual_length = input_ids.shape[1]
        if stop_token_id in input_ids[0, input_len:].tolist():
            actual_length = input_len + input_ids[0, input_len:].tolist().index(stop_token_id) + 1
        changed_token_indices = [entry for entry in changed_token_indices if entry["mismatch_index"] < actual_length]
        return input_ids, changed_token_indices

    @torch.no_grad()
    def eagenerate_with_statistics(
            self,
            input_ids,
            temperature=0.0,
            top_p=0.0,
            top_k=0.0,
            max_new_tokens=512,
            max_length: Optional[int] = None,
            is_llama3=None,
            classifier: Optional[callable] = None,
    ):
        """Same as EaModel.eagenerate, but it measures acceptance statistics and infers llama3/max_length"""
        assert input_ids.shape[0] == 1
        if is_llama3 is None:
            is_llama3 = any(x in self.base_model_name_or_path.lower() for x in ("llama-3", "llama3"))
        stop_token_id = None
        if is_llama3:
            warnings.warn("Model recognized as Llama 3.x")
            assert self.base_model.config.model_type == "llama"
            assert self.base_model.config.rope_scaling["rope_type"] == "llama3"
            stop_token_id = self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        else:
            warnings.warn("Model recognized as something other than Llama 3.x")
            assert input_ids.shape[0] == 1, "Only support batch size 1 for now!!"
            assert self.base_model.config.rope_scaling["rope_type"] != "llama3"
        max_length = input_ids.shape[1] + max_new_tokens if max_length is None else max_length

        if temperature > 1e-5:
            logits_processor = prepare_logits_processor(temperature=temperature, top_p=top_p, top_k=top_k)
        else:
            logits_processor = None
        # Avoid modifying the input_ids in-place
        padding = (torch.zeros(1, 1, dtype=torch.long) - 1).to(input_ids.device)
        input_ids = input_ids.clone()
        self.ea_layer.reset_kv()

        # Initialize the past key and value states
        if hasattr(self, "past_key_values"):
            past_key_values = self.past_key_values
            past_key_values_data = self.past_key_values_data
            current_length_data = self.current_length_data
            # Reset the past key and value states
            current_length_data.zero_()
        else:
            (
                past_key_values,
                past_key_values_data,
                current_length_data,
            ) = initialize_past_key_values(self.base_model, max_length=max_length)
            self.past_key_values = past_key_values
            self.past_key_values_data = past_key_values_data
            self.current_length_data = current_length_data

        input_len = input_ids.shape[1]
        reset_tree_mode(self)
        # prefill
        draft_tokens, retrieve_indices, tree_mask, tree_position_ids, logits, hidden_state, sample_token = initialize_tree(
            input_ids, self, past_key_values, logits_processor
        )
        new_token = 0
        max_length = max_length - self.ea_layer.total_tokens - 10
        step_statistics = []
        for _step in range(max_length):
            # with Timer("all"):
            self.base_model.model.tree_mask = tree_mask

            draft_tokens = draft_tokens.to(input_ids.device)
            # Target model forward, get logits
            logits, hidden_state_new, outputs = tree_decoding(
                self,
                draft_tokens,
                past_key_values,
                tree_position_ids,
                input_ids,
                retrieve_indices,
            )
            # retrieve_indices=tree_buffers["retrieve_indices"]
            # logits = logits[0, retrieve_indices]
            draft_tokens = torch.cat((draft_tokens, padding), dim=1)
            candidates = draft_tokens[0, retrieve_indices]
            # verification
            best_candidate, accept_length, sample_p = evaluate_posterior(
                logits, candidates, logits_processor
            )
            # <begin AutoJudge>
            if classifier is not None:
                assert best_candidate.item() == 0
                assert hidden_state_new.shape[0] == 1
                target_argmax_next_token_ids = logits.argmax(-1)  # [1, window_size]
                accepted_token_mask = candidates[..., 1:] == target_argmax_next_token_ids[..., :-1]
                assert accepted_token_mask.cumprod(-1).sum(-1).item() == accept_length

                while _step > 0 and accept_length + 1 < candidates.shape[1]:
                    if classifier(hidden_state_new[:, accept_length + 1, :]):
                        break # the token is important; keep it from target model; do not consider subsequent tokens
                    
                    # else the token is 'good enough'
                    accepted_token_mask[:, accept_length] = True  # mark this token as accepted
                    accept_length = accepted_token_mask.cumprod(-1).sum(-1).item()  # accept any subsequent matches
                    sample_p = logits[best_candidate, accept_length]  # for sampling the final token
            # </end AutoJudge>
            # Adjusting the input sequence, draft model forward
            input_ids, draft_tokens, retrieve_indices, tree_mask, tree_position_ids, new_token, hidden_state, sample_token = update_inference_inputs(
                input_ids,
                candidates,
                best_candidate,
                accept_length,
                retrieve_indices,
                logits_processor,
                new_token,
                past_key_values_data,
                current_length_data,
                self,
                hidden_state_new,
                sample_p
            )

            _prev_length = step_statistics[-1]["input_ids_length"] if step_statistics else input_len
            step_statistics.append(dict(
                _step=_step,
                accepted_tokens=input_ids.shape[1] - _prev_length,
                accept_length_internal=accept_length,
                input_ids_length=input_ids.shape[1]
            ))
            if is_llama3:
                if stop_token_id in input_ids[0, input_len:].tolist():
                    break

            if self.tokenizer.eos_token_id in input_ids[0, input_len:].tolist():
                break
            if new_token > max_new_tokens:
                break
            if input_ids.shape[1] > max_length:
                break
        return dict(input_ids=input_ids, new_token=new_token, step_statistics=step_statistics)


def verify_args(args):
    assert args.process_id < args.world_size, "--process_id must be < --world_size"


def get_args():
    parser = argparse.ArgumentParser(description="Arguments for important tokens selection algorithm")

    parser.add_argument('--draft_model', type=str, default='yuhuili/EAGLE-LLaMA3.1-Instruct-8B',
                        help='Path or identifier of the EAGLE draft model head.')
    parser.add_argument('--target_model', type=str, default='meta-llama/Llama-3.1-8B-Instruct',
                        help='Path or identifier of the target model.')
    parser.add_argument('--torch_dtype', type=str, choices=['float32', 'auto'], default='float32',
                        help='Data type for torch tensors.')
    parser.add_argument('--gsm8k_train_path', type=str,
                        default=os.path.join(parent_dir, 'data/gsm8k_train.json'),
                        help='Path to the GSM8K train dataset JSON file.')
    parser.add_argument('--random_seed', type=int, default=42,
                        help='Random seed for reproducibility.')
    parser.add_argument('--max_new_tokens', type=int, default=1024,
                        help='Maximum number of new tokens to generate.')
    parser.add_argument('--max_length', type=int, default=8192,
                        help='Maximum length for EAGLE purposes.')
    parser.add_argument('--output_folder', type=str, default='output',
                        help='Output folder name.')
    parser.add_argument('--output_file', type=str, default='important_tokens',
                        help='Output file name.')
    parser.add_argument('--num_shots', type=int, default=0, choices=[0, 8],
                        help='Number of shots to use.')
    parser.add_argument('--window_size', type=int, default=8,
                        help='Window size for EAGLE speculations')
    parser.add_argument('--world_size', type=int, default=1,
                        help='world size')
    parser.add_argument('--process_id', type=int, default=0,
                        help='Process ID')
    parser.add_argument('--dump_freq', type=int, default=64,
                        help='Dump frequency.')
    parser.add_argument('--write_index_offset', type=int, default=0,
                        help='Write index offset.')
    parser.add_argument('--process_saver_id', type=int, default=0)
    parser.add_argument('--local_world_size', type=int, default=8)

    args = parser.parse_args()
    verify_args(args)

    return args


def load_questions(args):
    with open(args.gsm8k_train_path) as f:
        gsm_questions = [json.loads(line) for line in f]

    gsm_questions = [
        {
            'question': i['question'],
            'answer': i['answer'][i['answer'].rfind('#### ') + 5:]
        }
        for i in gsm_questions
    ]

    return gsm_questions


if __name__ == "__main__":
    args = get_args()
    print('The script was run in the following way:')
    print("python script.py \\\n" + "\n".join(f"\t\t--{k} {v} \\" for k, v in vars(args).items()))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    assert args.max_new_tokens <= args.max_length

    if '70b' in args.target_model.lower():
        device_map = 'auto'
    else:
        device_map = device

    np.random.seed(args.random_seed)
    model = EaModelForAutoJudge.from_pretrained_with_tied_ref_model(
        base_model_path=args.target_model,
        ea_model_path=args.draft_model,
        use_eagle3="eagle3" in args.draft_model.lower(),
        torch_dtype="auto",  # was: torch.float16
        device_map=device_map,
        low_cpu_mem_usage=True,
        depth=args.window_size - 1,
        total_token=args.window_size,
        do_sample=False, top_p=None, top_k=1, temperature=None
    )
    device = next(model.parameters()).device
    tokenizer = model.get_tokenizer()
    tokenizer.pad_token_id = 128004  # <|finetune_right_pad_id|>
    model.base_model.generation_config.pad_token_id = tokenizer.pad_token_id
    model.ref_model.base_model.generation_config.pad_token_id = tokenizer.pad_token_id
    model.eval()

    gsm_questions = load_questions(args)
    logger.info(f'Read {len(gsm_questions)} questions from {args.gsm8k_train_path}')

    n_samples = len(gsm_questions)
    shard_len = (n_samples + args.world_size - 1) // args.world_size
    shard_start = args.process_id * shard_len
    shard_end = min((args.process_id + 1) * shard_len, n_samples)
    logger.info(
        f'Process {args.process_id} {n_samples=}, {args.world_size=}, {shard_len=}, {shard_start=}, {shard_end=}')
    logger.info(
        f'Process {args.process_id} will process {shard_end - shard_start} questions: [{shard_start}; {shard_end})')

    if args.num_shots == 0:
        prompt_with_shots = GSM8KPrompts.prompt_with_0_shots
        logger.info(f'{args.num_shots=}, using prompt_with_0_shots')
    else:
        prompt_with_shots = GSM8KPrompts.prompt_with_8_shots
        logger.info(f'{args.num_shots=}, using prompt_with_8_shots')

    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)
        logger.info(f'Created output folder {args.output_folder}')
    else:
        logger.info(f'Output folder {args.output_folder} already exists')

    output_file_path = os.path.join(args.output_folder, args.output_file + f'_{args.process_id}.pt')
    if os.path.exists(output_file_path):
        important_tokens_data = torch.load(output_file_path)
        logger.info(
            f'Loaded {len(important_tokens_data)} important tokens from {output_file_path}, starting from {shard_start + len(important_tokens_data)}-th sample')
    else:
        important_tokens_data = []
        logger.info(f'No important tokens found in {output_file_path}, will generate them')

    if 'NV_YT_OPERATION_ID' in os.environ:
        logger.info('NY_YT_OPERATION_ID found in os.environ')
    else:
        logger.info('NY_YT_OPERATION_ID not found in os.environ')

    ##### FAST REBUTTAL QWEN PATCH START #####

    if 'qwen' in args.target_model.lower():
        raise NotImplementedError("Qwen not tested with EAGLE")
        tokenizer.bos_token_id = tokenizer.eos_token_id  # based on: https://huggingface.co/Qwen/Qwen2-7B-Instruct/discussions/15
        with open("data/qwens/gsm8k-cot-llama.yaml", "r") as f:
            config = yaml.safe_load(f)

        fewshot_samples = config.get("fewshot_config", {}).get("samples", [])

        format_prompt = (
            "Given the following problem, reason and give a final answer to the problem.\n"
            "Problem: {question}\n"
            'Your response should end with "The final answer is [answer]" where [answer] is the response to the problem.'
        )

        few_shot_turns = []
        for sample in fewshot_samples:
            question = sample["question"]
            target = sample["target"]

            formatted_question = format_prompt.format(question=question)

            few_shot_turns.append({
                "role": "user",
                "content": formatted_question,
            })
            few_shot_turns.append({
                "role": "assistant",
                "content": target,
            })
    ##### FAST REBUTTAL QWEN PATCH END #####

    if shard_end > shard_start:
        with tqdm(total=shard_end - shard_start) as pbar:
            pbar.update(len(important_tokens_data))
            for sample_idx in range(shard_start + len(important_tokens_data), shard_end):
                question_sample = gsm_questions[sample_idx]
                answer = question_sample['answer']
                question = question_sample['question']
                prompt = prompt_with_shots + question + "\n" + GSM8KPrompts.formatting_prompt + llama_assistant_turn_end
                batch_input_ids = tokenizer(prompt, return_tensors='pt', add_special_tokens=False)['input_ids'].to(device)

                ##### FAST REBUTTAL QWEN PATCH START #####
                if 'qwen' in args.target_model.lower():
                    gsm_prompt_start = "Given the following problem, reason and give a final answer to the problem.\nProblem:"

                    prompt = gsm_prompt_start + " " + question + "\n" + GSM8KPrompts.formatting_prompt

                    batch_input_ids = tokenizer.apply_chat_template(
                        few_shot_turns + \
                        [{'role': 'user', 'content': prompt}]
                        , return_tensors='pt', add_generation_prompt=True, tokenize=True).to(device)
                ##### FAST REBUTTAL QWEN PATCH END #####

                current_response, changed_token_indices = model.find_important_tokens_greedy(
                    input_ids=batch_input_ids, max_new_tokens=args.max_new_tokens, max_length=args.max_length,
                )
                changed_token_indices = [
                    (d["mismatch_index"], d["is_important"], d["mismatch_target_token"], d["mismatch_draft_token"])
                    for d in changed_token_indices
                ]
                important_tokens_dict = dict(
                    current_response=current_response.data.cpu(), changed_token_indices=changed_token_indices,
                )

                important_tokens_dict['id'] = sample_idx + args.write_index_offset

                important_tokens_data.append(important_tokens_dict)

                if ((sample_idx % args.dump_freq == (args.dump_freq - 1)) or (sample_idx == shard_end - 1)):
                    torch.save(important_tokens_data, output_file_path)

                pbar.update(1)
                pbar.set_description(f'Process {args.process_id}')
    else:
        torch.save(important_tokens_data, output_file_path)

    done_file = os.path.join(args.output_folder, f"done_{args.process_id}.txt")
    with open(done_file, "w") as f:
        f.write("done\n")
    logger.info(f"Process {args.process_id} has finished. Created {done_file}")

    if args.process_id == args.process_saver_id and args.world_size > 1:
        logger.info("Process 0 is waiting for all other processes to finish...")
        while True:
            print("Process-saver tries to save snapshot:")
            done_files = [f"done_{i}.txt" for i in
                          range(args.process_saver_id + 1, args.process_saver_id + args.local_world_size)]
            print(f"{done_files=}")
            print(f"Done files:\n{[os.path.exists(os.path.join(args.output_folder, f)) for f in done_files]}")
            all_done = all(os.path.exists(os.path.join(args.output_folder, f)) for f in done_files)

            if all_done:
                logger.info("All processes finished. Concatenating important tokens data...")
                all_data = []
                for i in range(args.process_saver_id, args.process_saver_id + args.local_world_size):
                    # if i * shard_len >= shard_end:
                    #     continue
                    process_output_file = os.path.join(args.output_folder, f"{args.output_file}_{i}.pt")
                    all_data.extend(torch.load(process_output_file))
                    logger.info(f"Removing {process_output_file}")
                    os.remove(process_output_file)

                    done_file = os.path.join(args.output_folder, f"done_{i}.txt")
                    logger.info(f"Removing {done_file}")
                    os.remove(done_file)

                final_output_file_path = os.path.join(args.output_folder,
                                                      args.output_file + f'_{args.process_saver_id}' + '.pt')
                logger.info(f"Saving important tokens data to {final_output_file_path}...")
                torch.save(all_data, final_output_file_path)
                logger.info("Done.")
                break

            time.sleep(5)
