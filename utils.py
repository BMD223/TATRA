import argparse, csv, json, random, re
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Callable

from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


def parse_args():
    ap = argparse.ArgumentParser(
        description="Random search over paraphrases (n), in-context examples (k), and runs; unweighted voting only."
    )
    ap.add_argument("--model-path", type=str, default="Qwen/Qwen2.5-7B-Instruct",
                    help="HF id or local path to the model.")
    ap.add_argument("--generator-model-path", type=str, default=None,
                    help="HF id or local path to the generator model (overrides --model-path).")
    ap.add_argument("--evaluator-model-path", type=str, default=None,
                    help="HF id or local path to the evaluator model (overrides --model-path).")
    ap.add_argument("--eval-dataset-path", type=str, default="dataset/sst2_test.jsonl",
                    help="Dataset to evaluate end-to-end.")
    ap.add_argument("--limit", type=int, default=0,
                    help="Limit number of items evaluated from --eval-dataset-path (0 = all).")

    ap.add_argument("--para-temperature", type=float, default=0.9, help="Paraphraser temperature.")
    ap.add_argument("--para-top-p", type=float, default=0.95, help="Paraphraser top-p.")
    ap.add_argument("--para-max-new", type=int, default=256, help="Paraphraser max new tokens.")

    ap.add_argument("--max-new-tokens-proposer", type=int, default=0,
                    help="Override proposer max tokens (default = 48*k or 64).")

    ap.add_argument("--batch-size", type=int, default=64,
                    help="Cross-item batch size for paraphrasing and labeling.")

    ap.add_argument("--use-vllm", type=int, default=1, help="Use vLLM if available (1=yes, 0=no).")
    ap.add_argument("--tp", type=int, default=1, help="vLLM tensor parallel size.")
    ap.add_argument("--gpu-mem-util", type=float, default=0.60, help="vLLM GPU memory utilization (optimized for GH200 96GB) (safe default for 40GB GPUs).")
    ap.add_argument("--generator-gpu-mem-util", type=float, default=None,
                    help="vLLM GPU memory utilization for generator (overrides --gpu-mem-util).")
    ap.add_argument("--evaluator-gpu-mem-util", type=float, default=None,
                    help="vLLM GPU memory utilization for evaluator (overrides --gpu-mem-util).")
    ap.add_argument("--max-model-len", type=int, default=2048, help="vLLM max model length.")
    ap.add_argument("--quantization", type=str, default=None, help="vLLM quantization (e.g., awq, gptq).")

    ap.add_argument("--report-every", type=int, default=50,
                    help="Progress print frequency during evaluation (0 = silent).")

    ap.add_argument("--run-random-search", action="store_true",
                    help="Run random search over (n,k,runs) and seeds.")

    ap.add_argument("--grid-n", type=str, default="0,1,2,5,10,15",
                    help="Candidate values for paraphrases-per-item.")
    ap.add_argument("--grid-k", type=str, default="4,8,12,16,32",
                    help="Candidate values for in-context example counts.")
    ap.add_argument("--grid-runs", type=str, default="1,3,5,10,15",
                    help="Candidate values for repeated run counts.")
    ap.add_argument("--grid-seeds", type=str, default="1,2,3,4",
                    help="Seeds to test per sampled spec (seed 1 used for early stop).")

    ap.add_argument("--random-results-csv", type=str, default="results/sst2_random.csv",
                    help="Where to save random search results CSV.")
    ap.add_argument("--random-specs", type=int, default=50_000_000,
                    help="Number of random hyperparameter specs to evaluate.")
    ap.add_argument("--random-hparam-seed", type=int, default=12345,
                    help="RNG seed for random spec sampling.")
    ap.add_argument("--early-stop-threshold", type=float, default=0.93,
                    help="If seed==1 accuracy < threshold, skip remaining seeds for that spec.")
    return ap.parse_args()


def order_seeds(seeds: List[int]) -> List[int]:
    return [1] + [s for s in seeds if s != 1] if 1 in seeds else list(seeds)


def append_csv_row(path: str, row: Dict):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    write_hdr = not p.exists()
    with open(p, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_hdr:
            w.writeheader()
        w.writerow(row)


def _parse_int_list(s: str) -> List[int]:
    out: List[int] = []
    for tok in s.split(","):
        tok = tok.strip()
        if tok:
            out.append(int(tok))
    return out


def set_global_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_jsonl(path: str) -> List[Dict]:
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    items.append(json.loads(line))
                except Exception:
                    pass
    return items


def init_qwen_transformers(model_name: str = "Qwen/Qwen2.5-7B-Instruct"):
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, use_fast=True, padding_side='left')
    model = None
    for attn in ("sdpa", "eager", None):
        try:
            kw = dict(trust_remote_code=True, torch_dtype="auto", device_map="auto")
            if attn:
                kw["attn_implementation"] = attn
            model = AutoModelForCausalLM.from_pretrained(model_name, **kw)
            break
        except Exception:
            continue
    if model is None:
        model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
    torch.set_float32_matmul_precision("high")
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
    model.eval()
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer, model


def init_qwen_vllm(model_path: str, tp: int, util: float, max_len: int, quantization: Optional[str]):
    from vllm import LLM
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, padding_side='left')
    llm = LLM(
        model=model_path,
        trust_remote_code=True,
        tensor_parallel_size=tp,
        gpu_memory_utilization=util,
        max_model_len=max_len,
        dtype="auto",
        quantization=quantization,
        enable_prefix_caching=True,
    )
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer, llm


def vllm_generate_texts(
    llm,
    prompts: List[str],
    *,
    temperature: float,
    top_p: float,
    max_tokens: int,
    seed: Optional[int],
    stop: Optional[List[str]] = None,
) -> List[str]:
    from vllm import SamplingParams
    sp = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        seed=None if seed is None else int(seed),
        stop=stop or [],
    )
    outs = llm.generate(prompts, sp, use_tqdm=False)
    return [o.outputs[0].text for o in outs]


def gen_with_messages_transformers(tokenizer, model, messages: List[Dict[str, str]], *, temperature: float, top_p: float, max_new_tokens: int, seed: Optional[int]) -> str:
    if seed is not None:
        set_global_seed(seed)
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    device = next(model.parameters()).device
    enc = tokenizer(prompt, return_tensors="pt").to(device)
    do_sample = temperature > 0.01
    gen_kw = dict(max_new_tokens=max_new_tokens, pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                  eos_token_id=tokenizer.eos_token_id, use_cache=True)
    if do_sample:
        gen_kw.update(do_sample=True, temperature=temperature, top_p=top_p)
    else:
        gen_kw["do_sample"] = False
    with torch.inference_mode():
        out = model.generate(**enc, **gen_kw)
    return tokenizer.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True)


def _split_n_lines(text: str, n: int) -> List[str]:
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    cleaned = [ln.lstrip("-*•").lstrip().lstrip("0123456789").lstrip(".)]" ).strip() for ln in lines][:n]
    while len(cleaned) < n and cleaned:
        cleaned.append(cleaned[-1])
    if len(cleaned) < n:
        raise RuntimeError(f"Model returned fewer than {n} paraphrases:\n{text}")
    return cleaned


def batch_generate_paraphrases_transformers(sentences: List[str], n: int, tokenizer, model, prompt_fn: Callable[[str, int], str],
                                              temperature=0.9, top_p=0.95, max_new_tokens=256, seed: Optional[int] = None, mini_batch_size: int = 4) -> List[List[str]]:
    if not sentences or n <= 0:
        return [[] for _ in sentences]
    if seed is not None:
        set_global_seed(seed)
    prompts = [prompt_fn(s, n) for s in sentences]
    device = next(model.parameters()).device
    do_sample = temperature > 0.01
    gens = []
    for mb_start in range(0, len(prompts), mini_batch_size):
        mb = prompts[mb_start:mb_start + mini_batch_size]
        mb_sent = sentences[mb_start:mb_start + mini_batch_size]
        enc = tokenizer(mb, return_tensors="pt", padding=True, truncation=True)
        enc = {k: v.to(device) for k, v in enc.items()}
        gen_kw = dict(max_new_tokens=max_new_tokens, pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                      eos_token_id=tokenizer.eos_token_id, use_cache=True)
        if do_sample:
            gen_kw.update(do_sample=True, temperature=temperature, top_p=top_p)
        else:
            gen_kw["do_sample"] = False
        with torch.inference_mode():
            out = model.generate(**enc, **gen_kw)
        inp_len = enc["attention_mask"].sum(dim=1)
        for j in range(out.size(0)):
            text = tokenizer.decode(out[j, inp_len[j]:], skip_special_tokens=True)
            try:
                gens.append(_split_n_lines(text, n))
            except RuntimeError:
                gens.append([mb_sent[j]] * n)
        del enc, out
        torch.cuda.empty_cache()
    return gens


def batch_generate_paraphrases_vllm(sentences: List[str], n: int, llm, prompt_fn: Callable[[str, int], str],
                                     temperature=0.9, top_p=0.95, max_new_tokens=256, seed: Optional[int] = None) -> List[List[str]]:
    if not sentences or n <= 0:
        return [[] for _ in sentences]
    prompts = [prompt_fn(s, n) for s in sentences]
    texts = vllm_generate_texts(llm, prompts, temperature=temperature, top_p=top_p, max_tokens=max_new_tokens, seed=seed)
    results = []
    for i, t in enumerate(texts):
        try:
            results.append(_split_n_lines(t, n))
        except RuntimeError:
            results.append([sentences[i]] * n)
    return results


def build_label_prompts(sentences: List[str], prefix: str, suffix: str) -> List[str]:
    prompts = []
    for s in sentences:
        s_clean = s.replace('"', "'")
        prompts.append(prefix + s_clean + suffix)
    return prompts


def normalize_to_allowed_label(text: str, allowed_labels: List[str]) -> Optional[str]:
    t = (text or "").strip().lower()
    for lab in allowed_labels:
        if lab.lower() in t:
            return lab
    t0 = re.sub(r"[^a-z]", "", t)
    for lab in allowed_labels:
        if t0.startswith(re.sub(r"[^a-z]", "", lab.lower())):
            return lab
    return None


def batch_predict_labels_transformers(tokenizer, model, sentences: List[str], *, prefix: str, suffix: str,
                                        allowed_labels: List[str], max_new_tokens: int = 4, mini_batch_size: int = 8) -> List[Optional[str]]:
    if not sentences:
        return []
    prompts = build_label_prompts(sentences, prefix, suffix)
    order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))
    sorted_prompts = [prompts[i] for i in order]
    device = next(model.parameters()).device
    preds_sorted = []
    for mb_start in range(0, len(sorted_prompts), mini_batch_size):
        mb = sorted_prompts[mb_start:mb_start + mini_batch_size]
        enc = tokenizer(mb, return_tensors="pt", padding=True, truncation=True)
        enc = {k: v.to(device) for k, v in enc.items()}
        inp_len = enc["attention_mask"].sum(dim=1)
        with torch.inference_mode():
            out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                                  eos_token_id=tokenizer.eos_token_id,
                                  pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id, use_cache=True)
        for j in range(out.size(0)):
            text = tokenizer.decode(out[j, inp_len[j]:], skip_special_tokens=True)
            preds_sorted.append(normalize_to_allowed_label(text, allowed_labels))
        del enc, out
        torch.cuda.empty_cache()
    preds = [None] * len(preds_sorted)
    for rank, idx in enumerate(order):
        preds[idx] = preds_sorted[rank]
    return preds


def batch_predict_labels_vllm(llm, sentences: List[str], *, prefix: str, suffix: str,
                               allowed_labels: List[str], max_new_tokens: int = 4) -> List[Optional[str]]:
    if not sentences:
        return []
    prompts = build_label_prompts(sentences, prefix, suffix)
    texts = vllm_generate_texts(llm, prompts, temperature=0.0, top_p=1.0, max_tokens=max_new_tokens, seed=None)
    return [normalize_to_allowed_label(t, allowed_labels) for t in texts]


def format_examples_for_evaluator(examples: List[Dict[str, str]], drop_index: Optional[int] = None) -> str:
    lines, idx = [], 1
    for i, ex in enumerate(examples):
        if drop_index is not None and i == drop_index:
            continue
        sentence = ex['sentence'].replace('"', "'")
        lines.extend([f"Example{idx}:", f'Sentence: "{sentence}"', f"Label: {ex['label']}", ""])
        idx += 1
    return "\n".join(lines).rstrip()


def sentence_count_plan(k: int) -> List[int]:
    base = [1, 2, 1, 3, 2, 1, 3, 2]
    return [base[i % len(base)] for i in range(k)]


def parse_examples_from_text(
    text: str,
    k: int,
    allowed_labels: List[str],
    *,
    example_header_re: Optional[re.Pattern] = None,
    sentence_line_re: Optional[re.Pattern] = None,
) -> List[Dict[str, str]]:
    if example_header_re is None:
        example_header_re = re.compile(r"^Example\s*(\d+)\s*:\s*", re.IGNORECASE | re.MULTILINE)
    if sentence_line_re is None:
        sentence_line_re = re.compile(
            r'^\s*Sentence\s*:\s*"(.*?)"\s*$',
            re.IGNORECASE | re.MULTILINE | re.DOTALL,
        )
    lbl_pat = "|".join(re.escape(l) for l in allowed_labels)
    label_re = re.compile(rf'^\s*Label\s*:\s*({lbl_pat})\b', re.IGNORECASE | re.MULTILINE)

    blocks: List[Tuple[int, str]] = []
    headers = list(example_header_re.finditer(text))
    for i, m in enumerate(headers):
        start = m.start()
        end = headers[i + 1].start() if i + 1 < len(headers) else len(text)
        blocks.append((int(m.group(1)), text[start:end]))

    tuples: List[Tuple[int, str, str]] = []
    for idx, block in blocks:
        sents = [m.group(1).strip() for m in sentence_line_re.finditer(block)]
        if not sents:
            continue
        sentence = " ".join(sents)
        m_lab = label_re.search(block)
        if not m_lab:
            continue
        lab_raw = m_lab.group(1).strip()

        lab_norm = None
        for lab in allowed_labels:
            if lab.lower() == lab_raw.lower():
                lab_norm = lab
                break
        if lab_norm is None:
            continue

        tuples.append((idx, sentence, lab_norm))

    tuples.sort(key=lambda x: x[0])
    dedup: List[Dict[str, str]] = []
    seen = set()
    for _, sent, lab in tuples:
        key = (sent, lab)
        if key in seen:
            continue
        seen.add(key)
        dedup.append({"sentence": sent, "label": lab})
        if len(dedup) >= k:
            break
    return dedup


def majority_vote_generic(labels: List[str], allowed_labels: List[str], *, tie_breaker: Optional[str] = None, default_label: Optional[str] = None) -> str:
    counts = {lab: 0 for lab in allowed_labels}
    for lab in labels:
        if lab in counts:
            counts[lab] += 1
    best = max(counts.values()) if counts else 0
    winners = [lab for lab, c in counts.items() if c == best]
    if len(winners) == 1:
        return winners[0]
    if tie_breaker in allowed_labels:
        return tie_breaker
    if default_label in allowed_labels:
        return default_label
    return allowed_labels[0] if allowed_labels else ""


def prepare_items(
    dataset_path: str,
    limit: int,
    allowed_labels: List[str],
    *,
    solution_field: str = "solution",
    messages_field: str = "messages",
    role_field: str = "role",
    user_role_value: str = "user",
    content_field: str = "content",
) -> List[Tuple[str, str]]:
    data_all = load_jsonl(dataset_path)
    items: List[Tuple[str, str]] = []
    allowed_lower = {lab.lower() for lab in allowed_labels}

    for obj in data_all:
        sol_raw = str(obj.get(solution_field, "")).strip()
        sol_norm = None
        for lab in allowed_labels:
            if sol_raw.lower() == lab.lower():
                sol_norm = lab
                break
        if sol_norm is None:
            continue

        user_turn = next(
            (m for m in obj.get(messages_field, []) if m.get(role_field) == user_role_value),
            None,
        )
        if not user_turn:
            continue
        sentence = str(user_turn.get(content_field, "")).strip()
        if not sentence:
            continue

        items.append((sentence, sol_norm))
        if limit and len(items) >= limit:
            break
    return items


def eval_run_transformers_collect(
    tokenizer,
    model,
    *,
    label_prefix: str,
    label_suffix: str,
    items: List[Tuple[str, str]],
    paraphrases_per_item: int,
    base_seed: int,
    para_temperature: float,
    para_top_p: float,
    para_max_new: int,
    batch_size: int,
    report_every: int,
    paraphrase_prompt_fn: Callable[[str, int], str],
    allowed_labels: List[str],
) -> Tuple[List[List[str]], List[Tuple[float, float]], List[List[str]]]:
    N = len(items)
    per_item_top1: List[List[str]] = [[] for _ in range(N)]
    per_item_weight_sums: List[Tuple[float, float]] = [(0.0, 0.0) for _ in range(N)]
    per_item_paraphrases: List[List[str]] = [[] for _ in range(N)]
    total = 0
    correct_top1 = 0

    num_batches = (N + batch_size - 1) // batch_size
    for b in tqdm(range(num_batches), desc="Evaluating (HF; collect)", unit="batch"):
        start_idx = b * batch_size
        end_idx = min(N, start_idx + batch_size)
        chunk = items[start_idx:end_idx]
        sentences = [s for s, _ in chunk]

        if paraphrases_per_item > 0:
            par_lists = batch_generate_paraphrases_transformers(
                sentences,
                paraphrases_per_item,
                tokenizer,
                model,
                prompt_fn=paraphrase_prompt_fn,
                temperature=para_temperature,
                top_p=para_top_p,
                max_new_tokens=para_max_new,
                seed=base_seed + b,
            )
        else:
            par_lists = [[] for _ in sentences]

        all_candidates: List[str] = []
        idx_slices: List[Tuple[int, int]] = []
        for orig, pars in zip(sentences, par_lists):
            cands = [orig] + pars
            s = len(all_candidates)
            all_candidates.extend(cands)
            idx_slices.append((s, s + len(cands)))

        preds_all = batch_predict_labels_transformers(
            tokenizer,
            model,
            all_candidates,
            prefix=label_prefix,
            suffix=label_suffix,
            allowed_labels=allowed_labels,
            max_new_tokens=4,
        )

        for j, (_, solution) in enumerate(chunk):
            s, e = idx_slices[j]
            preds_clean = [p for p in preds_all[s:e] if p in allowed_labels]
            run_pred = majority_vote_generic(
                preds_clean,
                allowed_labels,
                tie_breaker=preds_clean[0] if preds_clean else None,
                default_label=allowed_labels[0],
            )
            per_item_top1[start_idx + j].extend(preds_clean)
            per_item_paraphrases[start_idx + j].extend(par_lists[j])
            total += 1
            if run_pred == solution:
                correct_top1 += 1

        if report_every > 0 and (total % report_every == 0):
            print(
                f"[EVAL][HF] processed={total} | running top1-acc={correct_top1/total:.4f} "
                f"({correct_top1}/{total})"
            )

    print(f"[EVAL][HF] run top1-acc: { (correct_top1/total) if total else 0.0 :.4f}  ({correct_top1}/{total})")
    return per_item_top1, per_item_weight_sums, per_item_paraphrases


def eval_run_vllm_collect(
    llm,
    *,
    generator_llm=None,
    evaluator_llm=None,
    label_prefix: str,
    label_suffix: str,
    items: List[Tuple[str, str]],
    paraphrases_per_item: int,
    base_seed: int,
    para_temperature: float,
    para_top_p: float,
    para_max_new: int,
    batch_size: int,
    report_every: int,
    paraphrase_prompt_fn: Callable[[str, int], str],
    allowed_labels: List[str],
) -> Tuple[List[List[str]], List[Tuple[float, float]], List[List[str]]]:
    N = len(items)
    per_item_top1: List[List[str]] = [[] for _ in range(N)]
    per_item_weight_sums: List[Tuple[float, float]] = [(0.0, 0.0) for _ in range(N)]
    per_item_paraphrases: List[List[str]] = [[] for _ in range(N)]
    total = 0
    correct_top1 = 0

    num_batches = (N + batch_size - 1) // batch_size
    for b in tqdm(range(num_batches), desc="Evaluating (vLLM; collect)", unit="batch"):
        start_idx = b * batch_size
        end_idx = min(N, start_idx + batch_size)
        chunk = items[start_idx:end_idx]
        sentences = [s for s, _ in chunk]

        if paraphrases_per_item > 0:
            par_lists = batch_generate_paraphrases_vllm(
                sentences,
                paraphrases_per_item,
                generator_llm if generator_llm else llm,
                prompt_fn=paraphrase_prompt_fn,
                temperature=para_temperature,
                top_p=para_top_p,
                max_new_tokens=para_max_new,
                seed=base_seed + b,
            )
        else:
            par_lists = [[] for _ in sentences]

        all_candidates: List[str] = []
        idx_slices: List[Tuple[int, int]] = []
        for orig, pars in zip(sentences, par_lists):
            cands = [orig] + pars
            s = len(all_candidates)
            all_candidates.extend(cands)
            idx_slices.append((s, s + len(cands)))

        preds_all = batch_predict_labels_vllm(
            evaluator_llm if evaluator_llm else llm,
            all_candidates,
            prefix=label_prefix,
            suffix=label_suffix,
            allowed_labels=allowed_labels,
            max_new_tokens=4,
        )

        for j, (_, solution) in enumerate(chunk):
            s, e = idx_slices[j]
            preds_clean = [p for p in preds_all[s:e] if p in allowed_labels]
            run_pred = majority_vote_generic(
                preds_clean,
                allowed_labels,
                tie_breaker=preds_clean[0] if preds_clean else None,
                default_label=allowed_labels[0],
            )
            per_item_top1[start_idx + j].extend(preds_clean)
            per_item_paraphrases[start_idx + j].extend(par_lists[j])
            total += 1
            if run_pred == solution:
                correct_top1 += 1

        if report_every > 0 and (total % report_every == 0):
            print(
                f"[EVAL][vLLM] processed={total} | running top1-acc={correct_top1/total:.4f} "
                f"({correct_top1}/{total})"
            )

    print(f"[EVAL][vLLM] run top1-acc: { (correct_top1/total) if total else 0.0 :.4f}  ({correct_top1}/{total})")
    return per_item_top1, per_item_weight_sums, per_item_paraphrases
