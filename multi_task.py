import os, re, time, random
from typing import List, Dict, Tuple, Optional

os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")
os.environ.setdefault("PYTORCH_ENABLE_FLASH_SDP", "0")
os.environ.setdefault("PYTORCH_ENABLE_MEM_EFFICIENT_SDP", "0")
os.environ.setdefault("PYTORCH_ENABLE_SOMA", "0")

from utils import (
    parse_args, order_seeds, append_csv_row, _parse_int_list, init_qwen_vllm,
    prepare_items, format_examples_for_evaluator, sentence_count_plan,
    parse_examples_from_text, vllm_generate_texts, gen_with_messages_transformers,
    eval_run_transformers_collect, eval_run_vllm_collect, majority_vote_generic, set_global_seed,
)

TASK_CONFIGS = {
    "sst2": {
        "labels": ["positive", "negative"],
        "default": "negative",
        "dataset": "dataset/sst2_test.jsonl",
        "csv": "results/sst2_random.csv",
        "task_header": "Please perform Sentiment Classification task. Given the sentence, assign a sentiment label from ['negative', 'positive']. Return label only without any other text.",
        "topics": {
            "positive": ["acting", "direction", "screenplay", "cinematography", "pacing", "soundtrack", "effects", "themes", "character", "emotion"],
            "negative": ["acting", "direction", "screenplay", "cinematography", "pacing", "soundtrack", "effects", "themes", "character", "emotion"],
        },
        "style_rules": {
            "positive": "- Label MUST be exactly: positive.\n- Write with clearly POSITIVE sentiment: praise, enjoyment, strengths.\n- Use favorable adjectives/adverbs.\n",
            "negative": "- Label MUST be exactly: negative.\n- Write with clearly NEGATIVE sentiment: criticism, disappointment, weakness.\n- Use critical adjectives/adverbs.\n",
        },
        "para_prompt": "You paraphrase movie-review sentences while preserving the original sentiment and meaning.\nProduce exactly {n} diverse paraphrases of the sentence below. Keep tone and sentiment identical.\nOutput ONLY the paraphrases, one per line, with no numbering, bullets, or extra commentary.\n\nSentence:\n{s}",
    },
    "cr": {
        "labels": ["positive", "negative"],
        "default": "negative",
        "dataset": "dataset/cr_test.jsonl",
        "csv": "results/cr_random.csv",
        "task_header": "Please perform Sentiment Classification task. Given the sentence, assign a sentiment label from ['negative', 'positive']. Return label only without any other text.",
        "topics": {
            "positive": ["product quality", "customer service", "value for money", "durability", "ease of use", "design", "performance", "reliability", "features", "shipping"],
            "negative": ["product defects", "poor service", "overpriced", "fragility", "complexity", "ugly design", "slow performance", "unreliable", "missing features", "shipping issues"],
        },
        "style_rules": {
            "positive": "- Label MUST be exactly: positive.\n- Write with clearly POSITIVE sentiment: praise, satisfaction, recommendation.\n- Use favorable adjectives/adverbs.\n",
            "negative": "- Label MUST be exactly: negative.\n- Write with clearly NEGATIVE sentiment: criticism, disappointment, complaints.\n- Use critical adjectives/adverbs.\n",
        },
        "para_prompt": "You paraphrase product review sentences while preserving the original sentiment and meaning.\nProduce exactly {n} diverse paraphrases of the sentence below. Keep tone and sentiment identical.\nOutput ONLY the paraphrases, one per line, with no numbering, bullets, or extra commentary.\n\nSentence:\n{s}",
    },
    "mr": {
        "labels": ["positive", "negative"],
        "default": "negative",
        "dataset": "dataset/mr_test.jsonl",
        "csv": "results/mr_random.csv",
        "task_header": "Please perform Sentiment Classification task. Given the sentence, assign a sentiment label from ['negative', 'positive']. Return label only without any other text.",
        "topics": {
            "positive": ["acting/performance", "direction", "screenplay/dialogue", "cinematography", "pacing", "soundtrack/music", "visual effects", "themes/message", "character development", "emotional impact"],
            "negative": ["acting/performance", "direction", "screenplay/dialogue", "cinematography", "pacing", "soundtrack/music", "visual effects", "themes/message", "character development", "emotional impact"],
        },
        "style_rules": {
            "positive": "- Label MUST be exactly: positive.\n- Write with clearly POSITIVE sentiment: praise, enjoyment, strengths.\n- Use favorable adjectives/adverbs.\n",
            "negative": "- Label MUST be exactly: negative.\n- Write with clearly NEGATIVE sentiment: criticism, disappointment, weakness.\n- Use critical adjectives/adverbs.\n",
        },
        "para_prompt": "You paraphrase movie-review sentences while preserving the original sentiment and meaning.\nProduce exactly {n} diverse paraphrases of the sentence below. Keep tone and sentiment identical.\nOutput ONLY the paraphrases, one per line, with no numbering, bullets, or extra commentary.\n\nSentence:\n{s}",
    },
    "sst5": {
        "labels": ["terrible", "bad", "okay", "good", "great"],
        "default": "okay",
        "dataset": "dataset/sst5_test.jsonl",
        "csv": "results/sst5_random.csv",
        "task_header": "Please perform Fine-grained Sentiment Classification task. Given the sentence, assign a sentiment label from ['terrible', 'bad', 'okay', 'good', 'great']. Return label only without any other text.",
        "topics": {
            "terrible": ["acting", "plot", "direction", "dialogue", "pacing"],
            "bad": ["acting", "plot", "direction", "dialogue", "pacing"],
            "okay": ["acting", "plot", "direction", "dialogue", "pacing"],
            "good": ["acting", "plot", "direction", "dialogue", "pacing"],
            "great": ["acting", "plot", "direction", "dialogue", "pacing"],
        },
        "style_rules": {
            "terrible": "- Label MUST be exactly: terrible.\n- Write extremely negative: disgust, hatred, complete failure.\n",
            "bad": "- Label MUST be exactly: bad.\n- Write negative: disappointment, flaws, below average.\n",
            "okay": "- Label MUST be exactly: okay.\n- Write neutral/mixed: mediocre, average, some good some bad.\n",
            "good": "- Label MUST be exactly: good.\n- Write positive: enjoyable, well-made, recommended.\n",
            "great": "- Label MUST be exactly: great.\n- Write extremely positive: masterpiece, outstanding, must-see.\n",
        },
        "para_prompt": "You paraphrase movie-review sentences while preserving the original sentiment intensity.\nProduce exactly {n} diverse paraphrases. Keep the sentiment level (terrible/bad/okay/good/great) identical.\nOutput ONLY the paraphrases, one per line.\n\nSentence:\n{s}",
    },
    "news": {
        "labels": ["World", "Sports", "Business", "Tech"],
        "default": "World",
        "dataset": "dataset/news_test.jsonl",
        "csv": "results/news_random.csv",
        "task_header": "Please perform News Topic Classification task. Given the headline/article, assign a topic label from ['World', 'Sports', 'Business', 'Tech']. Return label only without any other text.",
        "topics": {
            "World": ["international politics", "war/conflict", "diplomacy", "elections", "human rights"],
            "Sports": ["football", "basketball", "olympics", "tennis", "soccer"],
            "Business": ["stock market", "mergers", "economy", "earnings", "banking"],
            "Tech": ["software", "hardware", "internet", "AI", "startups"],
        },
        "style_rules": {
            "World": "- Label MUST be exactly: World.\n- Write about international news, politics, global events.\n",
            "Sports": "- Label MUST be exactly: Sports.\n- Write about athletic competitions, teams, players, games.\n",
            "Business": "- Label MUST be exactly: Business.\n- Write about companies, markets, finance, economy.\n",
            "Tech": "- Label MUST be exactly: Tech.\n- Write about technology, software, gadgets, science.\n",
        },
        "para_prompt": "You paraphrase news headlines while preserving the original topic and meaning.\nProduce exactly {n} diverse paraphrases. Keep the topic category identical.\nOutput ONLY the paraphrases, one per line.\n\nHeadline:\n{s}",
    },
    "trec": {
        "labels": ["Abbreviation", "Entity", "Description", "Human", "Location", "Number"],
        "default": "Entity",
        "dataset": "dataset/trec_test.jsonl",
        "csv": "results/trec_random.csv",
        "task_header": "Please perform Question Type Classification task. Given the question, assign a type label from ['Abbreviation', 'Entity', 'Description', 'Human', 'Location', 'Number']. Return label only without any other text.",
        "topics": {
            "Abbreviation": ["acronym meaning", "abbreviation expansion"],
            "Entity": ["product", "animal", "color", "invention", "food"],
            "Description": ["definition", "manner", "reason"],
            "Human": ["person name", "inventor", "author", "discoverer"],
            "Location": ["city", "country", "mountain", "address"],
            "Number": ["date", "count", "distance", "money", "percentage"],
        },
        "style_rules": {
            "Abbreviation": "- Label MUST be exactly: Abbreviation.\n- Ask about what an abbreviation/acronym stands for.\n",
            "Entity": "- Label MUST be exactly: Entity.\n- Ask about a thing, object, animal, event, etc.\n",
            "Description": "- Label MUST be exactly: Description.\n- Ask for a definition, explanation, or reason.\n",
            "Human": "- Label MUST be exactly: Human.\n- Ask about a person's identity or name.\n",
            "Location": "- Label MUST be exactly: Location.\n- Ask about a place, city, country, or address.\n",
            "Number": "- Label MUST be exactly: Number.\n- Ask about a quantity, date, distance, or measurement.\n",
        },
        "para_prompt": "You paraphrase questions while preserving the question type and intent.\nProduce exactly {n} diverse paraphrases. Keep the answer type requirement identical.\nOutput ONLY the paraphrases, one per line.\n\nQuestion:\n{s}",
    },
}

EXAMPLE_HEADER_RE = re.compile(r"^Example\s*(\d+)\s*:\s*", re.I | re.M)
SENTENCE_LINE_RE = re.compile(r'^\s*Sentence\s*:\s*"(.*?)"\s*$', re.I | re.M | re.S)


def build_cls_prefix_suffix(task: str, examples_block: str):
    hdr = TASK_CONFIGS[task]["task_header"]
    return f'{hdr}\n\n{examples_block}\n\nSentence: "', '"\nLabel:'


def _make_para_prompt(task: str):
    tpl = TASK_CONFIGS[task]["para_prompt"]
    return lambda s, n: tpl.format(n=n, s=s.strip())


def _build_label_messages(task: str, num_ex: int, label: str, counts: List[int]):
    cfg = TASK_CONFIGS[task]
    topics = cfg["topics"][label]
    plan = "\n".join(
        f"- Example{i+1}: write exactly {counts[i]} sentence{'s' if counts[i]>1 else ''}; topic: {topics[i % len(topics)]}."
        for i in range(num_ex)
    )
    return [
        {"role": "system", "content": "You are a data generator that writes high-quality in-context learning examples for classification."},
        {"role": "user", "content": f"""Create exactly {num_ex} training examples in THIS STRICT format only:

Example1:
Sentence: "<text>"
Label: {label}
...
Example{num_ex}:
Sentence: "<text>"
Label: {label}

Diversity plan (MUST FOLLOW):
{plan}

Rules:
- Each example's "Sentence" must contain exactly the number of sentences specified above (1–3).
- Keep sentences concise: typically 3–14 words each.
- Use only ASCII characters. Do NOT include double quotes inside the text.
- Do NOT wrap output in Markdown/code fences.
{cfg['style_rules'][label]}
- Output ONLY the examples in the exact format above; no extra text.
"""}
    ]


def generate_examples_balanced(task: str, tokenizer, model_or_llm, k: int, seed: Optional[int], max_new_tokens: Optional[int], use_vllm: bool):
    cfg = TASK_CONFIGS[task]
    labels = cfg["labels"]
    per_label = [k // len(labels)] * len(labels)
    for i in range(k % len(labels)):
        per_label[i] += 1

    counts = sentence_count_plan(k)
    max_tok = max_new_tokens if max_new_tokens and max_new_tokens > 0 else max(64, 48 * max(per_label))

    combined = []
    for idx, label in enumerate(labels):
        lk = per_label[idx]
        if lk == 0:
            continue
        lc = counts[:lk] + [1] * max(0, lk - len(counts[:lk]))
        counts = counts[lk:]
        msgs = _build_label_messages(task, lk, label, lc)
        if use_vllm:
            prompt = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            text = vllm_generate_texts(model_or_llm, [prompt], temperature=0.6, top_p=0.9, max_tokens=max_tok, seed=seed + idx if seed else None)[0]
        else:
            text = gen_with_messages_transformers(tokenizer, model_or_llm, msgs, temperature=0.6, top_p=0.9, max_new_tokens=max_tok, seed=seed + idx if seed else None)
        parsed = parse_examples_from_text(text, lk, labels, example_header_re=EXAMPLE_HEADER_RE, sentence_line_re=SENTENCE_LINE_RE)
        combined.extend(ex for ex in parsed if ex["label"].lower() == label.lower())

    rng = random.Random(seed + 99991 if seed else None)
    rng.shuffle(combined)
    examples = combined[:k]
    return format_examples_for_evaluator(examples), examples


def run_single_spec(task, args, seed, n_para, k_examples, num_runs, gen_tokenizer, gen_llm, eval_llm, model):
    cfg = TASK_CONFIGS[task]
    set_global_seed(seed)
    used_vllm, fell_back = gen_llm is not None, model is not None

    items = prepare_items(args.eval_dataset_path, args.limit, cfg["labels"])
    golds = [sol for _, sol in items]
    N = len(items)
    print(f"[INFO] Prepared {N} item(s) from: {args.eval_dataset_path}")

    aggregated = [[] for _ in range(N)]
    para_fn = _make_para_prompt(task)
    run_accs = []

    for run_idx in range(num_runs):
        rs = seed + 10000 * run_idx
        print(f"\n{'='*80}\n[RUN {run_idx+1}/{num_runs}] run_seed={rs}\n{'='*80}")

        model_or = gen_llm or model
        examples_block, _ = generate_examples_balanced(task, gen_tokenizer, model_or, k_examples, rs, args.max_new_tokens_proposer, gen_llm is not None)
        print(f"[INFO] Generated examples:\n{examples_block}\n")

        pfx, sfx = build_cls_prefix_suffix(task, examples_block)
        eval_args = dict(label_prefix=pfx, label_suffix=sfx, items=items, paraphrases_per_item=n_para,
                         base_seed=rs, para_temperature=args.para_temperature, para_top_p=args.para_top_p,
                         para_max_new=args.para_max_new, batch_size=args.batch_size, report_every=args.report_every,
                         paraphrase_prompt_fn=para_fn, allowed_labels=cfg["labels"])

        if eval_llm:
            top1, _, _ = eval_run_vllm_collect(eval_llm, generator_llm=gen_llm, evaluator_llm=eval_llm, **eval_args)
        else:
            top1, _, _ = eval_run_transformers_collect(gen_tokenizer, model, **eval_args)

        run_ok = sum(
            1 for i in range(N)
            if top1[i] and majority_vote_generic([p for p in top1[i] if p in cfg["labels"]], cfg["labels"],
                                                  tie_breaker=top1[i][0], default_label=cfg["default"]) == golds[i]
        )
        run_accs.append(run_ok / N if N else 0.0)
        print(f"[RUN {run_idx+1}] Accuracy: {run_accs[-1]:.4f} ({run_ok}/{N})")
        for i in range(N):
            aggregated[i].extend(top1[i])

    correct = sum(
        1 for i in range(N)
        if majority_vote_generic([p for p in aggregated[i] if p in cfg["labels"]], cfg["labels"],
                                  tie_breaker=aggregated[i][0] if aggregated[i] else None,
                                  default_label=cfg["default"]) == golds[i]
    )
    acc = correct / N if N else 0.0
    print(f"\n{'='*80}\n[FINAL] Accuracy: {acc:.4f} ({correct}/{N})")
    print(f"[RUN SCORES] {', '.join(f'{a:.4f}' for a in run_accs)}\n{'='*80}")
    return correct, N, acc, used_vllm, fell_back


def main():
    import argparse, sys
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--task", required=True, choices=list(TASK_CONFIGS.keys()))
    known, rest = pre.parse_known_args()
    task = known.task
    sys.argv = [sys.argv[0]] + rest

    args = parse_args()
    cfg = TASK_CONFIGS[task]
    if args.eval_dataset_path == "dataset/sst2_test.jsonl":
        args.eval_dataset_path = cfg["dataset"]
    if args.random_results_csv == "results/sst2_random.csv":
        args.random_results_csv = cfg["csv"]
    if not args.run_random_search:
        print(f"Re-run with --run-random-search. Task={task}, dataset={args.eval_dataset_path}, csv={args.random_results_csv}")
        return
    n_list = _parse_int_list(args.grid_n)
    k_list = _parse_int_list(args.grid_k)
    runs_list = _parse_int_list(args.grid_runs)
    seeds = order_seeds(_parse_int_list(args.grid_seeds))
    print(f"[RANDOM] Task={task} | n={n_list} | k={k_list} | runs={runs_list} | seeds={seeds}")
    print(f"[RANDOM] Dataset: {args.eval_dataset_path} | Limit: {args.limit or 'ALL'}")

    gen_llm, eval_llm, model, gen_tokenizer = None, None, None, None

    try:
        if not args.use_vllm:
            raise ImportError("vLLM disabled by flag.")
        gp, ep = args.generator_model_path, args.evaluator_model_path
        if gp and ep:
            if gp == ep:
                util = min(args.gpu_mem_util * 1.75, 0.92)
                print(f"[INIT] Single vLLM model (shared): {gp} util={util}")
                gen_tokenizer, gen_llm = init_qwen_vllm(gp, args.tp, util, args.max_model_len, args.quantization)
                eval_llm = gen_llm
            else:
                gu = args.generator_gpu_mem_util or args.gpu_mem_util
                eu = args.evaluator_gpu_mem_util or args.gpu_mem_util
                print(f"[INIT] Generator vLLM: {gp} util={gu}")
                gen_tokenizer, gen_llm = init_qwen_vllm(gp, args.tp, gu, args.max_model_len, args.quantization)
                print(f"[INIT] Evaluator vLLM: {ep} util={eu}")
                _, eval_llm = init_qwen_vllm(ep, args.tp, eu, args.max_model_len, args.quantization)
        else:
            print(f"[INIT] Single vLLM model: {args.model_path}")
            gen_tokenizer, gen_llm = init_qwen_vllm(args.model_path, args.tp, args.gpu_mem_util, args.max_model_len, args.quantization)
            eval_llm = gen_llm
        print("[INIT] vLLM ready.")
    except Exception as e:
        msg = str(e)
        print(f"\n{'='*80}\n[ERROR] vLLM init failed: {msg}\n{'='*80}")
        if "out of memory" in msg.lower():
            print(f"[HINT] Reduce --gpu-mem-util (current: {args.gpu_mem_util})")
        print("[CRITICAL] Not falling back to Transformers. Fix config and retry.")
        raise RuntimeError("vLLM init failed") from e

    rng = random.Random(args.random_hparam_seed)
    specs = set()
    for _ in range(args.random_specs * 20):
        if len(specs) >= args.random_specs:
            break
        specs.add((rng.choice(n_list), rng.choice(k_list), rng.choice(runs_list)))
    print(f"[RANDOM] Unique specs: {len(specs)}")

    results = []
    for sid, (n, k, r) in enumerate(specs, 1):
        print(f"\n{'#'*100}\n[SPEC #{sid}] n={n} | k={k} | runs={r}\n{'#'*100}")
        early = False
        for seed in seeds:
            t0 = time.time()
            correct, total, acc, used_vllm, fell_back = run_single_spec(
                task, args, seed, n, k, r, gen_tokenizer, gen_llm, eval_llm, model)
            dt = time.time() - t0
            row = {"task": task, "spec_id": sid, "seed": seed, "n_paraphrases": n, "k_examples": k,
                   "runs": r, "acc": acc, "correct": correct, "total": total, "elapsed_sec": round(dt, 3),
                   "use_vllm": int(used_vllm), "fell_back_to_hf": int(fell_back), "early_stop": 0}
            results.append(row)
            append_csv_row(args.random_results_csv, row)
            print(f"[RANDOM] spec#{sid} seed={seed} | n={n} k={k} runs={r} | acc={acc:.4f} ({correct}/{total}) | {dt:.2f}s")
            if seed == 1 and acc < args.early_stop_threshold:
                print(f"[EARLY-STOP] spec#{sid}: acc={acc:.4f} < {args.early_stop_threshold:.4f}")
                row["early_stop"] = 1
                append_csv_row(args.random_results_csv, row)
                early = True
                break
        if early:
            continue

    if results:
        avg_acc = sum(r["acc"] for r in results) / len(results)
        print(f"\n{'='*80}\n[SUMMARY] {task.upper()} avg={avg_acc*100:.2f}% top={max(r['acc'] for r in results)*100:.2f}%\n{'='*80}")
    for i, r in enumerate(sorted(results, key=lambda x: (-x["acc"], x["elapsed_sec"]))[:2], 1):
        print(f"  #{i}: spec={r['spec_id']} seed={r['seed']} | n={r['n_paraphrases']} k={r['k_examples']} runs={r['runs']} | acc={r['acc']:.4f}")


if __name__ == "__main__":
    main()
