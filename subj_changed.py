import json, os, random, re, time

for k, v in [("TOKENIZERS_PARALLELISM", "true"), ("PYTORCH_ENABLE_FLASH_SDP", "0"),
             ("PYTORCH_ENABLE_MEM_EFFICIENT_SDP", "0"), ("PYTORCH_ENABLE_SOMA", "0")]:
    os.environ.setdefault(k, v)

from utils import (
    _parse_int_list,
    append_csv_row,
    eval_run_transformers_collect,
    eval_run_vllm_collect,
    format_examples_for_evaluator,
    gen_with_messages_transformers,
    init_qwen_transformers,
    init_qwen_vllm,
    order_seeds,
    parse_args,
    parse_examples_from_text,
    prepare_items,
    sentence_count_plan,
    set_global_seed,
    vllm_generate_texts,
)

ALLOWED_LABELS = ["subjective", "objective", "unsure"]
FINAL_LABELS = ["subjective", "objective"]

EXAMPLE_HEADER_PATTERN = re.compile(r"^Example\s*(\d+)\s*:\s*", re.IGNORECASE | re.MULTILINE)
SENTENCE_LINE_PATTERN = re.compile(r'^\s*Sentence\s*:\s*"(.*?)"\s*$', re.IGNORECASE | re.MULTILINE | re.DOTALL)

CLASSIFICATION_INSTRUCTION = (
    "You are an expert text classifier for subjectivity analysis.\n\n"
    "DEFINITIONS:\n"
    "- subjective: Personal opinion, evaluation, judgment, or emotional expression\n"
    "- objective: Factual statement, description of events, or verifiable information\n"
    "- unsure: Use ONLY when genuinely ambiguous between subjective and objective\n\n"
    "CRITICAL DISTINCTION:\n"
    "Describing events or facts (even emotional ones) is OBJECTIVE.\n"
    "Expressing personal opinions or judgments is SUBJECTIVE.\n\n"
    "OUTPUT: Return exactly one label (subjective/objective/unsure)."
)

SUBJECTIVE_TOPICS = [
    "opinion on acting", "judgment of direction", "critique of dialogue",
    "assessment of cinematography", "evaluation of pacing", "review of soundtrack",
    "opinion on visual effects", "judgment of set design", "assessment of tone",
    "opinion on themes", "evaluation of casting", "critique of character development",
    "assessment of humor", "emotional reaction",
]

OBJECTIVE_TOPICS = [
    "plot summary", "character actions", "story events", "character relationships",
    "plot twist description", "setting description", "character background",
    "story conflict", "character motivation", "narrative arc", "scene description",
    "character introduction", "plot setup", "story resolution",
]


def compute_majority_vote_with_fallback(predictions, fallback_label="objective", threshold=0.6):
    # 3-label voting with unsure fallback
    votes = {"subjective": 0, "objective": 0, "unsure": 0}
    for p in predictions:
        if p in votes:
            votes[p] += 1

    total = sum(votes.values())
    if total == 0:
        return fallback_label, votes, False

    winner = max(votes, key=votes.get)
    confidence = votes[winner] / total

    if winner == "unsure":
        definite = {k: v for k, v in votes.items() if k != "unsure"}
        if definite and max(definite.values()) > 0:
            if votes["subjective"] == votes["objective"]:
                return fallback_label, votes, True
            best = max(definite, key=definite.get)
            return best, votes, confidence >= threshold
        return fallback_label, votes, True

    return winner, votes, False


def build_classification_prompt_parts(examples_block):
    prefix = f'{CLASSIFICATION_INSTRUCTION}\n\n{examples_block}\n\nSentence: "'
    return prefix, '"\nLabel:'


def create_paraphrase_prompt(sentence, count):
    return (
        "You paraphrase sentences while preserving the original MEANING and SUBJECTIVITY TYPE "
        "(whether it expresses opinion or describes facts).\n"
        f"Produce exactly {count} diverse paraphrases of the sentence below. "
        "Keep the subjectivity classification identical. "
        "Output ONLY the paraphrases, one per line, without numbering or commentary.\n\n"
        f"Sentence:\n{sentence.strip()}"
    )


def build_example_generation_messages(example_count, target_label, sentence_counts, topic_list):
    diversity_plan = "\n".join(
        f"- Example{i+1}: {sentence_counts[i]} sentence{'s' if sentence_counts[i] > 1 else ''}; topic: {topic_list[i % len(topic_list)]}."
        for i in range(example_count)
    )

    if target_label == "subjective":
        label_rules = (
            "- Label MUST be: subjective\n"
            "- Write the reviewer's OPINION about the movie\n"
            "- Use evaluative words: great, terrible, brilliant, boring, masterpiece, disappointing\n"
            "- Express assessment of quality or personal reaction to the film"
        )
    else:
        label_rules = (
            "- Label MUST be: objective\n"
            "- Write factual descriptions of plot events, character actions, story elements\n"
            "- Plot summaries are OBJECTIVE even when describing emotional story events\n"
            "- Describe WHAT HAPPENS, not your opinion about it"
        )

    user_message = f"""Create exactly {example_count} training examples in this exact format:

Example1:
Sentence: "<text>"
Label: {target_label}

Example2:
Sentence: "<text>"
Label: {target_label}

...

Diversity plan (follow exactly):
{diversity_plan}

Rules:
- Each Sentence must contain exactly the specified number of sentences (1-3)
- Keep sentences concise: 3-14 words each
- Include variety: at least one short (<=5 words) and one longer (10-14 words)
- Use only ASCII characters, no quotes inside the text
- Put multiple sentences in the same quotes separated by spaces
- No markdown or code fences
{label_rules}
- Output ONLY the examples, no extra text"""

    return [
        {"role": "system", "content": "You generate training examples for subjectivity classification. "
         "SUBJECTIVE = opinions about the movie. OBJECTIVE = factual plot/story descriptions."},
        {"role": "user", "content": user_message},
    ]


def distribute_sentence_counts(total_examples):
    subj_count = total_examples // 2
    obj_count = total_examples - subj_count
    all_counts = sentence_count_plan(total_examples)
    subj_counts = [all_counts[i] for i in range(0, total_examples, 2)][:subj_count]
    obj_counts = [all_counts[i] for i in range(1, total_examples, 2)][:obj_count]
    while len(subj_counts) < subj_count:
        subj_counts.append(all_counts[-1])
    while len(obj_counts) < obj_count:
        obj_counts.append(all_counts[-1])
    return subj_counts, obj_counts


def parse_and_filter_examples(raw_text, expected_count, required_label):
    parsed = parse_examples_from_text(
        raw_text, expected_count, ALLOWED_LABELS,
        example_header_re=EXAMPLE_HEADER_PATTERN, sentence_line_re=SENTENCE_LINE_PATTERN
    )
    return [ex for ex in parsed if ex["label"].lower() == required_label]


def generate_balanced_examples_transformers(tokenizer, model, total_examples, seed=None, max_new_tokens=None):
    # Generate balanced subj/obj examples with retry
    subj_count, obj_count = total_examples // 2, total_examples - total_examples // 2
    subj_sc, obj_sc = distribute_sentence_counts(total_examples)
    max_new_tokens = max_new_tokens or max(64, 48 * max(subj_count, obj_count))
    params = {"temperature": 0.6, "top_p": 0.9, "max_new_tokens": max_new_tokens}

    subj_msgs = build_example_generation_messages(subj_count, "subjective", subj_sc, SUBJECTIVE_TOPICS)
    obj_msgs = build_example_generation_messages(obj_count, "objective", obj_sc, OBJECTIVE_TOPICS)

    subj_raw = gen_with_messages_transformers(tokenizer, model, subj_msgs, seed=seed, **params)
    obj_raw = gen_with_messages_transformers(tokenizer, model, obj_msgs, seed=seed + 1 if seed else None, **params)

    combined = parse_and_filter_examples(subj_raw, subj_count, "subjective") + \
               parse_and_filter_examples(obj_raw, obj_count, "objective")

    missing_subj = subj_count - sum(1 for ex in combined if ex["label"].lower() == "subjective")
    missing_obj = obj_count - sum(1 for ex in combined if ex["label"].lower() == "objective")

    if missing_subj > 0:
        retry_msgs = build_example_generation_messages(missing_subj, "subjective", subj_sc[:missing_subj], SUBJECTIVE_TOPICS)
        retry_raw = gen_with_messages_transformers(tokenizer, model, retry_msgs, seed=seed + 2 if seed else None, **params)
        combined += parse_and_filter_examples(retry_raw, missing_subj, "subjective")

    if missing_obj > 0:
        retry_msgs = build_example_generation_messages(missing_obj, "objective", obj_sc[:missing_obj], OBJECTIVE_TOPICS)
        retry_raw = gen_with_messages_transformers(tokenizer, model, retry_msgs, seed=seed + 3 if seed else None, **params)
        combined += parse_and_filter_examples(retry_raw, missing_obj, "objective")

    rng = random.Random(seed + 99991 if seed else None)
    rng.shuffle(combined)
    final = combined[:total_examples]
    return format_examples_for_evaluator(final), final


def generate_balanced_examples_vllm(tokenizer, llm, total_examples, seed=None, max_new_tokens=None):
    # Generate balanced subj/obj examples with retry (vLLM)
    subj_count, obj_count = total_examples // 2, total_examples - total_examples // 2
    subj_sc, obj_sc = distribute_sentence_counts(total_examples)
    max_new_tokens = max_new_tokens or max(64, 48 * max(subj_count, obj_count))
    params = {"temperature": 0.6, "top_p": 0.9, "max_tokens": max_new_tokens}

    subj_msgs = build_example_generation_messages(subj_count, "subjective", subj_sc, SUBJECTIVE_TOPICS)
    obj_msgs = build_example_generation_messages(obj_count, "objective", obj_sc, OBJECTIVE_TOPICS)
    subj_prompt = tokenizer.apply_chat_template(subj_msgs, tokenize=False, add_generation_prompt=True)
    obj_prompt = tokenizer.apply_chat_template(obj_msgs, tokenize=False, add_generation_prompt=True)

    subj_raw = vllm_generate_texts(llm, [subj_prompt], seed=seed, **params)[0]
    obj_raw = vllm_generate_texts(llm, [obj_prompt], seed=seed + 1 if seed else None, **params)[0]

    combined = parse_and_filter_examples(subj_raw, subj_count, "subjective") + \
               parse_and_filter_examples(obj_raw, obj_count, "objective")

    missing_subj = subj_count - sum(1 for ex in combined if ex["label"].lower() == "subjective")
    missing_obj = obj_count - sum(1 for ex in combined if ex["label"].lower() == "objective")

    if missing_subj > 0:
        retry_msgs = build_example_generation_messages(missing_subj, "subjective", subj_sc[:missing_subj], SUBJECTIVE_TOPICS)
        retry_prompt = tokenizer.apply_chat_template(retry_msgs, tokenize=False, add_generation_prompt=True)
        retry_raw = vllm_generate_texts(llm, [retry_prompt], seed=seed + 2 if seed else None, **params)[0]
        combined += parse_and_filter_examples(retry_raw, missing_subj, "subjective")

    if missing_obj > 0:
        retry_msgs = build_example_generation_messages(missing_obj, "objective", obj_sc[:missing_obj], OBJECTIVE_TOPICS)
        retry_prompt = tokenizer.apply_chat_template(retry_msgs, tokenize=False, add_generation_prompt=True)
        retry_raw = vllm_generate_texts(llm, [retry_prompt], seed=seed + 3 if seed else None, **params)[0]
        combined += parse_and_filter_examples(retry_raw, missing_obj, "objective")

    rng = random.Random(seed + 99991 if seed else None)
    rng.shuffle(combined)
    final = combined[:total_examples]
    return format_examples_for_evaluator(final), final





def generate_examples_for_run(gen_tokenizer, model, gen_llm, use_vllm, example_count, seed, max_new_tokens):
    if use_vllm:
        return generate_balanced_examples_vllm(gen_tokenizer, gen_llm, example_count, seed=seed, max_new_tokens=max_new_tokens)
    return generate_balanced_examples_transformers(gen_tokenizer, model, example_count, seed=seed, max_new_tokens=max_new_tokens)


def evaluate_with_paraphrases(gen_tokenizer, model, eval_llm, gen_llm, use_vllm, prefix, suffix, items, paraphrase_count, run_seed, args):
    params = {
        "label_prefix": prefix, "label_suffix": suffix, "items": items,
        "paraphrases_per_item": paraphrase_count, "base_seed": run_seed,
        "para_temperature": args.para_temperature, "para_top_p": args.para_top_p,
        "para_max_new": args.para_max_new, "batch_size": args.batch_size,
        "report_every": args.report_every, "paraphrase_prompt_fn": create_paraphrase_prompt,
        "allowed_labels": ALLOWED_LABELS,
    }
    if use_vllm:
        return eval_run_vllm_collect(eval_llm, generator_llm=gen_llm, evaluator_llm=eval_llm, **params)
    return eval_run_transformers_collect(gen_tokenizer, model, **params)


def write_detailed_logs(log_paths, items, aggregated_predictions, aggregated_paraphrases, run_details, args, config):
    # Writes detailed logs for debugging and analysis
    log_long, log_short, log_prompts = log_paths
    n = len(items)

    with open(log_prompts, "w") as f:
        json.dump({"config": config, "run_details": run_details}, f, ensure_ascii=False, indent=2)

    correct_count = fallback_count = 0
    positive_examples, negative_examples = [], []

    with open(log_long, "w") as lf, open(log_short, "w") as sf:
        sf.write(f"SUBJECTIVITY CLASSIFICATION | seed={config['seed']} n={config['n_paraphrases']} k={config['k_examples']} runs={config['num_runs']}\n")
        sf.write(f"Dataset: {args.eval_dataset_path}\n{'='*80}\n\n")

        for i in range(n):
            sentence, gold = items[i]
            valid_preds = [p for p in aggregated_predictions[i] if p in ALLOWED_LABELS]

            if valid_preds:
                pred, votes, used_fb = compute_majority_vote_with_fallback(valid_preds)
                fallback_count += int(used_fb)
            else:
                pred, votes, used_fb = "objective", {"subjective": 0, "objective": 0, "unsure": 0}, False

            is_correct = pred == gold
            correct_count += int(is_correct)
            total = len(valid_preds) or 1
            pcts = {k: votes[k] / total * 100 for k in votes}
            conf = max(pcts["subjective"], pcts["objective"])

            rec = {
                "id": i, "sentence": sentence, "gold": gold, "pred": pred, "correct": is_correct,
                "used_fallback": used_fb, "votes": {**votes, "total": len(valid_preds)},
                "percentages": {**pcts, "confidence": conf},
                "all_preds": valid_preds, "paraphrases": aggregated_paraphrases[i],
            }
            (positive_examples if is_correct else negative_examples).append(rec)

            lf.write(json.dumps({
                "id": i, "sentence": sentence, "gold": gold, "pred": pred, "correct": is_correct,
                "used_unsure_fallback": used_fb,
                "votes": {"subj": votes["subjective"], "obj": votes["objective"], "undef": votes["unsure"],
                          "total": len(valid_preds), "subj%": round(pcts["subjective"], 1),
                          "obj%": round(pcts["objective"], 1), "undef%": round(pcts["unsure"], 1), "conf%": round(conf, 1)},
                "all_preds": valid_preds, "paraphrases": aggregated_paraphrases[i],
                "paraphrase_prompt": create_paraphrase_prompt(sentence, config["n_paraphrases"]),
            }, ensure_ascii=False) + "\n")

            fb_mark = " [FALLBACK]" if used_fb else ""
            sym = "+" if is_correct else "X"
            sf.write(f"[{i+1}/{n}] {sym} | Gold:{gold:11s} Pred:{pred:11s}{fb_mark} | "
                     f"Votes: SUBJ={votes['subjective']:2d}({pcts['subjective']:4.1f}%) OBJ={votes['objective']:2d}({pcts['objective']:4.1f}%) "
                     f"UNDEF={votes['unsure']:2d}({pcts['unsure']:4.1f}%) Conf={conf:4.1f}%\n")
            sf.write(f"     {sentence[:90]}{'...' if len(sentence) > 90 else ''}\n")
            if len(valid_preds) <= 15:
                sf.write(f"     Predictions: {', '.join(valid_preds)}\n")
            sf.write("\n")

    return correct_count, n, fallback_count, positive_examples, negative_examples


def print_analysis_report(run_details, positive_examples, negative_examples, correct_count, total_count, example_count, num_runs):
    # Prints summary of classification results
    print("\n" + "=" * 100 + "\nDETAILED CLASSIFICATION ANALYSIS\n" + "=" * 100)
    print(f"\n{'-'*80}\nGENERATED EXAMPLES (k={example_count}, runs={num_runs}):\n{'-'*80}")
    for r in run_details:
        print(f"\n[RUN {r['run_idx']}] seed={r['run_seed']}\nExamples used for classification:")
        for i, ex in enumerate(r["generated_examples"], 1):
            print(f'  {i}. [{ex["label"].upper():10s}] "{ex["sentence"]}"')

    def print_example(ex, kind):
        icon = "[+]" if kind == "POSITIVE" else "[-]"
        print(f"\n{icon} {kind} EXAMPLE\n" + "-" * 60)
        print(f"ID: {ex['id']}\nSentence: \"{ex['sentence']}\"\nGold: {ex['gold']}\nPred: {ex['pred']}\nResult: {'CORRECT' if ex['correct'] else 'INCORRECT'}")
        if ex["used_fallback"]:
            print("WARNING: Used fallback (unsure dominated)")
        print(f"\nVOTING:\n  Subjective: {ex['votes']['subjective']:3d} ({ex['percentages']['subjective']:5.1f}%)")
        print(f"  Objective:  {ex['votes']['objective']:3d} ({ex['percentages']['objective']:5.1f}%)")
        print(f"  Unsure:     {ex['votes']['unsure']:3d} ({ex['percentages']['unsure']:5.1f}%)")
        print(f"  TOTAL:      {ex['votes']['total']:3d}\n  Confidence: {ex['percentages']['confidence']:5.1f}%")
        print("\nGENERATED PARAPHRASES:")
        for i, p in enumerate(ex["paraphrases"], 1):
            print(f'  {i}. "{p}"')
        print("\nALL PREDICTIONS:")
        preds_str = ", ".join(ex["all_preds"])
        if len(preds_str) > 100:
            for c in range(0, len(ex["all_preds"]), 10):
                print(f"  {', '.join(ex['all_preds'][c:c+10])}")
        else:
            print(f"  {preds_str}")

    print(f"\n{'='*80}\nPOSITIVE EXAMPLE (correctly classified)\n{'='*80}")
    if positive_examples:
        print_example(sorted(positive_examples, key=lambda x: x["percentages"]["confidence"], reverse=True)[0], "POSITIVE")
    else:
        print("No positive examples!")

    print(f"\n{'='*80}\nNEGATIVE EXAMPLE (incorrectly classified)\n{'='*80}")
    if negative_examples:
        print_example(sorted(negative_examples, key=lambda x: x["percentages"]["confidence"], reverse=True)[0], "NEGATIVE")
    else:
        print("No negative examples - all classifications correct!")

    print(f"\n{'='*100}\nSUMMARY: {correct_count}/{total_count} correct ({correct_count/total_count*100:.1f}%)")
    print(f"Positive examples: {len(positive_examples)}\nNegative examples: {len(negative_examples)}\n{'='*100}\n")


def run_single_specification(args, seed, paraphrase_count, example_count, num_runs, gen_tokenizer, gen_llm, eval_llm, model):
    # Run evaluation with given hyperparameters
    set_global_seed(seed)
    used_vllm, fell_back = gen_llm is not None, model is not None

    items = prepare_items(args.eval_dataset_path, args.limit, FINAL_LABELS)
    n = len(items)
    golds = [item[1] for item in items]
    print(f"[INFO] Prepared {n} item(s) from: {args.eval_dataset_path}")

    agg_preds = [[] for _ in range(n)]
    agg_paras = [[] for _ in range(n)]
    run_details, run_accs = [], []

    for run_idx in range(num_runs):
        run_seed = seed + 10000 * run_idx
        print(f"\n{'='*80}\n[RUN {run_idx+1}/{num_runs}] run_seed={run_seed}\n{'='*80}")

        examples_block, gen_examples = generate_examples_for_run(
            gen_tokenizer, model, gen_llm, used_vllm, example_count, run_seed, args.max_new_tokens_proposer)
        print(f"[INFO] Generated examples for evaluator:\n\n{examples_block}\n")

        subj_c, obj_c = example_count // 2, example_count - example_count // 2
        subj_sc, obj_sc = distribute_sentence_counts(example_count)
        subj_msgs = build_example_generation_messages(subj_c, "subjective", subj_sc, SUBJECTIVE_TOPICS)
        obj_msgs = build_example_generation_messages(obj_c, "objective", obj_sc, OBJECTIVE_TOPICS)

        shuffle_rng = random.Random(run_seed + 77777)
        shuffled = gen_examples.copy()
        shuffle_rng.shuffle(shuffled)

        run_details.append({
            "run_idx": run_idx + 1, "run_seed": run_seed,
            "example_generation_prompts": {
                "subjective_prompt": {"messages": subj_msgs, "count": subj_c},
                "objective_prompt": {"messages": obj_msgs, "count": obj_c},
            },
            "generated_examples": gen_examples, "examples_block": examples_block,
            "shuffled_examples": shuffled, "shuffled_block": format_examples_for_evaluator(shuffled),
        })

        prefix, suffix = build_classification_prompt_parts(examples_block)
        preds, _, paras = evaluate_with_paraphrases(
            gen_tokenizer, model, eval_llm, gen_llm, used_vllm, prefix, suffix, items, paraphrase_count, run_seed, args)

        for i in range(n):
            agg_preds[i].extend(preds[i])
            agg_paras[i].extend(paras[i])

        run_correct = sum(
            1 for i in range(n)
            if (compute_majority_vote_with_fallback([p for p in preds[i] if p in ALLOWED_LABELS])[0]
                if any(p in ALLOWED_LABELS for p in preds[i]) else "objective") == golds[i]
        )
        run_acc = run_correct / n if n else 0.0
        run_accs.append(run_acc)
        print(f"[RUN {run_idx+1}] Accuracy: {run_acc:.4f} ({run_correct}/{n})")

    os.makedirs("detailed_logs", exist_ok=True)
    log_base = f"detailed_logs/subj_s{seed}_n{paraphrase_count}_k{example_count}_r{num_runs}"
    log_paths = (f"{log_base}_LONG.jsonl", f"{log_base}_SHORT.txt", f"{log_base}_PROMPTS.json")
    config = {"seed": seed, "n_paraphrases": paraphrase_count, "k_examples": example_count,
              "num_runs": num_runs, "dataset": args.eval_dataset_path}

    correct, total, fb_count, pos_ex, neg_ex = write_detailed_logs(
        log_paths, items, agg_preds, agg_paras, run_details, args, config)
    print_analysis_report(run_details, pos_ex, neg_ex, correct, total, example_count, num_runs)

    acc = correct / total if total else 0.0
    print(f"\n{'='*80}\n[FINAL] Accuracy: {acc:.4f} ({correct}/{total}) | Unsure fallbacks used: {fb_count}")
    print(f"[RUN SCORES] {', '.join(f'{a:.4f}' for a in run_accs)}")
    print(f"[LOGS] {log_paths[0]}\n       {log_paths[1]}\n       {log_paths[2]}\n{'='*80}")
    return correct, total, acc, used_vllm, fell_back


def main():
    args = parse_args()

    if args.eval_dataset_path == "dataset/sst2_test.jsonl":
        args.eval_dataset_path = "dataset/subj_test.jsonl"
    if args.random_results_csv == "results/sst2_random.csv":
        args.random_results_csv = "results/subj_random.csv"

    if not args.run_random_search:
        print("This script is configured for random search. Re-run with --run-random-search.")
        print(f"eval_dataset_path={args.eval_dataset_path} | results_csv={args.random_results_csv}")
        return

    n_vals = _parse_int_list(args.grid_n)
    k_vals = _parse_int_list(args.grid_k)
    run_vals = _parse_int_list(args.grid_runs)
    seeds = order_seeds(_parse_int_list(args.grid_seeds))

    print(f"[RANDOM] candidate n={n_vals} | k={k_vals} | runs={run_vals} | seeds={seeds}")
    print(f"[RANDOM] Dataset: {args.eval_dataset_path} | Limit: {args.limit or 'ALL'}")
    print(f"[RANDOM] use_vllm={args.use_vllm} | tp={args.tp} | util={args.gpu_mem_util} | max_len={args.max_model_len} | quant={args.quantization}")
    print(f"[RANDOM] specs_to_sample={args.random_specs} | hparam_rng_seed={args.random_hparam_seed}")
    print(f"[RANDOM] early_stop_threshold={args.early_stop_threshold:.4f} (applies to seed=1 only)")

    gen_llm = eval_llm = model = gen_tokenizer = None
    try:
        if args.use_vllm:
            if args.generator_model_path and args.evaluator_model_path:
                if args.generator_model_path == args.evaluator_model_path:
                    util = min(args.gpu_mem_util * 1.75, 0.92)
                    print(f"[INIT] Initializing Single vLLM model (shared): {args.generator_model_path} with util={util}...")
                    gen_tokenizer, gen_llm = init_qwen_vllm(args.generator_model_path, args.tp, util, args.max_model_len, args.quantization)
                    eval_llm = gen_llm
                else:
                    gen_util = args.generator_gpu_mem_util or args.gpu_mem_util
                    eval_util = args.evaluator_gpu_mem_util or args.gpu_mem_util
                    print(f"[INIT] Initializing Generator vLLM model: {args.generator_model_path} with util={gen_util}...")
                    gen_tokenizer, gen_llm = init_qwen_vllm(args.generator_model_path, args.tp, gen_util, args.max_model_len, args.quantization)
                    print(f"[INIT] Initializing Evaluator vLLM model: {args.evaluator_model_path} with util={eval_util}...")
                    _, eval_llm = init_qwen_vllm(args.evaluator_model_path, args.tp, eval_util, args.max_model_len, args.quantization)
            else:
                print(f"[INIT] Initializing Single vLLM model: {args.model_path}...")
                gen_tokenizer, gen_llm = init_qwen_vllm(args.model_path, args.tp, args.gpu_mem_util, args.max_model_len, args.quantization)
                eval_llm = gen_llm
            print("[INIT] vLLM models ready.")
        else:
            raise ImportError("vLLM disabled by flag.")
    except Exception as e:
        msg = str(e)
        print(f"\n{'='*80}\n[ERROR] vLLM initialization failed!\n[ERROR] Reason: {msg}\n{'='*80}")
        if "out of memory" in msg.lower():
            print(f"[SOLUTION] CUDA Out of Memory detected!\n  1. Reduce --gpu-mem-util (currently: {args.gpu_mem_util})")
            print(f"     Try: --gpu-mem-util 0.35 or --gpu-mem-util 0.30")
            print(f"  2. If using same gen/eval model, actual util = {min(args.gpu_mem_util * 1.75, 0.92):.2f}")
        print(f"\n[CRITICAL] Refusing to fall back to Transformers (60-100x slower!).")
        print(f"[CRITICAL] Please fix the vLLM configuration and re-run.\n{'='*80}\n")
        raise RuntimeError("vLLM initialization failed. Fix configuration and retry.") from e

    rng = random.Random(args.random_hparam_seed)
    specs = set()
    for _ in range(args.random_specs * 20):
        if len(specs) >= args.random_specs:
            break
        specs.add((rng.choice(n_vals), rng.choice(k_vals), rng.choice(run_vals)))
    print(f"[RANDOM] Unique specs to evaluate: {len(specs)} (requested {args.random_specs})")

    results = []
    for spec_id, (n, k, runs) in enumerate(specs, 1):
        print(f"\n{'#'*100}\n[SPEC #{spec_id}] n={n} | k={k} | runs={runs}\n{'#'*100}")
        early_stopped = False
        for seed in seeds:
            t0 = time.time()
            correct, total, acc, used_vllm, fell_back = run_single_specification(
                args, seed, n, k, runs, gen_tokenizer, gen_llm, eval_llm, model)
            elapsed = time.time() - t0

            row = {"spec_id": spec_id, "seed": seed, "n_paraphrases": n, "k_examples": k, "runs": runs,
                   "acc": acc, "correct": correct, "total": total, "elapsed_sec": round(elapsed, 3),
                   "use_vllm": int(used_vllm), "fell_back_to_hf": int(fell_back), "early_stop": 0}
            results.append(row)
            append_csv_row(args.random_results_csv, row)

            print(f"[RANDOM] spec#{spec_id} seed={seed} | n={n} k={k} runs={runs} | "
                  f"acc={acc:.4f} (correct={correct}/{total}) | time={elapsed:.2f}s | "
                  f"use_vllm={row['use_vllm']} fall_back={row['fell_back_to_hf']} | CSV -> {args.random_results_csv}")

            if seed == 1 and acc < args.early_stop_threshold:
                print(f"[RANDOM][EARLY-STOP] spec#{spec_id}: seed=1 acc={acc:.4f} < {args.early_stop_threshold:.4f} — skipping remaining seeds.")
                row["early_stop"] = 1
                append_csv_row(args.random_results_csv, row)
                early_stopped = True
                break
        if early_stopped:
            continue

    ranked = sorted(results, key=lambda r: (-r["acc"], r["elapsed_sec"]))
    print("\n[RANDOM] Top 2 runs (by accuracy desc, elapsed asc):")
    for i, r in enumerate(ranked[:2], 1):
        print(f"  #{i}: spec_id={r['spec_id']} seed={r['seed']} | n={r['n_paraphrases']} k={r['k_examples']} runs={r['runs']} | "
              f"acc={r['acc']:.4f} time={r['elapsed_sec']:.2f}s (correct={r['correct']}/{r['total']}) use_vllm={r['use_vllm']}")


if __name__ == "__main__":
    main()
