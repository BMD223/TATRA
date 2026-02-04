#!/usr/bin/env python3
import os, re, time, random, argparse
from typing import List, Dict, Tuple, Optional

os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")
os.environ.setdefault("PYTORCH_ENABLE_FLASH_SDP", "0")
os.environ.setdefault("PYTORCH_ENABLE_MEM_EFFICIENT_SDP", "0")
os.environ.setdefault("PYTORCH_ENABLE_SOMA", "0")

from utils import (
    append_csv_row, init_qwen_vllm, init_qwen_transformers,
    vllm_generate_texts, gen_with_messages_transformers,
    set_global_seed, load_jsonl,
)

TASK_CONFIGS = {
    "medqa": {
        "labels": ["A", "B", "C", "D", "E"],
        "default": "A",
        "dataset": "MathResources/medqa_test.jsonl",
        "csv": "results/medqa_random.csv",
        "task_header": (
            "Please use your domain knowledge in medical area to "
            "solve the questions. Return only answer 'A', 'B', 'C', 'D' or 'E' without "
            "any other text."
        ),
        "eval_max_tokens": 8,
        "accuracy_type": "exact_label",
        "topics": {
            "A": ["diagnosis", "treatment", "pharmacology", "pathology", "anatomy"],
            "B": ["diagnosis", "treatment", "pharmacology", "pathology", "anatomy"],
            "C": ["diagnosis", "treatment", "pharmacology", "pathology", "anatomy"],
            "D": ["diagnosis", "treatment", "pharmacology", "pathology", "anatomy"],
            "E": ["diagnosis", "treatment", "pharmacology", "pathology", "anatomy"],
        },
        "para_prompt": (
            "You rephrase medical questions while preserving the clinical scenario and answer options.\n"
            "Produce exactly {n} diverse rephrasings of the question below. Keep the medical meaning identical.\n"
            "Output ONLY the rephrasings, one per line, with no numbering, bullets, or extra commentary.\n\n"
            "Question:\n{s}"
        ),
    },
    "gsm8k": {
        "labels": None,
        "default": "0",
        "dataset": "MathResources/gsm8k_test.jsonl",
        "csv": "results/gsm8k_random.csv",
        "task_header": (
            "Solve the problem. You may show your reasoning. "
            "Conclude with a line that says 'Final Answer: <integer>'. "
            "Your answer will be considered correct if it includes the correct integer anywhere."
        ),
        "eval_max_tokens": 512,
        "accuracy_type": "contains_answer",  # 1 if correct answer is in prediction
        "topics": ["arithmetic", "algebra", "word problems", "percentages", "ratios"],
        "para_prompt": (
            "You rephrase math word problems while preserving the mathematical structure and answer.\n"
            "Produce exactly {n} diverse rephrasings of the problem below. Keep numbers and relationships identical.\n"
            "Output ONLY the rephrasings, one per line, with no numbering, bullets, or extra commentary.\n\n"
            "Problem:\n{s}"
        ),
    },
    "deepmath": {
        "labels": None,
        "default": "0",
        "dataset": "MathResources/deepmath_test.jsonl",
        "csv": "results/deepmath_random.csv",
        "task_header": (
            "Solve the problem. You may show your reasoning. "
            "Conclude with a line that says 'Final Answer: <answer>'. "
            "If the final answer is a number, output just this number; if it is mathematical expression, "
            "use TeX-style math typesetting (e.g., 1/2 as \\frac{1}{2}). "
            "Your answer will be considered correct if it includes the correct expression anywhere."
        ),
        "eval_max_tokens": 128,
        "accuracy_type": "contains_answer",
        "topics": ["number theory", "calculus", "linear algebra", "probability", "combinatorics"],
        "para_prompt": (
            "You rephrase advanced mathematics problems while preserving the mathematical content.\n"
            "Produce exactly {n} diverse rephrasings of the problem below. Keep notation and meaning identical.\n"
            "Output ONLY the rephrasings, one per line, with no numbering, bullets, or extra commentary.\n\n"
            "Problem:\n{s}"
        ),
    },
    "math500": {
        "labels": None,
        "default": "0",
        "dataset": "MathResources/math500_test.jsonl",
        "csv": "results/math500_random.csv",
        "task_header": (
            "Solve the problem. You may show your reasoning. "
            "Conclude with a line that says 'Final Answer: <answer>'. "
            "If the final answer is a number, output just this number; if it is mathematical expression, "
            "use TeX-style math typesetting (e.g., 1/2 as \\frac{1}{2}). "
            "Your answer will be considered correct if it includes the correct expression anywhere."
        ),
        "eval_max_tokens": 256,
        "accuracy_type": "contains_answer",
        "topics": ["algebra", "geometry", "number theory", "counting", "precalculus"],
        "para_prompt": (
            "You rephrase competition mathematics problems while preserving the mathematical content.\n"
            "Produce exactly {n} diverse rephrasings of the problem below. Keep notation and meaning identical.\n"
            "Output ONLY the rephrasings, one per line, with no numbering, bullets, or extra commentary.\n\n"
            "Problem:\n{s}"
        ),
    },
}

EXAMPLE_RE = re.compile(r"^Example\s*(\d+)\s*:", re.I | re.M)
QUESTION_RE = re.compile(r'^\s*Question\s*:\s*"?(.+?)"?\s*$', re.I | re.M | re.S)
ANSWER_RE = re.compile(r'^\s*Answer\s*:\s*(.+?)\s*$', re.I | re.M)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--task", required=True, choices=list(TASK_CONFIGS.keys()))
    p.add_argument("--model-path", default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--generator-model-path", default=None)
    p.add_argument("--evaluator-model-path", default=None)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--n", type=int, default=2)
    p.add_argument("--k", type=int, default=8)
    p.add_argument("--runs", type=int, default=15)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--para-temperature", type=float, default=0.9)
    p.add_argument("--para-top-p", type=float, default=0.95)
    p.add_argument("--para-max-new", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--use-vllm", type=int, default=1)
    p.add_argument("--tp", type=int, default=2)
    p.add_argument("--gpu-mem-util", type=float, default=0.60)
    p.add_argument("--generator-gpu-mem-util", type=float, default=None)
    p.add_argument("--evaluator-gpu-mem-util", type=float, default=None)
    p.add_argument("--max-model-len", type=int, default=4096)
    p.add_argument("--quantization", default=None)
    p.add_argument("--report-every", type=int, default=50)
    return p.parse_args()


def prepare_items_math(dataset_path, limit):
    data = load_jsonl(dataset_path)
    items = []
    for obj in data:
        sol = str(obj.get("solution", "")).strip()
        if not sol:
            continue
        user = next((m for m in obj.get("messages", []) if m.get("role") == "user"), None)
        if user:
            q = str(user.get("content", "")).strip()
            if q:
                items.append((q, sol))
                if limit and len(items) >= limit:
                    break
    return items


def normalize_answer(ans):
    return re.sub(r'(\d),(\d)', r'\1\2', ans.strip().lower())


def check_contains_answer(pred, truth):
    return normalize_answer(truth) in normalize_answer(pred)


def check_exact_label(pred, truth, labels):
    p = pred.strip().upper()
    t = truth.strip().upper()
    for lbl in labels:
        if lbl.upper() in p:
            p = lbl.upper()
            break
    return p == t


def majority_vote_answers(answers, acc_type, labels=None):
    if not answers:
        return ""
    if acc_type == "exact_label" and labels:
        counts = {l: 0 for l in labels}
        for ans in answers:
            for l in labels:
                if l.upper() in ans.upper():
                    counts[l] += 1
                    break
        return max(counts, key=counts.get)
    counts = {}
    for a in answers:
        n = normalize_answer(a)
        counts[n] = counts.get(n, 0) + 1
    return max(counts, key=counts.get) if counts else (answers[0] if answers else "")


def build_example_generation_messages_medqa(k, label, counts, topics):
    plan = "\n".join(f"- Example{i+1}: topic: {topics[i % len(topics)]}." for i in range(k))
    return [
        {"role": "system", "content": "You are a data generator that writes high-quality medical multiple-choice questions for in-context learning."},
        {"role": "user", "content": f"""Create exactly {k} training examples in THIS STRICT format only:

Example1:
Question: "<medical question with 5 options A-E>"
Answer: {label}
...
Example{k}:
Question: "<medical question with 5 options A-E>"
Answer: {label}

Diversity plan (MUST FOLLOW):
{plan}

Rules:
- Each question MUST include options A:, B:, C:, D:, E:
- The correct answer is always: {label}
- Use realistic medical scenarios (diagnosis, treatment, pharmacology)
- Use only ASCII characters
- Do NOT wrap output in Markdown/code fences
- Output ONLY the examples in the exact format above; no extra text
"""}
    ]


def build_example_generation_messages_math(task, k, topics):
    plan = "\n".join(f"- Example{i+1}: topic: {topics[i % len(topics)]}." for i in range(k))
    if task == "gsm8k":
        sys = "You are a data generator that writes grade-school math word problems with integer answers."
        style = "- Problems should be simple word problems with clear numerical answers\n- Answers MUST be integers\n"
    elif task == "deepmath":
        sys = "You are a data generator that writes advanced mathematics problems."
        style = "- Problems can involve calculus, linear algebra, number theory, probability\n- Use TeX notation for math expressions\n"
    else:
        sys = "You are a data generator that writes competition-style mathematics problems."
        style = "- Problems should be challenging but solvable\n- Use TeX notation for math expressions\n"

    return [
        {"role": "system", "content": sys},
        {"role": "user", "content": f"""Create exactly {k} training examples in THIS STRICT format only:

Example1:
Question: "<math problem>"
Answer: <answer>
...
Example{k}:
Question: "<math problem>"
Answer: <answer>

Diversity plan (MUST FOLLOW):
{plan}

Rules:
- Each question must be a well-formed math problem
- Each answer must be the correct solution
{style}
- Use only ASCII characters (except for TeX math symbols)
- Do NOT wrap output in Markdown/code fences
- Output ONLY the examples in the exact format above; no extra text
"""}
    ]


def parse_examples_from_text(text, k, task):
    examples = []
    headers = list(EXAMPLE_RE.finditer(text))
    for i, m in enumerate(headers):
        block = text[m.start():headers[i+1].start() if i+1 < len(headers) else len(text)]
        qm = QUESTION_RE.search(block) or re.search(r'Question:\s*(.+?)(?=Answer:|$)', block, re.I | re.S)
        am = ANSWER_RE.search(block)
        if qm and am:
            examples.append({"question": qm.group(1).strip().strip('"'), "answer": am.group(1).strip()})
            if len(examples) >= k:
                break
    return examples


def format_examples_for_prompt(examples):
    lines = []
    for i, ex in enumerate(examples, 1):
        lines.extend([f"Example{i}:", f"Question: {ex['question']}", f"Answer: {ex['answer']}", ""])
    return "\n".join(lines).rstrip()


def generate_examples_balanced(task, tokenizer, model_or_llm, k, seed, max_new_tokens, use_vllm):
    cfg = TASK_CONFIGS[task]
    combined = []
    if task == "medqa":
        labels = cfg["labels"]
        per = k // len(labels)
        rem = k % len(labels)
        for idx, lbl in enumerate(labels):
            lk = per + (1 if idx < rem else 0)
            if lk == 0:
                continue
            msgs = build_example_generation_messages_medqa(lk, lbl, [1]*lk, cfg["topics"][lbl])
            if use_vllm:
                prompt = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
                text = vllm_generate_texts(model_or_llm, [prompt], temperature=0.6, top_p=0.9, max_tokens=max_new_tokens, seed=seed+idx if seed else None)[0]
            else:
                text = gen_with_messages_transformers(tokenizer, model_or_llm, msgs, temperature=0.6, top_p=0.9, max_new_tokens=max_new_tokens, seed=seed+idx if seed else None)
            combined.extend(parse_examples_from_text(text, lk, task))
    else:
        msgs = build_example_generation_messages_math(task, k, cfg["topics"])
        if use_vllm:
            prompt = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            text = vllm_generate_texts(model_or_llm, [prompt], temperature=0.6, top_p=0.9, max_tokens=max_new_tokens, seed=seed)[0]
        else:
            text = gen_with_messages_transformers(tokenizer, model_or_llm, msgs, temperature=0.6, top_p=0.9, max_new_tokens=max_new_tokens, seed=seed)
        combined = parse_examples_from_text(text, k, task)

    rng = random.Random(seed + 99991 if seed else None)
    rng.shuffle(combined)
    return format_examples_for_prompt(combined[:k]), combined[:k]


def generate_paraphrases_batch(sentences, n_para, tokenizer, model_or_llm, para_template, use_vllm, temperature, top_p, max_tokens, seed):
    if n_para <= 0:
        return [[] for _ in sentences]
    prompts = []
    for s in sentences:
        pp = para_template.format(n=n_para, s=s.strip())
        msgs = [{"role": "user", "content": pp}]
        prompts.append(tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True) if use_vllm else pp)

    if use_vllm:
        outputs = vllm_generate_texts(model_or_llm, prompts, temperature=temperature, top_p=top_p, max_tokens=max_tokens, seed=seed)
    else:
        outputs = [gen_with_messages_transformers(tokenizer, model_or_llm, [{"role": "user", "content": p}], temperature=temperature, top_p=top_p, max_new_tokens=max_tokens, seed=seed+i if seed else None) for i, p in enumerate(prompts)]

    result = []
    for out in outputs:
        lines = [re.sub(r'^\d+[\.\)]\s*', '', l.strip()) for l in out.strip().split('\n') if l.strip()]
        result.append([c for c in lines if c][:n_para])
    return result


def evaluate_batch(task, gen_tokenizer, gen_model_or_llm, eval_tokenizer, eval_model_or_llm, items, examples_block, n_para, use_vllm, args, run_seed):
    cfg = TASK_CONFIGS[task]
    n = len(items)
    preds = [[] for _ in range(n)]
    bs = args.batch_size

    for b in range((n + bs - 1) // bs):
        start, end = b * bs, min(n, b * bs + bs)
        questions = [q for q, _ in items[start:end]]
        para_lists = generate_paraphrases_batch(questions, n_para, gen_tokenizer, gen_model_or_llm, cfg["para_prompt"], use_vllm, args.para_temperature, args.para_top_p, args.para_max_new, run_seed + b) if n_para > 0 else [[] for _ in questions]

        all_prompts, slices = [], []
        for orig, pars in zip(questions, para_lists):
            s = len(all_prompts)
            for cand in [orig] + pars:
                prompt = f"{cfg['task_header']}\n\n{examples_block}\n\nQuestion: {cand}\nAnswer:"
                if use_vllm:
                    prompt = eval_tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True)
                all_prompts.append(prompt)
            slices.append((s, len(all_prompts)))

        if use_vllm:
            all_preds = vllm_generate_texts(eval_model_or_llm, all_prompts, temperature=0.0, top_p=1.0, max_tokens=cfg["eval_max_tokens"], seed=None)
        else:
            all_preds = [gen_with_messages_transformers(eval_tokenizer, eval_model_or_llm, [{"role": "user", "content": p}], temperature=0.0, top_p=1.0, max_new_tokens=cfg["eval_max_tokens"], seed=None) for p in all_prompts]

        for j, (s, e) in enumerate(slices):
            preds[start + j].extend(all_preds[s:e])
        if args.report_every > 0 and (end % args.report_every == 0 or end == n):
            print(f"[EVAL] Processed {end}/{n} items")
    return preds


def run_single_spec(task, args, seed, n_para, k_examples, num_runs):
    cfg = TASK_CONFIGS[task]
    set_global_seed(seed)
    gen_llm, gen_model, gen_tokenizer = None, None, None
    eval_llm, eval_model, eval_tokenizer = None, None, None
    use_vllm, fell_back = bool(args.use_vllm), False

    try:
        if not use_vllm:
            raise ImportError("vLLM disabled")
        gp, ep = args.generator_model_path, args.evaluator_model_path
        if gp and ep:
            if gp == ep:
                util = min(args.gpu_mem_util * 1.75, 0.92)
                print(f"[INIT] Single vLLM (shared): {gp} util={util}")
                gen_tokenizer, gen_llm = init_qwen_vllm(gp, args.tp, util, args.max_model_len, args.quantization)
                eval_tokenizer, eval_llm = gen_tokenizer, gen_llm
            else:
                gu = args.generator_gpu_mem_util or args.gpu_mem_util
                eu = args.evaluator_gpu_mem_util or args.gpu_mem_util
                print(f"[INIT] Generator vLLM: {gp} util={gu}")
                gen_tokenizer, gen_llm = init_qwen_vllm(gp, args.tp, gu, args.max_model_len, args.quantization)
                print(f"[INIT] Evaluator vLLM: {ep} util={eu}")
                eval_tokenizer, eval_llm = init_qwen_vllm(ep, args.tp, eu, args.max_model_len, args.quantization)
        else:
            print(f"[INIT] Single vLLM: {args.model_path}")
            gen_tokenizer, gen_llm = init_qwen_vllm(args.model_path, args.tp, args.gpu_mem_util, args.max_model_len, args.quantization)
            eval_tokenizer, eval_llm = gen_tokenizer, gen_llm
    except Exception as e:
        print(f"\n{'='*80}\n[INFO] vLLM not available: {e}\n[INFO] Using Transformers\n{'='*80}")
        fell_back, use_vllm = True, False
        mp = args.generator_model_path or args.model_path
        print(f"[INIT] Loading Transformers: {mp}")
        gen_tokenizer, gen_model = init_qwen_transformers(mp)
        if args.evaluator_model_path and args.evaluator_model_path != mp:
            print(f"[INIT] Loading Evaluator Transformers: {args.evaluator_model_path}")
            eval_tokenizer, eval_model = init_qwen_transformers(args.evaluator_model_path)
        else:
            eval_tokenizer, eval_model = gen_tokenizer, gen_model
        print(f"[INFO] Transformers loaded\n{'='*80}\n")
    gen_or = gen_llm or gen_model
    eval_or = eval_llm or eval_model
    items = prepare_items_math(cfg["dataset"], args.limit)
    golds = [sol for _, sol in items]
    n = len(items)
    print(f"[INFO] Loaded {n} items from {cfg['dataset']}")

    all_preds = [[] for _ in range(n)]
    run_accs = []
    max_tok = max(64, 96 * k_examples)

    for run in range(num_runs):
        rs = seed + 10000 * run
        print(f"\n{'='*80}\n[RUN {run+1}/{num_runs}] seed={rs}\n{'='*80}")
        examples_block, _ = generate_examples_balanced(task, gen_tokenizer, gen_or, k_examples, rs, max_tok, use_vllm)
        print(f"[INFO] Generated examples:\n{examples_block[:1000]}{'...' if len(examples_block) > 1000 else ''}\n")
        item_preds = evaluate_batch(task, gen_tokenizer, gen_or, eval_tokenizer, eval_or, items, examples_block, n_para, use_vllm, args, rs)

        correct = sum(1 for i in range(n) if item_preds[i] and (
            check_exact_label(majority_vote_answers(item_preds[i], cfg["accuracy_type"], cfg.get("labels")), golds[i], cfg["labels"]) if cfg["accuracy_type"] == "exact_label"
            else check_contains_answer(majority_vote_answers(item_preds[i], cfg["accuracy_type"], cfg.get("labels")), golds[i])
        ))
        acc = correct / n if n else 0.0
        run_accs.append(acc)
        print(f"[RUN {run+1}] Accuracy: {acc:.4f} ({correct}/{n})")
        for i in range(n):
            all_preds[i].extend(item_preds[i])

    correct = sum(1 for i in range(n) if all_preds[i] and (
        check_exact_label(majority_vote_answers(all_preds[i], cfg["accuracy_type"], cfg.get("labels")), golds[i], cfg["labels"]) if cfg["accuracy_type"] == "exact_label"
        else check_contains_answer(majority_vote_answers(all_preds[i], cfg["accuracy_type"], cfg.get("labels")), golds[i])
    ))
    acc = correct / n if n else 0.0
    print(f"\n{'='*80}\n[FINAL] Accuracy: {acc:.4f} ({correct}/{n})")
    print(f"[RUNS] {', '.join(f'{a:.4f}' for a in run_accs)}\n{'='*80}")
    return correct, n, acc, use_vllm, fell_back


def main():
    args = parse_args()
    cfg = TASK_CONFIGS[args.task]
    print(f"{'='*100}\nTASK: {args.task.upper()}\n{'='*100}")
    print(f"Params: n={args.n}, k={args.k}, runs={args.runs}, seed={args.seed}")
    print(f"Dataset: {cfg['dataset']}\n{'='*100}")

    t0 = time.time()
    correct, total, acc, used_vllm, fell_back = run_single_spec(args.task, args, args.seed, args.n, args.k, args.runs)
    elapsed = time.time() - t0

    row = {"task": args.task, "seed": args.seed, "n_paraphrases": args.n, "k_examples": args.k, "runs": args.runs,
           "acc": acc, "correct": correct, "total": total, "elapsed_sec": round(elapsed, 3),
           "use_vllm": int(used_vllm), "fell_back_to_hf": int(fell_back)}
    append_csv_row(cfg["csv"], row)
    print(f"\n[DONE] {args.task} | seed={args.seed} | acc={acc:.4f} ({correct}/{total}) | {elapsed:.1f}s")
    print(f"[SAVED] {cfg['csv']}")


if __name__ == "__main__":
    main()