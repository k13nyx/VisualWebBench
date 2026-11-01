import os
import json
import yaml
import random
import argparse
import threading
from concurrent.futures import ThreadPoolExecutor

from tqdm import tqdm
import pandas as pd
import datasets

from model_adapters.base_adapter import BaseAdapter
from model_adapters.openrouter_adapter import OpenRouterAdapter
from utils import DEFAULT_PROMPTS
from utils import (
    eval_web_caption,
    eval_heading_ocr,
    eval_element_ocr,
    eval_action_prediction,
    eval_element_ground,
    eval_action_ground,
    eval_webqa,
)
import utils.constants as task


eval_metric = {
    task.CAPTION_TASK: eval_web_caption,
    task.HEADING_OCR_TASK: eval_heading_ocr,
    task.WEBQA_TASK: eval_webqa,
    task.ELEMENT_OCR_TASK: eval_element_ocr,
    task.ELEMENT_GROUND_TASK: eval_element_ground,
    task.ACTION_PREDICTION_TASK: eval_action_prediction,
    task.ACTION_GROUND_TASK: eval_action_ground,
}

# Order, display names and headline metric for the summary table
COLUMN_ORDER = [
    task.CAPTION_TASK, task.WEBQA_TASK, task.HEADING_OCR_TASK, task.ELEMENT_OCR_TASK,
    task.ELEMENT_GROUND_TASK, task.ACTION_PREDICTION_TASK, task.ACTION_GROUND_TASK,
]
COLUMN_NAME = {
    task.CAPTION_TASK: "Captioning",
    task.WEBQA_TASK: "WebQA",
    task.HEADING_OCR_TASK: "Heading OCR",
    task.ELEMENT_OCR_TASK: "Element OCR",
    task.ELEMENT_GROUND_TASK: "Element Grounding",
    task.ACTION_PREDICTION_TASK: "Action Prediction",
    task.ACTION_GROUND_TASK: "Action Grounding",
}
TASK_METRIC = {
    task.CAPTION_TASK: "rouge_l",
    task.WEBQA_TASK: "f1",
    task.HEADING_OCR_TASK: "rouge_l",
    task.ELEMENT_OCR_TASK: "rouge_l",
    task.ELEMENT_GROUND_TASK: "accuracy",
    task.ACTION_PREDICTION_TASK: "accuracy",
    task.ACTION_GROUND_TASK: "accuracy",
}


def evaluate(
    model_adapter: BaseAdapter,
    prompt: str,
    dataset: datasets.Dataset,
    task_type: str,
    **kwargs,
):
    print('='*80)
    print('Prompt: ', prompt)
    data_size = len(dataset)
    dataset_lock = threading.Lock()

    def build_prompt(sample):
        match task_type:
            case task.CAPTION_TASK | task.HEADING_OCR_TASK:
                return prompt
            case task.WEBQA_TASK:
                return prompt.format(question=sample['question'])
            case task.ELEMENT_OCR_TASK:
                return prompt.format(bbox_ratio=sample['bbox'])
            case task.ELEMENT_GROUND_TASK:
                return prompt.format(element_desc=sample['elem_desc'])
            case task.ACTION_PREDICTION_TASK:
                return prompt.format(bbox_ratio=sample['bbox'], choices_text=sample['options'])
            case task.ACTION_GROUND_TASK:
                return prompt.format(instruction=sample['instruction'])
            case _:
                raise NotImplementedError(f"Task type {task_type} not implemented.")
        
    def run(idx_):
        with dataset_lock:
            sample = dataset[idx_]

        # OpenRouter adapters return (answer, usage); local adapters return just the answer
        result = model_adapter.generate(build_prompt(sample), sample['image'], task_type)
        if isinstance(result, str):
            response, usage = result, {}
        else:
            response, usage = result

        return response, sample['answer'], usage

    # API calls run in parallel (kwargs['workers']); results keep the dataset order
    workers = kwargs.get('workers', 1)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(tqdm(pool.map(run, range(data_size)), total=data_size, desc=task_type))
    
    preds = [result[0] for result in results]
    golds = [result[1] for result in results]
    usage = [result[2] for result in results]

    scores = eval_metric[task_type](preds, golds)
    return scores, preds, golds, usage


def write_summary_table(summary, output_path):
    """Aggregate per-(model, task) rows into one row per model and write CSV + Excel."""
    df = pd.DataFrame(summary)
    rows = []

    for model in tqdm(df["model"].unique(), desc="Write Summary"):
        selected_row = df[df["model"] == model]
        row = { "model": model }

        for col in COLUMN_ORDER:
            pivot = selected_row.loc[selected_row["task"] == col, "score"]
            row[COLUMN_NAME[col]] = round(pivot.iloc[0], 2) if len(pivot) else None

        total_tokens = int(selected_row["total_tokens"].sum())
        total_cost = float(selected_row["total_cost_usd"].sum())

        row["avg_cost_per_1M"] = round((total_cost / total_tokens) * 1_000_000, 4) if total_tokens else -1
        row["total_cost_usd"] = round(total_cost, 4)
        row["n_items"] = int(selected_row["n_items"].sum())
        row["n_error"] = int(selected_row["errors"].sum())
        row["n_empty"] = int(selected_row["empty"].sum())
        rows.append(row)

    out = pd.DataFrame(rows).reindex(columns=[
        "model",
        *[COLUMN_NAME[t] for t in COLUMN_ORDER],
        "total_cost_usd",
        "avg_cost_per_1M",
        "n_items", "n_error", "n_empty"
    ])
    out = out.sort_values("model", ascending=False, na_position="last")

    csv_path = os.path.join(output_path, "summary.csv")
    xlsx_path = os.path.join(output_path, "summary.xlsx")
    out.to_csv(csv_path, index=False)
    out.to_excel(xlsx_path, index=False)

    print("\n" + out.to_string(index=False, na_rep=""))
    print(f"\nWrote {csv_path} and {xlsx_path}")


def main(args):
    model_config = yaml.load(open("models.yml"), Loader=yaml.FullLoader)
    env = model_config.get('env', {})
    models = model_config.get('models', {})

    base_url = env.get('base_url', 'https://openrouter.ai/api/v1')
    api_key = env.get('api_key', 'OPENROUTER_API_KEY')

    # Which models to run: 'all' or one/more keys from models.yml (comma-separated)
    if args.model_name == 'all':
        selected = list(models.items())
    else:
        items = [ it.strip() for it in args.model_name.split(',') ]
        missing = [ it for it in items if it not in models ]
        if missing:
            raise SystemExit(f"Unknown model key(s) not in models.yml: {missing}")
        selected = [(it, models[it]) for it in items]

    if ',' in args.task_type:
        task_types = [t.strip() for t in args.task_type.split(',')]
    else:
        task_types = [args.task_type]

    summary = []  # per-(model, task) rows for the final table
    for model_key, config in selected:
        model_adapter = OpenRouterAdapter(
            base_url=base_url,
            api_key=api_key,
            model=config['model_id'],

            max_tokens=config.get('max_tokens', 2048),
            temperature=config.get('temperature', 0.0),
            extra_body=config.get('extra_body')
        )
        print(f"\n=== Model: {model_key} ({config.get('name', config['model_id'])}) ===")

        out_dir = os.path.join(args.output_path, model_key)
        os.makedirs(out_dir, exist_ok=True)

        for task_type in task_types:
            prompt = DEFAULT_PROMPTS[f"{task_type}_prompt"]

            dataset = datasets.load_dataset(args.dataset_name_or_path, task_type)['test']
            if args.limit:  # pilot run: same random items for every model (fixed seed)
                idxs = list(range(len(dataset)))
                random.Random(args.seed).shuffle(idxs)
                dataset = dataset.select(sorted(idxs[:args.limit]))

            scores, preds, golds, usages = evaluate(
                model_adapter=model_adapter,
                prompt=prompt,
                dataset=dataset,
                task_type=task_type,
                workers=args.workers,
            )
            score_str = ', '.join([f"{k}: {v:.2f}" for k, v in scores.items()])

            # total tokens used
            total_in = sum((usg.get('prompt_tokens') or 0) for usg in usages if usg)
            total_out = sum((usg.get('completion_tokens') or 0) for usg in usages if usg)
            total_tokens = total_in + total_out

            # cost per item: OpenRouter's billed cost if returned, else tokens x list price
            price_in = float(config.get('price_in', 0))
            price_out = float(config.get('price_out', 0))
            total_cost = 0.0
            for usg in usages:
                if not usg:
                    continue
                if usg.get('cost_reported') is not None:
                    total_cost += usg['cost_reported']
                else:
                    total_cost += ((usg.get('prompt_tokens') or 0) * price_in
                                   + (usg.get('completion_tokens') or 0) * price_out) / 1e6

            # blended USD per 1M tokens for this task's run
            weighted_avg_price = (total_cost / total_tokens) * 1_000_000 if total_tokens else 0.0

            print(f"Model: {model_key}, Task: {task_type}, Scores: {score_str}, "
                  f"Cost: ${total_cost:.4f}, $/1M: ${weighted_avg_price:.4f}")

            summary.append({
                "model": model_key,
                "task": task_type,
                "score": scores[TASK_METRIC[task_type]],
                "total_cost_usd": total_cost,
                "total_tokens": total_tokens,
                "n_items": len(preds),
                "errors": sum(1 for usg in usages if usg and usg.get('error')),
                "empty": sum(1 for usg in usages if usg and usg.get('empty')),
            })

    write_summary_table(summary, args.output_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dataset_name_or_path',
        default="visualwebbench/VisualWebBench",
        type=str,
    )
    with open("models.yml") as f:
        _cfg = yaml.load(f, Loader=yaml.FullLoader)
    model_keys = ['all'] + list(_cfg.get('models', {}).keys())
    parser.add_argument(
        '--model_name',
        default='all',
        type=str,
        choices=model_keys,
    )
    parser.add_argument(
        '--task_type',
        default='web_caption',
        type=str,
        help="Task type can be one of web_caption, heading_ocr, element_ocr, action_prediction, element_ground, action_ground, webqa. Or several tasks separated by comma.",
    )
    parser.add_argument(
        '--output_path', 
        default='output', 
        type=str
    )
    parser.add_argument(
        "--gpus",
        default="0",
        type=str,
        help="A single GPU like 1 or multiple GPUs like 0,2",
    )
    parser.add_argument(
        "--limit",
        default=0,
        type=int,
        help="Pilot run: number of random items per task (0 = all items).",
    )
    parser.add_argument(
        "--seed",
        default=0,
        type=int,
        help="Seed for picking the --limit items.",
    )
    parser.add_argument(
        "--workers",
        default=1,
        type=int,
        help="Parallel API calls (API models only; keep 1 for local GPU models).",
    )
    args = parser.parse_args()

    os.makedirs(args.output_path, exist_ok=True)
    print(args)

    main(args)
