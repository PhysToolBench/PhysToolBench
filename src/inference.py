from model_api_transfer import generate_response
import os
import json
import sys
import argparse
import concurrent.futures
from functools import partial
from tqdm import tqdm

from dataset.dataset_reader import DatasetReader

def process_sample(sample_id, *, model_name, dataset_reader, api_url, api_key, output_dir, cot, resume, cot_only):
    output_filepath = os.path.join(output_dir, f"{str(sample_id+1).zfill(4)}.json")
    if resume and os.path.exists(output_filepath):
        print(f"Result for sample {sample_id+1} in {output_dir} already exists. Skipping.")
        return

    sample = dataset_reader[sample_id]
    cot_prompt, no_cot_prompt = sample['cot_prompt'], sample['no_cot_prompt']
    correct_tool = sample['correct_tool']
    image_path = sample['image_path']


    if cot:
        cot_response = generate_response(
            model_name,
            cot_prompt,
            image_path,
            api_url,
            api_key
        )
        if "### Answer" in cot_response:
            cot_response_short = cot_response.split("### Answer")[1].split("\'\'\'")[0].replace('```', '').replace('\n', '')
        else:
            cot_response_short = ""
    else:
        cot_response = ""
        cot_response_short = ""

    if not cot_only:
        no_cot_response = generate_response(
            model_name,
            no_cot_prompt,
            image_path,
            api_url,
            api_key
        )
    else:
        no_cot_response = ""

    print(f"Current Task ID: {sample_id+1}")
    print(f"COT Answer: {cot_response_short}")
    print(f"No COT Answer: {no_cot_response}")
    print(f"Correct Tool: {correct_tool}")
    print('-'*50)

    with open(output_filepath, "w") as f:
        json.dump({
            "cot_response_full": cot_response,
            "cot_response": cot_response_short,
            "no_cot_response": no_cot_response,
            "correct_tool": correct_tool
        }, f, indent=4)

def inference_all(model_name, dataset_reader_list, api_url, api_key, cot=True, resume=False, num_threads=1, cot_only=False):
    for dataset_reader in dataset_reader_list:
        task_num = len(dataset_reader)
        start_end = [0, task_num-1]
        output_dir = f"results/{dataset_reader.sequence_name}/{model_name}"

        os.makedirs(output_dir, exist_ok=True)

        inference(model_name, dataset_reader, api_url, api_key, output_dir, start_end, cot=cot, resume=resume, num_threads=num_threads, cot_only=cot_only)


def inference(model_name, dataset_reader, api_url, api_key, output_dir, start_end=[1, 132], cot=True, resume=False, num_threads=1, cot_only=False):
    total_samples = start_end[1] - start_end[0] + 1
    if num_threads > 1:
        process_func = partial(process_sample, model_name=model_name, dataset_reader=dataset_reader, api_url=api_url, api_key=api_key, output_dir=output_dir, cot=cot, resume=resume, cot_only=cot_only)
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
            list(tqdm(executor.map(process_func, range(start_end[0], start_end[1]+1)), total=total_samples, desc=f"Inference on {dataset_reader.sequence_name}"))
    else:
        for sample_id in tqdm(range(start_end[0], start_end[1]+1), total=total_samples, desc=f"Inference on {dataset_reader.sequence_name}"):
            process_sample(sample_id, model_name=model_name, dataset_reader=dataset_reader, api_url=api_url, api_key=api_key, output_dir=output_dir, cot=cot, resume=resume, cot_only=cot_only)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="gemma-3")
    parser.add_argument("--api_url", type=str, default="http://localhost:8003")
    parser.add_argument("--api_key", type=str, default="")
    parser.add_argument("--no_cot", default=False, action="store_true")
    parser.add_argument("--cot_only", default=False, action="store_true", help="Only run inference with CoT.")
    parser.add_argument("--resume", default=False, action="store_true", help="Resume inference, skip existing results.")
    parser.add_argument("--num_threads", type=int, default=1, help="Number of threads for parallel inference.")
    args = parser.parse_args()

    sequence_list = ['Medium-M1', 'Medium-M2', 'Medium-M3', 'Hard', 'Easy']
    
    dataset_reader_list = []
    for sequence in sequence_list:
        dataset_reader = DatasetReader(
            sequence_dir=f"dataset/images/{sequence}"
        )
        dataset_reader_list.append(dataset_reader)

    model_name = args.model_name
    api_url = args.api_url
    api_key = args.api_key
    cot = not args.no_cot
    cot_only = args.cot_only
    assert not (cot_only and args.no_cot), "cot_only and no_cot cannot be both True"
    if cot_only:
        cot = True
    
    inference_all(model_name, dataset_reader_list, api_url, api_key, cot, resume=args.resume, num_threads=args.num_threads, cot_only=cot_only)