import datasets
import json
import os
import argparse




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, default="../output_data/mm-dia")
    parser.add_argument("--meta_file", type=str, default="train_eval_test_hard_ids.json")
    parser.add_argument("--output_dir", type=str, default="../output_data/mm_dia_splits/")
    args = parser.parse_args()

    input_file = args.input_file
    meta_file = args.meta_file
    output_dir = args.output_dir

    #full_ds = datasets.load_dataset("json", data_files=input_file, split="train")
    full_ds = datasets.load_from_disk(input_file)

    metas = json.load(open(meta_file, "r"))
    for split, ids in metas.items():
        print(f"{split}: {len(ids)}")
        ds_split = full_ds.filter(lambda x: x["dialog_id"] in ids)
        print(f"Filtered ds split {split}: {len(ds_split)}")
        ds_split.save_to_disk(os.path.join(output_dir, split))