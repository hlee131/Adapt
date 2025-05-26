import os 

splits = ["test", "train"]
root_dir = "data/"
file_format = "gen_dataset_DPO_unpriv_verbose_%s_onlyAction.jsonl"

for split in splits:
    combined_filename = os.path.join(root_dir, "combined", file_format % split)
    output = open(combined_filename, 'w')

    for i in range(4):
        filename = os.path.join(root_dir, f"split{i}", file_format % split)
        with open(filename, 'r') as file:
            for line in file.readlines():
                output.write(line)

    output.close()
