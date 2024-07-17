import os


def combine_jsonl_files(
    directory: str, combined_file_name: str, index_file: str = 'index.txt'
):
    # Open the output files for writing
    with open(combined_file_name, 'w') as out, open(index_file, 'w') as index:
        # Iterate over each file in the directory
        for filename in os.listdir(directory):
            if filename.endswith('.jsonl'):
                # Write the filename to the index file
                index.write(f"Starting lines from: {filename}\n")
                start_line = out.tell()
                # Copy lines from the current JSONL file to the combined file
                with open(os.path.join(directory, filename), 'r') as f:
                    for line in f:
                        out.write(line)
                end_line = out.tell()
                # Write the start and end positions to the index file
                index.write(
                    f"Start position: {start_line}, End position: {end_line}\n\n"
                )
