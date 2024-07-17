import os


def decompose_combined_file(
    combined_file_path: str, index_file_path: str, output_directory: str
):
    os.makedirs(output_directory, exist_ok=True)
    # Open the combined file for reading
    with open(combined_file_path, 'r') as combined_file:
        # Open the index file for reading
        with open(index_file_path, 'r') as index_file:
            while True:
                # Read the filename from the index
                filename_line = index_file.readline().strip()
                if not filename_line:
                    break
                filename = filename_line.split(': ')[1]

                # Read the start and end positions from the index
                positions_line = index_file.readline().strip()
                start_position = int(positions_line.split(', ')[0].split(': ')[1])
                end_position = int(positions_line.split(', ')[1].split(': ')[1])

                # Move to the start position in the combined file
                combined_file.seek(start_position)

                # Read the content from the start to the end position
                content = combined_file.read(end_position - start_position)

                # Write the content to a new file
                with open(os.path.join(output_directory, filename), 'w') as output_file:
                    output_file.write(content)

                # Skip the empty line in the index file
                index_file.readline()

    print("Decomposition complete!")
