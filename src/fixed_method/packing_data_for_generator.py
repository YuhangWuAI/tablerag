import json

def process_files(input_file_1, input_file_2, input_file_3, output_file, pack_summary=False, pack_explanations=False):
    """
    Processes three input files and writes the results to an output file. Optionally, the fields summary and explanation can be set to empty.

    Parameters:
    - input_file_1: Path to the first input file (llm_filtered_data)
    - input_file_2: Path to the second input file (clarified_data)
    - input_file_3: Path to the third input file (raw small dataset)
    - output_file: Output file path
    - pack_summary: Whether to pack table_summary as empty
    - pack_explanations: Whether to pack terms_explanation as empty
    """
    # Read the first file e2ewtq_test.jsonl
    with open(input_file_1, 'r') as file1:
        lines1 = file1.readlines()

    # Initialize the results storage list
    results = []

    # Process the content of file 1 line by line
    for line_num, line in enumerate(lines1):
        data = json.loads(line)
        
        # If id == passage_id, proceed with further processing
        if data['id'] == data['result']['passage_id']:
            current_id = data['id']
            query = data['query']
            
            # Read the corresponding line from the second file clarified_data (line id + 1)
            with open(input_file_2, 'r') as file2:
                clarified_lines = file2.readlines()
                
                if current_id < len(clarified_lines):  # Ensure not to exceed the file line count
                    clarified_data = json.loads(clarified_lines[current_id])
                    
                    # Extract the necessary fields and choose whether to pack as empty based on parameters
                    table_title = clarified_data.get('table_title', '')
                    table_context = clarified_data.get('table_context', [])
                    table_summary = '' if pack_summary else clarified_data.get('table_summary', '')
                    terms_explanation = '' if pack_explanations else clarified_data.get('terms_explanation', '')
                    
                    # Read the formatted_table and label fields from the third file e2ewtq.jsonl
                    with open(input_file_3, 'r') as file3:
                        raw_lines = file3.readlines()

                        if current_id < len(raw_lines):  # Ensure not to exceed the file line count
                            raw_data = json.loads(raw_lines[current_id])
                            
                            # Extract the formatted_table field
                            formatted_table = raw_data.get('table', '')

                            # Extract the label field, combining label and alternativeLabel
                            label = []
                            if 'label' in raw_data:
                                label.append(raw_data['label'])
                            if 'alternativeLabel' in raw_data:
                                label.append(raw_data['alternativeLabel'])
                            
                            # Assign an empty value if the label list is empty
                            if not label:
                                label = None

                    # Pack the data
                    result = {
                        'id': current_id,
                        'query': query,
                        'table_title': table_title,
                        'table_context': table_context,
                        'table_summary': table_summary,  # Newly added controllable field
                        'formatted_table': formatted_table,
                        'terms_explanation': terms_explanation,  # Newly added controllable field
                        'label': label
                    }

                    # Add the result to the list
                    results.append(result)

    # Write the results to a new jsonl file
    with open(output_file, 'w') as output:
        for result in results:
            output.write(json.dumps(result, ensure_ascii=False) + '\n')

    print(f"Data processing complete, results saved to {output_file}")


# Usage example
input_file_1 = '/home/yuhangwu/Desktop/Projects/tablerag/data/processed/llm_filtered_data/e2ewtq.jsonl'
input_file_2 = '/home/yuhangwu/Desktop/Projects/tablerag/data/processed/clarified_data/e2ewtq.jsonl'
input_file_3 = '/home/yuhangwu/Desktop/Projects/tablerag/data/raw/small_dataset/e2ewtq.jsonl'
output_file = '/home/yuhangwu/Desktop/Projects/tablerag/data/processed/row_col_filtered_data/e2ewtq.jsonl'

# Call the function for processing, where pack_summary=True means packing summary as empty, and pack_explanations=False means keeping explanations
process_files(input_file_1, input_file_2, input_file_3, output_file, pack_summary=True, pack_explanations=False)
