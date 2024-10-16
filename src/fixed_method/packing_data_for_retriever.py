import json

def process_jsonl(input_file, output_file, pack_summary=False, pack_explanations=False, pack_suggestions=False):
    """
    Process a JSONL file, pack specific fields into the 'request' field, and optionally set the fields summary, explanations, and suggestions to empty.

    Parameters:
    - input_file: Path to the input JSONL file
    - output_file: Path to the output JSONL file
    - pack_summary: Whether to pack table_summary as empty
    - pack_explanations: Whether to pack terms_explanation as empty
    - pack_suggestions: Whether to pack query_suggestions as empty
    """
    # Initialize ID counter
    id_counter = 1
    # Open the input JSONL file
    with open(input_file, 'r', encoding='utf-8') as infile, open(output_file, 'w', encoding='utf-8') as outfile:
        # Read the JSONL file line by line
        for line in infile:
            # Load each JSON entry
            data = json.loads(line)
            
            # Choose whether to pack fields as empty based on parameters
            table_summary = '' if pack_summary else data.get('table_summary', '')
            terms_explanation = '' if pack_explanations else data.get('terms_explanation', '')
            query_suggestions = '' if pack_suggestions else data.get('query_suggestions', '')

            # Concatenate all field contents into a string and assign to the 'request' field
            data['request'] = (
                f"table_title: {data.get('table_title', '')}\n"
                f"table_context: {data.get('table_context', [])}\n"
                f"table_formatted: {data.get('table_formatted', '')}\n"
                f"table_summary: {table_summary}\n"
                f"terms_explanation: {terms_explanation}\n"
                f"query_suggestions: {query_suggestions}\n"
            )
            
            # Add an ID field
            data['id'] = str(id_counter)
            
            # Remove the original fields
            for key in ["table_title", "table_context", "table_formatted", "table_summary", "terms_explanation", "query_suggestions"]:
                data.pop(key, None)
            
            # Write the processed data to the output JSONL file
            outfile.write(json.dumps(data, ensure_ascii=False) + '\n')
            
            # Update the ID counter
            id_counter += 1

# Usage example
input_file = '/home/yuhangwu/Desktop/Projects/tablerag/data/processed/clarified_data/nqtables.jsonl'   # Name of the input JSONL file
output_file = '/home/yuhangwu/Desktop/Projects/tablerag/data/processed/packed_data/packed_nqtables_test.jsonl' # Name of the output JSONL file

# Call the function, where pack_summary=True means packing summary as empty, and pack_explanations=False means keeping explanations
process_jsonl(input_file, output_file, pack_summary=True, pack_explanations=True, pack_suggestions=True)
