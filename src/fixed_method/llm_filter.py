import traceback
import json
from src.llm.llm_generator.llm_generating import LLM_Generator

# Create an instance of LLM_Generator
llm_generator = LLM_Generator()

# ------------------------
# Step 1: Data Loading and Preprocessing
# ------------------------

def load_data(json_file):
    """Load JSON data file and return all data"""
    with open(json_file, 'r') as f:
        data = [json.loads(line) for line in f.readlines()]
    
    return data

def group_data_by_query(data):
    """Group data by query, where every 5 lines belong to the same query"""
    grouped_data = []
    for i in range(0, len(data), 5):
        grouped_data.append(data[i:i+5])  # Group every 5 lines
    return grouped_data

def extract_table_data(group):
    """Extract useful table information, ColBERT score, table ID, and correct table ID from each group"""
    query = group[0]['query']  # A single query corresponds to 5 different candidate results
    correct_id = group[0]['id']  # Unique ID of the correct table

    results = []
    for entry in group:
        result = entry['result']
        results.append({
            'table': result['content'],
            'passage_id': result['passage_id']  # Used to check if it matches the correct table
        })
    
    return query, results, correct_id

# ------------------------
# Step 2: Iterative Table Elimination
# ------------------------

def iterative_table_selection(select_best_table, query, tables):
    """Eliminate tables pairwise until the best table is selected"""
    
    # Set the first table as the initial best table
    current_best = tables[0]
    
    # Perform pairwise comparison from the second table onward
    for i in range(1, len(tables)):
        next_table = tables[i]
        
        # Call select_best_table to compare the current best table with the next table
        # Pass table content and passage_id together to the LLM for comparison
        better_table_passage_id = select_best_table(
            query, 
            f"table1['passage_id']: {current_best['passage_id']}\nTable1 Content: {current_best['table']}",
            f"table2['passage_id']: {next_table['passage_id']}\nTable2 Content: {next_table['table']}"
        )

        print('better_table_passage_id (from LLM):', better_table_passage_id, type(better_table_passage_id))
        
        # Print the next_table's passage_id and its type from data
        print('next_table["passage_id"] (from data):', next_table['passage_id'], type(next_table['passage_id']))

        # Update current_best as the better table based on returned better_table_passage_id
        if str(better_table_passage_id) == str(next_table['passage_id']):
            current_best = next_table
    
    # Return the final selected best table
    return current_best

# ------------------------
# Step 3: Save the Best Table to JSONL File
# ------------------------

def save_best_table_to_jsonl(best_table, original_group, output_file):
    """Save the original state of the best table to a new JSONL file"""
    # Find the original data of the best table within the original group
    for entry in original_group:
        if entry['result']['passage_id'] == best_table['passage_id']:
            with open(output_file, 'a') as f:  # Append mode
                f.write(json.dumps(entry) + '\n')  # Write the original data to file
            break

# ------------------------
# Main Program Logic
# ------------------------

if __name__ == "__main__":
    input_file = '/home/yuhangwu/Desktop/Projects/tablerag/data/processed/retrieval_results/e2ewtq.jsonl'
    output_file = '/home/yuhangwu/Desktop/Projects/tablerag/data/processed/llm_filtered_data/e2ewtq.jsonl'  # Output file for saving the best table
    
    # Load data
    data = load_data(input_file)
    
    # Group data by query, with each group containing 5 candidate tables
    grouped_data = group_data_by_query(data)
    
    total_queries = len(grouped_data)  # Total number of queries
    queries_with_correct_in_top5 = 0  # Number of queries where ColBERT hits@5 contains the correct answer
    correct_final_selection_in_top5 = 0  # Number of correct selections in hits@5 where correct answer is present
    correct_final_selection = 0  # Total number of correct final selections

    for group in grouped_data:  # Ensure all query groups are processed
        query, tables, correct_id = extract_table_data(group)
        
        # Check if ColBERT's top 5 contains the correct answer
        in_top5 = any(table['passage_id'] == correct_id for table in tables)
        if in_top5:
            queries_with_correct_in_top5 += 1
            
            # Perform iterative elimination to select the best table
            best_table = iterative_table_selection(llm_generator.select_best_table, query, tables)
            
            # Check if the final selected table is correct
            if best_table['passage_id'] == correct_id:
                correct_final_selection_in_top5 += 1
            
            # Save the original state of the best table in the group
            save_best_table_to_jsonl(best_table, group, output_file)
    
    # Ensure all groups are processed and results for each group are calculated
    if total_queries > 0:
        # Calculate accuracy when the correct table is included in the top 5
        if queries_with_correct_in_top5 > 0:
            accuracy_given_in_top5 = correct_final_selection_in_top5 / queries_with_correct_in_top5
        else:
            accuracy_given_in_top5 = 0
        
        # Calculate overall accuracy
        overall_accuracy = correct_final_selection_in_top5 / total_queries
        
        # Output results
        print(f"Accuracy of selecting the correct table when the correct table is present in ColBERT's top 5: {accuracy_given_in_top5:.2%}")
        print(f"Overall final accuracy: {overall_accuracy:.2%}")
    else:
        print("No data to process.")
