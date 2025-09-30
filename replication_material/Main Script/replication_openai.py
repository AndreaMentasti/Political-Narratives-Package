# -*- coding: utf-8 -*-
"""
Created on Mon Sep 29 14:26:10 2025

@author: AndreaMentasti
"""

#%% (0) Setting up

#----------------------------------------------------------------------------#
# Clean workspace
def clear_all_variables():
    globals_list = list(globals().keys())
    for var in globals_list:
        if var[0] == '_' or var in ['clear_all_variables', 'get_ipython']:
            continue
        del globals()[var]
clear_all_variables()
#----------------------------------------------------------------------------#

#----------------------------------------------------------------------------#
# Import functions and libraries
import sys
import os
import time
import pandas as pd
import json
import math
from openai import OpenAI

# Set your API key as an Environmental variable
client = OpenAI()

# Indicate path to your local functions module
module_directory = "C:/Users/AndreaMentasti/Documents/GitHub/climate-narratives/py"  # Update this path
# Add that directory to sys.path
if module_directory not in sys.path:
    sys.path.append(module_directory)
# Import the adjusted module of functions 
import functions as functions
#----------------------------------------------------------------------------#

#----------------------------------------------------------------------------#
# Define paths
output_folder = f'C:/Users/AndreaMentasti/Dropbox/climate_nature_narratives/output'
input_folder =  f'C:/Users/AndreaMentasti/Dropbox/climate_nature_narratives/input'

# Create necessary directories
#os.makedirs(os.path.join(output_folder, 'batch_input'), exist_ok=True)
#os.makedirs(os.path.join(output_folder, 'api_input'), exist_ok=True)
#os.makedirs(os.path.join(output_folder, 'api_output'), exist_ok=True)
#os.makedirs(os.path.join(output_folder, 'predictions'), exist_ok=True)
#os.makedirs(os.path.join(output_folder, 'batch_id'), exist_ok=True)
#----------------------------------------------------------------------------#

#%% (1) Loading dataset and chunk it
# Load the dataset
dataset_path = os.path.join(input_folder + '/openAI/c_3_fixed_snippets_dataset.xlsx') #CHANGE THIS WITH YOUR DATA
df = pd.read_excel(dataset_path)

# Assuming the dataset has columns 'snippet_id' and 'content', adjust if different
df = df[['snippet_id', 'content']]

# For testing purposes, select a small number of snippets
# Uncomment the following line to test with a small dataset
#df = df.sample(n=50, random_state=42).reset_index(drop=True)

# Divide the dataset into chunks (adjust the chunk size as needed)
chunk_size = 25  # Adjust based on your dataset size
num_chunk = math.ceil(len(df) / chunk_size)

# Save each chunk with consistent naming
for i in range(num_chunk):
    start_row = i * chunk_size
    end_row = min(start_row + chunk_size, len(df))  # Ensure last chunk may be smaller
    chunk_df = df.iloc[start_row:end_row]
    
    # Define filename with batch numbering
    batch_number = str(i+1).zfill(2)  # Pads numbers to ensure two digits (e.g., 01, 02)
    filename = output_folder + f'/data/openAI/batch_input/batch_{batch_number}.csv'
    
    # Save chunk to CSV
    chunk_df.to_csv(filename, index=False)
    print(f'Saved: {filename}')

#%% (2) Annotation loop
if __name__ == "__main__":
    start_batch = 1  # Change this if you want to start from a specific batch number
    total_chunks = num_chunk  # Total number of batches

    for i in range(start_batch, total_chunks + 1):
        batch_number = str(i).zfill(2)  # Formats the batch number to 2 digits

        # Prepare file paths for the current batch
        input_batch_file = os.path.join(output_folder + f'/data/openAI/batch_input/batch_{batch_number}.csv')
        stage1_input_jsonl = os.path.join(output_folder + f'/data/openAI/api_input/batch_{batch_number}_stg1_input.jsonl')
        stage1_output_file = os.path.join(output_folder + f'/data/openAI/api_output/batch_{batch_number}_stage1.csv')
        stage1_batch_id_file = os.path.join(output_folder + f'/data/openAI/batch_id/batch_{batch_number}_stg1_id.json')

        # Stage 1: Process the current batch
        print(f"Processing batch {batch_number} for stage 1!")

        # Load the batch
        df_r = pd.read_csv(input_batch_file)
        df_rel = df_r.drop_duplicates(subset=['snippet_id'], keep='first')

        # Load SYSTEM_MESSAGE for stage 1
        with open(os.path.join(input_folder, 'openAI/newspaper_snippet_system_message_stage1.json'), 'r') as file:
            system_message_data = json.load(file)
        SYSTEM_MESSAGE_STAGE1 = system_message_data['SYSTEM_MESSAGE']

        # Prepare batch for API input
        functions.prepare_batch_input(df_rel, SYSTEM_MESSAGE_STAGE1, 0.1, stage1_input_jsonl)

        # Retry mechanism for creating the batch and saving the batch ID
        while True:
            try:
                functions.create_batch_and_save_id(stage1_input_jsonl, stage1_batch_id_file)
                break  # Exit the loop if successful
            except Exception as e:
                print(f"Error while creating batch for stage 1 (batch {batch_number}): {e}. Retrying in 5 Minutes.")
                time.sleep(300)  # Wait 5 Minutes and retry

        # Retrieve batch ID
        batch_id = functions.retrieve_batch_id(stage1_batch_id_file)
        print(f"Uploaded batch {batch_number} - identifier: {batch_id} - for stage 1!")

        # Check batch status without re-uploading
        while True:
            try:
                output_str = functions.check_batch_status_and_read_output(batch_id)
                if output_str is not None:
                    break  # Exit the loop if output is retrieved successfully
                print(f"Batch {batch_id} for stage 1 not yet completed. Waiting for 5 minutes.")
                time.sleep(300)  # Wait 5 minutes before checking again
            except Exception as e:
                print(f"Error while checking batch status for stage 1 (batch {batch_number}): {e}. Retrying in 5 minutes.")
                time.sleep(300)  # Wait 5 minutes and retry

        # Parse output and save results
        parsed_results = functions.parse_batch_output(output_str)
        functions.create_and_save_final_df_stage1(parsed_results, df_rel, stage1_output_file)
        
        time.sleep(5)
        
        # Proceed to Stage 2 (adjust the condition based on your criteria)
        df_stage1 = pd.read_csv(stage1_output_file)
        # For example, proceed to stage 2 for snippets marked as relevant
        df_stage1_for_stage2 = df_stage1[df_stage1['relevance'] == 3][['snippet_id', 'content']]

        if not df_stage1_for_stage2.empty:
            print(f"Processing batch {batch_number} for stage 2!")

            # Prepare file paths for stage 2
            stage2_input_jsonl = os.path.join(output_folder,  f'data/openAI/api_input/batch_{batch_number}_stg2_input.jsonl')
            stage2_output_file = os.path.join(output_folder, f'data/openAI/api_output/batch_{batch_number}_stage2.csv')
            stage2_batch_id_file = os.path.join(output_folder, f'data/openAI/batch_id/batch_{batch_number}_stg2_id.json')

            # Load SYSTEM_MESSAGE for stage 2
            with open(os.path.join(input_folder, 'openAI/newspaper_snippet_system_message_stage2.json'), 'r') as file:
                system_message_data = json.load(file)
            SYSTEM_MESSAGE_STAGE2 = system_message_data['SYSTEM_MESSAGE']

            # Prepare batch for Stage 2
            functions.prepare_batch_input(df_stage1_for_stage2, SYSTEM_MESSAGE_STAGE2, 0.5, stage2_input_jsonl)

            # Retry mechanism for creating the batch and saving the batch ID (Stage 2)
            while True:
                try:
                    functions.create_batch_and_save_id(stage2_input_jsonl, stage2_batch_id_file)
                    break  # Exit the loop if successful
                except Exception as e:
                    print(f"Error while creating batch for stage 2 (batch {batch_number}): {e}. Retrying in 2 minutes.")
                    time.sleep(300)  # Wait 5 minutes and retry

            # Retrieve batch ID and check status for Stage 2
            batch_id = functions.retrieve_batch_id(stage2_batch_id_file)
            print(f"Uploaded batch {batch_id} for stage 2!")
            while True:
                try:
                    output_str = functions.check_batch_status_and_read_output(batch_id)
                    if output_str is not None:
                        break
                    print(f"Batch {batch_id} for stage 2 not yet completed. Waiting for 2 minutes.")
                    time.sleep(30)  # Wait 20 minutes if status is not yet complete
                except Exception as e:
                    print(f"Error while checking batch status for stage 2 (batch {batch_number}): {e}. Retrying in 2 minutes.")
                    time.sleep(300)  # Wait 5 minutes and retry

            # Parse output and save results for Stage 2
            parsed_results = functions.parse_batch_output(output_str)
            functions.create_and_save_final_df_stage2(parsed_results, df_stage1_for_stage2, stage2_output_file)

        print(f"Finished processing batch {batch_number}.")
print("All batches processed successfully!")

#%% (3) Creating the final dataset
# Set paths for the input and output directories
final_output_file = os.path.join(output_folder, 'data/openAI/predictions/predictions_gpt_newspaper_snippets.csv')

# Initialize an empty list to store each merged batch's dataframe
all_batches = []

# Define the total number of chunks
total_chunks = num_chunk  # Adjust according to the number of batches you processed

for i in range(1, total_chunks + 1):
    batch_number = str(i).zfill(2)  # Formats the batch number to 2 digits
    
    # Define file paths for stage 1 and stage 2
    stage1_output_file = os.path.join(output_folder, f'data/openAI/api_output/batch_{batch_number}_stage1.csv')
    stage2_output_file = os.path.join(output_folder, f'data/openAI/api_output/batch_{batch_number}_stage2.csv')

    # Load Stage 1 and Stage 2 files
    df_stage1 = pd.read_csv(stage1_output_file)

    # Check if stage 2 output exists
    if os.path.exists(stage2_output_file):
        df_stage2 = pd.read_csv(stage2_output_file)
        # Drop 'content' from Stage 2 to avoid duplication
        df_stage2 = df_stage2.drop(columns=['content'])
        # Merge Stage 1 and Stage 2 on 'snippet_id'
        df_merged = pd.merge(df_stage1, df_stage2, on='snippet_id', how='left')
    else:
        df_merged = df_stage1

    # Append the merged dataframe to the list
    all_batches.append(df_merged)

# Concatenate all merged batches
final_df = pd.concat(all_batches, ignore_index=True)

# Save the final concatenated dataframe to a CSV file
final_df.to_csv(final_output_file, index=False)

print(f"Final concatenated output saved to: {final_output_file}")