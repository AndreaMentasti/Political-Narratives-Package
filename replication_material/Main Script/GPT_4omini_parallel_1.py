#%% Running Cell 1
# gpt_parallel.py
# --------------------------------------------------------------------
# Author: Andrea Mentasti (adaptation of Remo Agovic' code) 
# Date: 2025-10-16
#
# Description: Parallel classification with checkpoint-based resumption.
# --------------------------------------------------------------------


#----------------------------------------------------------------------------#
# Clean workplace
def clear_all_variables():
    globals_list = list(globals().keys())
    for var in globals_list:
        if var[0] == '_' or var in ['clear_all_variables', 'get_ipython']:
            continue
        del globals()[var]
clear_all_variables()
#----------------------------------------------------------------------------#

#pip install "openai==2.4.0"

import os
import time
import pandas as pd
import json
import openai
import re
from concurrent.futures import ThreadPoolExecutor
import pyarrow.feather as feather
from datetime import datetime

# A major update to v1+ changed the way to make calls <--
from openai import OpenAI
client = OpenAI()  # needs OPENAI_API_KEY in env

# Adapt this to your task
JSON_SCHEMA = {
    "name": "ArticleClassification",
    "schema": {
        "type": "object",
        "properties": {
            "items": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {"type": ["integer", "string"]},
                        "a": {"type": "integer", "enum": [0, 1, 2, 3, 4]},
                        "b": {"type": "integer", "enum": [0, 1, 2, 3, 4]},
                        "c": {"type": "integer", "enum": [0, 1, 2, 3, 4]},
                        "d": {"type": "integer", "enum": [0, 1, 2, 3, 4]},
                        "e": {"type": "integer", "enum": [0, 1, 2, 3, 4]},
                        "f": {"type": "integer", "enum": [0, 1, 2, 3, 4]},
                        "g": {"type": "integer", "enum": [0, 1, 2, 3, 4]},
                        "h": {"type": "integer", "enum": [0, 1, 2, 3, 4]},
                        "i": {"type": "integer", "enum": [0, 1, 2, 3, 4]},
                        "j": {"type": "integer", "enum": [0, 1, 2, 3, 4]}
                    },
                    "required": ["id","a","b","c","d","e","f","g","h","i","j"],
                    "additionalProperties": False
                }
            }
        },
        "required": ["items"],
        "additionalProperties": False
    },
    "strict": True
}

# -----------------------------
# (0) SETUP
# -----------------------------

# Adjust this path to your main folder:
#main = r"C:\Users\AndreaMentasti\Dropbox\Narratives, identity, zero-sum thinking and political reforms"
main = r"C:\Users\bracl\Dropbox\Narratives, identity, zero-sum thinking and political reforms"

# Base folders for output
output_data_folder = os.path.join(main, "output", "data", "openai_output")
final_data_folder = os.path.join(main, "output", "data", "openai_final")

# ===== NEW PART: Specify the resume folder =====
# If you want to resume an existing run, set resume_run_folder to its folder name (including the "run_" prefix).
# For example, if your existing folder is "run_20250207_140728":
# resume_run_folder = "run_20250207_140728"  # Change to None if starting a new run
resume_run_folder = None

if resume_run_folder:
    run_partial_results_folder = os.path.join(output_data_folder, resume_run_folder)
    run_final_results_folder = os.path.join(final_data_folder, resume_run_folder)
    print(f"Resuming from folder: {run_partial_results_folder}")
else:
    unique_run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_partial_results_folder = os.path.join(output_data_folder, f"run_{unique_run_id}")
    run_final_results_folder = os.path.join(final_data_folder, f"run_{unique_run_id}")
    print(f"Starting a new run in folder: {run_partial_results_folder}")

# Create the folders (if they don’t exist)
os.makedirs(run_partial_results_folder, exist_ok=True)
os.makedirs(run_final_results_folder, exist_ok=True)

# Define the checkpoint file path (this remains the same)
checkpoint_path = os.path.join(run_partial_results_folder, "checkpoint.txt")

# Path to your data and prompt
csv_path = os.path.join(main, "data", "output", "opinion_question.csv") # depending on your data you might want to convert .csv in .feather
path_df = os.path.join(main, "data", "output", "opinion_question.feather")

prompt_path = os.path.join(main, "code", "prompts", "system_message_peru.json") # This to be uploaded

# Load the system prompt
with open(prompt_path, 'r', encoding='utf-8') as file:
    prompt_data = json.load(file)
SYSTEM_MESSAGE = prompt_data['SYSTEM_MESSAGE']

# Checking the input dataset and its format. Saving the .csv as .feather 

df_csv = pd.read_csv(csv_path, encoding="latin1")
feather.write_feather(df_csv, path_df)

#%% Running Cell 2
# --------------------------------------------------------------------
# Helper Functions
# --------------------------------------------------------------------

def build_user_content(article_group):
    
    """
    Build the user content for the OpenAI API.
    You need each "article" to be a dictionary like {"id":1, ""headline": "bla bla bla...""}
    """
    
    user_content = (
        "Classify the following opinion(s) strictly per the system instructions. "
        "Respond with **only** a JSON object of the form {\"items\": [ ... ]}, "
        "where \"items\" is an array of objects (one per text, same order). "
        "Each object must include keys: id, a, b, c, d, e, f, g, h, i, j (a–j in 0–4). "
        "No extra text.\n\n"
    )
    for article in article_group:
        user_content += f"ID: {article['id']}\n"
        user_content += f"Tweet: {article['text']}\n"
    return user_content



def send_request(request_id, article_group, system_message):
    """
    Send one request to the OpenAI API for a single article_group.
    """
    user_content = build_user_content(article_group)

    body = {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_content}
        ],
        "max_tokens": 4096,
        "temperature": 0.2,
        "response_format": {
            "type": "json_schema",
            "json_schema": JSON_SCHEMA
        }
    }

    try:
        response = client.chat.completions.create(**body)
        print(f"Request {request_id} completed successfully.")
        return {"request_id": request_id, "response": response, "status": "success"}
    except Exception as e:
        print(f"Request {request_id} failed with error: {e}")
        return {"request_id": request_id, "response": None, "status": "failed", "error": str(e)}
    


def process_requests_in_parallel(grouped_articles, system_message):
    """
    Creates a ThreadPool, sends requests in parallel, 
    and collects results (with article_id attached).
    """
    results = []
    with ThreadPoolExecutor(max_workers=64) as executor: #max workers was 2000 but let's try with 64
        futures = [
            executor.submit(send_request, request_id, article_group, system_message)
            for request_id, article_group in enumerate(grouped_articles)
        ]
        
        # Attach article_id to each future's result
        for i, future in enumerate(futures):
            r = future.result()
            # each 'article_group' is just a list of length 1 in this usage:
            r["article_id"] = grouped_articles[i][0]["id"]
            results.append(r)

    return results



def _extract_content_from_response(response_obj):
    try:
        return response_obj.choices[0].message.content
    except Exception:
        return response_obj["choices"][0]["message"]["content"]
    


def flatten_results(batch_results):
    flattened_data, failed_data = [], []

    for result in batch_results:
        if result["status"] == "success" and result["response"]:
            try:
                response_obj = result["response"]
                response_content = _extract_content_from_response(response_obj)
                if not response_content or not isinstance(response_content, str):
                    raise ValueError("Empty or non-string response content")

                # parse
                data = json.loads(response_content)

                # accept {"items":[...]} or a raw list
                if isinstance(data, dict) and "items" in data:
                    json_list = data["items"]
                elif isinstance(data, list):
                    json_list = data
                elif isinstance(data, dict):
                    json_list = [data]
                else:
                    raise ValueError("Unexpected JSON top-level type")

                if not isinstance(json_list, list):
                    raise ValueError("items is not a list")

                for entry in json_list:
                    entry_id = entry.get("id", result.get("article_id"))
                    flattened_entry = {
                        "request_id": result["request_id"],
                        "article_id": result.get("article_id"),
                        "id": entry_id,
                        "a": entry.get("a", 0),
                        "b": entry.get("b", 0),
                        "c": entry.get("c", 0),
                        "d": entry.get("d", 0),
                        "e": entry.get("e", 0),
                        "f": entry.get("f", 0),
                        "g": entry.get("g", 0),
                        "h": entry.get("h", 0),
                        #"i": entry.get("i", 0),
                        #"j": entry.get("j", 0)
                    }
                    flattened_data.append(flattened_entry)

            except Exception as e:
                failed_data.append({
                    "request_id": result["request_id"],
                    "article_id": result.get("article_id"),
                    "error": f"Parse error: {e}"
                })
        else:
            failed_data.append({
                "request_id": result["request_id"],
                "article_id": result.get("article_id"),
                "error": result.get("error", "No response object")
            })

    return pd.DataFrame(flattened_data), pd.DataFrame(failed_data)

# --------------------------------------------------------------------
# The "Wrapper" logic for chunking + partial saving with checkpointing
# --------------------------------------------------------------------

def run_classification_in_chunks(chunk_size=100):
    print("Starting classification in chunks...")
    
    overall_start_time = time.time()
    
    # Load and preprocess data
    df_all = feather.read_feather(path_df)
    
    #df_all = df_all[df_all["retweet_dummy"] != 1]
    #df = df_all[["tweet_id", "english_text"]].rename(columns={"tweet_id": "id"})
    
    df = df_all[["id", "literal"]].copy()
    #df = df_all.sample(n=1000, random_state=13)  # Set random_state for reproducibility
    
    # Create articles list using just id and headline
    #articles = [{"id": row["id"], "headline": row["english_text"]} for _, row in df.iterrows()]
    articles = [{"id": row["id"], "text": row["literal"]} for _, row in df.iterrows()]
    
    total_articles = len(articles)
    print(f"Total articles to process: {total_articles}")
    
    cumulative_tokens = 0  # track total token usage across all batches
    total_processed_articles = 0  # track overall progress
    
    # --- Check for existing checkpoint ---
    last_completed_chunk = 0
    if os.path.exists(checkpoint_path):
        try:
            with open(checkpoint_path, "r", encoding="utf-8") as cp:
                last_completed_chunk = int(cp.read().strip())
            print(f"Resuming from chunk {last_completed_chunk + 1}")
        except Exception as e:
            print("Invalid checkpoint file. Starting from the beginning.")
            last_completed_chunk = 0
    else:
        print("No checkpoint file found. Starting from the beginning.")
    
    # Process articles in chunks
    for start_idx in range(0, total_articles, chunk_size):
        end_idx = min(start_idx + chunk_size, total_articles)
        chunk_index = (start_idx // chunk_size) + 1

        # Skip chunks that were already processed according to the checkpoint file
        if chunk_index <= last_completed_chunk:
            print(f"Chunk {chunk_index} already processed according to checkpoint. Skipping.")
            continue

        articles_in_chunk = articles[start_idx:end_idx]
        
        partial_json_path = os.path.join(
            run_partial_results_folder, f"batch_results_{chunk_index}.json"
        )
        partial_csv_path = os.path.join(
            run_partial_results_folder, f"flattened_results_{chunk_index}.csv"
        )
        partial_failed_path = os.path.join(
            run_partial_results_folder, f"failed_results_{chunk_index}.csv"
        )
        
        # If this batch was already processed (based on output files), skip.
        if os.path.exists(partial_json_path) and os.path.exists(partial_csv_path):
            print(f"Chunk {chunk_index} already processed (files exist). Skipping.")
            # Also update the checkpoint file just in case.
            with open(checkpoint_path, "w", encoding="utf-8") as cp:
                cp.write(str(chunk_index))
            continue

        # Build grouped_articles so each item has exactly one article
        grouped_articles = [[article] for article in articles_in_chunk]
        
        print(f"Processing chunk {chunk_index} ({len(articles_in_chunk)} articles)...")
        batch_start_time = time.time()
        
        # 1) Send requests in parallel
        batch_results = process_requests_in_parallel(grouped_articles, SYSTEM_MESSAGE)
        
        # 2) Serialize + Save the raw results
        serializable_results = []
        batch_tokens = 0
        for result in batch_results:
            if result['status'] == 'success' and result['response']:
                #response_data = result['response'].to_dict()
                response_data = result['response'].model_dump()
                result['response'] = response_data
                # track token usage
                batch_tokens += response_data.get("usage", {}).get("total_tokens", 0)
            serializable_results.append(result)
        
        # Ensure the directory for partial_json_path exists
        os.makedirs(os.path.dirname(partial_json_path), exist_ok=True)
        
        # Save raw results as JSON (once per chunk)
        with open(partial_json_path, "w", encoding="utf-8") as f:
            json.dump(serializable_results, f, indent=4, ensure_ascii=False)
        
        # 3) Flatten results + capture failures
        flattened_df, failed_df = flatten_results(batch_results)
        
        # 4) Save flattened "successful" classifications
        if flattened_df.empty:
            print(f"Warning: Flattened DataFrame is empty for chunk {chunk_index}.")
        else:
            flattened_df.to_csv(partial_csv_path, index=False, encoding="utf-8-sig")
        
        # 5) Save any failures in a separate CSV
        if not failed_df.empty:
            failed_df.to_csv(partial_failed_path, index=False, encoding="utf-8-sig")
            print(f"Failed requests saved to: {partial_failed_path}")
        
        # 6) Update tracking
        cumulative_tokens += batch_tokens
        total_processed_articles += len(articles_in_chunk)
        
        # 7) Log timing / pace
        batch_elapsed_time = time.time() - batch_start_time
        articles_per_minute = (
            len(articles_in_chunk) / (batch_elapsed_time / 60)
            if batch_elapsed_time > 0
            else 0
        )
        print(
            f"Finished chunk {chunk_index}. "
            f"Tokens used in this batch: {batch_tokens}. "
            f"Total tokens used so far: {cumulative_tokens}. "
            f"Elapsed time: {batch_elapsed_time:.2f} seconds. "
            f"Current pace: {articles_per_minute:.2f} articles/minute."
        )
        print(
            f"Batch {chunk_index}/{(total_articles - 1) // chunk_size + 1} completed. "
            f"Tokens this batch: {batch_tokens}, Total tokens: {cumulative_tokens}, "
            f"Articles/minute: {articles_per_minute:.2f}."
        )
        
        # 8) Update the checkpoint file with the current chunk index
        with open(checkpoint_path, "w", encoding="utf-8") as cp:
            cp.write(str(chunk_index))
    
    overall_elapsed_time = time.time() - overall_start_time
    print(f"All chunks processed successfully! Total elapsed time: {overall_elapsed_time:.2f} seconds.")
    print(f"Processing complete. Total elapsed time: {overall_elapsed_time:.2f} seconds.")

# --------------------------------------------------------------------
# (OPTIONAL) Final Merging Step
# --------------------------------------------------------------------
def merge_partial_results_into_final():
    """
    Merge all partial CSVs into a single final CSV in the unique `final_data_folder`.
    """
    partial_csv_files = [
        f for f in os.listdir(run_partial_results_folder)
        if f.startswith("flattened_results_") and f.endswith(".csv")
    ]
    partial_csv_paths = [os.path.join(run_partial_results_folder, f) for f in partial_csv_files]

    df_list = []
    for csv_path in partial_csv_paths:
        df_chunk = pd.read_csv(csv_path, encoding="utf-8-sig")
        df_list.append(df_chunk)

    if df_list:
        merged_df = pd.concat(df_list, ignore_index=True)
        final_csv_path = os.path.join(run_final_results_folder, "flattened_results_all.csv")
        merged_df.to_csv(final_csv_path, index=False, encoding="utf-8-sig")
        print(f"Final merged CSV saved at: {final_csv_path}")
    else:
        print("No partial CSV files found to merge.")





#%% Running Cell 3
# --------------------------------------------------------------------
# Running code
# --------------------------------------------------------------------
if __name__ == "__main__":
    # 1) Run classification in chunks of 100 (This means 100 aritcles at a time!)
    run_classification_in_chunks(chunk_size=100)

    # 2) (Optional) Merge partial results into one final CSV
    merge_partial_results_into_final()