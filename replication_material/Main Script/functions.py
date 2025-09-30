"""
@author: MatteoGrigoletto

Aim: Collect functions
"""

# IMPORTS -------------------------------------------------------------------#
import re
import os
import json
from collections import Counter
import matplotlib.pyplot as plt
import time
import pandas as pd
import emoji
import textstat

import spacy
nlp = spacy.load("en_core_web_sm")

from nrclex import NRCLex
from nltk.sentiment import SentimentIntensityAnalyzer

import nltk
nltk.download('stopwords')
nltk.download('punkt_tab')
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('vader_lexicon')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)  

from openai import OpenAI
api_key = os.environ.get('OPENAI_API_KEY')
client = OpenAI(api_key=api_key)
#-----------------------------------------------------------------------------#



#--------------------------       function       -----------------------------#
def change_paths(paths_list):
    '''
    Purpose:
        Create necessary paths for eveybody using the script.
    Input:
        List of paths; just add your own path in the list "paths_all". Path should take you to climate_change_narratives folder.
    Output:
        Relative paths.
    '''
    my_dir= os.getcwd()
    my_dir_list= re.split(r'\\|/', my_dir)
    for i, word in enumerate(my_dir_list):
        if "user" in word.lower():
            user= my_dir_list[i+1]
                                    
    for i in paths_list:
        if user in i:
            try:        
                root_dir = i
                input_folder = f"{root_dir}/input"
                python = f"{input_folder}/python"
                mTurk_tweets =  f"{input_folder}/mTurk/tweets"
                output_folder = f"{root_dir}/output"
                return root_dir, input_folder, python, mTurk_tweets, output_folder
            except:
                print("")
#-----------------------------------------------------------------------------#



#--------------------------       function       -----------------------------#
def process_text(text):
    '''
    Goal: Provide NLP of plain text, in the SpaCy sense.
    Input: text
    Output: NLP version of text
    '''
    return nlp(text)
#-----------------------------------------------------------------------------#



#--------------------------       function       -----------------------------#
def extract_entities_from_doc(doc, labels):
    '''
    Goal: Extract entities from NLP version of text.
    Input: NLP version of text, label of desired entity
    Output: Entities extracted
    '''
    return [ent.text for ent in doc.ents if ent.label_ in labels]
#-----------------------------------------------------------------------------#



#--------------------------       function       -----------------------------#
def get_top_entities(entities, top_n=30):
    '''
    Function to get the frequency of entities in the full dataset of data from the US
    '''
    all_entities = [entity for sublist in entities for entity in sublist]
    entity_counts = Counter(all_entities)
    return entity_counts.most_common(top_n)
#-----------------------------------------------------------------------------#



#--------------------------       function       -----------------------------#
def plot_top_entities(top_entities, title, output_path):
    '''
    Function to plot and save top entities.
    '''
    entities, counts = zip(*top_entities)
    plt.figure(figsize=(10, 8))
    plt.barh(entities, counts, color='skyblue')
    plt.xlabel('Frequency (Number of Tweets)')
    plt.ylabel('Entity')
    plt.title(title)
    plt.gca().invert_yaxis()  # Highest frequency on top
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()  # Clear the figure after saving
#-----------------------------------------------------------------------------#



#--------------------------       function       -----------------------------#
def count_keywords(text, keywords):
    '''
    Goal: Count the occurrences of specified keywords in text.
    Input: plain text, list of keywords
    Output: keyword count
    '''
    count = 0
    for keyword in keywords:
        count += text.lower().count(keyword.lower())
    return count
#-----------------------------------------------------------------------------#



#--------------------------       function       -----------------------------#
def plot_keyword_frequencies_with_note(df, keywords_dict, title):
    '''
    Goal: Plot the frequency of keywords in the dataset with a note.
    Input: DataFrame, dictionary of keywords, plot title
    Output: Bar plot of keyword frequencies with a note
    '''
    frequencies = {category: df[category].sum() for category in keywords_dict}
    sorted_frequencies = sorted(frequencies.items(), key=lambda item: item[1], reverse=True)
    categories, counts = zip(*sorted_frequencies)
    
    plt.figure(figsize=(10, 8))
    plt.barh(categories, counts, color='skyblue')
    plt.xlabel('Total Mentions in Tweets')
    plt.ylabel('Category')
    plt.title(title)
    plt.gca().invert_yaxis()  # Highest frequency on top

    # Return the plt object for further manipulation outside the function
    return plt
#-----------------------------------------------------------------------------#



#--------------------------       function       -----------------------------#
def preprocess(text):
    '''
    Goal: Remove punctuation and lemmatize the text.
    Input: plain text
    Output: list of tokens, lemmatized and without stopwords
    '''
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    
    # Tokenize the text
    tokens = word_tokenize(text.lower())
    
    # Remove stopwords and punctuation, and lemmatize
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word.isalpha() and word not in stop_words]
    
    return tokens
#-----------------------------------------------------------------------------#



#--------------------------       function       -----------------------------#
@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def completion_with_backoff(**kwargs):
    """ Makes an API call with exponential backoff on failure. """
    return client.chat.completions.create(**kwargs)
#-----------------------------------------------------------------------------#



#--------------------------       function       -----------------------------#
def predict_new_task(model_name, system_msg: str, user_msg: str, temperature: str) -> dict:
    """Predicts the relevance and character roles in a tweet with adjustable temperature settings."""
    
    # Set temperature based on input
    if temperature == "low":
        temp_value = 0
    elif temperature == "medium":
        temp_value = 0.3
    else:  # Default to high if anything else is specified
        temp_value = 0.8
    
    # Build the message payload
    messages = [{"role": "system", "content": system_msg}, {"role": "user", "content": user_msg}]
    
    try:
        response = completion_with_backoff(
            model=model_name,
            messages=messages,
            temperature=temp_value,
            max_tokens=130  # Adjusted to handle larger responses
        )
        #print(response)  # Optionally, keep this for debugging

        # Get the content of the response message
        response_content = response.choices[0].message.content

        # Clean the response if it contains code block formatting
        if response_content.startswith("```json"):
            response_content = response_content.strip("```json").strip("```")

        # Parse the cleaned JSON response
        prediction = json.loads(response_content)  # Load the response as a dictionary
        
        return prediction  # Return the full dictionary with all labels (keys)
    
    except json.JSONDecodeError as json_err:
        print(f"JSON decoding failed: {json_err}")
    except Exception as e:
        print(f"Exception {e}")
        if hasattr(e, 'response'):
            if hasattr(e.response, 'json'):
                print("Error response from API:", e.response.json())
        return None
#-----------------------------------------------------------------------------#



#--------------------------       function       -----------------------------#
# Main function to process the dataframe and obtain predictions
def main_openaiAPI(df_rpf, model_name, system_message, temperature="low"): 
    """Processes each tweet in a dataframe and predicts relevance and character roles, returning a DataFrame."""
    
    start_time = time.time()  # Start the timer
    predictions = []  # List to store predictions
    
    for index, row in df_rpf.iterrows():
        user_input = row['clean_text']
        tweet_id = row['tweet_id_str']
        print(f"Processing tweet ID {tweet_id}")
        
        # Get the prediction for the current tweet
        prediction = predict_new_task(model_name, system_message, user_input, temperature)
        
        # If the prediction is None, handle it gracefully
        if prediction is None:
            prediction = {"r": None, "a": None, "b": None, "c": None, "d": None, 
                          "e": None, "f": None, "g": None, "h": None, "i": None, 
                          "j": None, "k": None, "l": None}  # Default to None if no prediction is made
        
        # Append the results with the tweet ID, original text, and all the prediction keys
        predictions.append({
            'tweet_id_str': tweet_id,
            'original_text': user_input,
            'relevance': prediction.get('r', None),
            'developing': prediction.get('a', None),
            'democrats': prediction.get('b', None),
            'republicans': prediction.get('c', None),
            'corporations': prediction.get('d', None),
            'people': prediction.get('e', None),
            'pricing': prediction.get('f', None),
            'banning': prediction.get('g', None),
            'fossil': prediction.get('h', None),
            'green': prediction.get('i', None),
            'nuclear': prediction.get('j', None),
            'science': prediction.get('k', None),
            'media': prediction.get('l', None)
        })

    # Create a DataFrame from the predictions list
    predictions_df = pd.DataFrame(predictions)
    
    end_time = time.time()  # End the timer
    print(f"Execution time: {end_time - start_time} seconds")
    
    return predictions_df  # Return the dataframe with the predictions
#-----------------------------------------------------------------------------#



#--------------------------       function       -----------------------------#
def prepare_batch_input(df, system_message, temp, output_filename):

    # Create a list to hold the data
    data = []
    
    # Loop over each tweet in the dataframe and create the request data
    for index, row in df.iterrows():
        tweet_id = row['tweet_id_str']
        clean_text = row['clean_text']
        
        # Create the JSON structure for each tweet
        entry = {
            "custom_id": f"{tweet_id}",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": "gpt-4o-2024-08-06",
                "messages": [
                    {"role": "system", "content": f"{system_message}"},
                    {"role": "user", "content": clean_text}
                ],
                "temperature": temp,
                "max_tokens": 300
            }
        }
        
        data.append(entry)

    # Define the output path
    output_path = output_filename
    
    # Write the data to a JSONL file
    with open(output_path, 'w') as f:
        for entry in data:
            f.write(json.dumps(entry) + '\n')
    
    return output_path
#-----------------------------------------------------------------------------#



#--------------------------       function       -----------------------------#
def prepare_batch_input_profile(df, system_message, temp, output_filename):

    # Create a list to hold the data
    data = []
    
    # Loop over each tweet in the dataframe and create the request data
    for index, row in df.iterrows():
        tweet_id = row['author_id_str']
        clean_text = row['description']
        
        # Create the JSON structure for each tweet
        entry = {
            "custom_id": f"{tweet_id}",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": "gpt-3.5-turbo",
                "messages": [
                    {"role": "system", "content": f"{system_message}"},
                    {"role": "user", "content": clean_text}
                ],
                "temperature": temp,
                "max_tokens": 300
            }
        }
        
        data.append(entry)

    # Define the output path
    output_path = output_filename
    
    # Write the data to a JSONL file
    with open(output_path, 'w') as f:
        for entry in data:
            f.write(json.dumps(entry) + '\n')
    
    return output_path
#-----------------------------------------------------------------------------#



#--------------------------       function       -----------------------------#
def create_batch_and_save_id(input_filename, output_filename, description="Batch process for WA"):
 
    # Initialize the OpenAI client
    client = OpenAI()
    
    # Step 1: Upload the batch input file to OpenAI
    try:
        batch_input_file = client.files.create(
            file=open(input_filename, "rb"),
            purpose="batch"
        )
        batch_input_file_id = batch_input_file.id
        #print(f"Batch input file uploaded. File ID: {batch_input_file_id}")
    except Exception as e:
        print(f"Error uploading batch file: {e}")
        return None

    # Step 2: Create the batch
    try:
        batch = client.batches.create(
            input_file_id=batch_input_file_id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
            metadata={"description": description}
        )
        batch_id = batch.id
        #print(f"Batch created. Batch ID: {batch_id}")
    except Exception as e:
        print(f"Error creating batch: {e}")
        return None

    # Step 3: Save the batch ID to a file (JSON format)
    try:
        with open(output_filename, 'w') as f:
            json.dump({"batch_id": batch_id}, f)
        #print(f"Batch ID saved to {output_filename}")
    except Exception as e:
        print(f"Error saving batch ID to file: {e}")
        return None
    
    return "Well done! You just uploaded a batch!"
#-----------------------------------------------------------------------------#



#--------------------------       function       -----------------------------#
def retrieve_batch_id(output_filename):
    """
    Retrieves the batch ID from a saved file (JSON format).

    Args:
        output_folder (str): The folder where the batch ID file is stored.
        output_filename (str): Filename where the batch ID is saved (default: 'batch_id.json').

    Returns:
        str: The batch ID if found, or None if not.
    """
    try:
        with open(output_filename, 'r') as f:
            batch_info = json.load(f)
            batch_id = batch_info.get("batch_id")
        return batch_id
    except Exception as e:
        print(f"Error retrieving batch ID: {e}")
        return None
#-----------------------------------------------------------------------------#



#--------------------------       function       -----------------------------#
def check_batch_status_and_read_output(batch_id):
    """
    Check the status of a batch in the OpenAI API. If completed, retrieve the output and return it as a string.

    Args:
        batch_id (str): The ID of the batch to check.
        output_folder (str): Folder where the output will be saved.

    Returns:
        str: The batch output as a string, or None if the batch is not completed or there's an error.
    """
    
    # Initialize OpenAI client
    client = OpenAI()
    
    # Step 1: Retrieve the batch status
    try:
        batch_out = client.batches.retrieve(batch_id)
        
        if batch_out.status != 'completed':
            print(f"Batch {batch_id} is {batch_out.status}.")
            return None
        
        # Step 2: Retrieve the output file if batch is completed
        file_response = client.files.content(batch_out.output_file_id)
        # Use read() to convert the file content into a string
        output_str = file_response.read().decode("utf-8")
        return output_str
    
    except Exception as e:
        print(f"Error retrieving batch status or output file: {e}")
        return None
#-----------------------------------------------------------------------------#




#--------------------------       function       -----------------------------#
def parse_batch_output(output_str):
    """
    Parses the batch output string into a dictionary where 'custom_id' is the key and the content is parsed 
    as a dictionary.

    Args:
        output_str (str): The batch output as a string.

    Returns:
        dict: A dictionary with 'custom_id' as keys and corresponding parsed content (dictionary) as values.
    """
    
    try:
        # Initialize the result dictionary
        results = {}
        
        # Split the response text by line since it's a JSONL format
        parsed_responses = output_str.splitlines()

        for response in parsed_responses:
            # Skip empty lines or lines that only contain whitespace
            if not response.strip():
                continue
            
            try:
                # Load each line as a JSON dictionary
                response_json = json.loads(response)
                
                # Extract the custom_id and content from the choices
                custom_id = response_json.get("custom_id")
                content = response_json["response"]["body"]["choices"][0]["message"]["content"]
                
                # Clean the content by stripping any unwanted characters like ```json``` markers
                content_cleaned = content.strip("```json").strip("```").strip()
                
                # Replace Python-style None with JSON-compatible null
                content_cleaned = content_cleaned.replace("None", "null")
                
                # Print the cleaned content to inspect it
                #print(f"Custom ID: {custom_id}, Cleaned Content: {content_cleaned}")
                
                # Parse the cleaned content (which is currently a string) into a dictionary
                parsed_content = json.loads(content_cleaned)
                
                # Add to the dictionary with custom_id as the key and the parsed content as the value
                results[custom_id] = parsed_content
            
            except json.JSONDecodeError as e:
                print(f"Error parsing content for custom_id {custom_id}: {content_cleaned} - {e}")
                continue
        
        return results
    
    except Exception as e:
        print(f"Error parsing the batch output: {e}")
        return None
#-----------------------------------------------------------------------------#



#--------------------------       function       -----------------------------#
def create_and_save_final_df_stage1(parsed_results, origin_df, output_filename):
    """
    Converts the parsed results dictionary into a DataFrame with the custom_id as the first column and the
    prediction values as subsequent columns.

    Args:
        parsed_results (dict): Dictionary where custom_id is the key and the content is a dictionary of predictions.

    Returns:
        pd.DataFrame: A DataFrame containing tweet IDs (custom_ids) and associated prediction values.
    """
    # Initialize an empty list to store the rows
    data = []
    
    # Iterate over the parsed results dictionary
    for custom_id, prediction in parsed_results.items():
        # Append each row as a dictionary, where 'tweet_id_str' is the custom_id and the rest are the values
        row = {
            'tweet_id_str': custom_id,         # custom_id as tweet_id
            'relevance': prediction.get('r', None),
        }
        data.append(row)
    
    # Convert the list of rows into a DataFrame
    df = pd.DataFrame(data)
    
    # Merge with origin batch file
    df_stage1 = pd.merge(origin_df, df, on='tweet_id_str', how='inner')
    
    # Save
    df_stage1.to_csv(output_filename, index=False)

    return "Howdy! Created and saved the output of stage 1"
#-----------------------------------------------------------------------------#



#--------------------------       function       -----------------------------#
def create_and_save_final_df_stage2(parsed_results, origin_df, output_filename):
    """
    Converts the parsed results dictionary into a DataFrame with the custom_id as the first column and the
    prediction values as subsequent columns.

    Args:
        parsed_results (dict): Dictionary where custom_id is the key and the content is a dictionary of predictions.

    Returns:
        pd.DataFrame: A DataFrame containing tweet IDs (custom_ids) and associated prediction values.
    """
    # Initialize an empty list to store the rows
    data = []
    
    # Iterate over the parsed results dictionary
    for custom_id, prediction in parsed_results.items():
        # Append each row as a dictionary, where 'tweet_id_str' is the custom_id and the rest are the values
        row = {
            'tweet_id_str': custom_id,         # custom_id as tweet_id
            'developing': prediction.get('a', None),
            'democrats': prediction.get('b', None),
            'republicans': prediction.get('c', None),
            'corporations': prediction.get('d', None),
            'people': prediction.get('e', None),
            'pricing': prediction.get('f', None),
            'banning': prediction.get('g', None),
            'fossil': prediction.get('h', None),
            'green': prediction.get('i', None),
            'nuclear': prediction.get('j', None),
        }
        data.append(row)
    
    # Convert the list of rows into a DataFrame
    df = pd.DataFrame(data)
    
    # Merge with origin batch file
    df_stage2 = pd.merge(origin_df, df, on='tweet_id_str', how='inner')
    
    # Save
    df_stage2.to_csv(output_filename, index=False)

    return "Howdy! Created and saved the output of stage 2"
#-----------------------------------------------------------------------------#



#--------------------------       function       -----------------------------#
def create_and_save_final_df_profile(parsed_results, origin_df, output_filename):
    """
    Converts the parsed results dictionary into a DataFrame with the custom_id as the first column and the
    prediction values as subsequent columns.

    Args:
        parsed_results (dict): Dictionary where custom_id is the key and the content is a dictionary of predictions.

    Returns:
        pd.DataFrame: A DataFrame containing tweet IDs (custom_ids) and associated prediction values.
    """
    # Initialize an empty list to store the rows
    data = []
    
    # Iterate over the parsed results dictionary
    for custom_id, prediction in parsed_results.items():
        # Append each row as a dictionary, where 'author_id_str' is the custom_id and the rest are the values
        row = {
            'author_id_str': custom_id,         # custom_id as tweet_id
            'politics': prediction.get('a', None),
            'religion': prediction.get('b', None),
            'education': prediction.get('c', None),
            'children': prediction.get('d', None)
        }
        data.append(row)
    
    # Convert the list of rows into a DataFrame
    df = pd.DataFrame(data)
    
    # Merge with origin batch file
    df_profile = pd.merge(origin_df, df, on='author_id_str', how='inner')
    
    # Save
    df_profile.to_csv(output_filename, index=False)

    return "Howdy! Created and saved the output of profile annotation!"
#-----------------------------------------------------------------------------#



#--------------------------        class         -----------------------------#
class TweetPreprocessor:
    def __init__(self, emoji_dict, emoticon_dict):
        self.emoji_dict = emoji_dict
        self.emoticon_dict = emoticon_dict
        self.sia = SentimentIntensityAnalyzer()

    # Existing functions for text processing
    def mention_count(self, text):
        return len(re.findall(r'@\w+', text))

    def hashtag_count(self, text):
        return len(re.findall(r'#\w+', text))

    def emoji_and_emoticon_count(self, text):
        emoji_count = sum(text.count(emoji) for emoji in self.emoji_dict.keys())
        emoticon_count = sum(text.count(emoticon) for emoticon in self.emoticon_dict.keys())
        return emoji_count + emoticon_count

    def unique_word_count(self, text):
        words = re.findall(r'\w+', text)
        return len(set(words))

    def cap_word_count(self, text):
        return len(re.findall(r'\b[A-Z]{2,}\b', text))

    def char_count(self, text):
        return len(text)

    def word_count(self, text):
        return len(re.findall(r'\w+', text))

    def excl_count(self, text):
        return text.count('!')

    def punct_count(self, text):
        return text.count('?')

    def cap_letters_count(self, text):
        return len(re.findall(r'[A-Z]', text))

    def readability_score(self, text):
        return textstat.flesch_reading_ease(text)

    def named_entity_count(self, doc):
        return len(doc.ents)
    
    # NRC emotion analysis
    def analyze_emotions(self, text):
        """
        Analyze the text and return binary indicators for the presence of each emotion.

        Returns:
        dict: A dictionary with emotion labels as keys. Each value is 1 if the emotion is present in the text, 0 otherwise.
        """
        # Possible emotions according to NRCLex
        possible_emotions = ['fear', 'anger', 'trust', 'surprise', 'positive', 'negative', 
                            'sadness', 'disgust', 'joy', 'anticipation']

        # Get the affect frequencies from NRCLex
        nrc_lex = NRCLex(text)
        affect_frequencies = nrc_lex.affect_frequencies

        # Initialize the binary emotion counts
        binary_emotion_counts = {emotion: 0 for emotion in possible_emotions}

        # Set binary count to 1 if frequency is greater than 0
        for emotion in possible_emotions:
            freq = affect_frequencies.get(emotion, 0)
            binary_emotion_counts[emotion] = 1 if freq > 0 else 0

        return binary_emotion_counts
    
    def count_emotions(self, text):
        """
        Analyze the text and return the raw frequency counts of each emotion.

        Returns:
        dict: A dictionary with emotion labels as keys and their raw frequency counts as values.
        """
        # Initialize a dictionary with all possible emotions set to zero
        all_emotions = ['fear', 'anger', 'trust', 'surprise', 'positive', 'negative', 
                        'sadness', 'disgust', 'joy', 'anticipation']
        emotion_counts = {emotion: 0 for emotion in all_emotions}

        # Get the actual counts from NRCLex
        counts = NRCLex(text).raw_emotion_scores

        # Update the dictionary with the actual counts
        for emotion, count in counts.items():
            if emotion in emotion_counts:
                emotion_counts[emotion] = count

        return emotion_counts

    def get_emotion_scores(self, text):
        """
        Analyze the text and return the normalized emotion scores from the NRC Lexicon.

        Returns:
        dict: A dictionary with emotion labels as keys and their normalized frequency scores as values.
        """
        # Initialize a dictionary with all possible emotions set to zero
        all_emotions = ['fear', 'anger', 'trust', 'surprise', 'positive', 'negative', 
                        'sadness', 'disgust', 'joy', 'anticipation']
        emotions_score = {emotion: 0 for emotion in all_emotions}

        # Get the actual scores from NRCLex
        scores = NRCLex(text).affect_frequencies

        # Update the dictionary with the actual scores
        for emotion, score in scores.items():
            if emotion in emotions_score:
                emotions_score[emotion] = score

        return emotions_score
        
    # NLTK sentiment analysis
    def nltk_sentiment(self, text):
        return self.sia.polarity_scores(text)["compound"]

    # Complexity analysis functions
    def calculate_ttr(self, text):
        words = text.split()
        return len(set(words)) / len(words) if words else 0

    def calculate_lexical_density(self, text):
        tokens = nltk.word_tokenize(text)
        content_words = [word for word in tokens if word.isalnum()]
        return len(content_words) / len(tokens) if tokens else 0

    def calculate_avg_word_length(self, text):
        words = text.split()
        return sum(len(word) for word in words) / len(words) if words else 0

    def calculate_avg_sentence_length(self, text):
        sentences = nltk.sent_tokenize(text)
        return sum(len(sentence.split()) for sentence in sentences) / len(sentences) if sentences else 0
    
    # Apply SpaCy NLP to text once and return the processed doc and lemmatized text
    def nlp_process(self, text):
        doc = nlp(text)
        lemmatized_text = ' '.join([token.lemma_ for token in doc])
        return doc, lemmatized_text

    # Main preprocessing method that applies all analysis functions to a text DataFrame
    def preprocess(self, df, text_id, text_column):
        
        # Step 1: Apply basic text feature functions
        df['mention_count'] = df[text_column].apply(self.mention_count)
        df['hashtag_count'] = df[text_column].apply(self.hashtag_count)
        df['emoji_emotic_count'] = df[text_column].apply(self.emoji_and_emoticon_count)
        df['unique_word_count'] = df[text_column].apply(self.unique_word_count)
        df['cap_word_count'] = df[text_column].apply(self.cap_word_count)
        df['char_count'] = df[text_column].apply(self.char_count)
        df['word_count'] = df[text_column].apply(self.word_count)
        df['excl_count'] = df[text_column].apply(self.excl_count)
        df['punct_count'] = df[text_column].apply(self.punct_count)
        df['cap_letters_count'] = df[text_column].apply(self.cap_letters_count)
        df['readability_score'] = df[text_column].apply(self.readability_score)

        # Step 2: Apply SpaCy NLP processing (e.g., lemmatization) only once
        nlp_results = df[text_column].apply(self.nlp_process)
        # Store lemmatized text as the 'nlp' column
        df['nlp'] = nlp_results.apply(lambda x: x[1])
        # Count named entities using SpaCy NLP processing
        df['named_entity_count'] = nlp_results.apply(lambda x: self.named_entity_count(x[0]))

        # Step 3: Perform NRC emotion analysis on text
        # Generate emotion analysis columns from text
        emotion_columns = df[text_column].apply(self.analyze_emotions)
        # Convert emotion analysis output to individual columns
        emotion_df = emotion_columns.apply(pd.Series)
        df = pd.concat([df, emotion_df], axis=1)

        # Rename emotion columns to have 'senan_' prefix
        columns_to_rename = ['fear', 'anger', 'trust', 'surprise', 'positive', 'negative', 
                            'sadness', 'disgust', 'joy', 'anticipation', 'nltk_sent']
        renamed_columns = {col: 'senan_' + col for col in columns_to_rename}
        df.rename(columns=renamed_columns, inplace=True)

        # Step 4: Apply NRC emotion count analysis for word-based emotion counts
        count_emotion_columns = df[text_column].apply(self.count_emotions)
        count_emotion_df = count_emotion_columns.apply(pd.Series)
        df = pd.concat([df, count_emotion_df], axis=1)

        # Rename emotion count columns to have 'count_' prefix
        count_emotion_columns = ['fear', 'anger', 'trust', 'surprise', 'positive', 'negative',
                                'sadness', 'disgust', 'joy', 'anticipation']
        count_renamed_columns = {col: 'count_' + col for col in count_emotion_columns}
        df.rename(columns=count_renamed_columns, inplace=True)
        
        # Step 5: Calculate NRC emotion scores (e.g., sentiment intensity)
        emotion_score_columns = df[text_column].apply(self.get_emotion_scores)
        emotion_score_df = emotion_score_columns.apply(pd.Series)
        df = pd.concat([df, emotion_score_df], axis=1)

        # Rename emotion score columns to have 'score_' prefix
        score_emotion_columns = ['fear', 'anger', 'trust', 'surprise', 'positive', 'negative',
                                'sadness', 'disgust', 'joy', 'anticipation']
        score_renamed_columns = {col: 'score_' + col for col in score_emotion_columns}
        df.rename(columns=score_renamed_columns, inplace=True)

        # Step 6: Apply NLTK sentiment analysis to text
        df['nltk_sent'] = df[text_column].apply(self.nltk_sentiment)

        # Step 7: Apply text complexity and readability analyses
        # Calculate Flesch reading ease score
        df['clxty_reading_ease'] = df[text_column].apply(textstat.flesch_reading_ease)
        # Calculate Gunning Fog index
        df['clxty_gunning_fog'] = df[text_column].apply(textstat.gunning_fog)
        # Calculate type-token ratio (lexical diversity)
        df['clxty_type_token_ratio'] = df[text_column].apply(self.calculate_ttr)
        # Calculate lexical density
        df['clxty_lex_density'] = df[text_column].apply(self.calculate_lexical_density)
        # Calculate average word length
        df['clxty_avg_word_len'] = df[text_column].apply(self.calculate_avg_word_length)
        # Calculate average sentence length
        df['clxty_avg_sent_len'] = df[text_column].apply(self.calculate_avg_sentence_length)

        # Step 8: Reorder columns to place 'nlp' column after the original text column
        cols = list(df.columns)
        text_col_index = cols.index(text_column)
        cols.insert(text_col_index + 1, cols.pop(cols.index('nlp')))
        df = df[cols]

        # Return the processed DataFrame
        return df
#-----------------------------------------------------------------------------#