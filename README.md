# Political Narratives Package

The **Political Narratives** Package allows users to adapt the Political Narratives framework presented in *Gehring & Grigoletto (2025)* to their own research. By following the steps described in this repository, interested users will be able to identify the occurrence of political narratives in their data sources. 

**What is a Political Narrative?** 

We provide you with the code to query the OpenAI API, the prompts that can be used (or adapted) to retrieve Political Narratives, and guidelines to adopt the Political Narrative framework.  
Moreover, at the link [Launch the Political Narratives App](https://political-narratives-package-jagwm2r46rtwhevafwwga5.streamlit.app/) you will find an interactive website that guides you through the logic behind the Political Narrative Framework.
This guide allows you to navigate the steps to prepare your research: you can reflect on the main questions to ask yourself, check them, and annotate your progress. In addition, this interactive resource provides clarifying examples taken from the paper of reference.

Before diving to the instructions to run the code, here a detailed explanation of everything you find in this repository:
- ````app\````: this folder contains the code for the online guideline that interested users can access [here](https://political-narratives-package-jagwm2r46rtwhevafwwga5.streamlit.app/). This folder do not provide useful insights to the user.
- ````code and prompts\````: this is the core of the repository. Here the users can download the Python scripts and the prompts needed to perform the Political Narrative analysis.
- ````data\````: here is contained the paper by *Gehring & Grigoletto (2025)*. Users can access it and get a deeper understanding on how to shape a research using the framework.
  
### How to Proceed? ✅
There are two ways to approach this package:

1) The first one is the independent approach, where users advanced in their research can simply download the Python code to perform the annotation and apply it to their data. The steps to follow are listed below.
   
2) The second approach is useful for less expereinced users that need to completely shape their research. Through the [online guide](https://political-narratives-package-jagwm2r46rtwhevafwwga5.streamlit.app/) they can organize their research from scratch thanks to a set of very useful instructions and best practices.

## Instructions to adopt the Political Narrative framework ​🗂️​
We provide here a step-by-step explanation on how to adapt the code and run it on your machine.

- (For beginner Python users) The first step is to download Python at the [link](https://www.python.org/downloads/).

- (For beginner Python users) Then, we suggest you to download Anaconda and Anaconda Prompt at the [link](https://www.anaconda.com/docs/getting-started/anaconda/install).

- Download the files in ````code and prompts```` and the ````environment.yml````. Then, the user must build the following folder structure locally:
  ```bash
  <main>/
  ├─ code/
  │  ├─ annotation_openai_stage1.py
  │  ├─ annotation_openai_stage2.py
  │  └─ prompts/
  │     └─ system_message_stage1.json        # <-- your input message
  │     └─ system_message_stage2.json        # <-- your input message
  ├─ data/
  │  └─ output/
  │     └─ your_data_stage1.csv              # <-- your input CSV
  │     └─ your_data_stage2.csv              # <-- your input CSV
  ├─ output/
  │  └─ data/
  │     ├─ openai_output/             
  │     └─ openai_final/              
  └─ logs/
  └─ environment.yml                         # <-- your environment requiremnts
According to this structure, the user needs to fill the ````code/prompts/```` folder with the system message of the stage, and the ````data/output/```` folder with the ````.csv```` dataset. Put the ````environment.yml```` in the main project folder.  
IMPORTANT: the input ````.csv```` must contain a column called ````id```` and a column called ````text````.

- Open Anaconda Prompt and move into the main project folder:
  ````bash
  cd C:\Users\YourName\Documents\political_narrative_project

- Then, create the environment where the code will run:
  ````bash
  conda env create -f environment.yml

- Activate the environment
  ````bash
  conda activate political_narrative

- Install the correct version of ````openai````
  ````bash
  pip install "openai==2.4.0"

- Set the ````OPENAI_API_KEY```` as an environmental variable in your current environment:
    ````bash
    conda env config vars set OPENAI_API_KEY="sk-your-key-here"
    conda deactivate
    conda activate political_narrative

This step is crucial for the success of the annotation process. This code requires an individual OpenAI API key, that the user can retrieve in his [OpenAI personal page](https://platform.openai.com/api-keys). 
This key is personal and directly linked to the user's wallet, so it's important to keep it personal and hidden in the machine and not in the script.

- Open Anaconda Prompt and activate Spyder or your preferred Python IDE by running the commands:
  ````bash
  conda install spyder
  spyder
  
- Open the script of interest directly in spyder using the top left command bar. Here the user can choose one of the two Python scripts depending on the task that he needs to perform:
  - The **stage 1 script**  ````annotation_openai_stage1```` allows the user tho classify a text based on its relevance to the topic selected. This code returns an additional column to the input dataset that takes values from 0 to 3 (0 - irrelevant, 1 - assert, 2 - deny, 3 - relevant).
  - The **stage 2 script** ````annotation_openai_stage2```` is the core of the Political Narrative Package, and it allows the user to retrieve the character-role classification. The code returns a dataset with a column for each specified character taking values from 0 to 4, where 0 is no-mention of the character, 1 is Villain role, 2 is Hero role, 3 is Victim role, and 4 for appearence of the character in none of these roles (Neutral).

- Regarding the prompt, the user can adapt the prompt in the folder ````code/prompts/```` with his own instructions. The user can do this by accessing the "SYSTEM MESSAGE" in the prompt. To modify the task, the user must change the keys (in the example from a to j) to match the number of characters of his analysis. Moreover, descriptions of the characters are required in order for OpenAI to perform a meaningful classification. An example of the prompt is provided here:
  ````bash
  {
  "SYSTEM_MESSAGE": "You are an average US citizen. The user will provide a three-sentence US newspaper excerpt (2010–2021). 
  Analyze it in the context of US political discourse on climate change and respond in JSON format.

  1. Character Analysis: For each mentioned character, assign a role:
   - 1 = Villain: contributes to problems, opposes positive change, engages in harmful actions.
   - 2 = Hero: leads efforts to combat climate change, promotes environmental policies, acts commendably.
   - 3 = Victim: suffers unfairly, is attacked, or endures consequences of climate change or others’ actions.
   - 4 = No role: mentioned but not clearly cast as Villain/Hero/Victim, or context is neutral/ambiguous.

  2. Characters to evaluate (keys a–j):
   a: Developing Economies — emerging and poorer nations (incl. BRICS). Mentions of their governments, representatives, or citizens in the context of climate negotiations or responsibilities.
   b: US Democrats — politicians or public figures tied to the Democratic Party (e.g., Biden, Obama, Pelosi). References to Democratic climate policies, proposals, or positions.
   c: US Republicans — politicians or public figures tied to the Republican Party (e.g., Trump, McConnell, Cruz). Mentions of opposition or support to climate-related policies from the GOP side.
   d: Corporations and Industry — private-sector actors including large companies, SMEs, banks, CEOs, and industry lobbies. Includes mentions of energy, tech, finance, or manufacturing in the climate context.
   e: US People — the collective public, workers, voters, youth, and grassroots movements (e.g., Sunrise Movement, Extinction Rebellion). Includes references to citizens as beneficiaries, victims, or agents of climate action.
   f: Emission Pricing Tools — market-based policies like carbon taxes, cap-and-trade, carbon markets, or pollution credits. References to pricing carbon as a solution or a burden.
   g: Regulation Policies — government bans or regulations (e.g., banning fracking, fossil phaseouts, plastics bans, degrowth/anti-capitalist proposals). Mentions of regulatory tools to address climate change.
   h: Fossil Fuels — oil, gas, coal, and related infrastructure (power plants, pipelines). Mentions in the context of pollution, energy needs, or phaseout debates.
   i: Green Technologies — renewable and low-carbon solutions like solar, wind, EVs, hydrogen, batteries, geothermal. Mentions of their potential, adoption, or challenges.
   j: Nuclear Energy — nuclear power, fission, or fusion technologies. Mentions of nuclear plants or research in the climate policy debate.

  3. Final Output: Respond with a JSON object containing keys a–j. Each value must be 0 (not mentioned) or 1–4 (role as defined above)."
  }

- Modify the input dataset: the user needs to provide in the ````/data/output```` folder his dataset named ````your_data_stage2```` (or ````your_data_stage1````). This dataset must contain a column called **id** with the unique identifiers and a column called **text** containing the text for the classification.

- - Your ````.csv```` file must be UTF-8 encoded. If you see errors such ````unicodeDecodeError```` try saving again your file in UTF-8, or the script will automatically fall back to latin-1 encoding.

- Once prepared the folder structure, the prompt, and the dataset, the user can make minimal changes to the Python script. First, changing the directory
  ````bash
  main = r"D:\your directory"

- Second, change the JSON structure based on the number of characters: depending on your character selection, you might need to modify the structure of the ````.JSON```` file that is created by OpenAI. You can access it by changing the properties in the following part of the script:
  ````bash
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
                            "f": {"type": "integer", "enum": [0, 1, 2, 3, 4]}                   
                            },
                        "required": ["id","a","b","c","d","e","f"],
                        "additionalProperties": False
                    }
                }
             },
            "required": ["items"],
            "additionalProperties": False
        },
        "strict": True
  }

- Then, the user needs to change the user content to adapt it to his number of characters, as done before: 
  ````bash
  def build_user_content(article_group):
    
    """
    Build the user content for the OpenAI API.
    You need each "article" to be a dictionary like {"id":1, ""text": "bla bla bla...""}
    """
    
    user_content = (
        "Classify the following opinion(s) strictly per the system instructions. "
        "Respond with **only** a JSON object of the form {\"items\": [ ... ]}, "
        "where \"items\" is an array of objects (one per text, same order). "
        "Each object must include keys: id, a, b, c, d, e, f (a–f in 0–4). "
        "No extra text.\n\n"
    )
    for article in article_group:
        user_content += f"ID: {article['id']}\n"
        user_content += f"Text: {article['text']}\n"
    return user_content

- Lastly, he needs to adapt the `flatten_results()` function by changing this few lines of code: 
  ````bash
                          "request_id": result["request_id"],
                        "article_id": result.get("article_id"),
                        "id": entry_id,
                        "a": entry.get("a", 0),
                        "b": entry.get("b", 0),
                        "c": entry.get("c", 0),
                        "d": entry.get("d", 0),
                        "e": entry.get("e", 0),
                        "f": entry.get("f", 0)

## Inputs - Output map
Inputs:
- dataset with text snippets and observation id (in the code ```your_code_stage1.csv``` and ```your_code_stage2.csv```)
- prompt for stage 1 (in the code ```system_message_stage1.json```)
- prompt for stage 2 (in the code ```system_message_stage2.json```)
- API key as Environmental variable
  
Outputs:
- Dataset with characther-role flags (if you run the second script):
  
  | id   | text               | dev | dem | rep | corp | ppl | pric | ban | fos | green | nuc |
  |------|--------------------|-----|-----|-----|------|-----|------|-----|-----|-------|-----|
  | 1_1  | If you listen…     |  0  |  0  |  2  |  0   |  0  |  0   |  0  |  1  |   0   |  0  |
  | 1_2  | But is it po…      |  0  |  0  |  0  |  0   |  0  |  0   |  0  |  0  |   0   |  0  |
  | 1_3  | In fact, on…       |  0  |  1  |  0  |  0   |  0  |  2   |  0  |  0  |   0   |  0  |


