import pandas as pd
import numpy as np
import openai
from tqdm import tqdm
import time
from openai import OpenAI
import argparse
import google.generativeai as genai

def create_prompt(comment, prompt='simulate_abstract'):
    #function to create a separate LLM prompt for each Reddit post
    if prompt == 'simulate_abstract':
        return f"""You will be given a Reddit comment that references a research paper, but the research paper is not cited in the comment. Can you write an abstract for the paper the comment is discussing? Please only answer with the abstract.

        Even if you feel the comment is missing crucial details of the study design, please do your best to write what a potential abstract for the paper might be.
        
        Comment:

        {{{comment}}}
        """
    
    elif prompt == 'remove_text':
        return f"""You will be given a Reddit comment that references a research paper, but the research paper is not cited in the comment. The comment may also contain extraneous information that is not related to the research paper. Can you repeat the comment, but this time leaving out details that do not concern the referenced paper?

        Comment:

        {{{comment}}}
        """
    
    elif prompt == 'scientific_style':
        return f"""You will be given a Reddit comment that references a research paper. Can you rewrite the comment in a more scientific style?

        Comment:

        {{{comment}}}
        """
    
    elif prompt == 'search_query':
        return f"""You will be given a Reddit comment that references a research paper from ArXiv, but the research paper is not cited in the comment. You are a helpful assistant that will help us find the original research paper. Can you create a search query to find the original research paper from ArXiv? Please output the search query as a list of keywords, separated by commas.

        Comment:

        {{{comment}}}
        """
    
    else:
        raise ValueError("the prompt argument should be one of: 'simulate_abstract', 'remove_text', 'scientific_style', or 'search_query'")
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_path", type=str)
    parser.add_argument("--column_names", type=str, nargs="+")
    parser.add_argument("--llm", type=str)
    parser.add_argument("--prompt", type=str)
    args = parser.parse_args()

    file_path = args.file_path
    column_names = args.column_names
    prompt = args.prompt
    llm = args.llm



    #load the dataset
    posts_df = pd.read_csv(file_path)

    #create a column with stuff to prompt the llm with
    for column_name in column_names:
        posts_df[f'{column_name}_{prompt}'] = posts_df[column_name].apply(create_prompt, args=[prompt])

    
        if llm == 'gemini-1.5-flash':
            genai.configure(api_key='GOOGLE_API_KEY')
            model = genai.GenerativeModel(llm)

            responses = [] #set up list to store responses
            for prompt in tqdm(posts_df[f'{column_name}_{prompt}']):
                response = model.generate_content(prompt)
                responses.append(response.text)
                time.sleep(4) # wait to use the model so you don't get rate limited

            posts_df[f'{column_name}_{prompt}_{llm}_response'] = responses

        else:
            if llm == 'gpt-3.5-turbo' or llm == 'gpt-4o-mini':
                client = OpenAI(api_key="OPENAI_API_KEY") #use openai api key for openai models

            elif llm == 'llama3.1-70b-instruct-berkeley':
                client = OpenAI(api_key="LAMBDA_API_KEY",
                                base_url="https://api.lambdalabs.com/v1") #use lambda api for llama
                
            responses = []
            for prompt in tqdm(posts_df[f'{column_name}_{prompt}']):
                completion = client.chat.completions.create(model=llm, messages=[{"role": "user", "content": prompt}])
                responses.append(completion)

            posts_df[f'{column_name}_{prompt}_{llm}_response'] = responses
    
    filename = file_path.split(".")[0]
    posts_df.to_csv(f"{filename}_{prompt}_{llm}.csv", index=False)
            
