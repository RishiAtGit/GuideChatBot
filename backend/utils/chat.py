import json
import re
import os
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate, FewShotPromptTemplate
import google.generativeai as genai
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
import warnings

# Suppress specific FutureWarning
warnings.filterwarnings("ignore", category=FutureWarning, message="`resume_download` is deprecated")
warnings.filterwarnings("ignore", category=FutureWarning, module="sentence_transformers")

# Load environment variables
load_dotenv()

# Configure Gemini API
genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))

# Configure Pinecone
pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
index_name = "maharashtra-forts"
index = pc.Index(index_name)

# Model setup for query encoding
model_name = "sangmini/msmarco-cotmae-MiniLM-L12_en-ko-ja"
retriever = SentenceTransformer(model_name)

# File path for your prompts.json
file_path = "backend/inputs/prompts.json"

# In-memory storage for conversation history
conversation_history = {}

def read_json_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    
    context = data["context"]
    examples = data["examples"]
    formatted_examples = []
    
    for example in examples:
        formatted_examples.append({
            "input": example["human"],
            "output": example["assistant"]
        })
    
    return context, formatted_examples

def get_gemini_response(prompt):
    try:
        model = genai.GenerativeModel('gemini-2.0-flash')
        
        safety_settings = [
            {
                "category": "HARM_CATEGORY_HARASSMENT",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE"
            },
            {
                "category": "HARM_CATEGORY_HATE_SPEECH",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE"
            },
            {
                "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE"
            },
            {
                "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE"
            }
        ]

        response = model.generate_content(
            prompt,
            safety_settings=safety_settings,
            generation_config={
                "temperature": 0.7,
                "top_p": 0.9,
                "top_k": 40,
                "max_output_tokens": 2048,
            }
        )
        
        return response.text
    except Exception as e:
        print(f"Error in get_gemini_response: {e}")
        return "I'm having trouble right now. Can we try again?"

def format_response(response):
    def convert_table(match):
        lines = match.group(1).strip().split('\n')
        html_table = "<table style='border-collapse: collapse; width: 100%; font-size: 0.6em;'>"
        for i, line in enumerate(lines):
            if i == 1 and set(line.strip()) <= set('|-'):  # Skip separator line
                continue
            cells = [cell.strip() for cell in line.split('|') if cell.strip() != '']
            if not cells:  # Skip empty rows
                continue
            html_table += "<tr>"
            cell_tag = "th" if i == 0 else "td"
            for cell in cells:
                html_table += f"<{cell_tag} style='border: 1px solid #ddd; padding: 8px; text-align: left;'>{cell if cell else '&nbsp;'}</{cell_tag}>"
            html_table += "</tr>"
        html_table += "</table>"
        return html_table

    # Convert markdown tables to HTML
    response = re.sub(r'((?:\n\|.*)+)', lambda m: convert_table(m), response)

    # Convert numbered lists and bullet points
    response = re.sub(r'(?m)^(\d+)\.\s', r'<br>\1. ', response)
    response = re.sub(r'(?m)^•\s', r'<br>• ', response)

    # Add bold to headings and important phrases
    response = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', response)
    
    # Replace double line breaks first
    response = response.replace('\n\n', '<br><br>')

    # Add emphasis to key phrases
    response = re.sub(r'\*(.*?)\*', r'<em>\1</em>', response)

    # Replace single line breaks after replacing double line breaks
    response = response.replace('\n', '<br>')

    # Remove any leading/trailing whitespace and extra new lines
    response = response.strip()
    response = re.sub(r'^(<br>)+|(<br>)+$', '', response)

    # Remove extra spaces between HTML tags
    response = re.sub(r'>\s+<', '><', response)

    # Remove extra spaces after new lines
    response = re.sub(r'<br>\s+', '<br>', response)

    return response.strip()

def clean_response(response):
    response = re.sub(r'^(Human:|Assistant:)\s*', '', response, flags=re.IGNORECASE)
    response = re.sub(r'^\d+\)\s*', '', response)
    response = re.sub(r'^Let\'s approach this step-by-step:\s*', '', response, flags=re.IGNORECASE)
    return response.strip()

def get_relevant_forts(query, top_k=5):
    query_embedding = retriever.encode([query])[0].tolist()
    results = index.query(
        vector=query_embedding,
        top_k=top_k,
        include_metadata=True
    )
    
    relevant_forts = []
    for match in results['matches']:
        fort_info = match['metadata'].copy()
        fort_info['score'] = float(match['score'])
        
        for key, value in fort_info.items():
            fort_info[key] = str(value)
        
        relevant_forts.append(fort_info)
    
    return relevant_forts

def is_fort_related(query):
    fort_keywords = ['fort', 'castle', 'fortress', 'citadel', 'maharashtra', 'history', 'architecture']
    return any(keyword in query.lower() for keyword in fort_keywords)

def format_fort_info(forts):
    formatted_info = []
    for fort in forts[:3]:
        info = f"{fort.get('name', 'N/A')} (Title: {fort.get('title', 'N/A')}, " \
               f"Summary: {fort.get('summary', 'N/A')[:100]}...)"
        formatted_info.append(info)
    
    return ". ".join(formatted_info)

def generate_response(session_id, human_prompt):
    context, examples = read_json_file(file_path)
    
    # 1. Fetch RAG Context first
    if is_fort_related(human_prompt):
        relevant_forts = get_relevant_forts(human_prompt)
        forts_context = "Context Information from Database:\n" + format_fort_info(relevant_forts)
    else:
        forts_context = ""

    # 2. Update the Example Prompt
    example_prompt = PromptTemplate(
        input_variables=["input", "output"],
        template="Human: {input}\nAssistant: {output}"
    )

    # 3. Create the Suffix (Context + History + Input)
    # We inject the forts_context BEFORE the "Human:" input so the AI knows facts before answering.
    suffix_template = (
        f"{context}\n\n"
        f"{forts_context}\n\n"  # <--- FIXED: Context goes here
        f"Current conversation:\n{{history}}\n"
        f"Human: {{input}}\n"
        f"Assistant:"
    )

    few_shot_template = FewShotPromptTemplate(
        examples=examples,
        example_prompt=example_prompt,
        suffix=suffix_template,
        input_variables=["input", "history"]
    )

    history = get_conversation_history(session_id)
    
    # 4. Generate the final prompt string
    prompt = few_shot_template.format(input=human_prompt, history=history)
    
    # 5. Call Gemini
    response = get_gemini_response(prompt)
    response = clean_response(response)
    formatted_response = format_response(response)
    
    update_conversation_history(session_id, "Human", human_prompt)
    update_conversation_history(session_id, "Assistant", response)
    
    response_data = {
        "response": formatted_response,
    }
    return response_data

def get_conversation_history(session_id):
    if session_id not in conversation_history:
        return ""
    formatted_history = ""
    for entry in conversation_history[session_id]:
        formatted_history += f"{entry['sender']}: {entry['message']}\n"
    return formatted_history.strip()

def update_conversation_history(session_id, sender, message):
    if session_id not in conversation_history:
        conversation_history[session_id] = []
    conversation_history[session_id].append({"sender": sender, "message": message})
    conversation_history[session_id] = conversation_history[session_id][-10:]
