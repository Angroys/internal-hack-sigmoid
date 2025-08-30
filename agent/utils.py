from dependencies.openai import OpenAIClient
from dotenv import load_dotenv
import os, json
from dependencies.hugging_face_card_extractor import HFClient
from langchain_community.tools.tavily_search import TavilySearchResults
from agent.agent_state import AnalyzedNeeds
from dependencies.hugging_face_card_extractor import HFClient
load_dotenv()
hf_client = HFClient()

def search_for_models(model_recommendations: AnalyzedNeeds) -> dict:
    search_tool = TavilySearchResults(max_results=3)
    research_results = {}
    
    # The value 'model_list' here is a list of strings, not a single string.
    for model_type, model_list in model_recommendations.dict().items():
        if not model_list:
            continue

        all_found_models = []
        # We need to loop through each model name in the list.
        for model_name in model_list:
            if not model_name:
                continue

            print(f"--- Researching {model_type}: {model_name} ---")
            query = f"HuggingFace model card for '{model_name}' image generation"
            
            try:
                results = search_tool.invoke(query)
                
                for result in results:
                    url = result.get("url", "")
                    if "huggingface.co/" in url:
                        parts = url.strip('/').split('/')
                        # Ensure we have a user and model name after the domain
                        if len(parts) >= 4 and "huggingface.co" in parts[-3]:
                            model_id = f"{parts[-2]}/{parts[-1]}"
                            if model_id not in all_found_models:
                                all_found_models.append(model_id)
            except Exception as e:
                print(f"Error researching {model_name}: {e}")
        
        research_results[model_type] = all_found_models if all_found_models else "No valid Hugging Face models found."
    
    return research_results

def extract_model_cards(researched_models: dict) -> dict:
    model_cards = {}
    for model_type, models in researched_models.items():
        if isinstance(models, list):
            card_content = []
            for model_id in models:
                content = hf_client.extract_model_card(model_id)
                card_content.append(f"--- MODEL: {model_id} ---\n{content}")
            model_cards[model_type] = "\n\n".join(card_content)
    return model_cards