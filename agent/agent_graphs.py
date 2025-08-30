from typing import List
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic.v1 import BaseModel, Field
from langgraph.graph import StateGraph, END
# Updated import to reflect the new filename
from agent.agent_state import WorkflowCreator, AnalyzedNeeds, FinalModelChoice, CodeCreation
from agent.configuration import Configuration   
from agent.utils import search_for_models


def analyze_user_needs(state: WorkflowCreator) -> dict:
    print("--- Step 1: Analyzing User Needs ---")
    user_message = state["user_message"]

    prompt = ChatPromptTemplate.from_template(
        """You are a Deep Learning engineer with 20 years of experience in the gen AI field.
        Your task is to analyze the user's message and identify the most suitable AI tools and models for image generation.

        Based on the user's needs, provide a detailed recommendation for the following:
        1. Base model
        2. LoRA
        3. ControlNet
        4. IPAdapter

        User's Message:
        {user_message}

        If the task is simple, you don't need to fill all fields. If a basic model is enough, just recommend that.
        Prioritize models from Hugging Face and CivitAI. You must find/choose OpenSource models.

        Rules:
        1. The recommended models should use the same architecture
        """
    )
    llm = ChatOpenAI(model=Configuration.base_model, temperature=0)
    structured_llm_chain = prompt | llm.with_structured_output(AnalyzedNeeds)
    
    response = structured_llm_chain.invoke({"user_message": user_message})
    
    return {"analyzed_user_needs": response}

def research_models(state: WorkflowCreator) -> dict:
    """
    Node 2: Researches the recommended models to find their Hugging Face IDs.
    """
    print("--- Step 2: Researching Models ---")
    model_recommendations = state["analyzed_user_needs"]
    
    if not model_recommendations:
        return {"final_models": "No models were recommended."}

    researched_models = search_for_models(model_recommendations)
    
    return {"researched_models": researched_models}

def evaluate_models(state: WorkflowCreator) -> dict:
    """
    Node 3: Extracts model cards and uses an LLM to choose the single best model.
    """
    print("--- Step 3: Evaluating Model Cards ---")
    researched_models = state["researched_models"]
    user_message = state["user_message"]


    # Agent to decide the best model
    evaluation_prompt = ChatPromptTemplate.from_template(
        """You are a Deep Learning engineer with 20 years of experience in the gen AI field.
        Your task is to analyze the given models and choose the best one for each topic.

        The user's original request was:
        {user_message}

        Here are the model cards you have to evaluate:
        {researched_models}
        
        Rule:
        1. Make sure the models you choose for each topic has the same architecture as the base model.
        """
    )
    
    llm = ChatOpenAI(model=Configuration.base_model, temperature=0) 
    
    evaluator_chain = evaluation_prompt | llm.with_structured_output(FinalModelChoice)

    final_recommendation = evaluator_chain.invoke({
        "user_message": user_message,
        "researched_models": researched_models
    })
    
    return {"final_models": final_recommendation}

def write_diffusers_code(state: WorkflowCreator) -> dict:
    """
    Node 4: Write the code to run diffusers models.
    """
    print("--- Step 3: Evaluating Model Cards ---")
    models = state["final_models"]
    user_message = state["user_message"]


    # Agent to decide the best model
    evaluation_prompt = ChatPromptTemplate.from_template(
        """You are a Machine Learning engineer with 20 years of experience in the gen AI field.
        Your task is to analyze the given models and write python code using Diffusers to run those models.

        The user's original request was:
        {user_message}

        Here are the model cards you have to use in the code:
        {models}
        
        Rule:
        1. Write clean code, respecting OOP principles
        2. Return only the python code
        """
    )
    
    llm = ChatOpenAI(model=Configuration.base_model, temperature=0) 
    
    evaluator_chain = evaluation_prompt | llm.with_structured_output(CodeCreation)

    code = evaluator_chain.invoke({
        "user_message": user_message,
        "models": models
    })
    
    return {"code": code}


