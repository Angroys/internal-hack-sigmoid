from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.tools.tavily_search import TavilySearchResults
from agent.agent_state import WorkflowCreator, AnalyzedNeeds, FinalModelChoice, CodeCreation, SearchQueries
from agent.configuration import Configuration   
from agent.utils import search_for_models
import subprocess, os

def analyze_user_needs(state: WorkflowCreator) -> dict:
    """
    Analyzes the user's needs by first generating search queries,
    searching with Tavily, and then making a final recommendation.
    """
    print("--- Step 1: Analyzing User Needs (with Tavily) ---")
    user_message = state["user_message"]
    llm = ChatOpenAI(model=Configuration.base_model, temperature=0)
    
    print("--- Generating search queries... ---")
    query_generation_prompt = ChatPromptTemplate.from_template(
        """You are a research assistant. Your task is to generate a set of 3-5 search queries 
        that will help find the best open-source AI image generation models (Base Model, LoRA, ControlNet, etc.) 
        for a user's request.

        User's Message:
        {user_message}
        
        Generate concise queries for platforms like Hugging Face and CivitAI.
        """
    )
    
    query_generation_chain = query_generation_prompt | llm.with_structured_output(SearchQueries)
    search_queries = query_generation_chain.invoke({"user_message": user_message})
    print(f"Generated Queries: {search_queries.queries}")

    print("--- Executing search with Tavily... ---")
    search_tool = TavilySearchResults(max_results=3)
    search_results = []
    for query in search_queries.queries:
        results = search_tool.invoke({"query": query})
        search_results.extend(results)
    
    search_results_str = "\n\n".join([str(res) for res in search_results])
    print("--- Search complete. Analyzing results... ---")

    final_analysis_prompt = ChatPromptTemplate.from_template(
        """You are a Deep Learning engineer with 20 years of experience in the gen AI field.
        Your task is to analyze the user's message and the provided search results to recommend the most 
        suitable AI tools and models for image generation.

        Based on the user's needs and the search results, provide a detailed recommendation for:
        1. Base model
        2. LoRA
        3. ControlNet
        4. IPAdapter

        User's Message:
        {user_message}

        Search Results:
        {search_results}

        Rules:
        - Prioritize models from Hugging Face and CivitAI. You must choose OpenSource models.
        - Ensure the recommended models share the same architecture (e.g., all for SDXL).
        - If the task is simple, you don't need to fill all fields.
        """
    )

    structured_llm_chain = final_analysis_prompt | llm.with_structured_output(AnalyzedNeeds)
    
    response = structured_llm_chain.invoke({
        "user_message": user_message,
        "search_results": search_results_str
    })
    
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
    print("--- Step 4: Generating Code---")
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

def format_code(state: WorkflowCreator) -> dict:
    """
    Writes code to a file, formats it with Ruff, and returns the result.
    """
    code_model = state["code"]
    code_string = code_model.code
    file_path = "test/result.py"

    # Ensure the directory exists
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    # Use 'with open' to automatically handle closing the file
    with open(file_path, "w", encoding="utf-8") as file_object:
        file_object.write(code_string)

    # Run the formatter
    result = subprocess.run(
        ["ruff", "format", file_path], 
        capture_output=True, 
        text=True
    )

    # Return a dictionary with a key and a value
    return {"result": result}