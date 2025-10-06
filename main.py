
from langgraph.graph import StateGraph, END
from agent.agent_state import WorkflowCreator
from agent.agent_graphs import analyze_user_needs, evaluate_models, research_models, write_diffusers_code, format_code
from dotenv import load_dotenv


load_dotenv()

workflow = StateGraph(WorkflowCreator)

workflow.add_node("analyze_user_needs", analyze_user_needs)
workflow.add_node("research_models", research_models)
workflow.add_node("evaluate_models", evaluate_models)
workflow.add_node("generate_code", write_diffusers_code)
workflow.add_node("format_code", format_code)


workflow.set_entry_point("analyze_user_needs")
workflow.add_edge("analyze_user_needs", "research_models")
workflow.add_edge("research_models", "evaluate_models")
workflow.add_edge("evaluate_models", "generate_code")
workflow.add_edge("generate_code", "format_code")
workflow.add_edge("format_code", END)

app = workflow.compile()

if __name__ == "__main__":
    inputs = {"user_message": "I want to generate an image with an antropomorphic human with fur."}
    for output in app.stream(inputs):
        for key, value in output.items():
            print(f"Output from node '{key}':")
            print("---")
            print(value)
        print("\n---\n")
