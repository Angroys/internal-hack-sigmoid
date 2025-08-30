from typing_extensions import TypedDict
from pydantic.v1 import BaseModel, Field
from typing import List, Dict, Union 
from typing import List, Dict, Union, Optional 

class WorkflowCreatorUserInput(TypedDict):

    user_message: str


class AnalyzedNeeds(BaseModel):
    base_model: List[str] = Field(description="4 recommended base model for the user's needs.")
    lora: List[str]= Field(description="4 recommended LoRA models, if applicable.")
    controlnet: List[str] = Field(description="4 recommended ControlNet models, if applicable.")
    ip_adapter: List[str]= Field(description="4 recommended IPAdapter models, if applicable.")

class FinalModelChoice(BaseModel):
    """The final selection of models for the user's workflow."""
    base_models: str = Field(description="The single best base model ID selected, e.g., 'stabilityai/stable-diffusion-xl-base-1.0'.")
    loras: Optional[str] = Field(default=None, description="The single best LoRA model ID, if applicable.")
    controlnets: Optional[str] = Field(default=None, description="The single best ControlNet model ID, if applicable.")
    ip_adapters: Optional[str] = Field(default=None, description="The single best IPAdapter model ID, if applicable.")
    reasonings: str = Field(description="A brief explanation for why these models were chosen.")



class CodeCreation(BaseModel):
    code: str = Field(description="Generated code to run the final, chosen models.")
class WorkflowCreator(TypedDict):
    # Input 
    user_message:str

    # Intermmediary
    analyzed_user_needs: AnalyzedNeeds 
    researched_models: Dict[str, Union[List[str], str]]

    base_models: str = Field(description="The single best base model ID selected, e.g., 'stabilityai/stable-diffusion-xl-base-1.0'.")
    loras: Optional[str] = Field(default=None, description="The single best LoRA model ID, if applicable.")
    controlnets: Optional[str] = Field(default=None, description="The single best ControlNet model ID, if applicable.")
    ip_adapters: Optional[str] = Field(default=None, description="The single best IPAdapter model ID, if applicable.")
    reasonings: str = Field(description="A brief explanation for why these models were chosen.")


    base_model: List[str] = Field(description="4 recommended base model for the user's needs.")
    lora: List[str]= Field(description="4 recommended LoRA models, if applicable.")
    controlnet: List[str] = Field(description="4 recommended ControlNet models, if applicable.")
    ip_adapter: List[str]= Field(description="4 recommended IPAdapter models, if applicable.")

    final_models: str

    # Output
    code: str