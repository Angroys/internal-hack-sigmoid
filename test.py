from agent.utils import extract_model_cards, search_for_models
from agent.agent_state import AnalyzedNeeds
from dotenv import load_dotenv

load_dotenv()
mock_data = AnalyzedNeeds(
        base_model=["stabilityai/stable-diffusion-3-medium-diffusers"],
        lora=["Cagliostro-LORA-SDXL"],
        controlnet=["lllyasviel/control_v11p_sd15_canny"],
        ip_adapter=["h94/IP-Adapter"]
    )
search = search_for_models(mock_data)
print(search)
print(extract_model_cards(search))