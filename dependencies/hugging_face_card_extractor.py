from huggingface_hub import hf_hub_download
from huggingface_hub.utils import HfHubHTTPError
import os


class HFClient:
    def __init__(self):
        pass


    def extract_model_card(self, repo_id: str, filename: str = "README.md") -> str:
        
        model_card_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename
        )

            # Now, read the content from the downloaded file
        with open(model_card_path, 'r', encoding='utf-8') as f:
            model_card_content = f.read()

        return model_card_content
