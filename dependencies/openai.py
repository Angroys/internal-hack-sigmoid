import os
from openai import OpenAI, OpenAIError

class OpenAIClient:
    def __init__(self, api_key: str = None):
        if api_key is None:
            api_key = os.environ.get("OPENAI_API_KEY")

        if api_key is None:
            raise ValueError("OpenAI API key not found. Please provide it as an argument or set the OPENAI_API_KEY environment variable.")

        self.client = OpenAI(api_key=api_key)

    def send_request_and_receive_response(self, prompt: str, model: str = "gpt-5-nano") -> str:
        if not prompt:
            return "Error: Prompt cannot be empty."

        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful Machine Learning Engineer."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
                
            )
            return response.choices[0].message.content.strip()

        except OpenAIError as e:
            print(f"An OpenAI API error occurred: {e}")
            return f"Error: Could not get a response from the API. Details: {e}"
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            return "Error: An unexpected error occurred."
