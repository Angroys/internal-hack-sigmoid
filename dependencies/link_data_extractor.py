from langchain_community.document_loaders import WebBaseLoader

class LangChainClient:
    def __init__(self):
       pass

    def load_and_return_info(self, url: str):
        self.loader = WebBaseLoader(url)
        self.load_documents()
        return self.return_info()

    def load_documents(self):
        self.website_info= self.loader.load()
        
    def return_info(self):
        return self.website_info[0].page_content.strip("\n")    