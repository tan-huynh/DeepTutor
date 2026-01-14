import settings as root_settings
from settings import settings

class Env_Settings_Adapter:
    def __init__(self):
        pass

    @property
    def DATA_FOLDER_PATH(self):
        return settings.lego_rag.data_path

    @property
    def EMBEDDING_MODEL_PATH(self):
        return settings.lego_rag.embedding_model_path

    @property
    def VECTOR_DB_PATH(self):
        return settings.lego_rag.vector_db_path

    @property
    def KNOWLEDGE_GRAPH_PATH(self):
        return settings.lego_rag.knowledge_graph_path

    @property
    def COLLECTION_NAME(self):
        return settings.lego_rag.collection_name

    @property
    def DOCSTORE_PATH(self):
        return settings.lego_rag.docstore_path
    
    @property
    def GROQ_API_KEY(self):
        if settings.llm.binding == "groq":
             return settings.llm.api_key
        return ""

settings = Env_Settings_Adapter()
