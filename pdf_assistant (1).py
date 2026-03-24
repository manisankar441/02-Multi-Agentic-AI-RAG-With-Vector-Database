import typer
from typing import Optional, List
from phi.model.groq import Groq
from phi.agent import Agent
from phi.storage.agent.postgres import PgAgentStorage
from phi.knowledge.pdf import PDFUrlKnowledgeBase
from phi.vectordb.pgvector import PgVector2
from phi.embedder.google import GeminiEmbedder
from google.generativeai import configure
import os

# Set API Keys
os.environ["GROQ_API_KEY"] = "Your GROQ API Key"
os.environ["GOOGLE_API_KEY"] = "Your Google API Key"

# Explicitly configure Google Generative AI
configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Access API keys
print(f"GROQ_API_KEY loaded: {os.getenv('GROQ_API_KEY') is not None}")
print(f"GOOGLE_API_KEY loaded: {os.getenv('GOOGLE_API_KEY') is not None}")

db_url = "postgresql+psycopg://ai:ai@localhost:5532/ai"

# Initialize Knowledge Base
embedder = GeminiEmbedder(api_key=os.getenv("GOOGLE_API_KEY"))
knowledge_base = PDFUrlKnowledgeBase(
    urls=["https://phi-public.s3.amazonaws.com/recipes/ThaiRecipes.pdf"],
    vector_db=PgVector2(collection="reci", db_url=db_url, embedder=embedder),
)

try:
    knowledge_base.load(recreate=True)
except Exception as e:
    print(f"Error loading knowledge base: {e}")

storage = PgAgentStorage(table_name="pdf_assistant", db_url=db_url)

def pdf_assistant(new: bool = False, user: str = "user"):
    run_id: Optional[str] = None

    assistant = Agent(
        model=Groq(id="llama-3.3-70b-versatile", embedder=embedder),
        run_id=run_id,
        user_id=user,
        knowledge_base=knowledge_base,
        storage=storage,
        show_tool_calls=True,
        search_knowledge=True,
        read_chat_history=True,
    )

    if run_id is None:
        run_id = assistant.run_id
        print(f"Started Run: {run_id}\n")
    else:
        print(f"Continuing Run: {run_id}\n")

    assistant.cli_app(markdown=True)

if __name__ == "__main__":
    typer.run(pdf_assistant)
