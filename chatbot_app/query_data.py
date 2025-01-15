import argparse
import subprocess
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.prompts import ChatPromptTemplate

CHROMA_PATH = "chroma"

PROMPT_TEMPLATE = """
Answer the question based on the above context: {question}
"""

def run_llama3(prompt):
    try:
        result = subprocess.run(
            ["ollama", "run", "llama3"],
            input=prompt,
            text=True,
            capture_output=True,
            check=True,
            encoding="utf-8" 
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        print(f"Error running Llama3: {e.stderr}")
        return None

def main():
    # Create CLI.
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    args = parser.parse_args()
    query_text = args.query_text

    # Prepare the DB.
    embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    # Search the DB.
    results = db.similarity_search_with_relevance_scores(query_text, k=3)
    if len(results) == 0 or results[0][1] < 0.5 :
        print(f"Unable to find results.")
        return
  

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)
    print(f"Generated Prompt:\n{prompt}")

    # Use the Llama3 model
    response = run_llama3(prompt)
    if response:
        print(f"Response: {response}")
    else:
        print("Failed to generate a response.")

if __name__ == "__main__":
    main()
