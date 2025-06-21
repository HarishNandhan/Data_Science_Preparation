import os
import requests
from chromadb import HttpClient
from dotenv import load_dotenv
import numpy as np
load_dotenv()


EURI_API_KEY = os.getenv("EURI_API_KEY") # for generating the embeddings
client = HttpClient(host="localhost",port = 8000)

"""
- Creates an instance of HttpClient to connect to a ChromaDB server running locally.
- host="localhost": The server is running on the same machine.
- port=8000: The server listens on port 8000.
"""


collection = client.get_or_create_collection("harish_linkedin_data")
"""
- Attempts to retrieve a collection named "harish_linkedin_data" from the database.
- If the collection does not exist, it creates a new one with that name.
- Collections in ChromaDB are used to organize and store related data, such as embeddings or documents.

"""

print(client)


# Function to scrap the textual information 

def generate_embeddings(text_list):
    url = "https://api.euron.one/api/v1/euri/alpha/embeddings"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {EURI_API_KEY}"
    }
    payload = {
        "input": text_list,
        "model": "text-embedding-3-small"
    }

    response = requests.post(url, headers=headers, json=payload)
    data = response.json()
    
    # Convert to numpy array for vector operations
    embeddings = [item['embedding'] for item in response.json()["data"]]
    
    return embeddings


document = [
    "my name is harish nandhan shanmugam",
    "i used to learn tech by watching lectures in youtube",
    "harish nandhan total year of experience in data science and analytics domain is 5, as a student 6 as a masters in data science student 1",
    "harish nandhan loves teaching and explaining concepts and architecture",
    "harish nandhan likes to explore new large language models and gen ai systems",
    "harish nandhan started working as an ai engineering intern in beaconai",
    "I am a Data Science enthusiast with a passion for exploring data and uncovering insights that drive better decision-making. My academic background in Artificial Intelligence and Machine Learning provides me with a strong foundation in some of the key areas: machine learning, deep learning, Natural Language Processing, and Computer Vision. Curiosity and dedication are the driving forces as I learn and grow in all these domains, pushing me toward enriching my knowledge.",
    "I have a strong interest in data analytics, particularly the analysis of data for actionable patterns and trends. I've had experience with using Excel, Power BI, and Tableau to develop dashboards and present data visually in meaningful ways. I am adept at managing data through SQL and NoSQL databases, including MongoDB, ensuring that data is structured and readily accessible for any analysis.",
    "Besides technical expertise, I have hands-on experience in working on AWS, with projects worked on in the cloud platforms. I also have experience in working with version control systems, where I can work with others through Git.",
    "I am passionate about learning more on the advancement of the Data Science field and applying those new techniques to real-world problems. It would be very important to make near-accurate predictions that would yield valuable insights necessary for organizational success by integrating technical expertise with business process knowledge.",
    "Looking forward to contributing as a Data Scientist or Data Analyst in a firm that values data-driven decision-making, I am open to internship opportunities across the United States during Summer 2025. Feel free to reach out here on LinkedIn if you wish to discuss potential opportunities or just share your insights with regard to data and analytics.",
    "I am attending the Agentic AI Conference by Data Science Dojo on May 27 and 28, 2025.",
    "The conference speakers include thought leaders in industry who will talk about all aspects of building agentic AI applications - covering everything from cutting-edge agentic frameworks to retrieval systems, observability, and guardrails for safe, trustworthy AI."
]

# each sentence will convert into the embedding of size 1536


all_embeddings = generate_embeddings(document)
print(all_embeddings)
print(len(all_embeddings[0]))



for idx,(doc,emb) in enumerate(zip(document,all_embeddings)):
    collection.add(
        documents = [doc],
        embeddings = [emb],
        metadatas = [{"source":"harish_linkedin_data"}],
        ids = [f"doc_{idx}"]
    )
# zip will do one to one mapping 
# enumerate will create a seperate id

print("all data stored in chroma_db")

print(collection.get(include=["documents","embeddings"]))

# now it can able to retrieve all the ids,embeddings, metadata and documents