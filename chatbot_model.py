# maiway_chatbot/backend/chatbot_model.py
from sentence_transformers import SentenceTransformer, util
import json
import torch
import os
import re

class ChatbotModel:
    def __init__(self, data_path='data/faq_data.json', similarity_threshold=0.78):
        script_dir = os.path.dirname(__file__)
        self.faq_file_path = os.path.join(script_dir, data_path)
        self.similarity_threshold = similarity_threshold 

        self.model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

        self.map_related_keywords = [
            "route", "routes", "how to get to", "location", "address", 
            "map", "direction", "directions", "saan", "paano pumunta",
            "papunta", "transport",
            "where is", "find", "locate", "travel", "by foot", "walking"
        ]

        self._load_and_encode_data()

    def _load_data(self, path):
        """Loads JSON data from the specified path."""
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            print(f"Successfully loaded data from: {path}")
            return data
        except FileNotFoundError:
            print(f"Error: FAQ data file not found at {path}. Initializing with empty data.")
            return []
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON from {path}: {e}. Initializing with empty data.")
            return [] 

    def _save_data(self, path):
        """Saves JSON data to the specified path."""
        try:
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(self.faq_data, f, ensure_ascii=False, indent=4)
            print(f"Successfully saved data to: {path}")
        except IOError as e:
            print(f"Error saving data to {path}: {e}")

    def _preprocess_text(self, text):
        """
        Applies basic text normalization:
        - Converts to lowercase
        - Removes extra whitespace
        - Strips leading/trailing whitespace
        """
        if not isinstance(text, str): 
            return ""
        text = text.lower()
        text = re.sub(r'\s+', ' ', text).strip() 
        return text

    def _load_and_encode_data(self):
        """Loads data from the FAQ file and encodes the corpus into embeddings."""
        print(f"Loading and encoding chatbot data from: {self.faq_file_path}")
        self.faq_data = self._load_data(self.faq_file_path)

        self.corpus = [self._preprocess_text(item["question"]) for item in self.faq_data]
        if not self.corpus:
            self.corpus_embeddings = torch.tensor([]) 
            print("Warning: Corpus is empty, no embeddings generated.")
        else:
            self.corpus_embeddings = self.model.encode(self.corpus, convert_to_tensor=True)
        print(f"Data loaded and corpus encoded. Corpus size: {len(self.corpus)} questions.")

    def get_response(self, user_query):
        """
        Generates a response to the user's query based on semantic similarity.
        Includes debug information for troubleshooting.
        Prioritizes map-related queries.
        """
        if not user_query:
            return "Wala po kayong tinanong. Paano po ako makakatulong?"

        processed_query = self._preprocess_text(user_query)

        for keyword in self.map_related_keywords:
            if keyword in processed_query:
                print(f"Detected map-related query with keyword: '{keyword}'. Redirecting to MapScreen.")
                return "For questions about routes, locations, or directions, please refer to the MapScreen. You can use the search bar there to find places."

        if self.corpus_embeddings.numel() == 0: 
            print("Warning: Chatbot corpus embeddings are empty. Cannot generate response.")
            return "Pasensya na, wala pa po akong impormasyon. Pakisubukang magdagdag ng FAQs."

        query_embedding = self.model.encode(processed_query, convert_to_tensor=True)

        cosine_scores = util.cos_sim(query_embedding, self.corpus_embeddings)[0]

        best_match_idx = int(torch.argmax(cosine_scores).item()) 

        similarity_score = cosine_scores[best_match_idx].item()

        print(f"\n--- Chatbot Response Debug ---")
        print(f"User Query (Original): '{user_query}'")
        print(f"User Query (Processed): '{processed_query}'")
        if self.corpus and best_match_idx < len(self.corpus): 
            print(f"Best matching question in corpus: '{self.corpus[best_match_idx]}'")
            print(f"Original question text: '{self.faq_data[best_match_idx]['question']}'")
        print(f"Similarity Score: {similarity_score:.4f}")
        print(f"Configured Threshold: {self.similarity_threshold:.4f}")
        print(f"--- End Debug ---\n")

        if similarity_score >= self.similarity_threshold:
            return self.faq_data[best_match_idx]["answer"]
        else:
            return "Pasensya na, hindi ko po maintindihan ang tanong ninyo. Maaari po bang ulitin o magtanong ng iba? (Sorry, I don't understand your question. Could you please rephrase or ask something else?)"

    def add_faq(self, question, answer):
        """
        Adds a new FAQ to the knowledge base and updates embeddings.
        Includes a basic check for duplicate questions.
        """
        if not question or not answer:
            print("Error: Question or answer cannot be empty when adding FAQ.")
            return False

        processed_new_question = self._preprocess_text(question)

        if self.faq_data:
            for item in self.faq_data:
                if self._preprocess_text(item["question"]) == processed_new_question:
                    print(f"Question '{question}' (processed as '{processed_new_question}') already exists. Skipping.")
                    return False 

        self.faq_data.append({"question": question, "answer": answer})
        self._save_data(self.faq_file_path)
        self._load_and_encode_data() 
        print(f"New FAQ added: '{question}'. Chatbot knowledge base updated.")
        return True 

    def reload_data(self):
        """Manually triggers a reload of the FAQ data and re-encodes the corpus."""
        self._load_and_encode_data()
        print("Chatbot data reloaded manually.")

    def get_matching_questions(self, query_text, limit=5):
        """
        Finds questions in the FAQ data that partially match the query_text.
        """
        if not query_text:
            return []

        processed_query = self._preprocess_text(query_text)
        matches = []
        for item in self.faq_data:
            processed_question = self._preprocess_text(item["question"])
            if processed_query in processed_question:
                matches.append(item["question"])
            if len(matches) >= limit:
                break
        return matches