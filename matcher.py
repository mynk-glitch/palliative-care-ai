import json
import chromadb
from sentence_transformers import SentenceTransformer

class PalliativeCaseMatcher:
    def __init__(self, json_filepath: str = "cases_knowledge_base.json"):
        # 1. Load the lightweight clinical embedding model
        self.encoder = SentenceTransformer("TimKond/S-PubMedBert-MedQuAD")
        
        # 2. Initialize in-memory ChromaDB client
        self.chroma_client = chromadb.Client()
        
        # 3. Create or reset collection
        try:
            self.chroma_client.delete_collection("palliative_cases")
        except Exception:
            pass
        self.collection = self.chroma_client.create_collection(
            name="palliative_cases",
            metadata={"hnsw:space": "cosine"}
        )
        
        # 4. Load dataset and index
        self.cases_data = self._load_cases(json_filepath)
        self._index_cases()

    def _load_cases(self, filepath: str):
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)

    def _index_cases(self):
        ids = []
        documents = []
        metadatas = []
        
        for case in self.cases_data:
            # Index medical terms, common everyday terms, and symptoms together
            searchable_text = (
                f"Condition: {case['condition']}. "
                f"Everyday Terms: {case.get('common_terms', '')}. "
                f"Symptoms: {case['primary_symptom']}. "
                f"Palliative Stage: {case['patient_profile']['palliative_stage']}. "
                f"Vital Concerns: {case['patient_profile']['vital_concerns']}."
            )
            
            ids.append(case["case_id"])
            documents.append(searchable_text)
            metadatas.append({
                "condition": case["condition"],
                "primary_symptom": case["primary_symptom"],
                "evidence_reference": case["evidence_reference"]
            })
            
        # Encode strings to vectors
        embeddings = self.encoder.encode(documents).tolist()
        
        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas
        )

    def find_matching_cases(self, query: str, top_k: int = 1):
        query_embedding = self.encoder.encode([query]).tolist()
        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=top_k
        )
        
        matched_results = []
        for i in range(len(results["ids"][0])):
            matched_id = results["ids"][0][i]
            # Convert cosine distance to percentage similarity
            cosine_dist = results["distances"][0][i] if "distances" in results else 0.0
            similarity_score = max(0.0, min(1.0, 1.0 - cosine_dist))
            
            full_case = next(c for c in self.cases_data if c["case_id"] == matched_id)
            matched_results.append({
                "case_id": matched_id,
                "similarity_score": round(similarity_score * 100, 2),
                "case_details": full_case
            })
            
        return matched_results

# Simple terminal test block
if __name__ == "__main__":
    print("Loading AI Model and indexing cases (this may take a moment on the first run)...")
    matcher = PalliativeCaseMatcher()
    
    test_query = "Terminal lung cancer with extreme breathlessness and panic"
    print(f"\nSearching for patient scenario: '{test_query}'")
    
    results = matcher.find_matching_cases(test_query)
    for result in results:
        print(f"\nMatch Found: {result['case_id']} ({result['similarity_score']}% match)")
        print(f"Condition: {result['case_details']['condition']}")
        print(f"Evidence Ref: {result['case_details']['evidence_reference']}")