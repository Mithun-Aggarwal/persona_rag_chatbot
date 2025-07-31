# FILE: verify_pinecone.py
import os
import pinecone
from dotenv import load_dotenv

# --- Configuration ---
# Change this to the document ID you want to check for.
DOCUMENT_ID_TO_CHECK = "May-2025-PBAC-Meeting-v5"
NAMESPACE_TO_CHECK = "pbac-text" # This is our unified namespace

# --- Main Script ---
def verify_document_in_pinecone():
    """Connects to Pinecone and verifies if vectors for a specific doc_id exist."""
    print("--- Pinecone Data Verifier ---")
    
    # 1. Load Environment Variables
    load_dotenv()
    api_key = os.getenv("PINECONE_API_KEY")
    index_name = os.getenv("PINECONE_INDEX_NAME")

    if not api_key or not index_name:
        print("❌ ERROR: PINECONE_API_KEY or PINECONE_INDEX_NAME not set in your .env file.")
        return

    print(f"Connecting to index '{index_name}'...")
    try:
        pc = pinecone.Pinecone(api_key=api_key)
        index = pc.Index(index_name)
        stats = index.describe_index_stats()
        print(f"✅ Successfully connected. Index has {stats['total_vector_count']} total vectors.")
    except Exception as e:
        print(f"❌ ERROR: Could not connect to Pinecone. Details: {e}")
        return

    # 2. Perform the Metadata-Filtered Query
    print(f"\nSearching for vectors with doc_id: '{DOCUMENT_ID_TO_CHECK}' in namespace '{NAMESPACE_TO_CHECK}'...")
    
    try:
        # We don't need a real query vector; the filter is what matters.
        # We query for a zero vector of the correct dimension (768 for mpnet).
        query_vector = [0.0] * 768 
        
        response = index.query(
            namespace=NAMESPACE_TO_CHECK,
            vector=query_vector,
            top_k=10,
            include_metadata=True,
            filter={
                "doc_id": {"$eq": DOCUMENT_ID_TO_CHECK}
            }
        )

        # 3. Analyze and Report Results
        matches = response.get('matches', [])
        if not matches:
            print("\n" + "="*50)
            print(f"🔴 VERIFICATION FAILED: No vectors found for '{DOCUMENT_ID_TO_CHECK}'.")
            print("   This confirms the data was NOT uploaded correctly.")
            print("="*50 + "\n")
        else:
            print("\n" + "="*50)
            print(f"🟢 VERIFICATION SUCCESSFUL: Found {len(matches)} vectors for '{DOCUMENT_ID_TO_CHECK}'.")
            print("   The data is present in your Pinecone index.")
            print("\n--- Sample of first result's metadata ---")
            print(matches[0].metadata)
            print("="*50 + "\n")

    except Exception as e:
        print(f"❌ ERROR: An error occurred during the query. Details: {e}")

if __name__ == "__main__":
    verify_document_in_pinecone()