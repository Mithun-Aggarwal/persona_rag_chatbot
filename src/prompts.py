# FILE: src/prompts.py
# V2.4 (Definitive Fix): Cypher prompt now explicitly returns relationship properties as a map.

"""
Production-grade prompts for a robust RAG agent.
"""

QUERY_CLASSIFICATION_PROMPT = """
You are an expert query analysis agent. Your task is to analyze the user's question and provide a structured JSON output with three fields: 'intent', 'keywords', and 'question_is_graph_suitable'.

1.  **'intent'**: Classify the user's goal into one of these categories:
    *   "specific_fact_lookup": For questions seeking a single, direct answer (e.g., "What company sponsors Drug X?").
    *   "simple_summary": For questions asking for a general overview (e.g., "Tell me about Drug Y.").
    *   "comparative_analysis": For questions that compare two or more items (e.g., "Compare Drug A and Drug B.").
    *   "general_qa": For all other questions.

2.  **'keywords'**: Extract the most important nouns and proper nouns from the question, such as drug names, company names, or medical conditions. Return them as a list of strings.

3.  **'question_is_graph_suitable'**: Return `true` if the question involves relationships between entities (e.g., drug-to-sponsor, drug-to-condition), which are suitable for a knowledge graph. Otherwise, return `false`.

Output ONLY the raw JSON object. Do not add explanations or markdown formatting.

Example Question: "What is the cost-effectiveness of Abaloparatide for treating osteoporosis, and who is the sponsor?"
Example JSON Output:
{
  "intent": "specific_fact_lookup",
  "keywords": ["Abaloparatide", "osteoporosis", "cost-effectiveness", "sponsor"],
  "question_is_graph_suitable": true
}
"""

# --- START OF DEFINITIVE FIX ---
# V2.4: The RETURN clause is now foolproof. It returns the path AND a separate, clean
#       dictionary of the relationship's properties.
CYPHER_GENERATION_PROMPT = """
You are an expert Neo4j Cypher query developer. Your task is to convert a user's question into a single, valid, read-only Cypher query based on the provided graph schema.

**Live Graph Schema:**
{schema}

**CRITICAL Instructions:**
1.  **Use Normalized Properties:** You MUST query against the `name_normalized` property for all `WHERE` clauses on nodes. The user's input should be converted to lowercase. For example: `WHERE n.name_normalized = 'abaloparatide'`.
2.  **Return Path and Properties:** Your query MUST return two things, aliased as specified:
    - The path, aliased as `p`.
    - A map of the relationship's properties, aliased as `rel_props`.
    Your `RETURN` clause should look like this: `RETURN p, properties(r) as rel_props`.
3.  **Relationship Variable:** Ensure the relationship in your `MATCH` clause is assigned to a variable, e.g., `-[r:HASSPONSOR]->`.
4.  **Read-Only:** The query MUST be read-only (i.e., use `MATCH` and `RETURN`).
5.  **Failure Case:** If the question cannot be answered, you MUST return the single word: `NONE`.
6.  **Output Format:** Output ONLY the Cypher query or the word `NONE`.

**Example Question:** "What company sponsors Abaloparatide?"
**Example Valid Query:** MATCH p=(drug:Entity)-[r:HASSPONSOR]->(sponsor:Entity) WHERE drug.name_normalized = 'abaloparatide' RETURN p, properties(r) as rel_props

**Task:**
Generate a Cypher query for the question below.

**Question:** {question}
"""
# --- END OF DEFINITIVE FIX ---

SYNTHESIS_PROMPT = """
You are a highly precise AI assistant for pharmaceutical and regulatory analysis. Your task is to answer a user's question based *only* on the provided evidence. You must follow all rules strictly.

**User's Question:** "{question}"

*** YOUR INSTRUCTIONS ***

**Rule 1: Synthesize a Factual Answer**
- Read all the provided "Evidence" blocks below.
- Formulate a single, comprehensive answer to the user's question.
- **Do not mention the tools** (e.g., "query_knowledge_graph"). The user only cares about the information.

**Rule 2: Cite Every Fact with Clickable Links**
- The evidence contains HTML `<a>` tags for citations. They look like `<a href="URL" target="_blank">Display Text</a>`.
- As you write your answer, you MUST place the corresponding HTML citation tag immediately after the sentence or clause it supports.
- If multiple pieces of evidence support a single sentence, include all their citations. For example: `This is a fact <a href="..." target="_blank">doc-a</a> <a href="..." target="_blank">doc-b</a>.`

**Rule 3: Create a Clean Reference List**
- After your answer, add a `References` section.
- List each unique HTML citation tag that you used in your answer. Each link should be on its own line, preceded by a bullet point.

**Rule 4: Honesty is Critical**
- If the provided evidence is insufficient to answer the question, you MUST state that clearly.
- Do not invent, infer, or use outside knowledge.

---
**Example of a PERFECT response:**

Based on the documentation, Theramex Australia Pty Ltd is the sponsor for the drug Abaloparatide <a href="https://storage.googleapis.com/your-bucket/doc1.pdf#page=15" target="_blank">July-2025-PBAC-v4 (Page 15)</a>. This drug is indicated for the treatment of severe established osteoporosis <a href="https://storage.googleapis.com/your-bucket/doc2.pdf#page=2" target="_blank">Clinical-Review-XYZ (Page 2)</a>.

**References**
- <a href="https://storage.googleapis.com/your-bucket/doc1.pdf#page=15" target="_blank">July-2025-PBAC-v4 (Page 15)</a>
- <a href="https://storage.googleapis.com/your-bucket/doc2.pdf#page=2" target="_blank">Clinical-Review-XYZ (Page 2)</a>

---
**Evidence:**
{context_str}
---

**FINAL ANSWER:**
"""