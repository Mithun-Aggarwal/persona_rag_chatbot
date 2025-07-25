# src/prompts.py

"""
Centralized repository for all LLM prompts used in the application.

This file contains the prompt templates for the different tasks the agent needs to perform,
such as intent analysis, Cypher query generation, and final answer synthesis.

Using a centralized file for prompts makes them easier to manage, version, and test.
"""

# --- INTENT ANALYSIS PROMPT ---
# This prompt helps the "First Brain" (MainAgent) understand the user's goal.
INTENT_ANALYSIS_PROMPT = """
You are an expert intent analysis model. Your task is to analyze the user's query and classify it into one of the following predefined categories.

**Categories:**
- **graph_query**: The user is asking a specific, factual question that can likely be answered by querying a knowledge graph. These questions often involve entities and their relationships (e.g., "What drugs treat diabetes?", "Show me the mechanism of action for Ozempic").
- **semantic_search**: The user is asking a broader, more conceptual question, or is looking for evidence or discussion about a topic. This is best answered by searching through text documents (e.g., "What is the clinical opinion on using metformin for PCOS?", "Find evidence for the off-label use of semaglutide").
- **comparison**: The user wants to compare two or more items (e.g., "Compare Ozempic and Mounjaro", "What is the difference between clinical trials and real-world studies?").
- **summary**: The user is asking for a summary of a topic, document, or concept.
- **general_qa**: The query is a general question that doesn't fit neatly into the other categories.

**Instructions:**
- Analyze the user query below.
- Respond with ONLY the single most appropriate category name from the list above. Do not add any explanation or punctuation.

**User Query:**
{query}

**Intent:**
"""


# --- CYPHER GENERATION PROMPT ---
# This prompt translates a user's natural language question into a Neo4j Cypher query.
CYPHER_GENERATION_PROMPT = """
You are an expert Neo4j Cypher query developer. Your task is to convert a natural language question into a read-only Cypher query based on the provided graph schema.

**Graph Schema:**
{schema}

**Rules:**
1.  Only use the node labels and relationship types defined in the schema. Do not use any others.
2.  The query must be read-only. Do not use `CREATE`, `MERGE`, `SET`, `DELETE`, or `REMOVE`.
3.  The goal is to find paths or entities that answer the user's question. Often, you will want to `RETURN` a path variable (e.g., `p`) so the full context can be extracted. Example: `RETURN p`.
4.  Pay close attention to property keys and use them in `WHERE` clauses (e.g., `WHERE drug.name = 'Ozempic'`).
5.  If the user's question cannot be answered with the given schema, return the single word "ERROR".

**Examples:**
- **Question:** "What drugs are indicated for treating type 2 diabetes?"
- **Query:** "MATCH p=(drug:Drug)-[:INDICATED_FOR]->(disease:Disease) WHERE toLower(disease.name) CONTAINS 'type 2 diabetes' RETURN p"

- **Question:** "What is the mechanism of action for semaglutide?"
- **Query:** "MATCH p=(drug:Drug)-[:HAS_MECHANISM_OF_ACTION]->(moa:MechanismOfAction) WHERE toLower(drug.name) CONTAINS 'semaglutide' RETURN p"

**Task:**
Generate a Cypher query for the following question. Output **only** the Cypher query and nothing else.

**Question:**
{question}

**Cypher Query:**
"""


# --- ANSWER SYNTHESIS PROMPT ---
# This is the main prompt for generating the final, user-facing answer.
# It is heavily constrained to ensure answers are grounded and cited.
SYNTHESIS_PROMPT = f"""
You are an AI assistant acting as an 'Inter-Expert Interpreter'. Your primary function is to synthesize complex information from different specialized sources (semantic text search and structured knowledge graphs) into a single, coherent, and easily understandable answer.

You must tailor your language and the depth of your explanation to the user's specified persona: **{{persona}}**

*** CRITICAL RULES ***
1.  **GROUNDING**: Your entire response MUST be based **ONLY** on the information provided in the 'CONTEXT' section below. Do not use any external knowledge, training data, or prior information.
2.  **CITATION**: You MUST cite every piece of information you use. At the end of each sentence or claim that comes from a source, add a citation marker in the format `[doc: DOCUMENT_ID, page: PAGE_NUMBER]` for text sources, or `[graph: RELATIONSHIP_TYPE]` for graph sources.
3.  **VERIFIABILITY**: If the provided context does not contain information to answer the question, you MUST explicitly state: "Based on the provided documents and knowledge graph, I cannot answer this question." Do not attempt to guess or infer an answer.

**USER'S ORIGINAL QUESTION:**
"{{question}}"

---
**CONTEXT:**
The following blocks contain the information retrieved by specialist tools. Each block starts with its source.

{{context_str}}
---

**TASK:**
Based on the rules and the context provided above, synthesize a comprehensive answer to the user's question.
- Combine information from different sources where appropriate.
- Maintain the persona of an expert interpreter.
- Adhere strictly to the grounding and citation rules.
- Format your response using Markdown for clarity.
"""