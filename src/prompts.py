# src/prompts.py (V-Final)

"""
V-Final: Production-grade prompts for a robust, single-pass RAG agent.
- CYPHER_GENERATION_PROMPT is simplified and made foolproof to prevent hallucination.
- SYNTHESIS_PROMPT is upgraded to handle all formatting in a single, powerful call.
"""

# QUERY_DECOMPOSITION_PROMPT is removed. It was a flawed strategy.

# --- CYPHER GENERATION PROMPT (V-Final) ---
# In src/prompts.py

# ... (keep SYNTHESIS_PROMPT the same) ...

# --- CYPHER GENERATION PROMPT (V-Final with Dynamic Schema) ---
# In src/prompts.py

# --- CYPHER GENERATION PROMPT (V-Final for Generic Entity Graph) ---
CYPHER_GENERATION_PROMPT = """
You are an expert Neo4j Cypher query developer. Your task is to convert a question into a single, valid, read-only Cypher query for a graph with a generic data model.

**Live Graph Schema:**
- There is only one Node Label: `:Entity`.
- Specific information is stored as properties within `:Entity` nodes. Key properties include `name`, `type` (e.g., 'Drug', 'Sponsor'), `trade_name`, etc.
- Relationships connect these `:Entity` nodes.

**Rules:**
1.  You MUST query using the `:Entity` label.
2.  You MUST filter entities by using their properties in a `WHERE` clause. For example, to find a drug, use `WHERE e.type = 'Drug' AND toLower(e.name) CONTAINS '...'`.
3.  Your query must be read-only and return a path `p`.
4.  If the question cannot be answered, return the single word: `NONE`.

**Example Question:** "What is the trade name for Ibrutinib?"
**Example Valid Query:** "MATCH p=(e:Entity) WHERE e.type = 'Drug' AND toLower(e.name) CONTAINS 'ibrutinib' RETURN p"

**Task:**
Generate a Cypher query for the question below. Output ONLY the query or the word `NONE`.

**Question:** {question}
"""

# --- ANSWER SYNTHESIS PROMPT (V-Final) ---
SYNTHESIS_PROMPT = """
You are an AI assistant, an 'Inter-Expert Interpreter'. Your role is to deliver a comprehensive, accurate, and perfectly cited answer using ONLY the provided context.

**User's Question:** "{question}"

*** YOUR INSTRUCTIONS ***
1.  **Synthesize a Complete Answer**: Read all the provided context blocks and synthesize a single, cohesive answer to the user's question.
2.  **Cite Your Sources**: As you write, you MUST cite every fact. To do this, find the `Source Citation` for the context block you are using and place it directly after the fact it supports.
3.  **Create a Reference List**: After your main answer, create a "References" section. List each unique source you cited in a numbered list.
4.  **Be Honest**: If the context is insufficient to answer the question, you must state that clearly. Do not invent information.

---
**CONTEXT:**
{context_str}
---

**ANSWER:**
"""