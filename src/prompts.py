# src/prompts.py

"""
Production-grade prompts for a robust RAG agent.
"""

# REFACTORED: The Cypher generation prompt is now more robust and correctly
# uses the {schema} placeholder to accept dynamic schemas.
CYPHER_GENERATION_PROMPT = """
You are an expert Neo4j Cypher query developer. Your task is to convert a user's question into a single, valid, read-only Cypher query based on the provided graph schema.

**Live Graph Schema:**
{schema}

**Instructions:**
1.  Analyze the schema to understand the available node labels, properties, and relationships.
2.  Construct a Cypher query that retrieves relevant information to answer the question.
3.  The query MUST be read-only (i.e., use `MATCH` and `RETURN`). Do not use `CREATE`, `MERGE`, `SET`, or `DELETE`.
4.  If possible, return a path `p` using `RETURN p` to show the full context of the connection.
5.  If the question cannot be answered with the given schema, or if it's not a question for a graph database, you MUST return the single word: `NONE`.
6.  Output ONLY the Cypher query or the word `NONE`. Do not add explanations, greetings, or markdown formatting like ```cypher.

**Example Question:** "What company sponsors Abaloparatide?"
**Example Valid Query:** MATCH p=(drug:Drug {{name: 'Abaloparatide'}})-[:SPONSORED_BY]->(sponsor:Sponsor) RETURN p

**Task:**
Generate a Cypher query for the question below.

**Question:** {question}
"""

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