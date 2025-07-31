# FILE: src/prompts.py
# V3.0 (ReAct Architecture): Added prompts for multi-step reasoning.

"""
Production-grade prompts for a robust RAG agent. This version supports both
direct RAG and a multi-step ReAct (Reason+Act) style reasoning loop.
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
"""

CYPHER_GENERATION_PROMPT = """
You are an expert Neo4j Cypher query developer. Your task is to convert a user's question into a single, valid, read-only Cypher query based on the provided graph schema.

**Live Graph Schema:**
{schema}

**CRITICAL Instructions:**
1.  Your `RETURN` clause should look like this: `RETURN p, properties(r) as rel_props`.
2.  The relationship in your `MATCH` clause must be assigned to a variable `r`.
3.  Query against the `name_normalized` property for all `WHERE` clauses on nodes.
4.  If the question cannot be answered, you MUST return the single word: `NONE`.
5.  Output ONLY the Cypher query or the word `NONE`.

**Example Question:** "What company sponsors Abaloparatide?"
**Example Valid Query:** MATCH p=(drug:Entity)-[r:HASSPONSOR]->(sponsor:Entity) WHERE drug.name_normalized = 'abaloparatide' RETURN p, properties(r) as rel_props

**Question:** {question}
"""

# --- START OF DEFINITIVE FIX ---

DECOMPOSITION_PROMPT = """
You are a master query planner. Your goal is to determine if a user's question can be answered in a single step or if it requires decomposition into multiple, simpler sub-questions.

Analyze the user's question and the chat history.

**Decision Criteria:**
- **Single Step:** If the question asks for a direct fact, a summary of a single topic, or a simple definition.
- **Decomposition:** If the question requires comparing information from two or more distinct topics (e.g., two drugs, two meetings), finding an intersection of two sets of information (e.g., "sponsors who submitted in BOTH meetings"), or involves a sequence of steps.

**Output Schema:**
You MUST output a single, valid JSON object with two keys:
1.  `requires_decomposition`: A boolean (`true` or `false`).
2.  `plan`: A list of strings.
    - If `requires_decomposition` is `false`, the plan should contain a single item: the original question.
    - If `requires_decomposition` is `true`, the plan should contain two or more simple, answerable sub-questions that build on each other to answer the original question.

**Example 1 (Single Step):**
- User Question: "What is the use of Esketamine?"
- Your JSON Output:
{{
  "requires_decomposition": false,
  "plan": ["What is the use of Esketamine?"]
}}

**Example 2 (Decomposition):**
- User Question: "Which companies submitted drugs in both the March 2024 and May 2024 PBAC meetings?"
- Your JSON Output:
{{
  "requires_decomposition": true,
  "plan": [
    "List all sponsors who made submissions in the March 2024 PBAC meeting documents.",
    "List all sponsors who made submissions in the May 2024 PBAC meeting documents."
  ]
}}

**TASK:**
- Chat History: {chat_history}
- User Question: {question}

Now, generate the JSON output.
"""

# --- END OF DEFINITIVE FIX ---

REASONING_SYNTHESIS_PROMPT = """
You are a highly intelligent synthesis agent. Your task is to answer a user's complex original question based on a series of observations you have made by answering simpler sub-questions.

**User's Original Question:** "{question}"

**Your Observations (Scratchpad):**
---
{scratchpad}
---

**CRITICAL INSTRUCTIONS:**
1.  Read the user's original question and all your observations from the scratchpad.
2.  Synthesize a final, comprehensive answer to the original question.
3.  **Do not show your step-by-step reasoning.** Just provide the final, clean answer.
4.  If your observations are insufficient to answer the question, clearly state what information you found and why it is not enough.
5.  Include citations from your observations where appropriate.

**Final Answer:**
"""

# --- END: New Prompts for ReAct Agent ---


# --- START OF DEFINITIVE, HARDENED PROMPT ---
DIRECT_SYNTHESIS_PROMPT = """
You are a document analysis bot. Your ONLY job is to answer the user's question using the provided evidence.

**TASK:**
1.  Read the User's Question.
2.  Read the Evidence blocks. Each block has a piece of text and a citation link.
3.  Synthesize a direct answer to the question.
4.  **IMPERATIVE:** You MUST end your answer with the citation link from the evidence you used.
5.  If the evidence does not contain the answer, you MUST state "The provided evidence does not contain the answer." and nothing else.

**EXAMPLE:**
User's Question: "What is the sponsor for DrugX?"
Evidence:
---
Evidence from graph: DrugX has sponsor CompanyY.
Citation: <a href="..." target="_blank">Document A (Page 5)</a>
---
Your Answer:
The sponsor for DrugX is CompanyY. <a href="..." target="_blank">Document A (Page 5)</a>

---
**User's Question:** "{question}"
---
**Evidence:**
{context_str}
---
**Your Answer:**
"""
# --- END OF DEFINITIVE, HARDENED PROMPT ---