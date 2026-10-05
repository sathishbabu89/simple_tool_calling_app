# Simple Tool Calling App

A hands-on Generative AI application demonstrating **LLM Tool Calling** using Python, LangChain, Groq, Streamlit, and a simple RAG pipeline.

The application allows an LLM to decide whether it should use:

- A **RAG tool** to search an uploaded bank policy PDF
- An **Account Balance tool** to retrieve mock account information
- A **Recent Transactions tool** to retrieve mock transaction information

This project is designed as a learning example for understanding the progression from **RAG → Tool Calling → Agentic AI**.

---

## 🎯 What This Project Demonstrates

The application demonstrates how an LLM can move beyond simply generating text.

### Without tools

```text
User Question
      ↓
     LLM
      ↓
   Answer
```

The LLM can generate a response, but it cannot directly access external systems.

### With RAG

```text
User Question
      ↓
Retriever
      ↓
Relevant Documents
      ↓
     LLM
      ↓
   Answer
```

RAG gives the LLM access to external **knowledge**.

### With Tool Calling

```text
User Question
      ↓
     LLM
      ↓
Tool Selection
      ↓
Tool Execution
      ↓
Tool Result
      ↓
     LLM
      ↓
Final Answer
```

Tool Calling gives the LLM access to external **capabilities**.

---

# 🏦 Application Overview

The application is a simple banking AI assistant.

The user can upload a banking policy PDF and ask questions about it.

The LLM can also access mock banking tools for:

1. Account balance
2. Recent transactions

The application therefore demonstrates how a single LLM can choose different capabilities based on the user's question.

---

# 🏗️ Architecture

```text
                         User Question
                              │
                              ▼
                         ┌─────────┐
                         │   LLM   │
                         └────┬────┘
                              │
                       Tool Selection
                              │
          ┌───────────────────┼───────────────────┐
          │                   │                   │
          ▼                   ▼                   ▼
 search_bank_policy    get_account_balance   get_recent_transactions
          │                   │                   │
          ▼                   ▼                   ▼
        FAISS              Mock API            Mock API
          │                   │                   │
          ▼                   ▼                   ▼
     Policy PDF           Account Balance       Transactions
          │                   │                   │
          └───────────────────┼───────────────────┘
                              ▼
                             LLM
                              │
                              ▼
                         Final Answer
```

---

# 🔧 Tools Available

The application exposes three tools to the LLM.

## 1. `search_bank_policy()`

This tool searches the uploaded PDF using FAISS.

```text
User Question
      ↓
LLM
      ↓
search_bank_policy()
      ↓
FAISS
      ↓
Relevant PDF chunks
      ↓
LLM
      ↓
Answer
```

Use this tool when the user asks about information contained in the uploaded bank policy.

Example:

> How many business days does the bank have to acknowledge a complaint?

---

## 2. `get_account_balance()`

This tool represents a mock banking API.

```text
User Question
      ↓
LLM
      ↓
get_account_balance()
      ↓
Mock Banking API
      ↓
Account Balance
      ↓
LLM
      ↓
Answer
```

Example:

> What is my current account balance?

The demonstration account `12345` returns a mock balance of:

```text
£85,420.50
```

> **Note:** This is mock data for educational purposes only.

---

## 3. `get_recent_transactions()`

This tool represents another mock banking API.

```text
User Question
      ↓
LLM
      ↓
get_recent_transactions()
      ↓
Mock Banking API
      ↓
Transaction Data
      ↓
LLM
      ↓
Answer
```

Example:

> Show me my recent transactions.

The tool also demonstrates tool arguments.

For example:

> Show me my last 3 transactions.

The LLM can request:

```python
get_recent_transactions(
    account_id="12345",
    number_of_transactions=3
)
```

---

# 🧠 How Tool Calling Works

The important concept in this application is:

```text
LLM Decision
     ↓
Tool Call
     ↓
Application Executes Tool
     ↓
Tool Result
     ↓
LLM
     ↓
Final Response
```

The LLM does **not** directly execute Python functions.

Instead, the LLM generates a structured tool-call request.

The application receives that request and executes the corresponding Python function.

The result is then sent back to the LLM.

---

# 🔄 Tool Calling Flow

Consider this question:

> What is my current account balance?

### Step 1 — User asks the question

```text
"What is my current account balance?"
```

### Step 2 — LLM decides a tool is required

The LLM identifies that it needs account information.

It requests:

```text
get_account_balance
```

with:

```text
account_id = "12345"
```

### Step 3 — Application executes the tool

The Python application executes:

```python
get_account_balance(
    account_id="12345"
)
```

### Step 4 — Tool returns the result

```text
Account 12345 has a current balance
of £85,420.50.
```

### Step 5 — Result goes back to the LLM

The LLM receives the tool result.

### Step 6 — LLM generates the final response

```text
Your current account balance is £85,420.50.
```

---

# 🔑 Important Code Concept

Tools are defined using LangChain's `@tool` decorator.

Example:

```python
@tool
def get_account_balance(account_id: str) -> str:
    """
    Get the current account balance
    for a customer.
    """

    # Tool implementation
    ...
```

The tool's:

- Name
- Description
- Input parameters

help the LLM understand when and how the tool can be used.

---

# 🔗 Binding Tools to the LLM

The tools are provided to the LLM using:

```python
llm_with_tools = llm.bind_tools(tools)
```

Conceptually:

```text
                 LLM
                  │
       ┌──────────┼──────────┐
       │          │          │
       ▼          ▼          ▼
   Policy      Balance   Transactions
    Tool         Tool        Tool
```

The LLM can select an appropriate tool based on the user's question.

---

# 🔎 Tool Selection Examples

## Example 1 — Policy Question

```text
How many business days does the bank
have to acknowledge a complaint?
```

Expected tool:

```text
search_bank_policy
```

Flow:

```text
Question
   ↓
LLM
   ↓
search_bank_policy()
   ↓
FAISS
   ↓
PDF
   ↓
Policy Information
   ↓
LLM
   ↓
Answer
```

---

## Example 2 — Account Question

```text
What is my current account balance?
```

Expected tool:

```text
get_account_balance
```

Flow:

```text
Question
   ↓
LLM
   ↓
get_account_balance()
   ↓
Mock Banking API
   ↓
Balance
   ↓
LLM
   ↓
Answer
```

---

## Example 3 — Transaction Question

```text
Show me my recent transactions.
```

Expected tool:

```text
get_recent_transactions
```

---

## Example 4 — Tool Arguments

```text
Show me my last 3 transactions.
```

The LLM can generate tool arguments such as:

```json
{
  "account_id": "12345",
  "number_of_transactions": 3
}
```

This demonstrates that tool calling is not only about selecting a tool.

The LLM can also determine the **input arguments** required by the tool.

---

# 🆚 RAG vs Tool Calling

| | RAG | Tool Calling |
|---|---|---|
| Primary purpose | Access knowledge | Access capabilities |
| Data source | Documents / knowledge base | APIs / functions / systems |
| Example | Search bank policy | Get account balance |
| Retrieval | Vector search | Tool execution |
| Typical output | Relevant document context | Tool result |
| LLM involvement | Generate answer using context | Select tool and generate answer |

A useful mental model:

> **RAG = Knowledge**

> **Tools = Capabilities**

---

# 🤖 From Tool Calling to Agents

Tool Calling is an important building block for Agentic AI.

### Simple LLM

```text
User
 ↓
LLM
 ↓
Answer
```

### RAG

```text
User
 ↓
Retrieve Knowledge
 ↓
LLM
 ↓
Answer
```

### Tool Calling

```text
User
 ↓
LLM
 ↓
Tool
 ↓
Result
 ↓
LLM
 ↓
Answer
```

### Agent

An agent can dynamically decide what actions to take and potentially perform multiple tool calls.

```text
User Goal
    ↓
   LLM
    ↓
Choose Action
    ↓
  Tool
    ↓
Observe Result
    ↓
   LLM
    ↓
Choose Next Action
    ↓
  Tool
    ↓
   ...
    ↓
Final Answer
```

This project demonstrates the building blocks required to move toward an agentic system.

---

# 🧪 Demo Questions

Use the following questions to demonstrate the application.

### Policy / RAG

```text
How many business days does the bank have to acknowledge a complaint?
```

Expected:

```text
search_bank_policy
```

---

### Policy / RAG

```text
How long does the bank have to provide a final response to a standard complaint?
```

Expected:

```text
search_bank_policy
```

---

### Account

```text
What is my current account balance?
```

Expected:

```text
get_account_balance
```

---

### Transactions

```text
Show me my recent transactions.
```

Expected:

```text
get_recent_transactions
```

---

### Tool Arguments

```text
Show me my last 3 transactions.
```

Expected:

```text
get_recent_transactions(
    number_of_transactions=3
)
```

---

### Multiple Capabilities

```text
What is the complaint acknowledgement policy
and what is my current account balance?
```

This can demonstrate the use of multiple tools.

---

# 📂 Project Structure

```text
simple_tool_calling_app/
│
├── app.py
├── requirements.txt
├── README.md
├── .env.example
├── .gitignore
│
└── data/
    └── README.md
```

The application is intentionally kept simple so that learners can focus on the Tool Calling concept rather than navigating a large project structure.

---

# 🛠️ Technologies Used

- Python
- Streamlit
- LangChain
- LangChain Groq
- Groq
- Hugging Face Embeddings
- FAISS
- PyPDF
- python-dotenv

---

# 📋 Prerequisites

Make sure you have:

- Python 3.10+
- A Groq API key
- Internet access for downloading the embedding model
- Git

---

# 🚀 Setup

## 1. Clone the repository

```bash
git clone <YOUR_REPOSITORY_URL>
```

Navigate into the project:

```bash
cd simple_tool_calling_app
```

---

## 2. Create a virtual environment

### Windows

```bash
python -m venv .venv
```

Activate it:

```bash
.venv\Scripts\activate
```

### macOS / Linux

```bash
python3 -m venv .venv
```

Activate it:

```bash
source .venv/bin/activate
```

---

## 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

# 🔐 Environment Configuration

Create a `.env` file in the project root.

```text
GROQ_API_KEY=your_groq_api_key
GROQ_MODEL=llama-3.3-70b-versatile
```

You can use `.env.example` as a template.

**Never commit your actual `.env` file or API key to Git.**

---

# ▶️ Run the Application

Start Streamlit:

```bash
streamlit run app.py
```

The application will open in your browser.

---

# 📄 Upload a Policy PDF

Upload a bank policy PDF through the Streamlit interface.

The application will:

```text
PDF
 ↓
PyPDFLoader
 ↓
Text Splitting
 ↓
Hugging Face Embeddings
 ↓
FAISS
```

The resulting vector store is then used by the `search_bank_policy()` tool.

---

# 🔍 Tool Execution Trace

The application displays the tool-calling trace.

For example:

```text
Tool Call 1

{
  "tool": "search_bank_policy",
  "arguments": {
    "question": "How many business days does the bank have to acknowledge a complaint?"
  }
}
```

This allows learners to see that the LLM is requesting a tool rather than directly executing the Python function.

---

# ⚠️ Important: Mock Banking Tools

The following tools are **mock implementations**:

```text
get_account_balance()
get_recent_transactions()
```

They do not connect to a real banking system.

They exist only to demonstrate how an LLM can interact with external capabilities through Tool Calling.

A real banking implementation would require appropriate:

- Authentication
- Authorization
- Input validation
- API security
- Audit logging
- Error handling
- Rate limiting
- Transaction controls
- Privacy and compliance controls

---

# 🎓 Learning Takeaways

After completing this project, you should understand:

1. What LLM Tool Calling is
2. Why Tool Calling is different from RAG
3. How to define tools using LangChain
4. How tools are bound to an LLM
5. How an LLM selects a tool
6. How an LLM generates tool arguments
7. How the application executes the tool
8. How tool results are returned to the LLM
9. How the LLM generates the final response
10. How Tool Calling forms the foundation for Agentic AI

---

# 🧠 Simple Mental Model

Remember:

```text
LLM
 │
 ├── RAG
 │     └── "Give me knowledge"
 │
 └── Tools
       └── "Give me capabilities"
```

And:

```text
RAG
=
Knowledge
```

```text
Tool Calling
=
Capabilities
```

```text
Agents
=
Decision + Actions
```

---

# 🔮 What's Next?

This project is intentionally the step before building an Agent.

The natural progression is:

```text
LLM
 ↓
Prompt Engineering
 ↓
RAG
 ↓
Tool Calling
 ↓
Agent
 ↓
Multi-Agent
 ↓
Production Agentic AI
```

The next step is to evolve this application into an **Agentic AI application** where the system can dynamically decide:

- Which tool to use
- When to use it
- Whether another tool is required
- When the task is complete
- How to combine results from multiple tools

---

## ⚠️ Disclaimer

This project is created for **educational and demonstration purposes**.

The banking data and APIs used in this application are mock implementations and must not be used for real financial transactions or production banking systems.
