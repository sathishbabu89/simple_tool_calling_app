import os
import tempfile
import json

import streamlit as st
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_core.tools import tool
from langchain_core.messages import (
    HumanMessage,
    SystemMessage,
    ToolMessage,
)


# =========================================================
# 1. Load configuration
# =========================================================

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv(
    "GROQ_MODEL",
    "llama-3.3-70b-versatile"
)


# =========================================================
# 2. Streamlit configuration
# =========================================================

st.set_page_config(
    page_title="Bank AI Assistant - Tool Calling",
    page_icon="🏦",
    layout="wide"
)

st.title("🏦 Bank AI Assistant")
st.caption(
    "RAG + Tool Calling demonstration"
)


# =========================================================
# 3. Check API key
# =========================================================

if not GROQ_API_KEY:
    st.error(
        "GROQ_API_KEY was not found. "
        "Please add it to your .env file."
    )
    st.stop()


# =========================================================
# 4. Upload PDF
# =========================================================

st.subheader("1️⃣ Upload the bank policy")

uploaded_file = st.file_uploader(
    "Choose a PDF document",
    type=["pdf"],
    help="Upload the policy document used by the RAG tool."
)


# =========================================================
# 5. Create RAG index
# =========================================================

if uploaded_file is not None:

    if (
        st.session_state.get("document_name")
        != uploaded_file.name
    ):

        with st.spinner("Preparing the document..."):

            # -------------------------------------------------
            # Save PDF temporarily
            # -------------------------------------------------

            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=".pdf"
            ) as temp_file:

                temp_file.write(
                    uploaded_file.getbuffer()
                )

                pdf_path = temp_file.name

            # -------------------------------------------------
            # Load PDF
            # -------------------------------------------------

            loader = PyPDFLoader(pdf_path)

            documents = loader.load()

            # -------------------------------------------------
            # Split into chunks
            # -------------------------------------------------

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=700,
                chunk_overlap=100
            )

            chunks = text_splitter.split_documents(
                documents
            )

            # -------------------------------------------------
            # Create embeddings
            # -------------------------------------------------

            embeddings = HuggingFaceEmbeddings(
                model_name=(
                    "sentence-transformers/"
                    "all-MiniLM-L6-v2"
                )
            )

            # -------------------------------------------------
            # Create FAISS vector store
            # -------------------------------------------------

            vectorstore = FAISS.from_documents(
                chunks,
                embeddings
            )

            # -------------------------------------------------
            # Store in session state
            # -------------------------------------------------

            st.session_state["vectorstore"] = vectorstore
            st.session_state["document_name"] = uploaded_file.name
            st.session_state["chunk_count"] = len(chunks)

        st.success(
            f"✅ Document ready! "
            f"'{uploaded_file.name}' has been processed."
        )


# =========================================================
# 6. Tool definitions
# =========================================================

@tool
def get_account_balance(account_id: str) -> str:
    """
    Get the current account balance for a customer.

    Use this tool when the user asks about their
    current account balance.
    """

    # -----------------------------------------------------
    # Mock banking API
    # -----------------------------------------------------

    mock_accounts = {
        "12345": 85420.50,
        "67890": 12450.75
    }

    balance = mock_accounts.get(account_id)

    if balance is None:
        return (
            f"Account {account_id} was not found."
        )

    return (
        f"Account {account_id} has a current "
        f"balance of £{balance:,.2f}."
    )


# =========================================================
# Transaction tool
# =========================================================

@tool
def get_recent_transactions(
    account_id: str,
    number_of_transactions: int = 5
) -> str:
    """
    Get recent transactions for a customer account.

    Use this tool when the user asks to see,
    list, or review recent transactions.
    """

    # -----------------------------------------------------
    # Mock transaction API
    # -----------------------------------------------------

    mock_transactions = {
        "12345": [
            {
                "date": "2026-10-04",
                "description": "Tesco",
                "amount": "-45.20"
            },
            {
                "date": "2026-10-03",
                "description": "Salary Credit",
                "amount": "3200.00"
            },
            {
                "date": "2026-10-02",
                "description": "Amazon",
                "amount": "-82.50"
            },
            {
                "date": "2026-10-01",
                "description": "Electricity Bill",
                "amount": "-120.00"
            },
            {
                "date": "2026-09-30",
                "description": "Coffee Shop",
                "amount": "-5.80"
            }
        ]
    }

    transactions = mock_transactions.get(
        account_id
    )

    if transactions is None:
        return (
            f"No transaction data found for "
            f"account {account_id}."
        )

    transactions = transactions[
        :number_of_transactions
    ]

    return json.dumps(
        transactions,
        indent=2
    )


# =========================================================
# RAG tool
# =========================================================

def create_policy_search_tool(vectorstore):

    @tool
    def search_bank_policy(question: str) -> str:
        """
        Search the uploaded bank policy for information
        relevant to the user's question.

        Use this tool when the user asks about bank
        policies, procedures, rules, limits, complaint
        handling, leave, reimbursement, or other
        information contained in the uploaded policy.
        """

        retrieved_docs = vectorstore.similarity_search(
            question,
            k=3
        )

        if not retrieved_docs:
            return (
                "No relevant information was found "
                "in the uploaded policy."
            )

        context_parts = []

        for index, doc in enumerate(
            retrieved_docs,
            start=1
        ):

            page_number = (
                doc.metadata.get("page", 0) + 1
            )

            context_parts.append(
                f"[Policy Section {index} - "
                f"Page {page_number}]\n"
                f"{doc.page_content}"
            )

        return "\n\n".join(context_parts)

    return search_bank_policy


# =========================================================
# 7. Create LLM
# =========================================================

llm = ChatGroq(
    groq_api_key=GROQ_API_KEY,
    model=GROQ_MODEL,
    temperature=0
)


# =========================================================
# 8. Main question-answer section
# =========================================================

if "vectorstore" in st.session_state:

    st.subheader("2️⃣ Ask your question")

    account_id = st.text_input(
        "Demo Account ID",
        value="12345",
        help=(
            "Used only by the mock banking tools."
        )
    )

    question = st.text_input(
        "What would you like to know?",
        placeholder=(
            "Examples:\n"
            "• How many days do I have to acknowledge a complaint?\n"
            "• What is my current account balance?\n"
            "• Show my recent transactions."
        )
    )

    if st.button(
        "🤖 Ask AI Assistant",
        type="primary"
    ):

        if not question.strip():
            st.warning(
                "Please enter a question."
            )
            st.stop()

        with st.spinner(
            "The AI assistant is thinking..."
        ):

            # -------------------------------------------------
            # Create RAG tool
            # -------------------------------------------------

            policy_tool = create_policy_search_tool(
                st.session_state["vectorstore"]
            )

            # -------------------------------------------------
            # Register available tools
            # -------------------------------------------------

            tools = [
                policy_tool,
                get_account_balance,
                get_recent_transactions
            ]

            tools_by_name = {
                tool.name: tool
                for tool in tools
            }

            # -------------------------------------------------
            # Bind tools to LLM
            # -------------------------------------------------

            llm_with_tools = llm.bind_tools(
                tools
            )

            # -------------------------------------------------
            # System instructions
            # -------------------------------------------------

            system_message = SystemMessage(
                content="""
You are a banking AI assistant.

You have access to three tools:

1. search_bank_policy
   - Use this for questions about banking policies
     and procedures.

2. get_account_balance
   - Use this when the user asks about an account
     balance.

3. get_recent_transactions
   - Use this when the user asks about recent
     transactions.

Important rules:

- Use tools whenever they are appropriate.
- Do not invent account information.
- Do not invent policy information.
- The account ID provided by the application is the
  demo account currently being discussed.
- After receiving tool results, provide a concise
  natural-language answer.
"""
            )

            messages = [
                system_message,
                HumanMessage(
                    content=question
                )
            ]

            # -------------------------------------------------
            # Tool execution loop
            # -------------------------------------------------

            tool_trace = []

            max_iterations = 5

            for iteration in range(
                max_iterations
            ):

                response = (
                    llm_with_tools.invoke(
                        messages
                    )
                )

                # -------------------------------------------------
                # No tool call -> final answer
                # -------------------------------------------------

                if not response.tool_calls:

                    messages.append(response)

                    final_answer = response.content

                    break

                # -------------------------------------------------
                # LLM requested one or more tools
                # -------------------------------------------------

                messages.append(response)

                for tool_call in response.tool_calls:

                    tool_name = tool_call["name"]
                    tool_args = tool_call["args"]
                    tool_call_id = tool_call["id"]

                    tool_trace.append({
                        "tool": tool_name,
                        "arguments": tool_args
                    })

                    selected_tool = tools_by_name.get(
                        tool_name
                    )

                    if selected_tool is None:

                        tool_result = (
                            f"Unknown tool: "
                            f"{tool_name}"
                        )

                    else:

                        # -------------------------------------------------
                        # Inject demo account ID for account tools
                        # -------------------------------------------------

                        if tool_name in [
                            "get_account_balance",
                            "get_recent_transactions"
                        ]:

                            tool_args["account_id"] = (
                                account_id
                            )

                        tool_result = selected_tool.invoke(
                            tool_args
                        )

                    # -------------------------------------------------
                    # Send tool result back to LLM
                    # -------------------------------------------------

                    messages.append(
                        ToolMessage(
                            content=str(
                                tool_result
                            ),
                            tool_call_id=tool_call_id
                        )
                    )

            else:

                final_answer = (
                    "The assistant reached the maximum "
                    "number of tool-calling steps."
                )

        # =====================================================
        # 9. Display final answer
        # =====================================================

        st.subheader("3️⃣ Answer")

        st.success(
            final_answer
        )

        # =====================================================
        # 10. Display tool execution trace
        # =====================================================

        if tool_trace:

            with st.expander(
                "🔧 Show tool-calling trace",
                expanded=True
            ):

                st.caption(
                    "This shows which tools the LLM "
                    "decided to call."
                )

                for index, trace in enumerate(
                    tool_trace,
                    start=1
                ):

                    st.markdown(
                        f"### Tool Call {index}"
                    )

                    st.code(
                        json.dumps(
                            trace,
                            indent=2
                        ),
                        language="json"
                    )

        else:

            with st.expander(
                "ℹ️ Tool-calling trace"
            ):

                st.write(
                    "The LLM answered directly "
                    "without calling a tool."
                )

