import streamlit as st
import os
import re
import time
import asyncio
import warnings
import mysql.connector
import pandas as pd
from typing import TypedDict, Annotated, List, Dict, Optional
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

warnings.filterwarnings('ignore')

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser

from langgraph.graph import StateGraph, END, START
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver

from openai import AsyncOpenAI
from ragas.llms import llm_factory
from ragas.metrics.collections import AnswerAccuracy

from colorama import init, Fore
init(autoreset=True)

# Page config
st.set_page_config(
    page_title="Text-to-SQL Query Interface",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
.main-header {
    font-size: 2.4rem;
    font-weight: 700;
    margin-bottom: 0.5rem;
}
</style>
""", unsafe_allow_html=True)

# Session state
defaults = {
    "db_connected": False,
    "llm_initialized": False,
    "query_history": [],
    "current_results": None,
    "llm": None,
    "db": None,
    "schema": "",
    "groq_api_key": "",
}
for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


# Helpers
def clean_sql(text: str) -> str:
    """Strip markdown code fences and whitespace from LLM SQL output."""
    text = text.strip()
    match = re.search(r"```(?:sql)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if match:
        text = match.group(1)
    return text.strip()


def is_safe_sql(sql: str) -> bool:
    """Allow only a single read-only statement."""
    stripped = sql.strip().rstrip(";").strip()
    if not stripped or ";" in stripped:
        return False
    upper = stripped.upper()
    if not upper.startswith(("SELECT", "SHOW", "DESCRIBE", "DESC", "EXPLAIN")):
        return False
    forbidden = r"\b(DROP|DELETE|TRUNCATE|UPDATE|INSERT|ALTER|CREATE|REPLACE|GRANT|REVOKE|RENAME)\b"
    if re.search(forbidden, upper):
        return False
    return True


# Database Manager
class DatabaseManager:
    """Simple database connection & query execution"""

    def __init__(self, config: dict):
        self.config = config
        self.connection = None
        self.schema_text = ""

    def connect(self):
        """Connect to MySQL database"""
        try:
            self.connection = mysql.connector.connect(**self.config)
            return True
        except Exception as e:
            print(f"{Fore.RED}Connection error: {e}")
            return False

    def close(self):
        try:
            if self.connection and self.connection.is_connected():
                self.connection.close()
        except Exception:
            pass

    def get_schema(self) -> str:
        """Get simplified schema for LLM"""
        if not self.connection:
            self.connect()

        try:
            cursor = self.connection.cursor()
            cursor.execute("SHOW TABLES")
            tables = cursor.fetchall()

            schema_text = "DATABASE SCHEMA:\n\n"

            for (table_name,) in tables:
                schema_text += f"TABLE: {table_name}\n"

                # Get columns
                cursor.execute(f"DESCRIBE `{table_name}`")
                columns = cursor.fetchall()

                schema_text += "Columns:\n"
                for col in columns:
                    col_name, col_type, nullable, key, default, extra = col
                    schema_text += f"  - {col_name} ({col_type})"
                    if key == 'PRI':
                        schema_text += " PRIMARY KEY"
                    schema_text += "\n"

                # Get sample data (2 rows)
                cursor.execute(f"SELECT * FROM `{table_name}` LIMIT 2")
                samples = cursor.fetchall()
                if samples:
                    schema_text += "Sample rows:\n"
                    for row in samples:
                        schema_text += f"  {row}\n"
                schema_text += "\n"
            cursor.close()
            self.schema_text = schema_text
            return schema_text

        except Exception as e:
            print(f"{Fore.RED}Schema error: {e}")
            return ""

    def _run(self, query: str) -> tuple:
        """Run a single query on the current connection"""
        cursor = self.connection.cursor()
        try:
            cursor.execute(query)
            query_upper = query.strip().upper()
            if query_upper.startswith(('SELECT', 'SHOW', 'DESCRIBE', 'DESC', 'EXPLAIN')):
                results = cursor.fetchall()
                columns = [desc[0] for desc in cursor.description] if cursor.description else []
                return True, results, columns
            self.connection.commit()
            return True, "Query executed", None
        finally:
            cursor.close()

    def execute_query(self, query: str) -> tuple:
        """Execute SQL Query with auto-reconnect"""
        if not self.connection or not self.connection.is_connected():
            self.connect()

        try:
            return self._run(query)
        except Exception as e:
            # Reconnect and retry once
            if "Lost connection" in str(e) or "gone away" in str(e):
                self.connect()
                try:
                    return self._run(query)
                except Exception as e2:
                    return False, str(e2), None
            return False, str(e), None


# Agent State
class AgentState(TypedDict):
    """Simple state for the pipeline"""
    messages: Annotated[List, add_messages]
    user_query: str
    schema: str
    sql_query: Optional[str]
    results: Optional[Dict]
    answer: Optional[str]
    error: Optional[str]
    retry_count: int
    evaluation_score: Optional[float]


# Query validation
def validate_query(query: str, llm) -> tuple:
    """
    Validates if query should go through SQL pipeline
    Returns: (is_sql_query: bool, response: str)
    """

    validation_prompt = ChatPromptTemplate.from_template("""
You are a gatekeeper for a SQL database query system. Analyze the user's input:

User Input: {query}

STRICT RULES:
1. If this is a GENUINE DATABASE QUERY (asking to retrieve, count, show, list, find data from database) → Return ONLY the word: TRUE
2. If this is ANYTHING ELSE (greeting, gibberish, out-of-context question, general chat) → Return a natural language response

Examples of genuine database queries (return "TRUE"):
- Show me all students
- Count the records
- What is the highest CGPA?
- List all tables
- Find students with marks > 80

Examples of non-database queries (return natural response):
- Hello/Hi → "Hello! I can help you query your database. What data would you like to retrieve?"
- How are you → "I'm a database assistant. I can help you retrieve data from your database. What would you like to know?"
- What's 2+2 → "I'm specifically designed for database queries. Please ask about data you'd like to retrieve."
- asdfgh123 → "I couldn't understand that. Please ask a clear question about your database."
- Write Python code → "I only handle database queries. Please ask about retrieving data from your database."

Remember: Return ONLY "TRUE" (nothing else) for database queries, or a helpful response for everything else.

Response:""")

    chain = validation_prompt | llm | StrOutputParser()

    try:
        result = chain.invoke({"query": query}).strip()

        # Accept "TRUE", "TRUE.", "true\n" and similar
        if re.sub(r'[^\w]', '', result).upper() == "TRUE" and len(result) < 10:
            return True, None

        return False, result

    except Exception as e:
        print(f"{Fore.YELLOW}Validation error: {e}")
        # Default to processing as SQL if validation fails
        return True, None


# Pipeline nodes (no Streamlit access inside nodes; dependencies are passed in)
def generate_sql_node(state: AgentState, llm) -> Dict:
    """Generate SQL from natural language"""

    prompt = ChatPromptTemplate.from_template("""
You are a SQL expert. Generate a MySQL query for this request.

Database Schema:
{schema}

User Request: {query}

Rules:
1. Use ONLY tables and columns from the schema
2. Generate valid MySQL syntax
3. Use JOINs when needed
4. Return ONLY the SQL query, no explanations

SQL Query:""")

    chain = prompt | llm | StrOutputParser()

    try:
        sql = clean_sql(chain.invoke({
            "schema": state["schema"],
            "query": state["user_query"]
        }))

        print(f"{Fore.YELLOW}Generated SQL: {sql}")

        return {
            "sql_query": sql,
            "error": None,
            "messages": [AIMessage(content="SQL Generated")]
        }
    except Exception as e:
        return {
            "sql_query": None,
            "error": f"SQL generation failed: {str(e)}",
            "retry_count": state.get("retry_count", 0) + 1,
            "messages": [AIMessage(content=f"Error:{str(e)}")]
        }


def execute_sql_node(state: AgentState, llm, db) -> Dict:
    """Executing SQL with retry logic"""

    retry_count = state.get("retry_count", 0)

    if not state.get("sql_query"):
        return {"error": "No SQL to execute", "retry_count": retry_count + 1}

    sql = state["sql_query"]

    # Safety check: single read-only statement only. Do not retry unsafe SQL.
    if not is_safe_sql(sql):
        return {
            "error": "Unsafe SQL detected. Only a single read-only query is allowed.",
            "retry_count": 2,
            "messages": [AIMessage(content="Unsafe SQL blocked")]
        }

    # Execute query
    success, data, columns = db.execute_query(sql)

    if not success and retry_count < 2:
        # If failed, try to fix and retry
        fix_prompt = ChatPromptTemplate.from_template("""
The following SQL query failed with error: {error}

Original SQL: {sql}

Database Schema:
{schema}

Generate a corrected SQL query that fixes the error. Return ONLY the SQL:""")

        chain = fix_prompt | llm | StrOutputParser()

        try:
            fixed_sql = clean_sql(chain.invoke({
                "error": data,
                "sql": sql,
                "schema": state["schema"]
            }))

            print(f"{Fore.YELLOW}Retrying with fixed SQL: {fixed_sql}")

            if is_safe_sql(fixed_sql):
                success, data, columns = db.execute_query(fixed_sql)

                if success:
                    results = {
                        "columns": columns if columns else [],
                        "rows": data if columns else [],
                        "row_count": len(data) if columns else 0
                    }
                    return {
                        "results": results,
                        "sql_query": fixed_sql,
                        "error": None,
                        "retry_count": retry_count + 1,
                        "messages": [AIMessage(content="Query executed after retry")]
                    }
        except Exception as e:
            print(f"{Fore.YELLOW}Retry failed: {e}")

    if success:
        results = {
            "columns": columns if columns else [],
            "rows": data if columns else [],
            "row_count": len(data) if columns else 0
        }

        print(f"{Fore.GREEN}Query executed: {results['row_count']} rows")

        return {
            "results": results,
            "error": None,
            "messages": [AIMessage(content="Query executed successfully")]
        }

    return {
        "error": f"Execution failed: {data}",
        "retry_count": retry_count + 1,
        "messages": [AIMessage(content=f"Error: {data}")]
    }


def format_answer_node(state: AgentState, llm) -> Dict:
    """Format results as natural language"""

    if state.get("error"):
        return {
            "answer": f"I encountered an error: {state['error']}",
            "messages": [AIMessage(content="Error in processing")]
        }

    if not state.get("results"):
        return {
            "answer": "No results to display",
            "messages": [AIMessage(content="No results")]
        }

    results = state["results"]

    # Preparing data for LLM
    shown_count = 0
    if results["rows"]:
        # Format first 10 rows
        rows_text = f"Columns: {results['columns']}\n"
        for i, row in enumerate(results["rows"][:10]):
            rows_text += f"Row {i+1}: {row}\n"
            shown_count += 1
        if results["row_count"] > 10:
            rows_text += f"... {results['row_count'] - 10} more rows exist but are NOT shown to you"
    else:
        rows_text = "No data found"

    prompt = ChatPromptTemplate.from_template("""
Convert these SQL results into a clear answer for the user.

User Question: {question}
SQL Query: {sql}
Total rows returned by the query: {count}
Rows shown to you below: {shown}

{results}

STRICT RULES:
1. Use ONLY the values that appear above. Never invent, estimate or calculate averages, ranges, percentages or trends.
2. If the result is a single value (a count, average, maximum and so on), state it in one sentence and do NOT add a table.
3. Otherwise, start with one sentence giving the total number of rows. If the total is larger than the rows shown, say that the first {shown} are listed.
4. Then show the rows you were given as a markdown table with all the columns, copying the values exactly.
5. Do not add anything after the table.

Answer:""")
    chain = prompt | llm | StrOutputParser()

    try:
        answer = chain.invoke({
            "question": state["user_query"],
            "sql": state["sql_query"],
            "results": rows_text,
            "count": results["row_count"],
            "shown": shown_count
        })

        if not answer or not answer.strip():
            answer = f"Found {results['row_count']} rows. The table below shows the first 10."

        return {
            "answer": answer,
            "messages": [AIMessage(content="Answer formatted")]
        }
    except Exception:
        return {
            "answer": f"Results: {results['row_count']} rows found",
            "messages": [AIMessage(content="Basic answer provided")]
        }


def evaluate_node(state: AgentState, api_key: str) -> Dict:
    """Simple RAGAS Evaluation"""

    if state.get("error") or not state.get("answer") or not state.get("sql_query"):
        return {
            "evaluation_score": None,
            "messages": [AIMessage(content="Skipping Evaluation")]
        }

    try:
        results_retrieved = str(state.get("results"))[:2000]

        # What the user actually sees: the text answer plus the displayed table rows
        res = state.get("results")
        shown_rows = res["rows"][:10] if isinstance(res, dict) and res.get("rows") else []
        response_text = state["answer"] + f"\nTable displayed to the user (first 10 rows): {shown_rows}"

        reference = f"SQL: {state['sql_query']}" + "\n" + f"Results Retrieved after execution of Query: {results_retrieved}"

        async def run_score():
            # Setup RAGAS with Groq (OpenAI-compatible endpoint)
            groq_client = AsyncOpenAI(
                api_key=api_key,
                base_url="https://api.groq.com/openai/v1",
            )
            ragas_llm = llm_factory("openai/gpt-oss-20b", client=groq_client, max_tokens=4096)
            scorer = AnswerAccuracy(llm=ragas_llm)
            return await scorer.ascore(
                user_input=state["user_query"],
                response=response_text,
                reference=reference,
            )

        # Running the async scorer in its own thread/event loop
        with ThreadPoolExecutor(max_workers=1) as executor:
            result = executor.submit(asyncio.run, run_score()).result()

        # New ragas returns a result object; older ones return a float
        score = float(getattr(result, "value", result))

        print(f"{Fore.GREEN}\u2713 RAGAS Score: {score:.2f}")

        return {
            "evaluation_score": score,
            "messages": [AIMessage(content=f"Evaluation: {score:.2f}")]
        }

    except Exception as e:
        print(f"{Fore.YELLOW}Evaluation skipped: {str(e)}")
        return {
            "evaluation_score": None,
            "messages": [AIMessage(content="Evaluation skipped")]
        }


def create_graph(llm, db, api_key: str):
    """Create the LangGraph workflow"""

    workflow = StateGraph(AgentState)

    # Add nodes (dependencies are bound here, so nodes never touch st.session_state)
    workflow.add_node("generate_sql", lambda state: generate_sql_node(state, llm))
    workflow.add_node("execute_sql", lambda state: execute_sql_node(state, llm, db))
    workflow.add_node("format_answer", lambda state: format_answer_node(state, llm))
    workflow.add_node("evaluate", lambda state: evaluate_node(state, api_key))

    # Define flow
    workflow.add_edge(START, "generate_sql")
    workflow.add_edge("generate_sql", "execute_sql")

    # Conditional routing after execution
    def check_execution(state):
        if state.get("error") and state.get("retry_count", 0) < 2:
            return "generate_sql"  # Retry
        return "format_answer"

    workflow.add_conditional_edges(
        "execute_sql",
        check_execution,
        {
            "generate_sql": "generate_sql",
            "format_answer": "format_answer"
        }
    )

    workflow.add_edge("format_answer", "evaluate")
    workflow.add_edge("evaluate", END)

    # Compile
    memory = MemorySaver()
    return workflow.compile(checkpointer=memory)


# Main UI
st.markdown('<h1 class="main-header">🔍 Text-to-SQL Query Interface</h1>', unsafe_allow_html=True)

# Sidebar for configuration
with st.sidebar:
    st.markdown("## ⚙️ Configuration")

    # Groq API Key
    with st.expander("🤖 Groq API Configuration", expanded=not st.session_state.llm_initialized):
        groq_api_key = st.text_input(
            "Groq API Key",
            type="password",
            placeholder="gsk_...",
            help="Enter your Groq API key"
        )

        if st.button("Initialize LLM", disabled=st.session_state.llm_initialized):
            if groq_api_key:
                try:
                    os.environ["GROQ_API_KEY"] = groq_api_key
                    st.session_state.llm = ChatGroq(
                        model="openai/gpt-oss-20b",
                        temperature=0.0,
                        max_tokens=2048,
                        groq_api_key=groq_api_key,
                    )
                    st.session_state.groq_api_key = groq_api_key
                    st.session_state.llm_initialized = True
                    st.success("✅ LLM initialized successfully!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Failed to initialize LLM: {e}")
            else:
                st.warning("Please enter Groq API key")

    if st.session_state.llm_initialized:
        st.success("✅ LLM Ready")

    # Database Configuration
    with st.expander("🗄️ Database Configuration", expanded=not st.session_state.db_connected):
        db_host = st.text_input("Host", value="localhost")
        db_port = st.number_input("Port", value=3306, min_value=1)
        db_user = st.text_input("Username", placeholder="root")
        db_password = st.text_input("Password", type="password")
        db_name = st.text_input("Database Name", placeholder="your_database")

        if st.button("Connect to Database", disabled=st.session_state.db_connected):
            if all([db_host, db_user, db_name]):
                db_config = {
                    "host": db_host,
                    "user": db_user,
                    "password": db_password,
                    "database": db_name,
                    "port": int(db_port)
                }

                with st.spinner("Connecting to database..."):
                    new_db = DatabaseManager(db_config)
                    if new_db.connect():
                        st.session_state.db = new_db
                        st.session_state.db_connected = True
                        st.session_state.schema = new_db.get_schema()
                        st.success("✅ Database connected successfully!")
                        st.rerun()
                    else:
                        st.error("Failed to connect to database")
            else:
                st.warning("Please fill in all required fields")

    if st.session_state.db_connected:
        st.success("✅ Database Connected")

        if st.button("Disconnect Database"):
            if st.session_state.db:
                st.session_state.db.close()
            st.session_state.db_connected = False
            st.session_state.db = None
            st.session_state.schema = ""
            st.rerun()

    # Show schema if connected
    if st.session_state.db_connected:
        with st.expander("📊 Database Schema"):
            st.code(st.session_state.schema, language="sql")

    # Query History
    if st.session_state.query_history:
        with st.expander("📜 Query History"):
            for i, item in enumerate(reversed(st.session_state.query_history[-5:])):
                st.text(f"{i+1}. {item['query'][:50]}...")

# Main content area
if not st.session_state.llm_initialized or not st.session_state.db_connected:
    st.info("👈 Please configure Groq API and Database connection in the sidebar to get started")
else:
    # Query input section
    col1, col2 = st.columns([4, 1])
    with col1:
        user_query = st.text_area(
            "Enter your question in natural language:",
            placeholder="e.g., Show me all students with CGPA greater than 8.0",
            height=100
        )

    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        execute_button = st.button("🚀 Execute Query", type="primary", use_container_width=True)

    # Sample queries
    st.markdown("#### 💡 Sample Queries")
    sample_queries = [
        "Show me all female patients with heart disease from the heart table",
        "Show the age, cholesterol and max heart rate of patients older than 60 with cholesterol above 240.",
        "Count the number of patients",
        "Show the patient with the highest age."
    ]

    sample_cols = st.columns(len(sample_queries))
    for i, sample in enumerate(sample_queries):
        with sample_cols[i]:
            if st.button(f"💡 {i+1}", help=sample, use_container_width=True):
                st.session_state.sample_query = sample
                st.rerun()

    # Use sample query if selected
    if 'sample_query' in st.session_state:
        user_query = st.session_state.sample_query
        del st.session_state.sample_query
        execute_button = True

    # Execute query
    if execute_button and user_query:
        # Validation first
        with st.spinner("Validating query..."):
            is_sql_query, response_text = validate_query(user_query, st.session_state.llm)

        if not is_sql_query:
            # Non-SQL query - show response and stop
            st.info("💬 Direct Response")
            st.markdown(response_text)

            # Store in history as non-SQL
            st.session_state.query_history.append({
                "query": user_query,
                "sql": None,
                "answer": response_text,
                "timestamp": datetime.now(),
                "is_sql": False
            })
        else:
            # Create the graph
            app = create_graph(
                st.session_state.llm,
                st.session_state.db,
                st.session_state.groq_api_key or os.environ.get("GROQ_API_KEY", ""),
            )

            # Initialize state
            initial_state = {
                "messages": [HumanMessage(content=user_query)],
                "user_query": user_query,
                "schema": st.session_state.schema,
                "retry_count": 0
            }

            # Progress tracking
            progress_container = st.container()
            with progress_container:
                progress_bar = st.progress(0)
                status_text = st.empty()

            # Results container
            results_container = st.container()

            with results_container:
                # Execution stages (collapsible)
                execution_expander = st.expander("🛠️ Execution Pipeline", expanded=True)

                with execution_expander:
                    p_col1, p_col2, p_col3, p_col4 = st.columns(4)

                    with p_col1:
                        sql_gen_status = st.empty()
                        sql_gen_status.info("⏳ Generating SQL...")

                    with p_col2:
                        sql_exec_status = st.empty()
                        sql_exec_status.info("⏳ Waiting...")

                    with p_col3:
                        format_status = st.empty()
                        format_status.info("⏳ Waiting...")

                    with p_col4:
                        eval_status = st.empty()
                        eval_status.info("⏳ Waiting...")

                # Execute workflow
                graph_config = {"configurable": {"thread_id": f"query_{datetime.now().timestamp()}"}}

                try:
                    step_count = 0
                    total_steps = 4

                    for step in app.stream(initial_state, graph_config):
                        step_count += 1
                        progress_bar.progress(min(step_count / total_steps, 1.0))

                        for key, value in step.items():
                            status_text.text(f"Processing: {key}")
                            value = value or {}

                            if key == "generate_sql":
                                with p_col1:
                                    if value.get("error"):
                                        sql_gen_status.error(f"❌ {value['error'][:50]}")
                                    else:
                                        sql_gen_status.success("✔ SQL Generated")
                                        if value.get("sql_query"):
                                            st.code(value["sql_query"], language="sql")

                            elif key == "execute_sql":
                                with p_col2:
                                    if value.get("error"):
                                        sql_exec_status.error(f"❌ {value['error'][:50]}")
                                    else:
                                        sql_exec_status.success("✔ Query Executed")
                                        if value.get("results"):
                                            st.metric("Rows", value["results"]["row_count"])

                            elif key == "format_answer":
                                with p_col3:
                                    format_status.success("✔ Answer Formatted")

                            elif key == "evaluate":
                                with p_col4:
                                    score_val = value.get("evaluation_score")
                                    if score_val is not None:
                                        eval_status.success("✔ Evaluated")
                                        st.metric("Score", f"{score_val:.2f}")
                                    else:
                                        eval_status.warning("⚠ Evaluation skipped")

                    time.sleep(0.5)  # Small delay for visual effect

                    # Get final state
                    final_state = app.get_state(graph_config).values

                    # Display results
                    st.markdown("---")

                    if final_state.get("error"):
                        st.error(f"❌ Error: {final_state['error']}")

                    elif final_state.get("answer"):
                        # Success message
                        st.success("✔ Query executed successfully!")

                        # Display answer
                        st.markdown("### 💬 Answer")
                        st.markdown(final_state["answer"])

                        # Display RAGAS Evaluation Score
                        score = final_state.get("evaluation_score")
                        if score is not None:
                            st.markdown("### 📊 Quality Assessment")

                            # Create columns for score display
                            score_cols = st.columns([2, 1, 2])

                            with score_cols[1]:
                                # Determine score color and emoji
                                if score >= 0.8:
                                    score_color = "#4CAF50"  # Green
                                    score_emoji = "🌟"
                                    score_label = "Excellent"
                                elif score >= 0.6:
                                    score_color = "#ffd666"  # Yellow
                                    score_emoji = "☀️"
                                    score_label = "Good"
                                elif score >= 0.4:
                                    score_color = "#ff9800"  # Orange
                                    score_emoji = "⚡"
                                    score_label = "Fair"
                                else:
                                    score_color = "#f44336"  # Red
                                    score_emoji = "⚠️"
                                    score_label = "Poor"

                                # Display score in a circular badge style
                                st.markdown(f"""
                                <div style='text-align: center; padding: 1rem;'>
                                    <div style='
                                        width: 150px;
                                        height: 150px;
                                        margin: 0 auto;
                                        border-radius: 50%;
                                        background: linear-gradient(135deg, {score_color}22, {score_color}44);
                                        border: 3px solid {score_color};
                                        display: flex;
                                        flex-direction: column;
                                        justify-content: center;
                                        align-items: center;
                                        box-shadow: 0 4px 15px rgba(255, 214, 102, 0.3);
                                    '>
                                        <span style='font-size: 2.5rem; font-weight: bold; color: {score_color};'>
                                            {score:.1%}
                                        </span>
                                        <span style='font-size: 1rem; color: #ffd666;'>
                                            {score_emoji} {score_label}
                                        </span>
                                    </div>
                                    <p style='color: #ffd666; margin-top: 1rem; font-size: 0.9rem;'>
                                        RAGAS Answer Accuracy Score
                                    </p>
                                </div>
                                """, unsafe_allow_html=True)

                            # Add score breakdown in expandable section
                            with st.expander("📋 Score Details", expanded=False):
                                row_count = (final_state.get("results") or {}).get("row_count", 0)
                                st.markdown(f"""
**Evaluation Metrics:**
- **Answer Accuracy**: {score:.2%}
- **Query**: {final_state.get('user_query', 'N/A')}
- **SQL Generated**: `{final_state.get('sql_query', 'N/A')}`
- **Rows Retrieved**: {row_count}

**Score Interpretation:**
- 🌟 **80-100%**: Excellent - Answer highly accurate and relevant
- ☀️ **60-79%**: Good - Answer mostly accurate with minor gaps
- ⚡ **40-59%**: Fair - Answer partially accurate, needs improvement
- ⚠️ **0-39%**: Poor - Answer lacks accuracy or relevance
                                """)

                                st.progress(max(0.0, min(float(score), 1.0)))
                                st.caption("The RAGAS evaluation compares the generated answer against the actual query results to measure accuracy and relevance.")

                        # Display results table if available
                        if final_state.get("results") and final_state["results"]["rows"]:
                            st.markdown("### 📊 Results Table")

                            results = final_state["results"]
                            df = pd.DataFrame(results["rows"], columns=results["columns"])

                            # Display controls
                            show_all = st.checkbox("Show all rows", value=False)

                            if show_all:
                                st.dataframe(df, use_container_width=True)
                            else:
                                st.dataframe(df.head(10), use_container_width=True)
                                if len(df) > 10:
                                    st.info(f"Showing 10 of {len(df)} rows. Check 'Show all rows' to see more.")

                            # Download button
                            csv = df.to_csv(index=False)
                            st.download_button(
                                label="📥 Download as CSV",
                                data=csv,
                                file_name=f"query_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                mime="text/csv"
                            )

                        # Store in history
                        st.session_state.query_history.append({
                            "query": user_query,
                            "sql": final_state.get("sql_query"),
                            "answer": final_state.get("answer"),
                            "timestamp": datetime.now(),
                            "is_sql": True
                        })

                        # Store current results
                        st.session_state.current_results = final_state

                    # Clear progress indicators
                    progress_bar.empty()
                    status_text.empty()

                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")
                    progress_bar.empty()
                    status_text.empty()

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #ffd666; background: rgba(0,0,0,0.8); padding: 1rem; border-radius: 10px;'>
    <p>Built with LangGraph, Groq, and Streamlit | Text-to-SQL Pipeline with RAGAS Evaluation</p>
    </div>
    """,
    unsafe_allow_html=True
)
