import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import psycopg2
from psycopg2.extras import execute_values
import os
import pandas as pd
import re
from rapidfuzz import fuzz, process
import threading
from queue import Queue, Empty
import time
import html  # unescape &gt; &lt; &amp;

# ---------------- GLOBALS ----------------
mapping_vars = []
PROMPT_TEMPLATES = [
    "Compare columns",
    "Find unmatched IDs",
    "Custom SQL Assistant",  # NEW third template
]

# Logging queues for thread-safe UI updates
log_queue = Queue()
ui_queue = Queue()

# Will be attached after the UI creates the Text widget
log_box = None


def enqueue_ui(fn):
    """Enqueue a callable to execute on the Tk main thread."""
    ui_queue.put(fn)


def _append_to_logbox(message):
    """Append a message directly to the log Text widget (main thread only)."""
    global log_box
    if log_box is None or not log_box.winfo_exists():
        return
    log_box.config(state="normal")
    log_box.insert(tk.END, str(message) + "\n")
    log_box.see(tk.END)
    log_box.config(state="disabled")


def pump_queues():
    """Periodically pump log messages and UI callables to the main thread."""
    global log_box
    if log_box is None or (hasattr(log_box, "winfo_exists") and not log_box.winfo_exists()):
        root.after(150, pump_queues)
        return

    # Flush logs
    try:
        while True:
            msg = log_queue.get_nowait()
            _append_to_logbox(msg)
    except Empty:
        pass

    # Flush UI commands
    try:
        while True:
            fn = ui_queue.get_nowait()
            try:
                fn()
            except Exception as e:
                _append_to_logbox(f"[UI Error] {e}")
    except Empty:
        pass

    root.after(150, pump_queues)


def log(message):
    """Thread-safe logging function—safe to call from any thread."""
    log_queue.put(str(message))


# ---------------- SMALL SPINNER UTILITY ----------------
class Spinner:
    """Animate a label with a simple braille spinner while active, safely handling destroyed widgets."""
    FRAMES = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]

    def __init__(self, label: tk.Label, root: tk.Misc, interval_ms: int = 120):
        self.label = label
        self.root = root
        self.interval = int(interval_ms)
        self.active = False
        self._i = 0
        self._after_id = None
        self._prefix = ""
        try:
            self._original_text = label.cget("text")
        except tk.TclError:
            self._original_text = ""

    def _widget_alive(self) -> bool:
        try:
            return (self.label is not None) and bool(self.label.winfo_exists())
        except tk.TclError:
            return False

    def _root_alive(self) -> bool:
        try:
            return (self.root is not None) and bool(self.root.winfo_exists())
        except tk.TclError:
            return False

    def start(self, prefix: str = ""):
        if self.active:
            return
        if not self._widget_alive() or not self._root_alive():
            return
        self._prefix = prefix or ""
        try:
            self._original_text = self.label.cget("text")
        except tk.TclError:
            self._original_text = ""
        self.active = True
        self._i = 0
        self._schedule_next()

    def update_prefix(self, prefix: str = ""):
        self._prefix = prefix or ""

    def _schedule_next(self):
        if not self.active:
            return
        if not self._widget_alive() or not self._root_alive():
            self.stop()
            return
        try:
            self._after_id = self.root.after(self.interval, self._tick)
        except tk.TclError:
            self.stop()

    def _tick(self):
        if not self.active:
            return
        if not self._widget_alive() or not self._root_alive():
            self.stop()
            return

        frame = Spinner.FRAMES[self._i % len(Spinner.FRAMES)]
        self._i += 1

        try:
            text = f"{self._prefix} {frame}".strip()
            self.label.config(text=text)
        except tk.TclError:
            self.stop()
            return

        self._schedule_next()

    def stop(self):
        if self._after_id is not None:
            try:
                self.root.after_cancel(self._after_id)
            except Exception:
                pass
            self._after_id = None

        if self._widget_alive():
            try:
                self.label.config(text=self._original_text)
            except tk.TclError:
                pass

        self.active = False


# ---------------- SECURITY & SQL NORMALIZATION ----------------
START_RE = re.compile(r"^\s*(select|with)\b", re.IGNORECASE | re.DOTALL)
BACKTICK_IDENT = re.compile(r"`([^`]+)`")


def fix_mysql_quotes(sql_text: str) -> str:
    if not sql_text:
        return sql_text
    return BACKTICK_IDENT.sub(r'"\1"', sql_text)


IDENT_STARTS_WITH_DIGIT = re.compile(r'(?<!["\w])([0-9][\w$]*)')


def quote_digit_starting_idents(sql_text: str) -> str:
    def repl(m):
        token = m.group(1)
        if re.fullmatch(r'\d+(\.\d+)?', token):
            return token
        return f'"{token}"'
    return IDENT_STARTS_WITH_DIGIT.sub(repl, sql_text)


def _wrap_with_limit(q: str, limit: int = 500) -> str:
    """
    Ensure SELECT/CTE queries do not return unbounded rows.
    - For CTE (WITH ...): DO NOT wrap; append LIMIT to the final SELECT.
    - For plain SELECT: wrap in subquery + LIMIT.
    """
    q = q.strip().rstrip(";")
    if not START_RE.match(q):
        return q + ";"
    if re.search(r"\blimit\s+\d+\b", q, flags=re.IGNORECASE):
        return q + ";"
    if q.lower().startswith("with"):
        return f"{q} LIMIT {limit};"
    return f"SELECT * FROM ({q}) AS subq LIMIT {limit};"


def run_sql(conn_params, query):
    """Run a SELECT/WITH query safely in read-only mode with a timeout and result cap."""
    if not query:
        return ["SQL Error: empty query."]

    # Unescape any HTML entities (&gt;, &lt;, &amp;, …)
    query = html.unescape(query)

    # Normalize identifiers
    query = fix_mysql_quotes(query)
    query = quote_digit_starting_idents(query)

    if not START_RE.match(query or ""):
        return [f"SQL Error: only SELECT/WITH queries are allowed."]
    try:
        with psycopg2.connect(**conn_params) as conn:
            conn.set_session(readonly=True, autocommit=False)
            with conn.cursor() as cur:
                cur.execute("SET LOCAL statement_timeout = '10s';")
                cur.execute("SET LOCAL search_path = public;")
                safe_q = _wrap_with_limit(query, limit=500)
                cur.execute(safe_q)
                rows = cur.fetchall()
                return rows
    except Exception as e:
        return [f"SQL Error: {e}"]


# ---------------- UTILITIES ----------------
def table_exists(conn_params, table_name):
    try:
        with psycopg2.connect(**conn_params) as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT EXISTS (
                        SELECT 1 
                        FROM information_schema.tables 
                        WHERE table_schema='public' 
                        AND table_name=%s
                    );
                """, (table_name,))
                return cur.fetchone()[0]
    except Exception as e:
        log(f"Error checking table existence: {e}")
        return False


def get_tables(host, db, user, pwd, port):
    try:
        with psycopg2.connect(host=host, database=db, user=user, password=pwd, port=port) as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT table_name 
                    FROM information_schema.tables 
                    WHERE table_schema='public'
                    ORDER BY table_name;
                """)
                return [row[0].strip() for row in cur.fetchall()]
    except Exception as e:
        log(f"get_tables error: {e}")
        return []


def safety_preview_popup(root, summary_text):
    """
    Shows a modal confirmation popup with a summary.
    Returns True if user clicks Proceed, False if Cancel.
    """
    win = tk.Toplevel(root)
    win.title("Safety Preview")
    win.geometry("700x500")
    win.grab_set()   # Make modal

    tk.Label(win, text="Review Changes Before Writing to DB2",
             font=("Arial", 12, "bold")).pack(pady=10)

    text_box = tk.Text(win, height=20, width=80, wrap="word")
    text_box.pack(padx=10, pady=5)
    text_box.insert(tk.END, summary_text)
    text_box.config(state="disabled")

    result = {"value": False}

    def proceed():
        result["value"] = True
        win.destroy()

    def cancel():
        result["value"] = False
        win.destroy()

    btn_frame = tk.Frame(win)
    btn_frame.pack(pady=10)

    tk.Button(btn_frame, text="Proceed",
              width=15, bg="#4CAF50", fg="white",
              command=proceed).pack(side="left", padx=10)

    tk.Button(btn_frame, text="Cancel",
              width=15, bg="#E53935", fg="white",
              command=cancel).pack(side="left", padx=10)

    win.wait_window()
    return result["value"]


# ---------- ROBUST SQL EXTRACTOR (handles comments/missing semicolons/CTEs) ----------
TARGET_DB_RE = re.compile(r'--\s*db\s*:\s*(db1|db2)\b', flags=re.IGNORECASE)

def extract_sql_queries(text: str):
    """
    Extract SQL SELECT/CTE statements robustly.
    - Prefers fenced code blocks first: ```sql ...``` or ```...```
    - Carries a preceding `-- db: DB1|DB2` line into the block if present just above it.
    - Splits on semicolons found OUTSIDE parentheses; keeps CTEs intact.
    - Unescapes HTML entities.
    - Returns each statement ending with a semicolon.
    """
    if not text:
        return []

    # Unescape HTML entities early so >, <, & are valid
    text = html.unescape(text)

    # Capture blocks with context so we can see the line just before the fence
    fence_pat = re.compile(r'(^|\n)([^\S\r\n]*)(--[^\n]*\n)?```(?:sql)?\s*(.*?)```',
                           flags=re.DOTALL | re.IGNORECASE)

    def split_outside_parens(block: str):
        stmts, buf = [], []

        def flush_stmt(s: str):
            s = s.strip()
            if not s:
                return
            s_low = s.lower()
            if s_low.startswith("select") or s_low.startswith("with"):
                if not s.endswith(";"):
                    s += ";"
                stmts.append(s)

        joined = ""
        paren = 0
        for line in block.splitlines():
            raw = line.rstrip()
            no_comment = raw.split("--", 1)[0]

            # Update parenthesis balance
            for ch in no_comment:
                if ch == "(":
                    paren += 1
                elif ch == ")":
                    paren = max(0, paren - 1)

            buf.append(raw)
            joined = "\n".join(buf)

            # Split only on semicolons at paren==0
            parts, last, p = [], 0, 0
            for i, c in enumerate(joined):
                if c == "(":
                    p += 1
                elif c == ")":
                    p = max(0, p - 1)
                elif c == ";" and p == 0:
                    parts.append(joined[last:i+1])
                    last = i + 1
            tail = joined[last:]

            if parts:
                for s in parts[:-1]:
                    flush_stmt(s)
                buf = [parts[-1] + tail]

        tail_stmt = "\n".join(buf).strip()
        if tail_stmt:
            flush_stmt(tail_stmt)

        # Deduplicate preserving order
        seen, uniq = set(), []
        for q in stmts:
            if q not in seen:
                seen.add(q)
                uniq.append(q)
        return uniq

    queries = []
    any_fences = False

    # 1) Fenced blocks, importing a preceding -- db: line if present
    for m in fence_pat.finditer(text):
        any_fences = True
        maybe_db_line = (m.group(3) or "").strip()
        block = m.group(4).strip()
        # If previous line looks like -- db: DB1/DB2 and it's NOT already in the block, prepend it
        if TARGET_DB_RE.match(maybe_db_line) and not TARGET_DB_RE.search(block):
            block = f"{maybe_db_line}\n{block}"
        queries.extend(split_outside_parens(block))

    # 2) If no fences, consider whole text
    if not any_fences:
        queries.extend(split_outside_parens(text))

    return queries


def normalize(name):
    return name.lower().replace("_", "").replace("-", "").replace(" ", "")


def get_columns(conn_params, table_name):
    try:
        with psycopg2.connect(**conn_params) as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT column_name
                    FROM information_schema.columns
                    WHERE table_schema='public'
                    AND table_name=%s
                    ORDER BY ordinal_position;
                """, (table_name,))
                return [row[0] for row in cur.fetchall()]
    except Exception as e:
        log(f"get_columns error: {e}")
        return []


def compute_matches(tables1, tables2):
    tables2 = [t.strip() for t in tables2]
    matches = []
    normalized2 = {normalize(t): t for t in tables2}
    normalized2_keys = list(normalized2.keys())

    for t1 in tables1:
        norm1 = normalize(t1)
        if not normalized2_keys:
            matches.append((t1, "", 0))
            continue
        best = process.extractOne(norm1, normalized2_keys, scorer=fuzz.ratio)
        if best is None:
            matches.append((t1, "", 0))
            continue
        best_norm, score, _ = best
        best_real = normalized2.get(best_norm, "")
        matches.append((t1, best_real, score))
    return matches




def preprocess_db2_table(conn1, conn2, t1, t2, id_col, mapping, missing_in_db2, extra_in_db2):
    """
    Preprocess DB2 table using DB1 schema as source of truth.
    Safe, logged, and includes a user confirmation preview.
    """

    import pandas as pd

    log(f"[PROCESS] Starting preprocessing: DB1='{t1}'  DB2='{t2}'")

    # ----------------------------
    # 1. Load DB1 schema
    # ----------------------------
    cols1 = get_columns(conn1, t1)
    if not cols1:
        msg = f"[ERROR] DB1 columns not found for '{t1}'"
        log(msg)
        return msg

    log(f"[SCHEMA] DB1 columns: {cols1}")

    # ----------------------------
    # 2. Load DB2 into DataFrame
    # ----------------------------
    log(f"[LOAD] Reading DB2 table '{t2}'")
    try:
        rows = run_sql(conn2, f'SELECT * FROM "{t2}"')
        if isinstance(rows, list) and rows and isinstance(rows[0], str) and rows[0].startswith("SQL Error"):
            log("[SQL ERROR] " + rows[0])
            return rows[0]
    except Exception as e:
        msg = f"[SQL ERROR] Cannot read DB2 '{t2}': {e}"
        log(msg)
        return msg

    df = pd.DataFrame(rows, columns=get_columns(conn2, t2))
    log(f"[LOAD] Loaded {len(df)} rows and {len(df.columns)} columns from DB2")

    output_log = []

    # ----------------------------
    # 3. Apply column renames (DB2 -> DB1)
    # ----------------------------
    rename_map = {}
    for pair in mapping:
        try:
            left, right = pair.split(":")
            left, right = left.strip(), right.strip()
        except:
            log(f"[WARN] Bad mapping pair: '{pair}'")
            continue

        if right in df.columns:
            rename_map[right] = left
            msg = f"Rename '{right}' → '{left}'"
            output_log.append(msg)
            log("[RENAME] " + msg)
        else:
            log(f"[SKIP] Column '{right}' not in DB2 → cannot rename")

    df = df.rename(columns=rename_map)

    # ----------------------------
    # IMPORTANT FIX:
    # Update ID column if renamed
    # ----------------------------
    if id_col in rename_map:
        new_id = rename_map[id_col]
        log(f"[ID-FIX] ID column renamed: '{id_col}' → '{new_id}'")
        id_col = new_id
    else:
        # Case-insensitive fallback
        for old, new in rename_map.items():
            if old.lower() == id_col.lower():
                log(f"[ID-FIX] ID normalized: '{id_col}' → '{new}'")
                id_col = new
                break

    # ----------------------------
    # 4. Add missing columns (DB1-only)
    # ----------------------------
    for col in missing_in_db2:
        if col not in df.columns:
            df[col] = ""
            msg = f"Added missing DB1 column '{col}'"
            output_log.append(msg)
            log("[ADD] " + msg)

    # ----------------------------
    # 5. Drop extra DB2-only columns
    # ----------------------------
    for col in extra_in_db2:
        if col in df.columns:
            df = df.drop(columns=[col])
            msg = f"Dropped extra DB2 column '{col}'"
            output_log.append(msg)
            log("[DROP] " + msg)

    # ----------------------------
    # 6. Guarantee DB1 schema completeness
    # ----------------------------
    for col in cols1:
        if col not in df.columns:
            df[col] = ""
            log(f"[FIX] Added missing column before reorder: '{col}'")

    # ----------------------------
    # 7. Reorder columns to DB1 exact order
    # ----------------------------
    log("[ORDER] Reordering columns to DB1 order")
    df = df[[c for c in cols1]]

    # ----------------------------
    # 8. Deduplicate by ID
    # ----------------------------
    log(f"[DEDUP] Removing duplicates using ID='{id_col}'")

    if id_col not in df.columns:
        msg = f"[ERROR] ID column '{id_col}' missing AFTER preprocessing!"
        log(msg)
        return msg

    before = len(df)
    df = df.drop_duplicates(subset=[id_col], keep="first")
    removed = before - len(df)

    msg = f"Removed {removed} duplicate rows"
    output_log.append(msg)
    log("[DEDUP] " + msg)

    # ----------------------------
    # 9. Build safety preview
    # ----------------------------
    final_table = f"{t1}_processed"
    summary = "\n".join([
        f"DB1 Reference Table: {t1}",
        f"DB2 Source Table: {t2}",
        "",
        "Planned Changes:",
        *output_log,
        "",
        "Final Column Order:",
        ", ".join(df.columns),
        "",
        f"Rows After Dedup: {len(df)}",
        "",
        f"Target Table: {final_table}",
    ])

    log("[PREVIEW] Showing safety preview dialog…")

    if not safety_preview_popup(root, summary):
        log("[CANCEL] User cancelled preprocessing.")
        return "Operation cancelled."

    # ----------------------------
    # 10. Write processed table
    # ----------------------------
    try:
        log(f"[SAVE] Writing processed table '{final_table}'")

        with psycopg2.connect(**conn2) as conn:
            with conn.cursor() as cur:
                cur.execute(f'DROP TABLE IF EXISTS "{final_table}"')

                col_defs = ", ".join(f'"{c}" TEXT' for c in df.columns)
                cur.execute(f'CREATE TABLE "{final_table}" ({col_defs});')

                records = df.astype(str).values.tolist()
                columns_sql = ", ".join(f'"{c}"' for c in df.columns)
                insert_sql = f'INSERT INTO "{final_table}" ({columns_sql}) VALUES %s'
                execute_values(cur, insert_sql, records)

        log(f"[SUCCESS] Saved processed table '{final_table}' with {len(df)} rows.")
        return f"Saved processed table '{final_table}'."

    except Exception as e:
        msg = f"[ERROR] Failed writing processed table: {e}"
        log(msg)
        return msg

def find_data_files(folder_path):
    data_files = []
    if not folder_path:
        return data_files
    for root_dir, dirs, files in os.walk(folder_path):
        for f in files:
            if f.lower().endswith(".xlsx") or f.lower().endswith(".csv"):
                data_files.append(os.path.join(root_dir, f))
    return data_files


# CSV reading: robust delimiter & encoding
def read_csv_safely(file_path: str) -> pd.DataFrame:
    encodings_to_try = ["utf-8", "utf-8-sig", "latin1", "cp1252", "iso-8859-1"]
    for enc in encodings_to_try:
        try:
            return pd.read_csv(file_path, sep=None, engine="python", encoding=enc, on_bad_lines="skip")
        except UnicodeDecodeError:
            continue
        except Exception:
            continue
    return pd.read_csv(file_path, sep=";", encoding="latin1", on_bad_lines="skip")


IDENT_RE = re.compile(r"[^a-zA-Z0-9_]+")


def safe_ident(s: str, max_len: int = 63) -> str:
    s = str(s).strip().lower().replace(" ", "_")
    s = IDENT_RE.sub("_", s)
    s = s.strip("_")
    return s[:max_len] or "col"


def quote_ident_strict(ident: str) -> str:
    if ident is None:
        raise ValueError("Identifier is None")
    return '"' + str(ident).replace('"', '""') + '"'


def import_file_to_db(conn_params, file_path):
    try:
        log(f"Processing file: {file_path}")
        if file_path.lower().endswith(".csv"):
            df = read_csv_safely(file_path)
            log(f"Loaded CSV with shape {df.shape}")
        else:
            try:
                df = pd.read_excel(file_path, engine="openpyxl")
            except Exception:
                df = pd.read_excel(file_path)
            log(f"Loaded Excel with shape {df.shape}")

        raw_table = os.path.splitext(os.path.basename(file_path))[0]
        table_name = safe_ident(raw_table)

        # Deduplicate & sanitize column names
        cols = [c if isinstance(c, str) else f"col_{i}" for i, c in enumerate(df.columns)]
        safe_cols = []
        seen = set()
        for c in cols:
            base = safe_ident(c)
            name = base or "col"
            k = 1
            while name in seen:
                k += 1
                name = f"{base}_{k}"
            seen.add(name)
            safe_cols.append(name)

        if table_exists(conn_params, table_name):
            msg = f"SKIPPED: Table '{table_name}' already exists. File not imported."
            log(msg)
            return msg

        with psycopg2.connect(**conn_params) as conn:
            with conn.cursor() as cur:
                col_defs = ", ".join([f'"{c}" TEXT' for c in safe_cols])
                cur.execute(f'CREATE TABLE "{table_name}" ({col_defs});')

                records = []
                for _, r in df.iterrows():
                    records.append([None if pd.isna(v) else str(v) for v in r.tolist()])

                if records:
                    columns_sql = ", ".join([f'"{c}"' for c in safe_cols])
                    insert_sql = f'INSERT INTO "{table_name}" ({columns_sql}) VALUES %s'
                    execute_values(cur, insert_sql, records, page_size=500)

        msg = f"IMPORTED: {file_path} → {table_name}"
        log(msg)
        return msg

    except Exception as e:
        msg = f"ERROR importing {file_path}: {e}"
        log(msg)
        return msg


# ---------------- ID selection + compare helpers ----------------
def find_default_id_col(columns):
    for c in columns or []:
        if isinstance(c, str) and "id" in c.lower():
            return c
    return ""


def fetch_ids_as_text(conn_params, table_name, id_col="id", cap=500_000):
    if not table_name:
        return ["SQL Error: table name is empty"]
    if not id_col:
        return ["SQL Error: id column is not selected"]

    cols = get_columns(conn_params, table_name)
    if not isinstance(cols, list) or len(cols) == 0:
        return [f"SQL Error: could not fetch columns for table '{table_name}'"]

    resolved_col = None
    if id_col in cols:
        resolved_col = id_col
    else:
        ci_matches = [c for c in cols if c.lower() == id_col.lower()]
        if len(ci_matches) == 1:
            resolved_col = ci_matches[0]

    if not resolved_col:
        return [f"SQL Error: column '{id_col}' not found in table '{table_name}'. Available columns: {cols}"]

    table_q = quote_ident_strict(table_name)
    col_q = quote_ident_strict(resolved_col)
    sql = f"SELECT DISTINCT {col_q}::text AS id FROM {table_q} ORDER BY 1 LIMIT {cap};"

    rows = run_sql(conn_params, sql)
    if isinstance(rows, list) and rows and isinstance(rows[0], str) and rows[0].startswith("SQL Error:"):
        return rows
    try:
        return {r[0] for r in rows}
    except Exception as e:
        return [f"SQL Error: {e}"]


def compare_ids_across_dbs(conn1, conn2, t1, t2, id1, id2):
    ids1 = fetch_ids_as_text(conn1, t1, id_col=id1)
    if isinstance(ids1, list) and ids1 and isinstance(ids1[0], str) and ids1[0].startswith("SQL Error:"):
        return ids1
    ids2 = fetch_ids_as_text(conn2, t2, id_col=id2)
    if isinstance(ids2, list) and ids2 and isinstance(ids2[0], str) and ids2[0].startswith("SQL Error:"):
        return ids2
    db1_only = sorted(ids1 - ids2)
    db2_only = sorted(ids2 - ids1)
    summary = {
        "db1_only": len(db1_only),
        "db2_only": len(db2_only),
        "db1_total": len(ids1),
        "db2_total": len(ids2),
    }
    return db1_only, db2_only, summary


# ---------- DB TARGET DETECTION FOR SQL QUERIES ----------
# Allows LLM to specify which DB to run with a simple comment: -- db: DB1 or -- db: DB2
def choose_conn_for_query(q: str, conn1, conn2, t1_local: str, t2_local: str):
    m = TARGET_DB_RE.search(q or "")
    if m:
        db = m.group(1).lower()
        return (conn1 if db == "db1" else conn2), db.upper()
    ql = (q or "").lower()
    if t1_local and t1_local.lower() in ql:
        return conn1, "DB1"
    if t2_local and t2_local.lower() in ql:
        return conn2, "DB2"
    return conn2, "DB2"


# ---------------- MAPPING UI ----------------
from llm_engine import run_llm, list_llms

def render_mapping_ui(matches, tables2, conn1, conn2):
    map_win = tk.Toplevel(root)
    map_win.title("Table Mapping (DB1 → DB2)")
    map_win.geometry("1700x950")

    local_spinners = []

    # Scrollable canvas
    canvas = tk.Canvas(map_win)
    canvas.pack(side="left", fill="both", expand=True)
    scrollbar = tk.Scrollbar(map_win, orient="vertical", command=canvas.yview)
    scrollbar.pack(side="right", fill="y")
    canvas.configure(yscrollcommand=scrollbar.set)

    inner_frame = tk.Frame(canvas)
    canvas.create_window((0, 0), window=inner_frame, anchor="nw")

    def on_configure(event):
        canvas.configure(scrollregion=canvas.bbox("all"))

    inner_frame.bind("<Configure>", on_configure)

    # ---------------- GLOBAL TEMPLATE CONTROLS ----------------
    global_template_var = tk.StringVar()
    global_template_box = ttk.Combobox(
        inner_frame,
        textvariable=global_template_var,
        values=PROMPT_TEMPLATES,
        state="readonly",
        width=25
    )
    global_template_box.grid(row=1, column=6, padx=10, pady=10, sticky="e")

    apply_all_button = tk.Button(inner_frame, text="Copy Template to All")
    apply_all_button.grid(row=1, column=7, padx=10, pady=10, sticky="e")

    run_all_button = tk.Button(inner_frame, text="Run All")
    run_all_button.grid(row=1, column=8, padx=10, pady=10, sticky="e")

    # NEW: LLM selection dropdown
    llm_var = tk.StringVar(value=list_llms()[0])  # default first model
    llm_box = ttk.Combobox(
        inner_frame,
        textvariable=llm_var,
        values=list_llms(),
        state="readonly",
        width=20
    )
    llm_box.grid(row=1, column=9, padx=10, pady=10, sticky="e")
    tk.Label(inner_frame, text="Select LLM").grid(row=1, column=10, padx=10, pady=10, sticky="e")

    # --- NEW: helper to call LLM with selected model ---
    def call_llm_with_selected_model(prompt: str):
        """
        Call llm_engine.run_llm using the model selected in the dropdown.
        Tries common keyword variants if the engine signature differs.
        Falls back to run_llm(prompt) if no kwargs are supported.
        """
        try:
            from llm_engine import run_llm as _run_llm
        except Exception as e:
            raise RuntimeError(f"LLM engine not available: {e}")

        selected = (llm_var.get() or "").strip()

        # Try common keyword variants
        for kw in ("model", "engine", "model_name", "llm"):
            try:
                if selected:
                    return _run_llm(prompt, **{kw: selected})
            except TypeError:
                # Signature didn't accept this kw; try next one
                continue
            except Exception:
                # If the engine raised something else, surface it
                raise

        # Fallback: original signature without any model kw
        return _run_llm(prompt)

    def on_process_data(pb, rb, t1_local, t2_local, id2_local):
        """
        LLM column comparison → safe parsing → column mapping summary →
        preprocessing with safety preview → final DB write.
        """

        def worker():
            try:
                rb.delete("1.0", tk.END)

                if not t2_local:
                    enqueue_ui(lambda: rb.insert(tk.END, "[ERROR] No DB2 table selected.\n"))
                    return

                if not id2_local:
                    enqueue_ui(lambda: rb.insert(tk.END, "[ERROR] No DB2 ID column selected.\n"))
                    return

                # ------------------------------------------------
                # 1. Build strict LLM prompt
                # ------------------------------------------------
                cols1 = get_columns(conn1, t1_local)
                cols2 = get_columns(conn2, t2_local)

                prompt = (
                    f"Compare the columns of DB1 table '{t1_local}' and DB2 table '{t2_local}'.\n\n"
                    f"DB1 columns:\n{cols1}\n\n"
                    f"DB2 columns:\n{cols2}\n\n"
                    "Return EXACTLY these 3 lines and NOTHING ELSE:\n"
                    "missing_in_db2: [...]\n"
                    "missing_in_db1: [...]\n"
                    "possible_mappings: ['DB1col:DB2col', ...]\n"
                    "If names match ignoring case, include a mapping pair."
                )

                log(f"[LLM] Running Compare Columns for {t1_local} → {t2_local}")

                llm_out = call_llm_with_selected_model(prompt)

                enqueue_ui(lambda: rb.insert(tk.END, "LLM raw output:\n" + llm_out + "\n\n"))

                # ------------------------------------------------
                # 2. Parse LLM output robustly
                # ------------------------------------------------
                lines = [l.strip() for l in llm_out.splitlines() if l.strip()]
                parsed = {"missing_in_db2": [], "missing_in_db1": [], "possible_mappings": []}

                for line in lines:
                    if line.startswith("missing_in_db2"):
                        try:
                            parsed["missing_in_db2"] = eval(line.split(":", 1)[1].strip())
                        except:
                            pass
                    elif line.startswith("missing_in_db1"):
                        try:
                            parsed["missing_in_db1"] = eval(line.split(":", 1)[1].strip())
                        except:
                            pass
                    elif line.startswith("possible_mappings"):
                        try:
                            parsed["possible_mappings"] = eval(line.split(":", 1)[1].strip())
                        except:
                            pass

                missing_in_db2 = parsed["missing_in_db2"]
                extra_in_db2 = parsed["missing_in_db1"]
                possible_mappings = parsed["possible_mappings"]

                # ------------------------------------------------
                # 3. Auto‑mapping fallback
                # ------------------------------------------------
                if not possible_mappings or any(":" not in p for p in possible_mappings):
                    log("[WARN] Invalid LLM mappings → generating automatic fallback mappings.")

                    possible_mappings = []
                    for c1 in cols1:
                        for c2 in cols2:
                            if c1.lower() == c2.lower():
                                possible_mappings.append(f"{c1}:{c2}")

                    enqueue_ui(lambda: rb.insert(tk.END,
                                                 "[WARNING] Using fallback auto‑mappings:\n" +
                                                 str(possible_mappings) + "\n\n"
                                                 ))

                # ------------------------------------------------
                # 4. Display column matching summary
                # ------------------------------------------------
                summary = []
                summary.append("===== COLUMN MATCHING SUMMARY =====\n")
                summary.append("Missing in DB2 (add):")
                summary.append(str(missing_in_db2))
                summary.append("\nExtra in DB2 (drop):")
                summary.append(str(extra_in_db2))
                summary.append("\nMappings (DB2 → DB1):")
                for pair in possible_mappings:
                    left, right = pair.split(":")
                    summary.append(f"{right.strip()} → {left.strip()}")
                summary.append("\n===================================\n\n")

                enqueue_ui(lambda: rb.insert(tk.END, "\n".join(summary)))

                # ------------------------------------------------
                # 5. Run preprocessing pipeline
                # ------------------------------------------------
                log(f"[PROCESS] Running preprocessing for table '{t2_local}'")

                output = preprocess_db2_table(
                    conn1, conn2,
                    t1=t1_local,
                    t2=t2_local,
                    id_col=id2_local,
                    mapping=possible_mappings,
                    missing_in_db2=missing_in_db2,
                    extra_in_db2=extra_in_db2
                )

                enqueue_ui(lambda: rb.insert(tk.END, "\nProcessing Result:\n" + str(output)))

            except Exception as e:
                log(f"[ERROR] on_process_data exception: {e}")
                enqueue_ui(lambda: rb.insert(tk.END, f"[ERROR] {e}\n"))

        # Run non-blocking
        threading.Thread(target=worker, daemon=True).start()

    # ---------------- HEADER ----------------
    headers = ["DB1 Table", "DB2 Table", "Accuracy", "Template", "DB1 ID", "DB2 ID"]
    for col, text in enumerate(headers):
        tk.Label(inner_frame, text=text, font=("Arial", 10, "bold")).grid(
            row=2, column=col, padx=10, pady=5, sticky="w"
        )

    row_widgets = []
    row = 3

    # ---------------- ROWS ----------------
    for (t1, best, score) in matches:
        clean_best = (best or "").strip()

        tk.Label(inner_frame, text=t1, width=30, anchor="w").grid(
            row=row, column=0, padx=10, pady=5, sticky="w"
        )

        options = [t.strip() for t in tables2]
        if clean_best and clean_best not in options:
            options.append(clean_best)

        db2_var = tk.StringVar(value=clean_best)
        mapping_vars.append(db2_var)

        db2_dropdown = ttk.Combobox(
            inner_frame, textvariable=db2_var, values=options,
            state="readonly", width=30
        )
        db2_dropdown.grid(row=row, column=1, padx=10, sticky="w")

        tk.Label(inner_frame, text=f"{float(score):.1f}%", width=10).grid(
            row=row, column=2, padx=10, sticky="w"
        )

        template_var = tk.StringVar()
        template_box = ttk.Combobox(
            inner_frame, textvariable=template_var,
            values=PROMPT_TEMPLATES, state="readonly", width=25
        )
        template_box.grid(row=row, column=3, padx=10, sticky="w")

        cols1 = get_columns(conn1, t1)
        cols2 = get_columns(conn2, clean_best) if clean_best else []

        default_id1 = find_default_id_col(cols1)
        default_id2 = find_default_id_col(cols2)

        id1_var = tk.StringVar(value=default_id1)
        id2_var = tk.StringVar(value=default_id2)

        id1_box = ttk.Combobox(inner_frame, textvariable=id1_var, values=cols1 or [""], state="readonly", width=20)
        id1_box.grid(row=row, column=4, padx=10, sticky="w")
        id2_box = ttk.Combobox(inner_frame, textvariable=id2_var, values=cols2 or [""], state="readonly", width=20)
        id2_box.grid(row=row, column=5, padx=10, sticky="w")

        # Prompt + result + run button + status
        prompt_box = tk.Text(inner_frame, height=6, width=70, wrap="word")
        prompt_box.grid(row=row + 1, column=0, padx=10, pady=2, sticky="nw")

        result_box = tk.Text(inner_frame, height=20, width=100, wrap="word")
        result_box.grid(row=row + 1, column=1, padx=10, pady=2, sticky="nw")

        time_label = tk.Label(inner_frame, text="Time: 0.00s")
        time_label.grid(row=row + 1, column=3, padx=10, sticky="nw")

        status_label = tk.Label(inner_frame, text="", width=10, anchor="w")
        status_label.grid(row=row + 1, column=3, padx=(110, 10), sticky="nw")
        row_spinner = Spinner(status_label, root, interval_ms=120)
        local_spinners.append(row_spinner)

        # ---- PROMPT HELPERS ----
        def crisp_sql_helper(db1_table, db2_table, db1_cols, db2_cols, id1, id2):
            """
            Prompt scaffold shown when the user picks Custom SQL Assistant.
            Strongly nudges the LLM to return only executable SQL blocks with the DB directive inside the fence.
            """
            return (
                "Task: Generate runnable SQL for the question at the end.\n\n"
                "Hard rules (must follow):\n"
                "1) Output ONLY SQL inside one or more ```sql fenced blocks. No prose anywhere.\n"
                "2) The FIRST line INSIDE each block MUST be the target DB comment, exactly:\n"
                "   -- db: DB1   or   -- db: DB2\n"
                "3) SQL must be read-only (SELECT / WITH only). No DDL/DML.\n"
                "4) Each block MUST be fully self-contained (include any CTEs it needs).\n"
                "5) Always include a LIMIT if the query can return many rows.\n"
                "6) Use real operators (>, <, >=, <=, =) — do not use HTML entities like &gt; or &lt;.\n"
                "7) Quote identifiers that start with digits using double quotes, e.g. \"20241022niderspannungstrecken\".\n"
                "8) Use PostgreSQL syntax.\n"
                "9) If the question asks for both DBs, return two separate ```sql blocks (one per DB), each with its own -- db: line.\n\n"
                "Context:\n"
                f"- DB1 table: {db1_table}\n"
                f"- DB2 table: {db2_table}\n"
                f"- DB1 columns: {db1_cols}\n"
                f"- DB2 columns: {db2_cols}\n"
                f"- Likely ID DB1: {id1}\n"
                f"- Likely ID DB2: {id2}\n\n"
                "Now answer the question, returning only the SQL blocks:\n"
                "(Write your analysis question here.)"
            )

        def build_custom_sql_prompt(db1_table, db2_table, conn1_local, conn2_local, user_question):
            """
            Auto-wrap a plain question into the crisp helper with row-specific context
            so the model returns executable SQL blocks even when the user types only a question.
            """
            cols1_local = get_columns(conn1_local, db1_table)
            cols2_local = get_columns(conn2_local, db2_table) if db2_table else []
            default_id1_local = find_default_id_col(cols1_local)
            default_id2_local = find_default_id_col(cols2_local)

            return (
                "Task: Generate runnable SQL for the question at the end.\n\n"
                "Hard rules (must follow):\n"
                "1) Output ONLY SQL inside one or more ```sql fenced blocks. No prose anywhere.\n"
                "2) The FIRST line INSIDE each block MUST be the target DB comment, exactly:\n"
                "   -- db: DB1   or   -- db: DB2\n"
                "3) SQL must be read-only (SELECT / WITH only). No DDL/DML.\n"
                "4) Each block MUST be fully self-contained (include any CTEs it needs).\n"
                "5) Always include a LIMIT if the query can return many rows.\n"
                "6) Use real operators (>, <, >=, <=, =) — do not use HTML entities like &gt; or &lt;.\n"
                "7) Quote identifiers that start with digits using double quotes, e.g. \"20241022niderspannungstrecken\".\n"
                "8) Use PostgreSQL syntax.\n"
                "9) If the question asks for both DBs, return two separate ```sql blocks (one per DB), each with its own -- db: line.\n\n"
                "Context:\n"
                f"- DB1 table: {db1_table}\n"
                f"- DB2 table: {db2_table or '(not selected)'}\n"
                f"- DB1 columns: {cols1_local}\n"
                f"- DB2 columns: {cols2_local}\n"
                f"- Likely ID DB1: {default_id1_local or '(none)'}\n"
                f"- Likely ID DB2: {default_id2_local or '(none)'}\n\n"
                "Now answer the question, returning only the SQL blocks:\n"
                f"{user_question or '(Write your analysis question here.)'}"
            )

        # ---- TEMPLATE SELECTION ----
        def on_template_selected(event, t1_local=t1, t2_var=db2_var, pb=prompt_box):
            template_name = event.widget.get()
            t2_value = t2_var.get()
            if template_name == "Compare columns":
                cols1_local = get_columns(conn1, t1_local)
                cols2_local = get_columns(conn2, t2_value) if t2_value else []
                prompt = (
                    f"Compare the columns of DB1 table '{t1_local}' and DB2 table '{t2_value}'.\n\n"
                    f"DB1 columns:\n{cols1_local}\n\n"
                    f"DB2 columns:\n{cols2_local}\n\n"
                    "Return ONLY these 3 lines in this exact order and format, no prose, no markdown:\n"
                    "missing_in_db2: [comma-separated DB1 column names that are not in DB2]\n"
                    "missing_in_db1: [comma-separated DB2 column names that are not in DB1]\n"
                    "possible_mappings: [pairs like DB1Col:DB2Col, where names are similar]\n"
                    "Output MUST be exactly 3 lines. If a list is empty, write []."
                )
            elif template_name == "Find unmatched IDs":
                prompt = (
                    f"(This action compares IDs across DB1 '{t1_local}' and DB2 '{t2_value}' in Python—no cross-DB SQL.)\n"
                    f"Select the correct ID columns above (DB1 ID / DB2 ID), then click Run."
                )
            elif template_name == "Custom SQL Assistant":
                cols1_local = get_columns(conn1, t1_local)
                cols2_local = get_columns(conn2, t2_value) if t2_value else []
                default_id1_local = find_default_id_col(cols1_local)
                default_id2_local = find_default_id_col(cols2_local)
                prompt = crisp_sql_helper(
                    db1_table=t1_local,
                    db2_table=t2_value or "(not selected)",
                    db1_cols=cols1_local,
                    db2_cols=cols2_local,
                    id1=default_id1_local or "(none)",
                    id2=default_id2_local or "(none)",
                )
            else:
                prompt = ""
            pb.delete("1.0", tk.END)
            pb.insert(tk.END, prompt)

        template_box.bind("<<ComboboxSelected>>", on_template_selected)

        # ---------------- RUNNERS ----------------
        def run_find_unmatched_ids(pb, rb, tl, t1_local, t2_local, spinner: Spinner, id1_sel, id2_sel, conn1, conn2):
            def worker():
                try:
                    enqueue_ui(lambda: spinner.start(prefix="Running"))
                    id1_name = id1_sel.get().strip()
                    id2_name = id2_sel.get().strip()
                    if not id1_name or not id2_name:
                        enqueue_ui(lambda: (
                            rb.delete("1.0", tk.END),
                            rb.insert(tk.END, "Error: Please select ID columns for both DB1 and DB2 before running 'Find unmatched IDs'."),
                            spinner.stop()
                        ))
                        return
                    start = time.time()
                    result = compare_ids_across_dbs(conn1, conn2, t1_local, t2_local, id1_name, id2_name)
                    end = time.time()
                    if isinstance(result, list) and result and isinstance(result[0], str) and result[0].startswith("SQL Error:"):
                        text = result[0]
                    else:
                        db1_only, db2_only, summary = result
                        CLIP = 1000
                        parts = []
                        parts.append(f"IDs in DB1 ({t1_local}.{id1_name}) not in DB2 ({t2_local}.{id2_name}):")
                        parts.extend(db1_only[:CLIP])
                        if len(db1_only) > CLIP:
                            parts.append(f"... (truncated, total {len(db1_only)})")
                        parts.append("")
                        parts.append(f"IDs in DB2 ({t2_local}.{id2_name}) not in DB1 ({t1_local}.{id1_name}):")
                        parts.extend(db2_only[:CLIP])
                        if len(db2_only) > CLIP:
                            parts.append(f"... (truncated, total {len(db2_only)})")
                        parts.append("")
                        parts.append("Summary:")
                        parts.append(str(summary))
                        text = "\n".join(str(x) for x in parts)
                    elapsed = end - start
                    enqueue_ui(lambda rb=rb, tl=tl, s=spinner, t=text, el=elapsed: (
                        rb.delete("1.0", tk.END),
                        rb.insert(tk.END, t),
                        tl.config(text=f"Time: {el:.2f}s"),
                        s.stop()
                    ))
                except Exception as e:
                    enqueue_ui(lambda rb=rb, s=spinner: (
                        rb.delete("1.0", tk.END),
                        rb.insert(tk.END, f"Error: {e}"),
                        s.stop()
                    ))
            threading.Thread(target=worker, daemon=True).start()

        def run_llm_for_row_async(pb, rb, tl, t1_local, t2_local, spinner: Spinner, template_name: str):
            def worker():
                try:
                    enqueue_ui(lambda: spinner.start(prefix=f"Running ({(llm_var.get() or '').strip()})"))
                    # Lazy import to allow script to run without llm_engine during dev
                    try:
                        from llm_engine import run_llm
                    except Exception as e:
                        output = f"LLM engine not available: {e}"
                        enqueue_ui(lambda rb=rb, tl=tl, s=spinner, out=output: (
                            rb.delete("1.0", tk.END),
                            rb.insert(tk.END, out),
                            tl.config(text="Time: 0.00s"),
                            s.stop()
                        ))
                        return

                    raw = pb.get("1.0", tk.END).strip()
                    if not raw:
                        enqueue_ui(lambda rb=rb, s=spinner: (
                            rb.delete("1.0", tk.END),
                            rb.insert(tk.END, "Please enter or select a prompt.\n"),
                            s.stop()
                        ))
                        return

                    # If Custom SQL Assistant & no code fence, auto-wrap the user question with the crisp helper
                    if template_name == "Custom SQL Assistant" and "```" not in raw:
                        prompt = build_custom_sql_prompt(
                            db1_table=t1_local,
                            db2_table=t2_local,
                            conn1_local=conn1,
                            conn2_local=conn2,
                            user_question=raw
                        )
                    else:
                        prompt = raw

                    start = time.time()
                    response = call_llm_with_selected_model(prompt)
                    end = time.time()

                    queries = extract_sql_queries(response)
                    output = ""

                    if queries:
                        output += "Detected SQL queries:\n\n"
                        for idx, q in enumerate(queries, start=1):
                            conn_for_q, chosen_db = choose_conn_for_query(q, conn1, conn2, t1_local, t2_local)
                            rows = run_sql(conn_for_q, q)
                            output += (
                                f"--- Query {idx} (target {chosen_db}) ---\n{q}\n\n"
                                f"--- Result {idx} ---\n{rows}\n\n"
                            )
                    else:
                        output = response

                    elapsed = end - start
                    enqueue_ui(lambda rb=rb, tl=tl, s=spinner, out=output, el=elapsed: (
                        rb.delete("1.0", tk.END),
                        rb.insert(tk.END, out),
                        tl.config(text=f"Time: {el:.2f}s"),
                        s.stop()
                    ))
                except Exception as e:
                    enqueue_ui(lambda rb=rb, s=spinner: (
                        rb.delete("1.0", tk.END),
                        rb.insert(tk.END, f"Error: {e}"),
                        s.stop()
                    ))

            threading.Thread(target=worker, daemon=True).start()

        def on_run_click(pb=prompt_box, rb=result_box, tl=time_label,
                         t1_local=t1, t2_var=db2_var, sp=row_spinner,
                         tmpl_var=template_var, id1v=id1_var, id2v=id2_var):
            template_name = tmpl_var.get()
            t2_value = t2_var.get()
            if template_name == "Find unmatched IDs":
                run_find_unmatched_ids(pb, rb, tl, t1_local, t2_value, sp, id1v, id2v, conn1, conn2)
            else:
                run_llm_for_row_async(pb, rb, tl, t1_local, t2_value, sp, template_name)

        run_button = tk.Button(inner_frame, text="Run", command=on_run_click)
        run_button.grid(row=row + 1, column=2, padx=10, sticky="nw")

        process_button = tk.Button(inner_frame, text="Process Data",
                                   command=lambda pb=prompt_box, rb=result_box, t1=t1,
                                                  t2=db2_var, id1=id1_var, id2=id2_var:
                                   on_process_data(pb, rb, t1, t2.get(), id2.get()))
        process_button.grid(row=row + 1, column=4, padx=10, sticky="nw")

        row_widgets.append({
            "t1": t1,
            "t2_var": db2_var,
            "prompt_box": prompt_box,
            "result_box": result_box,
            "time_label": time_label,
            "spinner": row_spinner,
            "id1_var": id1_var,
            "id2_var": id2_var,
            "template_var": template_var,
        })

        row += 2

    # ---------------- APPLY TEMPLATE TO ALL ----------------
    def apply_template_to_all():
        template_name = global_template_var.get()
        if not template_name:
            messagebox.showwarning("No Template", "Please select a template first.")
            return

        for row in row_widgets:
            t1_local = row["t1"]
            t2_local = row["t2_var"].get()
            pb = row["prompt_box"]

            if template_name == "Compare columns":
                cols1 = get_columns(conn1, t1_local)
                cols2 = get_columns(conn2, t2_local) if t2_local else []
                prompt = (
                    f"Compare the columns of DB1 table '{t1_local}' and DB2 table '{t2_local}'.\n\n"
                    f"DB1 columns:\n{cols1}\n\n"
                    f"DB2 columns:\n{cols2}\n\n"
                    "Return ONLY these 3 lines in this exact order and format, no prose, no markdown:\n"
                    "missing_in_db2: [comma-separated DB1 column names that are not in DB2]\n"
                    "missing_in_db1: [comma-separated DB2 column names that are not in DB1]\n"
                    "possible_mappings: [pairs like DB1Col:DB2Col, where names are similar]\n"
                    "Output MUST be exactly 3 lines. If a list is empty, write []."
                )

            elif template_name == "Find unmatched IDs":
                prompt = (
                    f"(This action compares IDs across DB1 '{t1_local}' and DB2 '{t2_local}' in Python—no cross-DB SQL.)\n"
                    f"Select the correct ID columns above (DB1 ID / DB2 ID), then click Run."
                )

            elif template_name == "Custom SQL Assistant":
                cols1 = get_columns(conn1, t1_local)
                cols2 = get_columns(conn2, t2_local) if t2_local else []
                default_id1 = find_default_id_col(cols1)
                default_id2 = find_default_id_col(cols2)
                prompt = (
                    "Task: Generate runnable SQL for the question at the end.\n\n"
                    "Hard rules (must follow):\n"
                    "1) Output ONLY SQL inside one or more ```sql fenced blocks. No prose anywhere.\n"
                    "2) The FIRST line INSIDE each block MUST be the target DB comment, exactly:\n"
                    "   -- db: DB1   or   -- db: DB2\n"
                    "3) SQL must be read-only (SELECT / WITH only). No DDL/DML.\n"
                    "4) Each block MUST be fully self-contained (include any CTEs it needs).\n"
                    "5) Always include a LIMIT if the query can return many rows.\n"
                    "6) Use real operators (>, <, >=, <=, =) — do not use HTML entities like &gt; or &lt;.\n"
                    "7) Quote identifiers that start with digits using double quotes, e.g. \"20241022niderspannungstrecken\".\n"
                    "8) Use PostgreSQL syntax.\n"
                    "9) If the question asks for both DBs, return two separate ```sql blocks (one per DB), each with its own -- db: line.\n\n"
                    "Context:\n"
                    f"- DB1 table: {t1_local}\n"
                    f"- DB2 table: {t2_local or '(not selected)'}\n"
                    f"- DB1 columns: {cols1}\n"
                    f"- DB2 columns: {cols2}\n"
                    f"- Likely ID DB1: {default_id1 or '(none)'}\n"
                    f"- Likely ID DB2: {default_id2 or '(none)'}\n\n"
                    "Now answer the question, returning only the SQL blocks:\n"
                    "(Write your analysis question here.)"
                )
            else:
                prompt = ""

            pb.delete("1.0", tk.END)
            pb.insert(tk.END, prompt)

    apply_all_button.config(command=apply_template_to_all)

    # ---------------- RUN ALL (threaded with spinners) ----------------
    def run_all_llms_async():
        def worker():
            try:
                try:
                    from llm_engine import run_llm
                    llm_available = True
                except Exception as e:
                    llm_available = False
                    first_row = row_widgets[0] if row_widgets else None
                    if first_row:
                        rb = first_row["result_box"]
                        enqueue_ui(lambda rb=rb, msg=e: (
                            rb.delete("1.0", tk.END),
                            rb.insert(tk.END, f"LLM engine not available: {msg}\n")
                        ))

                for roww in row_widgets:
                    pb = roww["prompt_box"]
                    rb = roww["result_box"]
                    tl = roww["time_label"]
                    sp = roww.get("spinner")
                    t1_local = roww["t1"]
                    t2_local = roww["t2_var"].get()
                    tmpl_var = roww.get("template_var")
                    id1v = roww.get("id1_var")
                    id2v = roww.get("id2_var")

                    template_name = (tmpl_var.get() if tmpl_var else "") or global_template_var.get() or ""
                    raw_prompt = pb.get("1.0", tk.END).strip()

                    if not template_name and not raw_prompt:
                        enqueue_ui(lambda rb=rb: (
                            rb.delete("1.0", tk.END),
                            rb.insert(tk.END, "Please select a template or enter a prompt for this row.\n")
                        ))
                        if sp:
                            enqueue_ui(lambda s=sp: s.stop())
                        continue

                    if sp:
                        enqueue_ui(lambda s=sp: s.start(prefix="Running"))

                    if template_name == "Find unmatched IDs":
                        id1 = id1v.get().strip() if id1v else ""
                        id2 = id2v.get().strip() if id2v else ""
                        if not id1 or not id2:
                            enqueue_ui(lambda rb=rb, s=sp: (
                                rb.delete("1.0", tk.END),
                                rb.insert(tk.END, "Error: Please select ID columns for both DB1 and DB2 before running 'Find unmatched IDs'."),
                                s.stop() if s else None
                            ))
                            continue

                        start = time.time()
                        result = compare_ids_across_dbs(conn1, conn2, t1_local, t2_local, id1, id2)
                        end = time.time()

                        if isinstance(result, list) and result and isinstance(result[0], str) and result[0].startswith("SQL Error:"):
                            out_text = result[0]
                        else:
                            db1_only, db2_only, summary = result
                            CLIP = 1000
                            parts = []
                            parts.append(f"IDs in DB1 ({t1_local}.{id1}) not in DB2 ({t2_local}.{id2}):")
                            parts.extend(db1_only[:CLIP])
                            if len(db1_only) > CLIP:
                                parts.append(f"... (truncated, total {len(db1_only)})")
                            parts.append("")
                            parts.append(f"IDs in DB2 ({t2_local}.{id2}) not in DB1 ({t1_local}.{id1}):")
                            parts.extend(db2_only[:CLIP])
                            if len(db2_only) > CLIP:
                                parts.append(f"... (truncated, total {len(db2_only)})")
                            parts.append("")
                            parts.append("Summary:")
                            parts.append(str(summary))
                            out_text = "\n".join(str(x) for x in parts)

                        elapsed = end - start
                        enqueue_ui(lambda rb=rb, tl=tl, s=sp, out=out_text, el=elapsed: (
                            rb.delete("1.0", tk.END),
                            rb.insert(tk.END, out),
                            tl.config(text=f"Time: {el:.2f}s"),
                            s.stop() if s else None
                        ))
                        continue

                    if not llm_available:
                        enqueue_ui(lambda rb=rb, s=sp: (
                            rb.delete("1.0", tk.END),
                            rb.insert(tk.END, "LLM engine not available.\n"),
                            s.stop() if s else None
                        ))
                        continue

                    # Auto-wrap question if needed for Custom SQL Assistant
                    if template_name == "Custom SQL Assistant" and "```" not in raw_prompt:
                        prompt = build_custom_sql_prompt(
                            db1_table=t1_local,
                            db2_table=t2_local,
                            conn1_local=conn1,
                            conn2_local=conn2,
                            user_question=raw_prompt
                        )
                    else:
                        prompt = raw_prompt

                    try:
                        from llm_engine import run_llm
                    except Exception as e:
                        enqueue_ui(lambda rb=rb, s=sp, msg=e: (
                            rb.delete("1.0", tk.END),
                            rb.insert(tk.END, f"LLM engine not available: {msg}\n"),
                            s.stop() if s else None
                        ))
                        continue

                    start = time.time()
                    response = call_llm_with_selected_model(prompt)
                    end = time.time()

                    queries = extract_sql_queries(response)
                    output = ""

                    if queries:
                        output += "Detected SQL queries:\n\n"
                        for idx, q in enumerate(queries, start=1):
                            conn_for_q, chosen_db = choose_conn_for_query(q, conn1, conn2, t1_local, t2_local)
                            rows = run_sql(conn_for_q, q)
                            output += (
                                f"--- Query {idx} (target {chosen_db}) ---\n{q}\n\n"
                                f"--- Result {idx} ---\n{rows}\n\n"
                            )
                    else:
                        output = response

                    elapsed = end - start
                    enqueue_ui(lambda rb=rb, tl=tl, out=output, el=elapsed, s=sp: (
                        rb.delete("1.0", tk.END),
                        rb.insert(tk.END, out),
                        tl.config(text=f"Time: {el:.2f}s"),
                        s.stop() if s else None
                    ))
            except Exception as e:
                log(f"Run All Error: {e}")

        threading.Thread(target=worker, daemon=True).start()

    run_all_button.config(command=run_all_llms_async)

    def _on_close():
        for sp in local_spinners:
            try:
                sp.stop()
            except Exception:
                pass
        try:
            map_win.destroy()
        except Exception:
            pass

    map_win.protocol("WM_DELETE_WINDOW", _on_close)


# ---------------- FOLDER SELECTION ----------------
def browse_folder(entry_widget):
    folder_selected = filedialog.askdirectory()
    if folder_selected:
        entry_widget.delete(0, tk.END)
        entry_widget.insert(0, folder_selected)


# ---------------- SUBMIT (threaded, with spinner) ----------------
def submit():
    submit_btn.config(state="disabled")
    submit_status.config(text="Working")
    global_submit_spinner.start(prefix="Working")
    log("----- Starting Import Process -----")

    # DB1 fields
    h1 = host1.get()
    d1 = db1.get()
    u1 = user1.get()
    p1 = pwd1.get()
    pt1 = port1.get()

    # DB2 fields
    h2 = host2.get()
    d2 = db2.get()
    u2 = user2.get()
    p2 = pwd2.get()
    pt2 = port2.get()

    # Folders
    f1 = folder1.get()
    f2 = folder2.get()

    log(f"DB1 folder: {f1}")
    log(f"DB2 folder: {f2}")

    conn1 = {"host": h1, "database": d1, "user": u1, "password": p1, "port": pt1}
    conn2 = {"host": h2, "database": d2, "user": u2, "password": p2, "port": pt2}

    def worker():
        try:
            files1 = find_data_files(f1)
            log(f"Found {len(files1)} files for DB1")
            for file in files1:
                result = import_file_to_db(conn1, file)
                log(result)

            files2 = find_data_files(f2)
            log(f"Found {len(files2)} files for DB2")
            for file in files2:
                result = import_file_to_db(conn2, file)
                log(result)

            log("Refreshing table lists…")
            tables1 = get_tables(h1, d1, u1, p1, pt1)
            tables2 = get_tables(h2, d2, u2, p2, pt2)
            matches = compute_matches(tables1, tables2)

            enqueue_ui(lambda: render_mapping_ui(matches, tables2, conn1, conn2))

            log(f"DB1 tables: {tables1}")
            log(f"DB2 tables: {tables2}")
            log("----- Import Completed -----")
        finally:
            def _done():
                submit_btn.config(state="normal")
                global_submit_spinner.stop()
                submit_status.config(text="Done")
            enqueue_ui(_done)

    threading.Thread(target=worker, daemon=True).start()


# ---------------- GUI ----------------
root = tk.Tk()
root.title("PostgreSQL Table Viewer + Excel Importer")
root.geometry("1200x850")

title = tk.Label(root, text="Compare PostgreSQL Databases + Import Excel Files", font=("Arial", 16, "bold"))
title.pack(pady=10)

frame = tk.Frame(root)
frame.pack(pady=10)

# ---------- DB1 ----------
db1_frame = tk.LabelFrame(frame, text="Database 1", padx=10, pady=10)
db1_frame.grid(row=0, column=0, padx=20)

tk.Label(db1_frame, text="Host").grid(row=0, column=0, sticky="w")
host1 = tk.Entry(db1_frame)
host1.insert(0, "localhost")
host1.grid(row=0, column=1)

tk.Label(db1_frame, text="Database").grid(row=1, column=0, sticky="w")
db1 = tk.Entry(db1_frame)
db1.insert(0, "db1")
db1.grid(row=1, column=1)

tk.Label(db1_frame, text="User").grid(row=2, column=0, sticky="w")
user1 = tk.Entry(db1_frame)
user1.insert(0, "postgres")
user1.grid(row=2, column=1)

tk.Label(db1_frame, text="Password").grid(row=3, column=0, sticky="w")
pwd1 = tk.Entry(db1_frame, show="*")
pwd1.insert(0, "postgres")
pwd1.grid(row=3, column=1)

tk.Label(db1_frame, text="Port").grid(row=4, column=0, sticky="w")
port1 = tk.Entry(db1_frame)
port1.insert(0, "5432")
port1.grid(row=4, column=1)

# ---------- DB2 ----------
db2_frame = tk.LabelFrame(frame, text="Database 2", padx=10, pady=10)
db2_frame.grid(row=0, column=1, padx=20)

tk.Label(db2_frame, text="Host").grid(row=0, column=0, sticky="w")
host2 = tk.Entry(db2_frame)
host2.insert(0, "localhost")
host2.grid(row=0, column=1)

tk.Label(db2_frame, text="Database").grid(row=1, column=0, sticky="w")
db2 = tk.Entry(db2_frame)
db2.insert(0, "db2")
db2.grid(row=1, column=1)

tk.Label(db2_frame, text="User").grid(row=2, column=0, sticky="w")
user2 = tk.Entry(db2_frame)
user2.insert(0, "postgres")
user2.grid(row=2, column=1)

tk.Label(db2_frame, text="Password").grid(row=3, column=0, sticky="w")
pwd2 = tk.Entry(db2_frame, show="*")
pwd2.insert(0, "postgres")
pwd2.grid(row=3, column=1)

tk.Label(db2_frame, text="Port").grid(row=4, column=0, sticky="w")
port2 = tk.Entry(db2_frame)
port2.insert(0, "5432")
port2.grid(row=4, column=1)

# ---------- Folder Inputs ----------
folder_frame = tk.Frame(root)
folder_frame.pack(pady=10)

tk.Label(folder_frame, text="Folder for DB1 Excel Files").grid(row=0, column=0, sticky="w")
folder1 = tk.Entry(folder_frame, width=60)
folder1.grid(row=0, column=1, padx=5)
browse1 = tk.Button(folder_frame, text="Browse", command=lambda: browse_folder(folder1))
browse1.grid(row=0, column=2)

tk.Label(folder_frame, text="Folder for DB2 Excel Files").grid(row=1, column=0, sticky="w")
folder2 = tk.Entry(folder_frame, width=60)
folder2.grid(row=1, column=1, padx=5)
browse2 = tk.Button(folder_frame, text="Browse", command=lambda: browse_folder(folder2))
browse2.grid(row=1, column=2)

# Submit
submit_btn = tk.Button(root, text="Submit", command=submit, font=("Arial", 12))
submit_btn.pack(pady=(15, 5))

submit_status = tk.Label(root, text="", font=("Arial", 10))
submit_status.pack(pady=(0, 5))
global_submit_spinner = Spinner(submit_status, root, interval_ms=120)

# Logs
log_frame = tk.LabelFrame(root, text="Logs", padx=10, pady=10)
log_frame.pack(fill="both", expand=True, padx=10, pady=10)

log_box = tk.Text(log_frame, height=10, width=120, state="disabled")
log_box.pack(side="left", fill="both", expand=True)

scrollbar = tk.Scrollbar(log_frame, command=log_box.yview)
scrollbar.pack(side="right", fill="y")
log_box.config(yscrollcommand=scrollbar.set)

pump_queues()
root.mainloop()