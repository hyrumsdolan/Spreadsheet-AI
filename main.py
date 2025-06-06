import streamlit as st
import pandas as pd
import csv
import asyncio
import io
import config
from streamlit_local_storage import LocalStorage

# --------------------------------------------------
# Page setup & custom styling
# --------------------------------------------------
st.set_page_config(
    page_title="Universal AI Processing Tool",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    div.stButton > button {background:#28a745;color:#fff;border-radius:8px;padding:10px 16px;}
    div.stButton > button:hover {background:#218838;}
    div[data-testid="stFileUploader"] button {background:#007BFF;color:#fff;border-radius:8px;padding:10px 16px;}
    div[data-testid="stFileUploader"] button:hover {background:#0056b3;}
    div[data-testid="stDownloadButton"] button {background:#28a745;color:#fff;border-radius:8px;padding:10px 16px;}
    div[data-testid="stDownloadButton"] button:hover {background:#218838;}
    </style>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------
# Async helpers
# --------------------------------------------------
async def _process_row(context, prompt, idx, model, temp, client, service):
    try:
        if service == "openai":
            resp = await client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": f"{prompt}\n{context}"}],
                temperature=temp,
            )
            return idx, resp.choices[0].message.content.strip()
        else:
            msg = await client.messages.create(
                model=model,
                messages=[{"role": "user", "content": f"{prompt}\n{context}"}],
                max_tokens=512,
                temperature=temp,
            )
            text = "".join(block.text for block in msg.content)
            return idx, text.strip()
    except Exception as err:
        return idx, f"ERROR: {err}"

async def _run_tasks(df, cols, prompt, model, temp, concurrency, client, service):
    sem = asyncio.Semaphore(concurrency)
    tasks = []

    for i, row in df.iterrows():
        ctx = "\n".join(f"{c}: {row[c]}" for c in cols)

        async def worker(idx=i, context=ctx):
            async with sem:
                return await _process_row(
                    context, prompt, idx, model, temp, client, service
                )

        tasks.append(worker())

    results, total = [], len(tasks)
    progress = st.progress(0, text="Processing AI calls…")
    for coro in asyncio.as_completed(tasks):
        idx, out = await coro
        results.append((idx, out))
        progress.progress(len(results) / total)
    progress.empty()
    return sorted(results, key=lambda x: x[0])

# --------------------------------------------------
# CSV helper
# --------------------------------------------------
def to_csv(results):
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["Index", "AI Output"])
    writer.writerows(results)
    return buf.getvalue()

# --------------------------------------------------
# One provider page (inside its tab)
# --------------------------------------------------
def provider_page(
    service_code: str,
    service_label: str,
    default_api_key: str,
    model_options: list[str],
    AIClientClass,
    get_models_func,
    api_key_url: str,
    max_temp: float,
):
    localS = LocalStorage()

    # ---------- Collapsible settings ----------
    with st.expander("⚙️  Config", expanded=False):
        # --- Logic and data prep ---
        session_key = f"{service_code}_api_key_input"
        
        def save_key():
            localS.setItem(f"{service_code}_api_key", st.session_state[session_key])

        local_storage_key = localS.getItem(f"{service_code}_api_key")
        api_key_default = local_storage_key if local_storage_key else default_api_key

        final_api_key = st.session_state.get(session_key, api_key_default)

        current_models = model_options
        error_message = None
        if final_api_key:
            current_models, error_message = get_models_func(final_api_key)

        # --- UI Layout ---
        # Row for titles
        col1, col2, col3, col4 = st.columns((3, 2, 1, 1))

        with col1:
            st.markdown(f"""
                <div style="display: flex; align-items: center; justify-content: space-between; height: 100%;">
                    <span style="font-weight: bold;">API Key</span>
                    <a href="{api_key_url}" target="_blank" style="font-size: 0.8em;">Get Key</a>
                </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("**Model**")

        with col3:
            st.markdown("**Temp**")

        with col4:
            st.markdown("**Concurrency**")

        # Row for widgets
        col1_w, col2_w, col3_w, col4_w = st.columns((3, 2, 1, 1))

        with col1_w:
            api_key = st.text_input(
                "API Key",
                value=api_key_default,
                type="password",
                label_visibility="collapsed",
                key=session_key,
                on_change=save_key,
            )
        
        with col2_w:
            default_index = 0
            if current_models:
                for i, model_name in enumerate(current_models):
                    if "latest" in model_name:
                        default_index = i
                        break
            
            model_choice = st.selectbox(
                "",
                current_models,
                index=default_index,
                label_visibility="collapsed",
                key=f"{service_code}_model",
            )

        with col3_w:
            temp = st.slider(
                "", 0.0, max_temp, 0.7, 0.1, label_visibility="collapsed", key=f"{service_code}_temp"
            )

        with col4_w:
            concurrency = st.number_input(
                "", 1, 1000, 100, 1, label_visibility="collapsed", key=f"{service_code}_concurrency"
            )

        if error_message:
            st.warning(f"{error_message} Using default model list.")

    # Use the selected values
    final_api_key = api_key if api_key else api_key_default
    client = AIClientClass(api_key=final_api_key)

    # ---------- 1 Upload ----------
    file = st.file_uploader(
        "CSV or Excel", type=["csv", "xlsx"], key=f"{service_code}_file"
    )
    if file is None:
        return

    df = pd.read_csv(file) if file.name.endswith(".csv") else pd.read_excel(file)
    df = df.reset_index(drop=True)
    st.dataframe(df, height=300, use_container_width=True)

    # ---------- 2 Configure & run ----------
    cols = st.multiselect(
        "Context columns", df.columns.tolist(), key=f"{service_code}_cols"
    )
    prompt = st.text_area(
        "AI instructions",
        value="",  # empty so nothing is pre‑filled
        placeholder="Enter System Message",
        key=f"{service_code}_prompt",
    )
    can_run = bool(cols and prompt.strip())

    if st.button(
        "🚀 Process Rows",
        disabled=not can_run,
        key=f"{service_code}_run",
    ):
        st.session_state[f"{service_code}_results"] = asyncio.run(
            _run_tasks(
                df,
                cols,
                prompt,
                model_choice,
                temp,
                concurrency,
                client,
                service_code,
            )
        )

    # ---------- 3 Results & download ----------
    results_key = f"{service_code}_results"
    if results_key in st.session_state and st.session_state[results_key]:
        option = st.radio(
            "Download format",
            ["AI‑only CSV", "Append to original"],
            horizontal=True,
            key=f"{service_code}_download_format",
        )

        if option == "AI‑only CSV":
            csv_str = to_csv(st.session_state[results_key])
            st.download_button(
                "Download CSV", csv_str, "ai_output.csv", key=f"{service_code}_dl_ai"
            )
            preview_df = pd.read_csv(io.StringIO(csv_str))
        else:
            out_df = df.copy()
            out_df["AI Output"] = [o for _, o in st.session_state[results_key]]
            if file.name.endswith(".csv"):
                buf = out_df.to_csv(index=False)
                mime, fname = "text/csv", "with_output.csv"
            else:
                buf_io = io.BytesIO()
                with pd.ExcelWriter(buf_io, engine="xlsxwriter") as w:
                    out_df.to_excel(w, index=False)
                buf = buf_io.getvalue()
                mime = (
                    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
                fname = "with_output.xlsx"
            st.download_button(
                "Download File", buf, fname, mime, key=f"{service_code}_dl_full"
            )
            preview_df = out_df

        st.dataframe(preview_df, height=400, use_container_width=True)

# --------------------------------------------------
# Main entry
# --------------------------------------------------
def main():
    st.title("Spreadsheet AI")

    openai_tab, anthropic_tab = st.tabs(["OpenAI", "Anthropic"])

    with openai_tab:
        from openai import AsyncOpenAI as OpenAIClient
        provider_page(
            service_code="openai",
            service_label="OpenAI",
            default_api_key=config.OPENAI_API_KEY,
            model_options=config.OPENAI_MODELS,
            AIClientClass=OpenAIClient,
            get_models_func=config.get_openai_models,
            api_key_url="https://platform.openai.com/account/api-keys",
            max_temp=2.0,
        )

    with anthropic_tab:
        from anthropic import AsyncAnthropic as AnthropicClient
        provider_page(
            service_code="anthropic",
            service_label="Anthropic",
            default_api_key=config.ANTHROPIC_API_KEY,
            model_options=config.CLAUDE_MODELS,
            AIClientClass=AnthropicClient,
            get_models_func=config.get_claude_models,
            api_key_url="https://console.anthropic.com/dashboard",
            max_temp=1.0,
        )

if __name__ == "__main__":
    main()
