import streamlit as st
import pandas as pd
import csv
import asyncio
import io
import config

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
async def _process_row(context, prompt, idx, model, client, service):
    try:
        if service == "openai":
            resp = await client.responses.create(
                model=model, instructions=prompt, input=context
            )
            return idx, resp.output[0].content[0].text.strip()
        else:
            msg = await client.messages.create(
                model=model,
                messages=[{"role": "user", "content": f"{prompt}\n{context}"}],
                max_tokens=512,
            )
            text = "".join(block.text for block in msg.content)
            return idx, text.strip()
    except Exception as err:
        return idx, f"ERROR: {err}"

async def _run_tasks(df, cols, prompt, model, concurrency, client, service):
    sem = asyncio.Semaphore(concurrency)
    tasks = []

    for i, row in df.iterrows():
        ctx = "\n".join(f"{c}: {row[c]}" for c in cols)

        async def worker(idx=i, context=ctx):
            async with sem:
                return await _process_row(
                    context, prompt, idx, model, client, service
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
):
    # ---------- Collapsible settings ----------
    with st.expander("⚙️  Config", expanded=False):
        col1, col2, col3 = st.columns((3, 2, 1))

        with col1:
            st.markdown("**API Key**")
            api_key = st.text_input(
                "",
                value=default_api_key,
                type="password",
                label_visibility="collapsed",
                placeholder=f"{service_label} API Key",
                key=f"{service_code}_api_key",
            )

        with col2:
            st.markdown("**Model**")
            model_choice = st.selectbox(
                "",
                model_options,
                label_visibility="collapsed",
                key=f"{service_code}_model",
            )

        with col3:
            st.markdown("**Concurrency**")
            concurrency = st.number_input(
                "",
                min_value=1,
                max_value=1000,
                value=100,
                step=1,
                label_visibility="collapsed",
                key=f"{service_code}_concurrency",
            )

    client = AIClientClass(api_key=api_key)

    # ---------- 1 Upload ----------
    file = st.file_uploader(
        "CSV or Excel", type=["csv", "xlsx"], key=f"{service_code}_file"
    )
    if file is None:
        return

    df = pd.read_csv(file) if file.name.endswith(".csv") else pd.read_excel(file)
    df = df.reset_index(drop=True)
    st.dataframe(df, height=300, use_container_width=True)

    # ---------- 2 Configure & run ----------
    cols = st.multiselect(
        "Context columns", df.columns.tolist(), key=f"{service_code}_cols"
    )
    prompt = st.text_area(
        "AI instructions",
        value="",  # empty so nothing is pre‑filled
        placeholder="Enter System Message",
        key=f"{service_code}_prompt",
    )
    can_run = bool(cols and prompt.strip())

    if st.button(
        "🚀 Process Rows",
        disabled=not can_run,
        key=f"{service_code}_run",
    ):
        st.session_state[f"{service_code}_results"] = asyncio.run(
            _run_tasks(
                df,
                cols,
                prompt,
                model_choice,
                concurrency,
                client,
                service_code,
            )
        )

    # ---------- 3 Results & download ----------
    results_key = f"{service_code}_results"
    if results_key in st.session_state and st.session_state[results_key]:
        option = st.radio(
            "Download format",
            ["AI‑only CSV", "Append to original"],
            horizontal=True,
            key=f"{service_code}_download_format",
        )

        if option == "AI‑only CSV":
            csv_str = to_csv(st.session_state[results_key])
            st.download_button(
                "Download CSV", csv_str, "ai_output.csv", key=f"{service_code}_dl_ai"
            )
            preview_df = pd.read_csv(io.StringIO(csv_str))
        else:
            out_df = df.copy()
            out_df["AI Output"] = [o for _, o in st.session_state[results_key]]
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
                "Download File", buf, fname, mime, key=f"{service_code}_dl_full"
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
        )

    with anthropic_tab:
        from anthropic import AsyncAnthropic as AnthropicClient
        provider_page(
            service_code="anthropic",
            service_label="Anthropic",
            default_api_key=config.ANTHROPIC_API_KEY,
            model_options=config.CLAUDE_MODELS,
            AIClientClass=AnthropicClient,
        )

if __name__ == "__main__":
    main()
