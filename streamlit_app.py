"""
Streamlit application for configuring and running multimodal model
evaluations.  The UI exposes the same set of parameters that are
present in the provided shell script (e.g. ROOT_DIR, METHODS,
model/architecture, BUDGETS, MODEL_PATH, MODEL_NAME and TASKS) and
provides a simple way to kick off evaluations from the browser.

The application also includes a results explorer which will scan the
selected ROOT_DIR for completed runs and display any metrics or
logged samples it finds.  This allows you to quickly inspect the
outcome of different method/task/budget combinations without digging
through the filesystem manually.

Note: the subprocess calls rely on the underlying environment being
correctly configured (e.g. `accelerate` installed, correct CUDA
devices visible, etc.).  When running inside this notebook the
evaluation commands will likely fail due to missing dependencies or
lack of GPU resources, but the code illustrates the intended usage.
"""

import json
import os
# Use HF mirror for Hugging Face API requests when Streamlit starts.
# This ensures the application uses the mirror endpoint every time it is launched.
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
import subprocess
import textwrap
from pathlib import Path

import streamlit as st


def parse_budget_input(budget_str: str) -> list[float]:
    """Parse a comma‑separated list of floats from user input.

    Returns a list of floats.  Any unparsable entries are ignored.
    """
    budgets: list[float] = []
    for entry in budget_str.split(','):
        entry = entry.strip()
        if not entry:
            continue
        try:
            budgets.append(float(entry))
        except ValueError:
            # ignore bad entries
            continue
    return budgets


def load_method_configs() -> dict[str, dict[str, str]]:
    """Return a dictionary of preset method configurations.

    Each entry maps a human‑friendly method name to a dictionary
    containing the filename portion and any additional model_args.

    These presets mirror the examples found in the shell script.
    """
    return {
        # Qwen2‑VL KV cache methods
        "h2o": {
            "filename": "h2o",
            "additional": "head_adaptive=True,use_flash_attention_2=true,device_map=auto",
        },
        "snapkv": {
            "filename": "snapkv",
            "additional": "head_adaptive=True,pooling=avgpool,use_flash_attention_2=true",
        },
        "pyramidkv": {
            "filename": "pyramidkv",
            "additional": "head_adaptive=True,pooling=avgpool,use_flash_attention_2=true",
        },
        "look-m": {
            "filename": "look-m",
            "additional": "merge=True,use_flash_attention_2=true",
        },
        "vl-cache": {
            "filename": "vl-cache",
            "additional": "vlcache_different_window_per_layer=False,vlcache_head_adaptive=True,layer_adaptive=True,use_flash_attention_2=true",
        },
        "random": {
            "filename": "random",
            "additional": "use_flash_attention_2=true",
        },
        "streamingllm": {
            "filename": "streamingllm",
            "additional": "use_flash_attention_2=true",
        },
        # Qwen2‑VL token prune methods
        "fastv": {
            "filename": "fastv",
            "additional": "use_flash_attention_2=true",
        },
        "visionzip": {
            "filename": "visionzip",
            "additional": "use_flash_attention_2=true",
        },
        "prumerge+": {
            "filename": "prumerge+",
            "additional": "use_flash_attention_2=true",
        },
    }


def construct_command(
    base_command: str,
    task: str,
    output_path: Path,
    model_args: str,
) -> str:
    """Build the full command line string for the evaluation.

    Parameters
    ----------
    base_command: str
        The constant portion of the command (accelerate launch, model
        architecture, batch size, etc.).
    task: str
        The evaluation task name (e.g. "textvqa").
    output_path: Path
        Directory where results will be saved.
    model_args: str
        Comma‑separated key=value pairs passed to the model.

    Returns
    -------
    str
        A fully constructed command string ready for execution.
    """
    cmd = (
        f"{base_command} "
        f"--tasks {task} "
        f"--output_path {output_path} "
        f"--log_samples_suffix {task} "
        f"--model_args \"{model_args}\""
    )
    return cmd


def run_evaluation(
    root_dir: Path,
    model_name: str,
    model_arch: str,
    model_path: Path,
    budgets: list[float],
    tasks: list[str],
    methods: list[str],
    num_processes: int,
    method_cfg: dict[str, dict[str, str]],
):
    """Run evaluations for all combinations of tasks/methods/budgets.

    This helper iterates through the chosen combinations, constructs the
    appropriate output directories and command lines, and executes
    `lmms_eval` via `accelerate`.  It uses Streamlit's progress
    bar and status messaging to display feedback to the user.

    Parameters mirror those configured in the UI.
    """
    # Build the constant portion of the command.
    base_command = (
        "python3 -m accelerate.commands.launch "
        "--main_process_port=28176 "
        "--mixed_precision=bf16 "
        f"--num_processes={num_processes} "
        "-m lmms_eval "
        f"--model {model_arch} "
        "--batch_size 1 "
        "--log_samples"
    )

    # Initialize progress bar to track number of runs.
    total_runs = len(tasks) * len(methods) * len(budgets)
    progress = st.progress(0)
    run_count = 0

    for task in tasks:
        for method in methods:
            cfg = method_cfg[method]
            filename = cfg["filename"]
            additional = cfg.get("additional", "")
            for budget in budgets:
                run_count += 1
                progress.progress(run_count / total_runs)
                output_path = root_dir / f"{model_name}_{method}_{task}_{budget}_{filename}"
                # Skip if the folder already exists.
                if output_path.exists():
                    st.info(f"Skipping existing directory {output_path}")
                    continue
                # Create output directory ahead of time to avoid race conditions.
                output_path.mkdir(parents=True, exist_ok=True)
                model_args = f"pretrained={model_path},method={method},budgets={budget}"
                if additional:
                    model_args = f"{model_args},{additional}"

                cmd = construct_command(base_command, task, output_path, model_args)
                with st.expander(f"Run {run_count}/{total_runs}: {method} | {task} | budget={budget}"):
                    st.code(cmd, language="bash")
                    # Execute the command and capture output.  This may take
                    # considerable time and use GPU resources.  Capturing
                    # stdout/stderr allows the user to debug failures.
                    with st.spinner("Running evaluation… this may take a while"):
                        try:
                            result = subprocess.run(
                                cmd,
                                shell=True,
                                capture_output=True,
                                text=True,
                                check=False,
                            )
                            if result.stdout:
                                st.text("stdout:\n" + result.stdout)
                            if result.stderr:
                                st.text("stderr:\n" + result.stderr)
                        except Exception as exc:
                            st.error(f"Error running command: {exc}")
    # Done
    progress.progress(1.0)
    st.success("All evaluations processed.")


def list_result_files(run_dir: Path) -> list[Path]:
    """Return a list of result files within a run directory.

    Result files are those ending in .json, .jsonl, .txt or .csv.  The
    function returns absolute paths which can later be read and
    displayed.
    """
    results: list[Path] = []
    # Search recursively so files stored under nested model-name folders are discovered
    suffixes = {".json", ".jsonl", ".txt", ".csv", ".tsv"}
    for ext in suffixes:
        for p in run_dir.rglob(f"*{ext}"):
            if p.is_file():
                results.append(p)
    # Deduplicate and sort for stable display
    results = sorted(set(results))
    return results


def display_results(root_dir: Path, key_prefix: str = ""):
    """递归展示运行目录下的所有文件，保持目录结构"""
    if not root_dir.exists():
        st.warning(f"Root directory {root_dir} does not exist.")
        return

    subdirs = sorted([p for p in root_dir.iterdir() if p.is_dir()])
    if not subdirs:
        st.info("No completed runs found in the selected root directory.")
        return

    def show_dir(dir_path: Path, prefix: str):
        entries = sorted(dir_path.iterdir())
        files = [e for e in entries if e.is_file()]
        dirs = [e for e in entries if e.is_dir()]
        # 显示当前目录下的文件
        for file_path in files:
            cols = st.columns([8, 1])
            cols[0].write(file_path.relative_to(dir_path.parent))
            view_key = f"view::{prefix}{file_path.as_posix()}"
            if cols[1].button("View", key=view_key):
                _display_file_contents(file_path)
        # 对每个子目录递归
        for child_dir in dirs:
            with st.expander(child_dir.name):
                show_dir(child_dir, prefix)

    for subdir in subdirs:
        st.subheader(subdir.name)
        show_dir(subdir, key_prefix + "::")


def _display_file_contents(file_path: Path):
    """Helper to read and render a single file's contents."""
    try:
        suffix = file_path.suffix.lower()
        if suffix == ".json":
            with open(file_path, "r", encoding="utf-8") as f:
                content = json.load(f)
            # For dicts/lists, display as JSON (clickable/expandable in Streamlit)
            if isinstance(content, (dict, list)):
                st.json(content)
            else:
                st.text(str(content))
        elif suffix == ".jsonl":
            with open(file_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
            st.write(f"Showing first {min(50, len(lines))} of {len(lines)} lines:")
            for i, line in enumerate(lines[:50]):
                try:
                    obj = json.loads(line)
                    st.json(obj)
                except Exception:
                    st.text(line.strip())
        else:
            # Generic text files
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
            if len(content) > 5000:
                st.text(content[:5000] + "\n... (truncated)")
            else:
                st.text(content)
    except Exception as exc:
        st.error(f"Could not read {file_path}: {exc}")


def extract_results_fields(root_dir: Path) -> Path:
    """Scan each run folder under `root_dir` and extract the top-level
    "results" field from any .json files. Writes extracted content to
    `root_dir/extracted_results/<run_name>/*.json` and returns that path.

    The function is idempotent and will overwrite existing extracted files
    for the same run/file.
    """
    extracted_root = root_dir / "extracted_results"
    extracted_root.mkdir(parents=True, exist_ok=True)

    # Each run directory under root_dir corresponds to a single evaluation
    for run_dir in sorted([p for p in root_dir.iterdir() if p.is_dir()]):
        # create a subfolder per run under extracted_root
        target_run_dir = extracted_root / run_dir.name
        target_run_dir.mkdir(parents=True, exist_ok=True)

        # find all top-level JSON files in the run (search recursively)
        for json_file in run_dir.rglob("*.json"):
            try:
                with open(json_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception:
                # skip unreadable/invalid JSON
                continue

            if not isinstance(data, dict):
                continue
            
            if "results" in data:
                results_content = data["results"]
            elif "overall_acc" in data:
                # mmbench style results are at the top level
                results_content = data
            else:
                continue
            # write to target file named <origname>_results.json
            out_name = json_file.stem + "_results.json"
            out_path = target_run_dir / out_name
            try:
                with open(out_path, "w", encoding="utf-8") as out_f:
                    json.dump(results_content, out_f, ensure_ascii=False, indent=2)
            except Exception:
                # ignore write errors for now
                continue

    return extracted_root


def kill_all_processes():
    """Kill all running evaluation processes."""
    # Kill accelerate launch processes and lmms_eval
    # We use pkill -f to match the full command line
    subprocess.run(["pkill", "-f", "lmms_eval"], check=False)
    subprocess.run(["pkill", "-f", "accelerate"], check=False)
    st.sidebar.warning("Evaluation processes terminated!")


def main():
    st.set_page_config(page_title="LMMs Evaluation Configurator", layout="wide")
    st.title("🔧 Large Multimodal Model Evaluation Dashboard")

    # Sidebar configuration inputs
    st.sidebar.header("Evaluation Parameters")
    # Root directory fixed to workspace results folder — users only need
    # to create a new subfolder under this path for each run.
    default_root = "/home/lyli/EffiVLM-Bench/result"
    root_dir_str = st.sidebar.text_input("Root directory (base result folder)", value=default_root)
    root_dir = Path(root_dir_str)

    # Model selection
    model_options = {
        "Qwen2-VL": "qwen2_vl_with_kvcache",
        "InternVL2.5-38B": "internvl2_with_kvcache",
        "LLaVA-OneVision": "llava_onevision_with_kvcache",
    }
    model_name = st.sidebar.selectbox(
        "Model name", options=list(model_options.keys()), index=0
    )
    model_arch = model_options[model_name]
    # Default model paths per selection
    default_model_paths = {
        "Qwen2-VL": "/home/lyli/models/Qwen2-VL-7B-Instruct",
        "InternVL2.5-38B": "/home/lyli/models/InternVL2.5-38B",
        "LLaVA-OneVision": "/home/lyli/models/LLaVA-OneVision-7B",
    }
    default_model_path = default_model_paths.get(model_name, "/home/lyli/models/Qwen2-VL-7B-Instruct")
    model_path = st.sidebar.text_input(
        "Model path",
        value=default_model_path,
    )

    # Budgets
    budget_str = st.sidebar.text_input("Budgets (comma separated)", value="0.05")
    budgets = parse_budget_input(budget_str)
    if not budgets:
        st.sidebar.error("Please enter at least one valid numeric budget.")

    # Tasks
    # Provide a few common tasks as suggestions; user can also type custom ones.
    suggested_tasks = [
        "docvqa",
        "chartqa",
        "textvqa",
        "ocrbench",
        "ai2d",
        "gqa",
        "mmmu",
        "mme",
        "realworldqa",
        "mmstar",
        "mathvista",
        "llava_wilder",
        "mmbench",
        "mmvet" 
    ]
    selected_tasks = st.sidebar.multiselect(
        "Tasks", options=suggested_tasks, default=["textvqa"]
    )
    # Allow users to enter custom tasks via a text box.
    custom_task = st.sidebar.text_input("Add custom task (optional)")
    if custom_task:
        selected_tasks.append(custom_task.strip())

    # Methods
    method_cfg = load_method_configs()
    method_names = list(method_cfg.keys())
    selected_methods = st.sidebar.multiselect(
        "Methods", options=method_names, default=method_names[:3]
    )

    # Number of processes
    num_processes = int(
        st.sidebar.number_input(
            "Number of GPU processes", min_value=1, max_value=16, value=10, step=1
        )
    )

    # Run folder name: user provides a name for a new subfolder under the
    # fixed base result path. The app will create this folder if it does not
    # already exist. This keeps all runs under /home/lyli/EffiVLM-Bench/result/
    run_name = st.sidebar.text_input("Run folder name (create under base result folder)", value="")

    # Action button
    if st.sidebar.button("▶️ Run evaluation"):
        if not run_name or not run_name.strip():
            st.sidebar.error("Please provide a run folder name (a new subdirectory under the base result folder).")
        elif not selected_tasks:
            st.sidebar.error("Please select at least one task to run.")
        elif not selected_methods:
            st.sidebar.error("Please select at least one method to run.")
        elif not budgets:
            st.sidebar.error("Please specify at least one budget value.")
        else:
            # Build the run root under the fixed base result folder
            run_root = root_dir / run_name.strip()
            try:
                run_root.mkdir(parents=True, exist_ok=True)
            except Exception as exc:
                st.sidebar.error(f"Could not create run folder {run_root}: {exc}")
                run_root = root_dir

            run_evaluation(
                root_dir=run_root,
                model_name=model_name,
                model_arch=model_arch,
                model_path=Path(model_path),
                budgets=budgets,
                tasks=selected_tasks,
                methods=selected_methods,
                num_processes=num_processes,
                method_cfg=method_cfg,
            )

    if st.sidebar.button("🛑 Kill running processes"):
        kill_all_processes()

    # Extraction UI: extract "results" fields from saved JSONs
    if st.sidebar.button("🔍 Extract results fields"):
        with st.spinner("Extracting results fields..."):
            extracted_dir = extract_results_fields(root_dir)
        st.success(f"Extracted results saved to: {extracted_dir}")
        st.sidebar.markdown(f"- Extracted path: **{extracted_dir}**")

    st.header("📊 Evaluation Results Viewer")
    display_results(root_dir, key_prefix="orig")

    # Show extracted results (if present) in a separate section for quick access
    extracted_dir = root_dir / "extracted_results"
    if extracted_dir.exists():
        st.header("📁 Extracted Results")
        # reuse the same display logic so files are shown with View buttons
        display_results(extracted_dir, key_prefix="extracted")


if __name__ == "__main__":
    main()
