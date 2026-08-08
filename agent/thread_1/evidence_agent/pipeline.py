from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from perception_layer.agent.thread_1.evidence_agent.agent import EvidenceSubagent, print_colored
from perception_layer.agent.thread_1.evidence_agent.utils import (
    EVIDENCE_GROUPS,
    build_windowed_evidence_inputs,
    extract_anomaly_ranges,
    generate_highlighted_images,
)
from perception_layer.path_config import DEFAULT_CONFIG_PATH


def filter_and_distribute_evidence(
    parquet_path: str | Path,
    context: str = "",
    window_size_rows: int | None = 600,
    window_side: int | None = None,
    window_size_seconds: float = 5.0,
    sample_rate_hz: float = 50.0,
    overlap_rows: int | None = 50,
    overlap_fraction: float = 0.0,
    start_row: int = 0,
    end_row: int | None = None,
    model: str | None = None,
    timeout: int = 180,
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    debug: bool = False,
) -> dict[str, str]:
    """
    Filter the feature groups and distribute them to their corresponding subagents.
    Each subagent's output is then passed to a final agent for a consolidated conclusion.
    """
    
    t_calc_start = time.time()
    filtered_evidence_buckets = build_windowed_evidence_inputs(
        parquet_path=parquet_path,
        window_size_rows=window_size_rows,
        window_side=window_side,
        window_size_seconds=window_size_seconds,
        sample_rate_hz=sample_rate_hz,
        overlap_rows=overlap_rows,
        overlap_fraction=overlap_fraction,
        start_row=start_row,
        end_row=end_row,
    )
    t_calc_end = time.time()
    
    if debug:
        for group in EVIDENCE_GROUPS:
            print_colored(f"TOOL ANALYST OUTPUT (TABLE): {group.upper()}", filtered_evidence_buckets[group], "\033[96m")  # Cyan
        print_colored("TIMING", f"EVIDENCE CALCULATION TIME: {t_calc_end - t_calc_start:.2f} seconds", "\033[96m")

    agent = EvidenceSubagent(
        model=model,
        timeout=timeout,
        config_path=config_path,
    )

    group_outputs: dict[str, str] = {}
    
    t_sub_start = time.time()
    with ThreadPoolExecutor(max_workers=len(EVIDENCE_GROUPS)) as executor:
        futures = {
            executor.submit(
                agent.run_group, group, filtered_evidence_buckets[group], context, debug
            ): group
            for group in EVIDENCE_GROUPS
        }
        for future in as_completed(futures):
            group = futures[future]
            group_outputs[group] = future.result()
    t_sub_end = time.time()
    
    if debug:
        print_colored("TIMING", f"3 SUBAGENTS PARALLEL EXECUTION TIME: {t_sub_end - t_sub_start:.2f} seconds", "\033[96m")

    t_final_start = time.time()
    final_output = agent.run_final(group_outputs=group_outputs, context=context, debug=debug)
    t_final_end = time.time()
    
    if debug:
        print_colored("TIMING", f"FINAL SYNTHESIS AGENT EXECUTION TIME: {t_final_end - t_final_start:.2f} seconds", "\033[96m")
    
    anomaly_ranges_str = extract_anomaly_ranges(final_output)
    highlighted_image_folder = ""
    
    if anomaly_ranges_str != "No specific anomaly ranges identified.":
        t_img_start = time.time()
        try:
            if debug:
                print_colored("PIPELINE", "Generating new images with distinct colored highlights for candidate ranges...", "\033[93m")
            out_dir, updated_ranges_str = generate_highlighted_images(Path(parquet_path), anomaly_ranges_str)
            highlighted_image_folder = str(out_dir)
            anomaly_ranges_str = updated_ranges_str
            
            t_img_end = time.time()
            if debug:
                print_colored("TIMING", f"IMAGE HIGHLIGHT GENERATION TIME: {t_img_end - t_img_start:.2f} seconds", "\033[96m")
        except Exception as e:
            if debug:
                print(f"Warning: Failed to generate highlighted images: {e}")
                
    return {
        **group_outputs,
        "final_output": final_output,
        "anomaly_ranges_str": anomaly_ranges_str,
        "highlighted_image_folder": highlighted_image_folder,
    }
