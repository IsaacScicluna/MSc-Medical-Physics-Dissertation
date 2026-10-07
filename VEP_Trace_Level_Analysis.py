"""
Batch VEP Trace Analysis
-----------------------------------------------------------------------

Purpose:
-------
To process ALL normal volunteer folders in VEP_Project/data_raw automatically.
For each VEP text file, the script:

1. Extracts Channel 2 only: Oz-Fz.
2. Extracts all individual Oz-Fz traces from the repeated sweep sections.
3. Computes the initial average waveform using ALL traces.
4. Reads the NATUS/system marker values for Channel 2 from the text file:
      Marker 4 = N75
      Marker 5 = P100
      Marker 6 = N145
5. If two repeated runs are present, averages the system marker latencies and amplitudes.
6. Uses the mean system P100 latency as the reference.
7. For each individual trace, detects a candidate P100 by searching +/-25 ms around
   the system P100 latency and finding the strongest P100-polarity deflection.
8. Retains only traces whose detected P100 latency is within +/-5 ms of the system P100 latency.
9. Computes a refined average from retained traces.
10. Saves one PDF report per volunteer, per-recording CSV outputs, and one master summary CSV.

Expected project structure:
--------------------------
VEP_Project/
    code/
        VEP Trace Analysis Code.py
    data_raw/
        V001-Name/
            V001_L_12x16.txt
            V001_L_48x64.txt
            V001_R_12x16.txt
            V001_R_48x64.txt
        V002-Name/
            ...
    results/

Important notes:
---------------
- The plotted waveform is inverted to match the NATUS-style display convention.
- The plotted trace time axis is corrected by subtracting 12 ms (stimulator latency).
- System marker latencies from the text file are treated as the reference latencies.
- Amplitudes in the text file are stored in mV and converted to uV.
- The script does not ask for manual marker placement (the process is automatised).
- Completed volunteers are skipped automatically when their PDF report and all per-recording CSV outputs already exist.
- Set FORCE_REPROCESS = True to deliberately rerun every volunteer.
"""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


# =========================
# USER SETTINGS: This section defines the main parameters used throughout the analysis.
# =========================

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data_raw"
RESULTS_DIR = PROJECT_DIR / "results" / "VEP_Trace_Analysis_results"

CHANNEL_TO_EXTRACT = 2
SITE_LABEL_TO_EXTRACT = "Oz-Fz"

SWEEP_DURATION_MS = 250.0
STIMULATOR_LATENCY_CORRECTION_MS = 12.0
INVERT_DISPLAY = True

# Trace-selection settings.
P100_TRACE_SEARCH_HALF_WINDOW_MS = 25.0   # wider search region around system P100.
P100_ACCEPTANCE_TOLERANCE_MS = 5.0        # final keep/reject condition.
MIN_RETAINED_TRACES = 5

# Report settings.
PDF_FIGSIZE = (11.69, 8.27)  # A4 landscape.
DPI = 300

# Resume/skip settings.
SKIP_ALREADY_PROCESSED = True
FORCE_REPROCESS = False


# =========================
# DATA STRUCTURES: This section defines how the N75, P100, and N145 latency and amplitude values are organised and
# stored for individual NATUS runs and for the mean values calculated across repeated runs.
# =========================

@dataclass
class MarkerSet:
    run_index: int
    n75_latency_ms: float
    n75_amplitude_uV_raw: float
    p100_latency_ms: float
    p100_amplitude_uV_raw: float
    n145_latency_ms: float
    n145_amplitude_uV_raw: float

    @property
    def n75_p100_amplitude_uV(self) -> float:
        return abs(self.p100_amplitude_uV_raw - self.n75_amplitude_uV_raw)


@dataclass
class MeanSystemMarkers:
    n_runs: int
    n75_latency_ms: float
    n75_amplitude_uV_raw: float
    p100_latency_ms: float
    p100_amplitude_uV_raw: float
    n145_latency_ms: float
    n145_amplitude_uV_raw: float
    n75_p100_amplitude_uV: float


# =========================
# FILE READING / PARSING: This section reads the NATUS text files, extracts the individual Oz-Fz traces, and retrieves
# the N75, P100, and N145 marker values stored by the NATUS system.
# =========================

def read_text_file(path: Path) -> str:
    """Read Viking/Natus text exports using several likely encodings."""
    encodings = ["utf-8-sig", "utf-16", "utf-16-le", "utf-16-be", "latin-1"]
    for enc in encodings:
        try:
            return path.read_text(encoding=enc)
        except UnicodeError:
            continue
    return path.read_text(errors="replace")


def parse_numbers_from_trace_block(block: str) -> np.ndarray:
    """Extract numeric samples after Sweep Data(...)= in a trace block."""
    match = re.search(r"Sweep\s+Data\([^)]*\)<\d+>\s*=", block, flags=re.IGNORECASE)
    if not match:
        return np.array([], dtype=float)

    data_part = block[match.end():]
    data_part = data_part.replace("/", " ")
    numbers = re.findall(r"[-+]?\d*\.\d+|[-+]?\d+", data_part)
    return np.array([float(x) for x in numbers], dtype=float)


def extract_channel_sweeps(text: str, channel_number: int = CHANNEL_TO_EXTRACT) -> List[List[np.ndarray]]:
    """Extract trace arrays for the requested channel, split by sweep-data section."""
    lines = text.splitlines()
    sweeps: List[List[np.ndarray]] = []
    current_channel: Optional[int] = None
    current_sweep_traces: Optional[List[np.ndarray]] = None
    collecting_trace = False
    trace_lines: List[str] = []

    def finish_trace() -> None:
        nonlocal trace_lines, collecting_trace, current_sweep_traces
        if collecting_trace and current_sweep_traces is not None:
            block = "\n".join(trace_lines)
            arr = parse_numbers_from_trace_block(block)
            if arr.size > 0:
                current_sweep_traces.append(arr)
        trace_lines = []
        collecting_trace = False

    def finish_sweep() -> None:
        nonlocal current_sweep_traces
        if current_sweep_traces is not None and len(current_sweep_traces) > 0:
            sweeps.append(current_sweep_traces)
        current_sweep_traces = None

    for line in lines:
        stripped = line.strip()

        if re.match(r"\[.*Sweep\s+Data.*\]", stripped, flags=re.IGNORECASE):
            finish_trace()
            finish_sweep()
            current_channel = None
            current_sweep_traces = None
            continue

        m_channel = re.match(r"Channel\s+Number\s*=\s*(\d+)", stripped, flags=re.IGNORECASE)
        if m_channel:
            finish_trace()
            current_channel = int(m_channel.group(1))
            current_sweep_traces = [] if current_channel == channel_number else None
            continue

        if re.match(r"\[.*Trace\s+Data.*\]", stripped, flags=re.IGNORECASE):
            finish_trace()
            if current_channel == channel_number and current_sweep_traces is not None:
                collecting_trace = True
                trace_lines = []
            continue

        if collecting_trace:
            if stripped.startswith("Algorithm="):
                finish_trace()
            else:
                trace_lines.append(line)

    finish_trace()
    finish_sweep()
    return [s for s in sweeps if len(s) > 0]


def split_into_sections(text: str) -> List[Tuple[str, str]]:
    """Return [(header, body), ...] for bracketed sections in the text export."""
    pattern = re.compile(r"^\s*\[([^\]]+)\]\s*$", flags=re.MULTILINE)
    matches = list(pattern.finditer(text))
    sections: List[Tuple[str, str]] = []
    for i, m in enumerate(matches):
        header = m.group(1).strip()
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[start:end]
        sections.append((header, body))
    return sections


def get_float_value(body: str, key: str) -> Optional[float]:
    """Find a float value after a key, e.g. Marker 5 Latency(ms)=100.0."""
    pattern = re.compile(re.escape(key) + r"\s*=\s*([-+]?\d*\.\d+|[-+]?\d+)", flags=re.IGNORECASE)
    m = pattern.search(body)
    return float(m.group(1)) if m else None


def body_has_channel_ozfz(body: str, channel_number: int = CHANNEL_TO_EXTRACT) -> bool:
    ch_match = re.search(r"Channel\s+Number\s*=\s*(\d+)", body, flags=re.IGNORECASE)
    if not ch_match or int(ch_match.group(1)) != channel_number:
        return False

    # Site Label is a useful additional safeguard, but some exports may format it differently.
    site_match = re.search(r"Site\s+Label\s*=\s*([^\n\r]+)", body, flags=re.IGNORECASE)
    if site_match:
        label = site_match.group(1).strip().lower().replace(" ", "")
        return label in {"oz-fz", "ozfz"}

    return True


def parse_marker_set_from_custom_body(body: str, run_index: int) -> Optional[MarkerSet]:
    """Parse Marker 4/5/6 latencies and amplitudes from a VEP Site Custom Results body."""
    n75_lat = get_float_value(body, "Marker 4 Latency(ms)")
    n75_amp_mV = get_float_value(body, "Marker 4 Amplitude(mV)")
    p100_lat = get_float_value(body, "Marker 5 Latency(ms)")
    p100_amp_mV = get_float_value(body, "Marker 5 Amplitude(mV)")
    n145_lat = get_float_value(body, "Marker 6 Latency(ms)")
    n145_amp_mV = get_float_value(body, "Marker 6 Amplitude(mV)")

    values = [n75_lat, n75_amp_mV, p100_lat, p100_amp_mV, n145_lat, n145_amp_mV]
    if any(v is None for v in values):
        return None

    return MarkerSet(
        run_index=run_index,
        n75_latency_ms=float(n75_lat),
        n75_amplitude_uV_raw=float(n75_amp_mV) * 1000.0,
        p100_latency_ms=float(p100_lat),
        p100_amplitude_uV_raw=float(p100_amp_mV) * 1000.0,
        n145_latency_ms=float(n145_lat),
        n145_amplitude_uV_raw=float(n145_amp_mV) * 1000.0,
    )


def extract_system_marker_sets_for_channel2(text: str) -> List[MarkerSet]:
    """
    Extract system/NATUS marker values for Channel 2 Oz-Fz.

    In the Viking export, each Channel 2 'VEP Site Results' section is followed by
    its corresponding 'VEP Site Custom Results' section. Marker 4/5/6 values are
    stored in that custom-results section.
    """
    sections = split_into_sections(text)
    marker_sets: List[MarkerSet] = []

    for i, (header, body) in enumerate(sections):
        if "VEP Site Results" not in header or "Custom" in header:
            continue
        if not body_has_channel_ozfz(body, CHANNEL_TO_EXTRACT):
            continue

        # Finding the next VEP Site Custom Results section.
        for j in range(i + 1, min(i + 4, len(sections))):
            next_header, next_body = sections[j]
            if "VEP Site Custom Results" in next_header:
                parsed = parse_marker_set_from_custom_body(next_body, run_index=len(marker_sets) + 1)
                if parsed is not None:
                    marker_sets.append(parsed)
                break

    return marker_sets


def mean_system_markers(marker_sets: List[MarkerSet]) -> MeanSystemMarkers:
    """Average system marker values across repeated runs/sweeps."""
    if not marker_sets:
        raise ValueError("No system marker sets were supplied.")

    return MeanSystemMarkers(
        n_runs=len(marker_sets),
        n75_latency_ms=float(np.mean([m.n75_latency_ms for m in marker_sets])),
        n75_amplitude_uV_raw=float(np.mean([m.n75_amplitude_uV_raw for m in marker_sets])),
        p100_latency_ms=float(np.mean([m.p100_latency_ms for m in marker_sets])),
        p100_amplitude_uV_raw=float(np.mean([m.p100_amplitude_uV_raw for m in marker_sets])),
        n145_latency_ms=float(np.mean([m.n145_latency_ms for m in marker_sets])),
        n145_amplitude_uV_raw=float(np.mean([m.n145_amplitude_uV_raw for m in marker_sets])),
        n75_p100_amplitude_uV=float(np.mean([m.n75_p100_amplitude_uV for m in marker_sets])),
    )


# =========================
# SIGNAL HELPERS: This section prepares the VEP data for analysis by creating the corrected time axis, inverting the
# waveforms to match the NATUS display, combining individual traces, and detecting the candidate P100 in each trace.
# =========================

def make_time_axis(trace_len: int) -> np.ndarray:
    raw_time = np.linspace(0, SWEEP_DURATION_MS, trace_len, endpoint=False)
    return raw_time - STIMULATOR_LATENCY_CORRECTION_MS


def display_signal(trace: np.ndarray) -> np.ndarray:
    return -trace if INVERT_DISPLAY else trace


def stack_traces(traces: List[np.ndarray]) -> np.ndarray:
    if not traces:
        raise ValueError("No traces provided.")
    min_len = min(len(t) for t in traces)
    return np.vstack([t[:min_len] for t in traces])


def nearest_index(time_ms: np.ndarray, latency_ms: float) -> int:
    return int(np.argmin(np.abs(time_ms - latency_ms)))


def amplitude_at_latency(time_ms: np.ndarray, waveform: np.ndarray, latency_ms: float) -> float:
    return float(waveform[nearest_index(time_ms, latency_ms)])


def detect_trace_p100_latency(
    time_ms: np.ndarray,
    display_trace_waveform: np.ndarray,
    system_p100_latency_ms: float,
) -> Tuple[float, float]:
    """
    Detect candidate P100 in an individual trace.

    The code searches +/- P100_TRACE_SEARCH_HALF_WINDOW_MS around the system-derived
    P100 latency and selects the local minimum in the displayed waveform, because the
    NATUS-style display used in this project treats P100 as a downward deflection.
    """
    lo = system_p100_latency_ms - P100_TRACE_SEARCH_HALF_WINDOW_MS
    hi = system_p100_latency_ms + P100_TRACE_SEARCH_HALF_WINDOW_MS
    mask = (time_ms >= lo) & (time_ms <= hi)
    if not np.any(mask):
        return np.nan, np.nan

    segment = display_trace_waveform[mask]
    segment_time = time_ms[mask]
    local_idx = int(np.argmin(segment))
    return float(segment_time[local_idx]), float(segment[local_idx])


def safe_name(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "_", text).strip("_")


def recording_output_paths(volunteer_dir: Path, recording_file: Path) -> Tuple[Path, Path]:
    """Return the expected per-recording CSV output paths."""
    base = f"{safe_name(volunteer_dir.name)}__{safe_name(recording_file.stem)}__OzFz"
    decisions_csv = RESULTS_DIR / "trace_decisions" / f"{base}_trace_p100_decisions.csv"
    averages_csv = RESULTS_DIR / "average_waveforms" / f"{base}_averages.csv"
    return decisions_csv, averages_csv


def volunteer_report_path(volunteer_dir: Path) -> Path:
    """Return the expected volunteer PDF report path."""
    return RESULTS_DIR / "volunteer_reports" / (
        f"{safe_name(volunteer_dir.name)}__p100_system_marker_report.pdf"
    )


def volunteer_is_fully_processed(volunteer_dir: Path, recording_files: List[Path]) -> bool:
    """
    A volunteer is considered complete only when:
      1. the volunteer PDF report exists and is non-empty; and
      2. both expected CSV outputs exist and are non-empty for every text file.
    """
    pdf_path = volunteer_report_path(volunteer_dir)
    if not pdf_path.exists() or pdf_path.stat().st_size == 0:
        return False

    for recording_file in recording_files:
        decisions_csv, averages_csv = recording_output_paths(volunteer_dir, recording_file)
        for output_path in (decisions_csv, averages_csv):
            if not output_path.exists() or output_path.stat().st_size == 0:
                return False

    return True


def read_existing_csv_rows(path: Path) -> Tuple[List[str], List[List[str]]]:
    """Read an existing CSV header and rows; return empty values when unavailable."""
    if not path.exists() or path.stat().st_size == 0:
        return [], []

    try:
        with path.open("r", newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            rows = list(reader)
    except (OSError, UnicodeError, csv.Error):
        return [], []

    if not rows:
        return [], []
    return rows[0], rows[1:]


# =========================
# OUTPUT HELPERS: This section saves the analysis results, including the trace acceptance/rejection decisions and
# initial and refined average waveforms, as CSV files. It also generates the four-panel graphical report for each
# recording.
# =========================

def save_trace_decisions_csv(
    path: Path,
    time_ms: np.ndarray,
    all_display_traces: np.ndarray,
    system_p100_latency_ms: float,
    retained_mask: np.ndarray,
) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "trace_number",
            "detected_p100_latency_ms",
            "detected_p100_amplitude_uV_display",
            "difference_from_system_p100_ms",
            "retained",
        ])
        for idx, tr in enumerate(all_display_traces, start=1):
            lat, amp = detect_trace_p100_latency(time_ms, tr, system_p100_latency_ms)
            writer.writerow([
                idx,
                lat,
                amp,
                lat - system_p100_latency_ms if np.isfinite(lat) else "",
                bool(retained_mask[idx - 1]),
            ])


def save_average_waveforms_csv(
    path: Path,
    time_ms: np.ndarray,
    initial_average: np.ndarray,
    refined_average: Optional[np.ndarray],
) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if refined_average is None:
            writer.writerow(["corrected_latency_ms", "initial_average_uV_display"])
            for t, a in zip(time_ms, initial_average):
                writer.writerow([t, a])
        else:
            writer.writerow(["corrected_latency_ms", "initial_average_uV_display", "refined_average_uV_display"])
            for t, a, r in zip(time_ms, initial_average, refined_average):
                writer.writerow([t, a, r])


def plot_recording_page(
    pdf: PdfPages,
    volunteer_name: str,
    recording_name: str,
    time_ms: np.ndarray,
    all_display_traces: np.ndarray,
    initial_average: np.ndarray,
    refined_average: Optional[np.ndarray],
    markers: MeanSystemMarkers,
    retained_mask: np.ndarray,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=PDF_FIGSIZE)
    fig.suptitle(
        f"{volunteer_name} | {recording_name} | Channel 2 Oz-Fz\n"
        f"System markers used | 12 ms latency-corrected trace axis | NATUS-style display polarity",
        fontsize=12,
    )

    # Panel 1: all traces + initial average + system markers.
    ax = axes[0, 0]
    for tr in all_display_traces:
        ax.plot(time_ms, tr, linewidth=0.35, alpha=0.15)
    ax.plot(time_ms, initial_average, linewidth=2.0, label="Initial average")
    ax.axhline(0, linewidth=0.8)
    for name, lat in [("N75", markers.n75_latency_ms), ("P100", markers.p100_latency_ms), ("N145", markers.n145_latency_ms)]:
        amp = amplitude_at_latency(time_ms, initial_average, lat)
        ax.axvline(lat, linestyle="--", linewidth=0.9)
        ax.plot(lat, amp, marker="o", markersize=4)
        ax.text(lat, amp, f" {name}\n {lat:.1f} ms", fontsize=7)
    ax.set_title(f"All traces and initial average (n={len(all_display_traces)})")
    ax.set_xlabel("Corrected latency (ms)")
    ax.set_ylabel(r"Amplitude (uV)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2)

    # Panel 2: initial average with system marker values.
    ax = axes[0, 1]
    ax.plot(time_ms, initial_average, linewidth=2.0)
    ax.axhline(0, linewidth=0.8)
    marker_info = [
        ("N75", markers.n75_latency_ms, markers.n75_amplitude_uV_raw),
        ("P100", markers.p100_latency_ms, markers.p100_amplitude_uV_raw),
        ("N145", markers.n145_latency_ms, markers.n145_amplitude_uV_raw),
    ]
    for name, lat, amp_raw in marker_info:
        amp_plot = amplitude_at_latency(time_ms, initial_average, lat)
        ax.axvline(lat, linestyle="--", linewidth=1.0)
        ax.plot(lat, amp_plot, marker="o", markersize=4)
        ax.text(lat, amp_plot, f" {name}\n {lat:.1f} ms", fontsize=8)
    ax.set_title(
        f"Mean system markers from {markers.n_runs} run(s)\n"
        f"System N75-P100 amplitude = {markers.n75_p100_amplitude_uV:.2f} uV"
    )
    ax.set_xlabel("Corrected latency (ms)")
    ax.set_ylabel(r"Amplitude (uV)")
    ax.grid(True, alpha=0.2)

    # Panel 3: retained/rejected decisions
    ax = axes[1, 0]
    detected_latencies = []
    for tr in all_display_traces:
        lat, _ = detect_trace_p100_latency(time_ms, tr, markers.p100_latency_ms)
        detected_latencies.append(lat)
    detected_latencies = np.array(detected_latencies, dtype=float)
    trace_numbers = np.arange(1, len(detected_latencies) + 1)
    ax.scatter(trace_numbers[retained_mask], detected_latencies[retained_mask], s=12, label="Retained")
    ax.scatter(trace_numbers[~retained_mask], detected_latencies[~retained_mask], s=12, label="Rejected")
    ax.axhline(markers.p100_latency_ms, linestyle="-", linewidth=1.0, label="System P100")
    ax.axhline(markers.p100_latency_ms - P100_ACCEPTANCE_TOLERANCE_MS, linestyle="--", linewidth=0.8)
    ax.axhline(markers.p100_latency_ms + P100_ACCEPTANCE_TOLERANCE_MS, linestyle="--", linewidth=0.8)
    ax.set_title(f"P100 latency selection: retained {retained_mask.sum()} / {len(retained_mask)}")
    ax.set_xlabel("Trace number")
    ax.set_ylabel("Detected P100 latency (ms)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2)

    # Panel 4: initial vs refined average.
    ax = axes[1, 1]
    ax.plot(time_ms, initial_average, linewidth=2.0, label="Initial average")
    if refined_average is not None:
        ax.plot(time_ms, refined_average, linewidth=2.0, label="Refined average")
    ax.axhline(0, linewidth=0.8)
    ax.axvline(markers.p100_latency_ms, linestyle="--", linewidth=0.9, label="System P100")
    ax.set_title("Initial vs refined average")
    ax.set_xlabel("Corrected latency (ms)")
    ax.set_ylabel(r"Amplitude (uV)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2)

    fig.tight_layout(rect=[0, 0, 1, 0.92])
    pdf.savefig(fig)
    plt.close(fig)


# =========================
# PROCESSING: This section performs the main VEP analysis for each recording. It calculates the initial average,
# detects the P100 of each individual trace, retains traces within +/-5ms of the system P100 latency, calculates the
# refined average, and saves the results.
# =========================

def process_recording(
    volunteer_dir: Path,
    recording_file: Path,
    volunteer_pdf: PdfPages,
    summary_writer: csv.writer,
    error_writer: csv.writer,
) -> None:
    print(f"  Processing: {recording_file.name}")
    try:
        text = read_text_file(recording_file)
        sweeps = extract_channel_sweeps(text, CHANNEL_TO_EXTRACT)
        marker_sets = extract_system_marker_sets_for_channel2(text)

        if not sweeps:
            raise RuntimeError("No Channel 2 Oz-Fz trace sweeps found.")
        if not marker_sets:
            raise RuntimeError("No Channel 2 Oz-Fz system marker values found.")

        markers = mean_system_markers(marker_sets)

        traces = [trace for sweep in sweeps for trace in sweep]
        raw_arr = stack_traces(traces)
        display_arr = display_signal(raw_arr)
        time_ms = make_time_axis(display_arr.shape[1])
        initial_average = display_arr.mean(axis=0)

        detected_latencies = []
        detected_amplitudes = []
        for tr in display_arr:
            lat, amp = detect_trace_p100_latency(time_ms, tr, markers.p100_latency_ms)
            detected_latencies.append(lat)
            detected_amplitudes.append(amp)
        detected_latencies = np.array(detected_latencies, dtype=float)

        retained_mask = np.abs(detected_latencies - markers.p100_latency_ms) <= P100_ACCEPTANCE_TOLERANCE_MS
        n_retained = int(retained_mask.sum())
        n_rejected = int(len(retained_mask) - n_retained)

        refined_average = None
        refined_n75_p100_amp = ""
        if n_retained > 0:
            refined_average = display_arr[retained_mask].mean(axis=0)
            refined_n75 = amplitude_at_latency(time_ms, refined_average, markers.n75_latency_ms)
            refined_p100 = amplitude_at_latency(time_ms, refined_average, markers.p100_latency_ms)
            refined_n75_p100_amp = abs(refined_p100 - refined_n75)

        decisions_csv, averages_csv = recording_output_paths(
            volunteer_dir, recording_file
        )

        save_trace_decisions_csv(decisions_csv, time_ms, display_arr, markers.p100_latency_ms, retained_mask)
        save_average_waveforms_csv(averages_csv, time_ms, initial_average, refined_average)

        plot_recording_page(
            pdf=volunteer_pdf,
            volunteer_name=volunteer_dir.name,
            recording_name=recording_file.name,
            time_ms=time_ms,
            all_display_traces=display_arr,
            initial_average=initial_average,
            refined_average=refined_average,
            markers=markers,
            retained_mask=retained_mask,
        )

        summary_writer.writerow([
            volunteer_dir.name,
            recording_file.name,
            len(sweeps),
            len(marker_sets),
            len(traces),
            n_retained,
            n_rejected,
            markers.n75_latency_ms,
            markers.p100_latency_ms,
            markers.n145_latency_ms,
            markers.n75_amplitude_uV_raw,
            markers.p100_amplitude_uV_raw,
            markers.n145_amplitude_uV_raw,
            markers.n75_p100_amplitude_uV,
            refined_n75_p100_amp,
            decisions_csv.name,
            averages_csv.name,
        ])

        print(f"    OK | traces={len(traces)}, retained={n_retained}, system P100={markers.p100_latency_ms:.2f} ms")
        if n_retained < MIN_RETAINED_TRACES:
            print(f"    WARNING: only {n_retained} retained traces; refined average may be unreliable.")

    except Exception as exc:
        print(f"    ERROR: {exc}")
        error_writer.writerow([volunteer_dir.name, recording_file.name, str(exc)])


# =========================
# FINAL RESULTS TABLE: This section reshapes the per-recording master summary into one row per volunteer.
# It assumes that only volunteers who have already passed the required screening/ophthalmological assessment
# are included in the input dataset. No normal/abnormal classification is performed here.
# =========================

def create_final_vep_results_csv(summary_csv_path: Path) -> Path:
    """
    Create the final VEP results dataset with one row per volunteer.

    All volunteers present in the analysed dataset are included.
    No normal/abnormal classification is performed in this script.

    Final values:
        P100 latency = mean NATUS/system P100 latency across repeated runs.
        N75-P100 amplitude = amplitude measured from the refined average
                             after trace rejection.

    The final CSV contains:
        LT 12x16 P100 latency and N75-P100 amplitude
        RT 12x16 P100 latency and N75-P100 amplitude
        LT 48x64 P100 latency and N75-P100 amplitude
        RT 48x64 P100 latency and N75-P100 amplitude
    """

    final_csv_path = RESULTS_DIR / "VEP_FINAL_RESULTS.csv"

    with summary_csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        summary_rows = list(reader)

    volunteer_data = {}

    for row in summary_rows:
        volunteer = row["volunteer_folder"]
        recording_file = Path(row["recording_file"]).stem

        # Expected filename formats include:
        # V001_L_12x16
        # V001_R_12x16
        # V001_L_48x64
        # V001_R_48x64
        #
        # LT and RT are also accepted.
        match = re.search(
            r"_(L|R|LT|RT)_(12x16|48x64)$",
            recording_file,
            flags=re.IGNORECASE,
        )

        if not match:
            print(
                f"WARNING: Could not identify eye/check size from "
                f"{row['recording_file']}"
            )
            continue

        eye = match.group(1).upper()
        check_size = match.group(2).lower()

        if eye in {"L", "LT"}:
            eye = "LT"
        else:
            eye = "RT"

        key = f"{eye}_{check_size}"

        try:
            p100_latency = float(row["system_mean_p100_latency_ms"])
            amplitude = float(row["refined_n75_p100_amplitude_uV_display"])
        except (ValueError, TypeError, KeyError):
            print(
                f"WARNING: Missing final result for "
                f"{volunteer} | {recording_file}"
            )
            continue

        volunteer_data.setdefault(volunteer, {})

        volunteer_data[volunteer][key] = {
            "p100_latency_ms": p100_latency,
            "n75_p100_amplitude_uV": amplitude,
        }

    final_rows = []

    required_recordings = [
        "LT_12x16",
        "RT_12x16",
        "LT_48x64",
        "RT_48x64",
    ]

    for volunteer, recordings in sorted(volunteer_data.items()):
        if not all(key in recordings for key in required_recordings):
            missing = [
                key for key in required_recordings
                if key not in recordings
            ]
            print(
                f"WARNING: {volunteer} does not have all four required "
                f"recordings and will not be included in VEP_FINAL_RESULTS.csv. "
                f"Missing: {', '.join(missing)}"
            )
            continue

        final_rows.append([
            volunteer,

            recordings["LT_12x16"]["p100_latency_ms"],
            recordings["LT_12x16"]["n75_p100_amplitude_uV"],

            recordings["RT_12x16"]["p100_latency_ms"],
            recordings["RT_12x16"]["n75_p100_amplitude_uV"],

            recordings["LT_48x64"]["p100_latency_ms"],
            recordings["LT_48x64"]["n75_p100_amplitude_uV"],

            recordings["RT_48x64"]["p100_latency_ms"],
            recordings["RT_48x64"]["n75_p100_amplitude_uV"],
        ])

    header = [
        "volunteer",

        "LT_12x16_P100_latency_ms",
        "LT_12x16_N75_P100_amplitude_uV",

        "RT_12x16_P100_latency_ms",
        "RT_12x16_N75_P100_amplitude_uV",

        "LT_48x64_P100_latency_ms",
        "LT_48x64_N75_P100_amplitude_uV",

        "RT_48x64_P100_latency_ms",
        "RT_48x64_N75_P100_amplitude_uV",
    ]

    with final_csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(final_rows)

    print(f"\nFinal VEP results CSV created: {final_csv_path}")
    print(f"Volunteers included in final table: {len(final_rows)}")

    return final_csv_path


# =========================
# MAIN: This section controls the overall batch-processing workflow. It finds all volunteer folders, checks which
# volunteers have already been processed, analyses the remaining volunteers automatically, and creates the PDF reports
# and master summary files.
# =========================

def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "volunteer_reports").mkdir(exist_ok=True)
    (RESULTS_DIR / "trace_decisions").mkdir(exist_ok=True)
    (RESULTS_DIR / "average_waveforms").mkdir(exist_ok=True)

    volunteer_dirs = sorted([p for p in DATA_DIR.glob("V*") if p.is_dir()])
    if not volunteer_dirs:
        raise RuntimeError(f"No volunteer folders found in {DATA_DIR}")

    summary_csv_path = RESULTS_DIR / "ALL_VOLUNTEERS_summary.csv"
    errors_csv_path = RESULTS_DIR / "ALL_VOLUNTEERS_errors.csv"

    summary_header = [
        "volunteer_folder",
        "recording_file",
        "n_trace_sweep_sections",
        "n_system_marker_runs",
        "n_all_traces",
        "n_retained_traces",
        "n_rejected_traces",
        "system_mean_n75_latency_ms",
        "system_mean_p100_latency_ms",
        "system_mean_n145_latency_ms",
        "system_mean_n75_amplitude_uV_raw",
        "system_mean_p100_amplitude_uV_raw",
        "system_mean_n145_amplitude_uV_raw",
        "system_mean_n75_p100_amplitude_uV",
        "refined_n75_p100_amplitude_uV_display",
        "trace_decisions_csv",
        "average_waveforms_csv",
    ]
    error_header = ["volunteer_folder", "recording_file", "error"]

    # Deciding which volunteers are complete before rewriting the master CSV files.
    volunteer_files = {
        volunteer_dir.name: sorted(volunteer_dir.glob("*.txt"))
        for volunteer_dir in volunteer_dirs
    }
    completed_volunteer_names = {
        volunteer_dir.name
        for volunteer_dir in volunteer_dirs
        if (
            SKIP_ALREADY_PROCESSED
            and not FORCE_REPROCESS
            and volunteer_files[volunteer_dir.name]
            and volunteer_is_fully_processed(
                volunteer_dir, volunteer_files[volunteer_dir.name]
            )
        )
    }

    # Preserving existing master rows only for volunteers that will be skipped.
    _, existing_summary_rows = read_existing_csv_rows(summary_csv_path)
    _, existing_error_rows = read_existing_csv_rows(errors_csv_path)

    preserved_summary_rows = [
        row for row in existing_summary_rows
        if row and row[0] in completed_volunteer_names
    ]
    preserved_error_rows = [
        row for row in existing_error_rows
        if row and row[0] in completed_volunteer_names
    ]

    print("Batch VEP Trace Filtering")
    print("================================================================")
    print(f"Project directory: {PROJECT_DIR}")
    print(f"Data directory:    {DATA_DIR}")
    print(f"Results directory: {RESULTS_DIR}")
    print(f"Volunteers found:  {len(volunteer_dirs)}")
    print(f"Already complete:  {len(completed_volunteer_names)}")
    if FORCE_REPROCESS:
        print("FORCE_REPROCESS=True: all volunteers will be processed again.")
    print()

    processed_count = 0
    skipped_count = 0

    with summary_csv_path.open("w", newline="", encoding="utf-8") as fsum, \
            errors_csv_path.open("w", newline="", encoding="utf-8") as ferr:

        summary_writer = csv.writer(fsum)
        error_writer = csv.writer(ferr)

        summary_writer.writerow(summary_header)
        error_writer.writerow(error_header)
        summary_writer.writerows(preserved_summary_rows)
        error_writer.writerows(preserved_error_rows)

        for volunteer_idx, volunteer_dir in enumerate(volunteer_dirs, start=1):
            recording_files = volunteer_files[volunteer_dir.name]
            print(
                f"[{volunteer_idx}/{len(volunteer_dirs)}] "
                f"{volunteer_dir.name} | files={len(recording_files)}"
            )

            if not recording_files:
                error_writer.writerow([
                    volunteer_dir.name,
                    "",
                    "No .txt files found in volunteer folder.",
                ])
                continue

            if volunteer_dir.name in completed_volunteer_names:
                skipped_count += 1
                print("  SKIPPED: PDF report and all recording CSV outputs already exist.")
                continue

            pdf_path = volunteer_report_path(volunteer_dir)
            with PdfPages(pdf_path) as volunteer_pdf:
                for recording_file in recording_files:
                    process_recording(
                        volunteer_dir=volunteer_dir,
                        recording_file=recording_file,
                        volunteer_pdf=volunteer_pdf,
                        summary_writer=summary_writer,
                        error_writer=error_writer,
                    )

            processed_count += 1

    # Creating the final one-row-per-volunteer results table.
    final_results_csv_path = create_final_vep_results_csv(summary_csv_path)

    print("\nDone.")
    print(f"Volunteers processed this run: {processed_count}")
    print(f"Volunteers skipped:            {skipped_count}")
    print(f"Master summary CSV: {summary_csv_path}")
    print(f"Final VEP results CSV: {final_results_csv_path}")
    print(f"Errors CSV:         {errors_csv_path}")
    print(f"Volunteer reports:  {RESULTS_DIR / 'volunteer_reports'}")
    print(
        "\nTo rerun everything deliberately, set FORCE_REPROCESS = True "
        "near the top of the script."
    )


if __name__ == "__main__":
    main()