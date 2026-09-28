#!/usr/bin/env python3
"""Render the paper's main tables and Setting B ablation from saved seed results.

Uses NumPy and the numeric CSVs in ``results/``.
All reported table entries are means over 100 independent seeds per cell.
"""

from __future__ import annotations

import argparse
import csv
import math
import statistics
from collections import defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
STRENGTHS = ("0.3", "0.5", "0.8")
METHODS = ("raw", "ae", "vae", "contrastive", "balanced", "ldm", "sprl")
SUMMARY_METHODS = ("raw", "balanced", "ldm", "sprl")
LAMBDA_METHODS = ("ldm", "sprl_lambda_0.5", "sprl", "sprl_lambda_2.0", "sprl_lambda_5.0")
T_99_975 = 1.9842169515086827

# Rounded entries transcribed from the corrected camera-ready main Tables 1–3.
# These constants are deliberately independent of the CSVs; --check catches drift.
EXPECTED_A = {
    "raw": ((100, "1.000"), (100, "1.000"), (100, "1.000")),
    "ae": ((19, "1.527"), (45, "1.718"), (64, "2.193")),
    "vae": ((0, "1.177"), (39, "1.182"), (100, "1.190")),
    "contrastive": ((7, "1.221"), (48, "1.249"), (97, "1.304")),
    "balanced": ((100, "0.997"), (100, "0.999"), (100, "1.001")),
    "ldm": ((100, "0.988"), (100, "0.991"), (100, "0.994")),
    "sprl": ((100, "1.001"), (100, "1.016"), (100, "1.053")),
}
EXPECTED_B = {
    "raw": (("1.00", "1.00", "0.990"), ("1.00", "1.00", "0.990"), ("1.00", "1.00", "0.991")),
    "ae": (("1.37", "2.25", "0.986"), ("1.32", "2.19", "0.986"), ("1.29", "2.16", "0.987")),
    "vae": (("1.19", "10.23", "0.987"), ("1.18", "8.76", "0.988"), ("1.05", "5.69", "0.989")),
    "contrastive": (("1.60", "4.99", "0.984"), ("1.50", "4.45", "0.985"), ("1.38", "4.09", "0.986")),
    "balanced": (("4.45", "9.33", "0.595"), ("4.30", "8.89", "0.579"), ("4.17", "9.10", "0.574")),
    "ldm": (("2.88", "4.92", "0.968"), ("2.83", "4.76", "0.969"), ("2.93", "5.01", "0.961")),
    "sprl": (("1.45", "2.66", "0.986"), ("1.26", "2.32", "0.991"), ("1.27", "2.48", "0.990")),
}
EXPECTED_3 = {
    "raw": ("1.00", "1.08", "0.88"),
    "balanced": ("4.30", "1.49", "1.20"),
    "ldm": ("2.83", "1.36", "1.62"),
    "sprl": ("1.26", "0.73", "1.10"),
}


def read_rows(name: str) -> list[dict[str, str]]:
    with (RESULTS / name).open(newline="") as file:
        return list(csv.DictReader(file))


def grouped(rows: list[dict[str, str]], *keys: str) -> dict[tuple[str, ...], list[dict[str, str]]]:
    out: dict[tuple[str, ...], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        out[tuple(row[key] for key in keys)].append(row)
    for group in out.values():
        group.sort(key=lambda row: int(row["seed"]))
    return out


def mean(rows: list[dict[str, str]], field: str) -> float:
    return statistics.mean(float(row[field]) for row in rows)


def truth(value: str) -> bool:
    if value not in ("True", "False"):
        raise ValueError(f"Unexpected Boolean value: {value!r}")
    return value == "True"


def mean_ci(values: list[float]) -> tuple[float, float, float]:
    """Pointwise two-sided 95% t interval over 100 independent seeds."""
    center = statistics.mean(values)
    half_width = T_99_975 * statistics.stdev(values) / math.sqrt(len(values))
    return center, center - half_width, center + half_width


def numeric(rows: list[dict[str, str]], field: str) -> list[float]:
    return [float(row[field]) for row in rows]


def paired_differences(left: list[dict[str, str]], right: list[dict[str, str]],
                       field: str) -> tuple[float, float, float]:
    left_by_seed = {int(row["seed"]): float(row[field]) for row in left}
    right_by_seed = {int(row["seed"]): float(row[field]) for row in right}
    return mean_ci([left_by_seed[seed] - right_by_seed[seed] for seed in sorted(left_by_seed)])


def wilson_interval(count: int, n: int) -> tuple[float, float]:
    z = 1.959963984540054
    p = count / n
    denominator = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denominator
    half_width = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return center - half_width, center + half_width


def paired_ratio_bootstrap(left: list[float], right: list[float], seed: int) -> tuple[float, float]:
    """Original 10,000-draw NumPy paired ratio-of-means bootstrap."""
    numerator = np.asarray(left, dtype=float)
    denominator = np.asarray(right, dtype=float)
    picks = np.random.default_rng(seed).integers(0, len(numerator), size=(10_000, len(numerator)))
    ratios = numerator[picks].mean(axis=1) / denominator[picks].mean(axis=1)
    low, high = np.quantile(ratios, [0.025, 0.975])
    return float(low), float(high)


def paired_difference_bootstrap(left: list[float], right: list[float], seed: int
                                ) -> tuple[float, float, float, float]:
    """Original 10,000-draw NumPy paired mean and median difference bootstrap."""
    differences = np.asarray(left, dtype=float) - np.asarray(right, dtype=float)
    picks = np.random.default_rng(seed).integers(0, len(differences), size=(10_000, len(differences)))
    sampled = differences[picks]
    mean_low, mean_high = np.quantile(sampled.mean(axis=1), [0.025, 0.975])
    median_low, median_high = np.quantile(np.median(sampled, axis=1), [0.025, 0.975])
    return float(mean_low), float(mean_high), float(median_low), float(median_high)


def validate(rows: list[dict[str, str]], keys: tuple[str, ...], expected_methods: tuple[str, ...],
             expected_strengths: tuple[str, ...], expected_seed_start: int) -> None:
    groups = grouped(rows, *keys)
    expected = {(w, method) for w in expected_strengths for method in expected_methods}
    if set(groups) != expected:
        raise ValueError(f"Unexpected cells: missing={expected - set(groups)}, extra={set(groups) - expected}")
    for cell, group in groups.items():
        seeds = sorted(int(row["seed"]) for row in group)
        required = list(range(expected_seed_start, expected_seed_start + 100))
        if seeds != required:
            raise ValueError(f"Cell {cell} does not contain each expected seed exactly once")


def build_tables() -> dict[str, list[dict[str, object]]]:
    a = read_rows("setting_a_per_seed.csv")
    b = read_rows("setting_b_per_seed.csv")
    semi = read_rows("benchmark_per_seed.csv")
    validate(a, ("w_u", "method"), METHODS, STRENGTHS, 1)
    validate(b, ("w_u", "method"),
             METHODS + tuple(method for method in LAMBDA_METHODS if method not in METHODS), STRENGTHS, 1)
    benchmark_groups = grouped(semi, "benchmark", "w_u", "method")
    required_benchmarks = {("ihdp", w, m) for w in STRENGTHS for m in SUMMARY_METHODS}
    required_benchmarks |= {("halfcheetah", "0.5", m) for m in SUMMARY_METHODS}
    if set(benchmark_groups) != required_benchmarks:
        raise ValueError("Unexpected benchmark conditions or methods")
    for cell, group in benchmark_groups.items():
        if sorted(int(row["seed"]) for row in group) != list(range(101, 201)):
            raise ValueError(f"Incomplete benchmark cell {cell}")

    a_groups = grouped(a, "w_u", "method")
    b_groups = grouped(b, "w_u", "method")
    table_1 = []
    table_2 = []
    for method in METHODS:
        row_a: dict[str, object] = {"method": method}
        row_b: dict[str, object] = {"method": method}
        for w in STRENGTHS:
            a_cell = a_groups[(w, method)]
            raw_width = mean(a_groups[(w, "raw")], "interval_width")
            row_a[f"incl_{w}"] = sum(truth(r["includes_target"]) for r in a_cell)
            row_a[f"width_ratio_{w}"] = mean(a_cell, "interval_width") / raw_width
            b_cell = b_groups[(w, method)]
            row_b[f"median_{w}"] = mean(b_cell, "amplification_median")
            row_b[f"p95_{w}"] = mean(b_cell, "amplification_p95")
            row_b[f"u_auc_{w}"] = mean(b_cell, "u_auc")
        table_1.append(row_a)
        table_2.append(row_b)

    table_3 = []
    for method in SUMMARY_METHODS:
        table_3.append({"method": method,
            "setting_b": mean(b_groups[("0.5", method)], "amplification_median"),
            "ihdp": mean(benchmark_groups[("ihdp", "0.5", method)], "fresh_amplification_median"),
            "halfcheetah": mean(benchmark_groups[("halfcheetah", "0.5", method)], "fresh_amplification_median")})

    ablation = []
    for method, lam in zip(LAMBDA_METHODS, ("0", "0.5", "1", "2", "5")):
        cell = b_groups[("0.3", method)]
        ablation.append({"lambda_prop": lam, "median_mean": mean(cell, "amplification_median"),
            "median_se": statistics.stdev(float(r["amplification_median"]) for r in cell) / math.sqrt(100),
            "p95_mean": mean(cell, "amplification_p95"),
            "transition_mse_mean": mean(cell, "transition_mse")})
    figure_strength = []
    for method in ("ldm", "sprl"):
        for w in STRENGTHS:
            cell = b_groups[(w, method)]
            figure_strength.append({"method": method, "w_u": w,
                "median_mean": mean(cell, "amplification_median"),
                "median_se": statistics.stdev(float(r["amplification_median"]) for r in cell) / math.sqrt(100),
                "u_auc_mean": mean(cell, "u_auc"),
                "u_auc_se": statistics.stdev(float(r["u_auc"]) for r in cell) / math.sqrt(100)})

    a_endpoints = []
    a_paired = []
    for w in STRENGTHS:
        for method in METHODS:
            cell = a_groups[(w, method)]
            inclusion = sum(truth(r["includes_target"]) for r in cell)
            inc_low, inc_high = wilson_interval(inclusion, 100)
            width, width_low, width_high = mean_ci(numeric(cell, "interval_width"))
            ratio_low, ratio_high = paired_ratio_bootstrap(
                numeric(cell, "interval_width"), numeric(a_groups[(w, "raw")], "interval_width"),
                20260925 + int(10 * float(w)))
            a_endpoints.append({"w_u": w, "method": method,
                "lower_mean": mean(cell, "interval_lower"),
                "upper_mean": mean(cell, "interval_upper"),
                "width_mean": width, "width_ci95_lower": width_low,
                "width_ci95_upper": width_high,
                "width_ratio_to_raw": width / mean(a_groups[(w, "raw")], "interval_width"),
                "width_ratio_bootstrap95_lower": ratio_low,
                "width_ratio_bootstrap95_upper": ratio_high, "inclusion_count": inclusion,
                "inclusion_wilson95_lower": inc_low, "inclusion_wilson95_upper": inc_high})
        for comparator in ("ldm", "balanced"):
            center, low, high = paired_differences(a_groups[(w, "sprl")],
                                                    a_groups[(w, comparator)], "interval_width")
            a_paired.append({"w_u": w, "comparison": f"sprl_minus_{comparator}",
                             "width_difference_mean": center, "ci95_lower": low, "ci95_upper": high})

    benchmark_methods = []
    benchmark_paired = []
    for benchmark, w in (("ihdp", "0.3"), ("ihdp", "0.5"), ("ihdp", "0.8"),
                         ("halfcheetah", "0.5")):
        raw = benchmark_groups[(benchmark, w, "raw")]
        raw_width = mean(raw, "interval_width") if benchmark == "ihdp" else None
        for method in SUMMARY_METHODS:
            cell = benchmark_groups[(benchmark, w, method)]
            amp, amp_low, amp_high = mean_ci(numeric(cell, "fresh_amplification_median"))
            p95 = numeric(cell, "fresh_amplification_p95")
            p95_mean, p95_low, p95_high = mean_ci(p95)
            reward, reward_low, reward_high = (("", "", "") if method == "raw" else
                                               mean_ci(numeric(cell, "fresh_reward_mse")))
            entry: dict[str, object] = {"benchmark": benchmark, "w_u": w, "method": method,
                "amplification_median_mean": amp, "amplification_median_ci95_lower": amp_low,
                "amplification_median_ci95_upper": amp_high,
                "amplification_p95_mean": p95_mean,
                "amplification_p95_ci95_lower": p95_low,
                "amplification_p95_ci95_upper": p95_high,
                "amplification_p95_seed_median": statistics.median(p95),
                "reward_mse_mean": reward, "reward_mse_ci95_lower": reward_low,
                "reward_mse_ci95_upper": reward_high}
            if benchmark == "ihdp":
                inclusion = sum(truth(r["includes_analytic_target"]) for r in cell)
                inc_low, inc_high = wilson_interval(inclusion, 100)
                width, width_low, width_high = mean_ci(numeric(cell, "interval_width"))
                ratio_low, ratio_high = paired_ratio_bootstrap(
                    numeric(cell, "interval_width"), numeric(raw, "interval_width"),
                    20260926 + round(float(w) * 10) * 100 + SUMMARY_METHODS.index(method))
                entry.update({"inclusion_count": inclusion,
                    "inclusion_wilson95_lower": inc_low, "inclusion_wilson95_upper": inc_high,
                    "width_mean": width, "width_ci95_lower": width_low,
                    "width_ci95_upper": width_high, "width_ratio_to_raw": width / raw_width,
                    "width_ratio_bootstrap95_lower": ratio_low,
                    "width_ratio_bootstrap95_upper": ratio_high})
            else:
                entry.update({"inclusion_count": "", "inclusion_wilson95_lower": "",
                    "inclusion_wilson95_upper": "", "width_mean": "", "width_ci95_lower": "",
                    "width_ci95_upper": "", "width_ratio_to_raw": "",
                    "width_ratio_bootstrap95_lower": "", "width_ratio_bootstrap95_upper": ""})
            benchmark_methods.append(entry)
        for comparator in ("ldm", "balanced"):
            sprl = benchmark_groups[(benchmark, w, "sprl")]
            other = benchmark_groups[(benchmark, w, comparator)]
            for field in ("fresh_amplification_median", "fresh_amplification_p95",
                          "fresh_reward_mse") + (("interval_width",) if benchmark == "ihdp" else ()):
                center, low, high = paired_differences(sprl, other, field)
                paired_seed_median = statistics.median(
                    float(a[field]) - float(b[field]) for a, b in zip(sprl, other))
                boot_mean_low, boot_mean_high, boot_median_low, boot_median_high = paired_difference_bootstrap(
                    numeric(sprl, field), numeric(other, field),
                    20260926 + round(float(w) * 10) * 1000 +
                    SUMMARY_METHODS.index(comparator) * 100 + sum(ord(char) for char in field))
                benchmark_paired.append({"benchmark": benchmark, "w_u": w,
                    "comparison": f"sprl_minus_{comparator}", "metric": field,
                    "difference_mean": center, "ci95_lower": low, "ci95_upper": high,
                    "paired_mean_bootstrap95_lower": boot_mean_low,
                    "paired_mean_bootstrap95_upper": boot_mean_high,
                    "paired_seed_median_difference": paired_seed_median,
                    "paired_median_bootstrap95_lower": boot_median_low,
                    "paired_median_bootstrap95_upper": boot_median_high})
    return {"table_1_setting_a": table_1, "table_2_setting_b": table_2,
            "table_3_summary": table_3, "figure_4a_lambda_sweep": ablation,
            "figure_4bc_strength": figure_strength,
            "appendix_setting_a_endpoints": a_endpoints,
            "appendix_setting_a_paired_width": a_paired,
            "appendix_benchmark_methods": benchmark_methods,
            "appendix_benchmark_paired": benchmark_paired}


def check(tables: dict[str, list[dict[str, object]]]) -> None:
    for row in tables["table_1_setting_a"]:
        for w, expected in zip(STRENGTHS, EXPECTED_A[str(row["method"])]):
            actual = (row[f"incl_{w}"], f"{row[f'width_ratio_{w}']:.3f}")
            if actual != expected:
                raise AssertionError(f"Table 1 {row['method']} w={w}: {actual} != {expected}")
    for row in tables["table_2_setting_b"]:
        for w, expected in zip(STRENGTHS, EXPECTED_B[str(row["method"])]):
            actual = (f"{row[f'median_{w}']:.2f}", f"{row[f'p95_{w}']:.2f}",
                      f"{row[f'u_auc_{w}']:.3f}")
            if actual != expected:
                raise AssertionError(f"Table 2 {row['method']} w={w}: {actual} != {expected}")
    for row in tables["table_3_summary"]:
        actual = tuple(f"{row[field]:.2f}" for field in ("setting_b", "ihdp", "halfcheetah"))
        expected = EXPECTED_3[str(row["method"])]
        if actual != expected:
            raise AssertionError(f"Table 3 {row['method']}: {actual} != {expected}")
    print("Saved seed counts and camera-ready rounded Tables 1–3: PASS")


def write_tables(tables: dict[str, list[dict[str, object]]], output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    for name, rows in tables.items():
        with (output / f"{name}.csv").open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="validate seeds and rounded camera-ready Tables 1–3")
    parser.add_argument("--output", type=Path, help="write table, figure-data, and appendix CSV files here")
    args = parser.parse_args()
    tables = build_tables()
    if args.check:
        check(tables)
    if args.output:
        write_tables(tables, args.output)
        print(f"Wrote {len(tables)} CSV files to {args.output}")
    if not args.check and not args.output:
        for name, rows in tables.items():
            print(f"\n{name}")
            writer = csv.DictWriter(__import__("sys").stdout, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    main()
