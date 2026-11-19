"""Prepare five fixed train/validation prefixes only after ROOT policy closure.

Default PENDING has no project/input/output IO. CSV/TSF text tokenisation is not
a cryptographic suffix seal; only predetermined prefix target tokens are ever
converted to numbers. There is no server connection, score access or test array.
"""
import argparse
import csv
import datetime as dt
import hashlib
import io
import itertools
import json
import math
from pathlib import Path
import stat
import zipfile

IDS = ["SteelIndustry6", "MetroPT15", "AEMO5", "GEFCOM2014Wind10", "MelbournePedestrian17"]
MODELS = ["OURS", "DLinear", "SparseTSF", "ModernTCN", "MixLinear", "PhaseFormer"]
STATIC = {
    "SteelIndustry6": {"raw_file": "raw/UCI851_SteelIndustry.zip", "raw_sha256": "d82d28b33780ff1582507fcf08ae764ff648af459d58234370c551e62aadeaef", "raw_bytes": 481973,
                       "rows": 35040, "bounds": [24528, 28032, 35040], "C": 6, "seconds": 900, "first": "2018-01-01T00:15:00", "last": "2019-01-01T00:00:00",
                       "channels": ["Usage_kWh", "Lagging_Current_Reactive.Power_kVarh", "Leading_Current_Reactive_Power_kVarh", "CO2(tCO2)", "Lagging_Current_Power_Factor", "Leading_Current_Power_Factor"]},
    "MetroPT15": {"raw_file": "raw/UCI791_MetroPT3.zip", "raw_sha256": "aab991a970e58210de853bb8078ce0e63abb4d9412fdc5c79792dae3d8e1721a", "raw_bytes": 218381995,
                  "rows": 30697, "bounds": [21487, 24557, 30697], "C": 15, "seconds": 600, "first": "2020-02-01T00:00:00", "last": "2020-09-01T04:00:00",
                  "channels": ["TP2", "TP3", "H1", "DV_pressure", "Reservoirs", "Oil_temperature", "Motor_current", "COMP", "DV_eletric", "Towers", "MPG", "LPS", "Pressure_switch", "Oil_level", "Caudal_impulses"]},
    "AEMO5": {"raw_file": "raw/australian_electricity_demand_dataset.zip", "raw_sha256": "b1439c28a631766bd05ee327f1f430a887b9f53b38a8cd5c0769ded1fd4aaf5a", "raw_bytes": 5770526,
              "rows": 87648, "bounds": [61353, 70118, 87648], "C": 5, "seconds": 1800, "first": "2010-01-01T00:00:00", "last": "2014-12-31T23:30:00", "channels": ["T1", "T2", "T3", "T4", "T5"]},
    "GEFCOM2014Wind10": {"raw_file": "raw/GEFCom2014_Elsevier_mmc1.zip", "raw_sha256": "d68d957270edd93b26a37d0f9b5e901f942abdf34c75eacbe14e417beb16e154", "raw_bytes": 126360077,
                        "rows": 16800, "bounds": [11760, 13440, 16800], "C": 10, "seconds": 3600, "first": "2012-01-01T01:00:00", "last": "2013-12-01T00:00:00", "channels": [str(i) for i in range(1, 11)]},
    "MelbournePedestrian17": {"raw_file": "raw/pedestrian_counts_dataset.zip", "raw_sha256": "6e81cb8cad43650e7e0f754a1103fc75c960e03387bae9b91146aa3241c9aa50", "raw_bytes": 4587054,
                             "rows": 43824, "bounds": [30676, 35059, 43824], "C": 17, "seconds": 3600, "first": "2015-01-01T00:00:01", "last": "2019-12-31T23:00:01",
                             "channels": ["T10", "T11", "T18", "T19", "T2", "T20", "T21", "T23", "T24", "T26", "T27", "T28", "T31", "T34", "T6", "T8", "T9"]},
}

def reject_links(path):
    path = Path(path).absolute()
    for part in (path, *path.parents):
        if part.is_symlink():
            raise RuntimeError("symlink leaf/ancestor rejected")
        try:
            st = part.lstat()
        except FileNotFoundError:
            continue
        if getattr(st, "st_file_attributes", 0) & 0x400:
            raise RuntimeError("Windows reparse leaf/ancestor rejected")
    return path

def scoped(base, relative):
    base = reject_links(base)
    relative = Path(relative)
    if relative.is_absolute() or ".." in relative.parts or not relative.parts:
        raise RuntimeError("nonlocal input/output path")
    path = reject_links(base / relative)
    if not path.is_relative_to(base):
        raise RuntimeError("path outside new scope")
    return path

def file_sha(path):
    path = reject_links(path)
    if not stat.S_ISREG(path.stat().st_mode):
        raise RuntimeError("not a regular file")
    h = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()

def load_json(path):
    path = reject_links(path)
    if not stat.S_ISREG(path.stat().st_mode):
        raise RuntimeError("not a regular JSON file")
    return json.loads(path.read_text(encoding="utf-8"))

def need(condition, text):
    if not condition:
        raise RuntimeError(text)

def sha_string(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)

def verify_policy(base, policy_sha):
    need(sha_string(policy_sha), "explicit real policy SHA required")
    policy_path = scoped(base, "ROOT_NEW5_PRE_SCORE_POLICY.json")
    need(file_sha(policy_path) == policy_sha, "ROOT policy SHA mismatch")
    p = load_json(policy_path)
    need(p.get("status") == "ROOT_FIXED_FIVE_BEFORE_NEW_SCORES", "ROOT fixed policy status missing")
    need(p.get("candidate_order") == IDS and p.get("models") == MODELS, "fixed five/six model identities differ")
    need(p.get("length") == 96 and p.get("horizons") == [96, 192, 336, 720] and p.get("seed") == 2026, "fixed science geometry differs")
    for k, value in {"H96_observation_only": True, "final_strict_metric_wins_required": 6, "metric_group_count": 8,
                     "test_numerical_values_decoded": False, "all_candidates_and_negative_results_retained": True,
                     "recipe_changes_or_score_driven_retraining_allowed": False}.items():
        need(type(p.get(k)) is type(value) and p.get(k) == value, "policy guard differs: " + k)
    need(file_sha(scoped(base, "LATEST_HUMAN_NEW5_SCOPE_AUTHORIZATION.json")) == p.get("human_authorization_sha256"), "human scope receipt identity differs")
    audit_path = scoped(base, "INDEPENDENT_NEW5_SOURCE_AND_PRIOR_EXPOSURE_AUDIT.json")
    need(file_sha(audit_path) == p.get("source_and_prior_exposure_audit_sha256"), "independent source review SHA differs")
    review = load_json(audit_path)
    need(review.get("status") == "PASS_INDEPENDENT_NEW5_SOURCE_AND_PRIOR_EXPOSURE_METADATA_AUDIT"
         and review.get("candidate_order") == IDS and review.get("all_five_new_nonpems") is True
         and review.get("failed_checks") == [] and review.get("check_count") == 112
         and len(review.get("checks", [])) == 112 and all(r.get("pass") is True for r in review["checks"]),
         "independent source/newness review incomplete")
    need(review.get("draft_policy_sha256") == p.get("root_pre_score_draft_sha256")
         == "01163f3a7afbc6cf40f2426272be06e05b7b87406e9b99e56f5f1a8c5301c277",
         "independently reviewed pre-score source draft identity differs")
    need(review.get("measurement_float_conversion") is False and review.get("research_scores_read") is False
         and review.get("test_numeric_measurements_read") is False, "source review scope differs")
    candidates = p.get("candidates")
    need(isinstance(candidates, list) and [c.get("dataset") for c in candidates] == IDS, "exact five source definitions required")
    for c in candidates:
        s = STATIC[c["dataset"]]
        for k in ["raw_file", "raw_sha256", "raw_bytes", "rows", "bounds", "channels"]:
            need(c.get(k) == s[k], "fixed source shape/identity differs: " + c["dataset"] + "/" + k)
        need(c.get("channels_count") == s["C"] and c.get("prepared_frequency_seconds") == s["seconds"], "source channel or frequency differs")
        need(c.get("grid_first") == s["first"] and c.get("grid_last") == s["last"], "source fixed native clock differs")
        raw = scoped(base, c["raw_file"])
        need(raw.stat().st_size == s["raw_bytes"] and file_sha(raw) == s["raw_sha256"], "raw size/SHA mismatch before any numeric read")
        for path_key, hash_key in [("structural_receipt_path", "structural_receipt_sha256"),
                                   ("clock_amendment_receipt_path", "clock_amendment_receipt_sha256"),
                                   ("clock_phase_receipt_path", "clock_phase_receipt_sha256"),
                                   ("opaque_download_receipt_path", "opaque_download_receipt_sha256")]:
            if path_key in c:
                need(file_sha(scoped(base, c[path_key])) == c.get(hash_key), "pre-numeric source metadata receipt SHA mismatch")
        with zipfile.ZipFile(raw) as z:
            need(all(not i.flag_bits & 1 for i in z.infolist()), "encrypted raw archive")
            need(z.testzip() is None, "raw ZIP CRC failure before numerical read")
    # Precheck every target before any prefix measurement decoding or writing.
    for sid in IDS:
        for name in [f"data/{sid}_trainval.npz", f"data/{sid}.metadata.json"]:
            need(not scoped(base, name).exists(), "existing prepared payload preserved; do not overwrite/reprepare")
    need(not scoped(base, "data/metadata.json").exists(), "existing combined metadata preserved")
    need(not scoped(base, "FIVE_PREFIX_PREPARATION_RECEIPT.json").exists(), "existing final prefix receipt preserved before any numerical read")
    return p

def numeric_token(text, stats):
    stats["numeric_prefix_target_tokens_considered"] += 1
    token = text.strip()
    if token in ("", "?", "NA", "N/A", "null", "None"):
        stats["explicit_missing_prefix_tokens"] += 1
        return float("nan")
    value = float(token)
    if not math.isfinite(value):
        stats["nonfinite_prefix_tokens"] += 1
        return float("nan")
    stats["finite_source_zero_prefix_tokens"] += int(value == 0)
    stats["finite_source_negative_prefix_tokens"] += int(value < 0)
    return value

def new_stats():
    return {"numeric_prefix_target_tokens_considered": 0, "explicit_missing_prefix_tokens": 0,
            "nonfinite_prefix_tokens": 0, "finite_source_zero_prefix_tokens": 0, "finite_source_negative_prefix_tokens": 0,
            "suffix_target_tokens_float_converted": 0, "NWP_or_calendar_tokens_float_converted": 0}

def read_steel(raw, config, stats):
    import numpy as np
    s = STATIC[config["dataset"]]
    val = s["bounds"][1]
    matrix = np.full((val, s["C"]), np.nan, dtype=np.float64)
    first = dt.datetime.fromisoformat(s["first"])
    amendments = 0
    rows = 0
    with zipfile.ZipFile(raw) as z:
        with z.open("Steel_industry_data.csv") as f:
            reader = csv.DictReader(io.TextIOWrapper(f, encoding="utf-8-sig", newline=""))
            need(set(s["channels"]).issubset(reader.fieldnames or []), "Steel target header mismatch")
            previous = None
            for idx, row in enumerate(reader):
                published = dt.datetime.strptime(row["date"], "%d/%m/%Y %H:%M")
                expected = first + dt.timedelta(seconds=s["seconds"] * idx)
                if published != expected:
                    need(published.hour == 0 and published.minute == 0 and published + dt.timedelta(days=1) == expected,
                         "unexpected Steel clock mismatch")
                    need(previous == published + dt.timedelta(hours=23, minutes=45), "Steel source-order midnight predecessor mismatch")
                    amendments += 1
                previous = published
                if idx < val:
                    matrix[idx] = [numeric_token(row[k], stats) for k in s["channels"]]
                rows += 1
    need(rows == 35040 and amendments == 365, "full Steel clock closure differs")
    stats.update({"source_structural_rows_rechecked": rows, "native_midnight_metadata_amendments": amendments,
                  "returned_measurement_last": (first + dt.timedelta(seconds=s["seconds"] * (val - 1))).isoformat(),
                  "measurement_source_row_order_preserved": True})
    return matrix

def right_bin(t, first, seconds):
    elapsed = int((t - first).total_seconds())
    return (elapsed + seconds - 1) // seconds

def read_metro(raw, config, stats):
    import numpy as np
    s = STATIC[config["dataset"]]
    val = s["bounds"][1]
    sums = np.zeros((val, s["C"]), dtype=np.float64)
    counts = np.zeros((val, s["C"]), dtype=np.int64)
    first = dt.datetime.fromisoformat(s["first"])
    limit = first + dt.timedelta(seconds=s["seconds"] * (val - 1))
    raw_rows = prefix_rows = 0
    max_returned = None
    last_time = None
    with zipfile.ZipFile(raw) as z:
        with z.open("MetroPT3(AirCompressor).csv") as f:
            reader = csv.DictReader(io.TextIOWrapper(f, encoding="utf-8-sig", newline=""))
            need((reader.fieldnames or [])[1:] == ["timestamp", *s["channels"]], "Metro target header mismatch")
            for row in reader:
                t = dt.datetime.fromisoformat(row["timestamp"])
                need(last_time is None or t > last_time, "Metro source clock not strictly increasing")
                last_time = t
                idx = right_bin(t, first, s["seconds"])
                need(0 <= idx < s["rows"], "Metro timestamp outside fixed grid")
                raw_rows += 1
                if idx >= val:
                    continue
                need(t <= limit, "Metro suffix numerical conversion rejected")
                values = np.array([numeric_token(row[k], stats) for k in s["channels"]], dtype=np.float64)
                finite = np.isfinite(values)
                sums[idx, finite] += values[finite]
                counts[idx, finite] += 1
                prefix_rows += 1
                max_returned = t
    need(raw_rows == 1516948 and last_time == dt.datetime(2020, 9, 1, 3, 59, 50), "Metro structural row/time closure differs")
    matrix = np.full_like(sums, np.nan)
    np.divide(sums, counts, out=matrix, where=counts > 0)
    stats.update({"source_structural_rows_rechecked": raw_rows, "returned_measurement_source_rows": prefix_rows,
                  "returned_measurement_last": max_returned.isoformat() if max_returned else None,
                  "prefix_empty_structural_bins": int((counts.sum(axis=1) == 0).sum()),
                  "aggregation": "Arithmetic mean of finite values in source-row order per right-closed10min/channel; digital targets become mean duty fractions"})
    return matrix

def wind_time(text):
    day, clock = text.split()
    hour, minute = map(int, clock.split(":"))
    need(minute == 0 and 0 <= hour <= 24, "invalid source wind clock")
    date = dt.datetime.strptime(day, "%Y%m%d")
    return date + dt.timedelta(hours=hour)

def read_wind(raw, config, stats):
    import numpy as np
    s = STATIC[config["dataset"]]
    val = s["bounds"][1]
    matrix = np.full((val, s["C"]), np.nan, dtype=np.float64)
    first = dt.datetime.fromisoformat(s["first"])
    with zipfile.ZipFile(raw) as z:
        with zipfile.ZipFile(io.BytesIO(z.read("GEFCom2014 Data/GEFCom2014-W_V2.zip"))) as w:
            taskbytes = w.read("Wind/Task 15/Task15_W_Zone1_10.zip")
    rows_per_zone = {}
    with zipfile.ZipFile(io.BytesIO(taskbytes)) as task:
        for column, zone in enumerate(range(1, 11)):
            count = 0
            with task.open(f"Task15_W_Zone1_10/Task15_W_Zone{zone}.csv") as f:
                reader = csv.DictReader(io.TextIOWrapper(f, encoding="utf-8-sig", newline=""))
                need(reader.fieldnames == ["ZONEID", "TIMESTAMP", "TARGETVAR", "U10", "V10", "U100", "V100"], "Wind header differs")
                for idx, row in enumerate(reader):
                    need(int(row["ZONEID"]) == zone, "Wind zone ID differs")
                    t = wind_time(row["TIMESTAMP"])
                    need(t == first + dt.timedelta(hours=idx), "Wind hourly row clock differs")
                    need(bool(row["TARGETVAR"].strip()), "Unexpected official blank target; do not impute unknown competition future")
                    if idx < val:
                        matrix[idx, column] = numeric_token(row["TARGETVAR"], stats)
                    count += 1
            need(count == 16800, "Wind historical rows differ")
            rows_per_zone[str(zone)] = count
    stats.update({"rows_per_zone_structurally_rechecked": rows_per_zone,
                  "returned_measurement_last": (first + dt.timedelta(hours=val - 1)).isoformat(),
                  "official_solutions_or_benchmarks_opened": False, "only_numeric_target_column": "TARGETVAR"})
    return matrix

def tsf_records(raw, member):
    with zipfile.ZipFile(raw) as archive:
        with archive.open(member) as f:
            attributes = []
            in_data = False
            for line in io.TextIOWrapper(f, encoding="cp1252", newline=""):
                line = line.strip()
                if not line:
                    continue
                if not in_data:
                    if line.lower() == "@data":
                        in_data = True
                    elif line.lower().startswith("@attribute "):
                        attributes.append(line.split()[1])
                    continue
                metadata, opaque_tokens = line.rsplit(":", 1)
                parts = metadata.split(":")
                need(len(parts) == len(attributes), "TSF metadata width differs")
                yield dict(zip(attributes, parts)), opaque_tokens

def read_tsf(raw, config, stats):
    import numpy as np
    s = STATIC[config["dataset"]]
    val = s["bounds"][1]
    matrix = np.full((val, s["C"]), np.nan, dtype=np.float64)
    meta = {r["id"]: r for r in config["selected_series_metadata"]}
    need(set(meta) == set(s["channels"]), "TSF exact selected series metadata differs")
    selected_seen = set()
    all_series_seen = set()
    for attrs, opaque in tsf_records(raw, config["member"]):
        sid = attrs["series_name"]
        need(sid not in all_series_seen, "duplicate TSF source ID")
        all_series_seen.add(sid)
        if sid not in meta:
            continue  # No numeric conversion for any unselected source series.
        m = meta[sid]
        need(attrs == m["metadata_attributes"] and opaque.count(",") + 1 == m["count"], "TSF structure differs from pre-score audit")
        start = dt.datetime.strptime(attrs["start_timestamp"], "%Y-%m-%d %H-%M-%S")
        offset = m["start_index_for_fixed_period"]
        need(start + dt.timedelta(seconds=offset * s["seconds"]) == dt.datetime.fromisoformat(s["first"]), "TSF original phase or calendar offset differs")
        need(m["stop_exclusive_index_for_fixed_period"] - offset == s["rows"] and offset >= 0, "TSF whole fixed period coverage differs")
        column = s["channels"].index(sid)
        tokens = itertools.islice(opaque.split(","), offset, offset + val)
        values = [numeric_token(t, stats) for t in tokens]
        need(len(values) == val, "TSF prefix incomplete")
        matrix[:, column] = values
        selected_seen.add(sid)
    need(selected_seen == set(s["channels"]), "TSF exact covering group incomplete")
    need(len(all_series_seen) == (5 if config["dataset"] == "AEMO5" else 66), "TSF original source series count differs")
    stats.update({"all_source_series_structurally_rechecked": len(all_series_seen), "selected_all_covering_series": s["channels"],
                  "unselected_series_tokens_float_converted": 0, "preperiod_tokens_float_converted": 0,
                  "returned_measurement_last": (dt.datetime.fromisoformat(s["first"]) + dt.timedelta(seconds=s["seconds"] * (val - 1))).isoformat(),
                  "original_native_phase_retained": True})
    return matrix

def fill_prefix(matrix, train_end):
    import numpy as np
    need(matrix.dtype == np.float64 and matrix.ndim == 2 and 0 < train_end < len(matrix), "prefix aggregation shape/dtype differs")
    finite = np.isfinite(matrix)
    training_valid_means = np.empty(matrix.shape[1], dtype=np.float64)
    for column in range(matrix.shape[1]):
        train_values = matrix[:train_end, column]
        valid = np.isfinite(train_values)
        need(valid.any(), "channel has no finite TRAIN values; do not silently drop it")
        training_valid_means[column] = np.mean(train_values[valid], dtype=np.float64)
    filled = matrix.copy()
    leading = 0
    for column in range(matrix.shape[1]):
        indices = np.maximum.accumulate(np.where(finite[:, column], np.arange(len(matrix)), -1))
        early = indices < 0
        leading += int(early.sum())
        filled[~early, column] = matrix[indices[~early], column]
        filled[early, column] = training_valid_means[column]
    payload = filled.astype(np.float32)
    need(np.isfinite(payload).all(), "nonfinite prepared prefix")
    mean = payload[:train_end].mean(axis=0, dtype=np.float64).astype(np.float32)
    std = payload[:train_end].std(axis=0, dtype=np.float64).astype(np.float32)
    std[std < 1e-6] = 1
    missing = int((~finite).sum())
    return payload, {"missing_aggregated_prefix_elements": missing, "leading_train_mean_fill_elements": leading,
                     "causal_forward_fill_elements": missing - leading,
                     "finite_aggregated_zero_prefix_elements": int(((matrix == 0) & finite).sum()),
                     "finite_aggregated_negative_prefix_elements_retained": int(((matrix < 0) & finite).sum()),
                     "prepared_zero_prefix_elements": int((payload == 0).sum()),
                     "training_valid_mean_float64_sha256": hashlib.sha256(training_valid_means.tobytes()).hexdigest(),
                     "raw_filled_array_sha256": hashlib.sha256(payload.tobytes()).hexdigest(),
                     "scaler_mean_sha256": hashlib.sha256(mean.tobytes()).hexdigest(),
                     "scaler_std_sha256": hashlib.sha256(std.tobytes()).hexdigest(),
                     "scaler_std_lt_1e_minus6_maps_to1": True}

def json_new(path, body):
    path = reject_links(path)
    with path.open("x", encoding="utf-8", newline="\n") as file:
        json.dump(body, file, ensure_ascii=False, indent=2, allow_nan=False)
        file.write("\n")

def prepare(base, policy_sha):
    import numpy as np
    base = reject_links(base)
    policy = verify_policy(base, policy_sha)  # All five guards close before any numeric measurement read.
    prepared = []
    source_sha = file_sha(Path(__file__))
    for config in policy["candidates"]:
        sid = config["dataset"]
        s = STATIC[sid]
        raw = scoped(base, s["raw_file"])
        stats = new_stats()
        if sid == "SteelIndustry6":
            aggregated = read_steel(raw, config, stats)
        elif sid == "MetroPT15":
            aggregated = read_metro(raw, config, stats)
        elif sid == "GEFCOM2014Wind10":
            aggregated = read_wind(raw, config, stats)
        else:
            aggregated = read_tsf(raw, config, stats)
        need(aggregated.shape == (s["bounds"][1], s["C"]), "prefix exact shape differs")
        payload, filling = fill_prefix(aggregated, s["bounds"][0])
        train, val, T = s["bounds"]
        metadata = {
            "dataset": sid, "channels": s["C"], "channel_ids": s["channels"], "rows": T,
            "bounds": s["bounds"], "trainval_rows": val, "payload_rows": val, "path": f"data/{sid}_trainval.npz",
            "test_rows_in_payload": False, "split_counts_len": 2, "raw_sha256": s["raw_sha256"], "raw_bytes": s["raw_bytes"],
            "grid_first": s["first"], "grid_last": s["last"], "frequency_seconds": s["seconds"],
            "train_last": (dt.datetime.fromisoformat(s["first"]) + dt.timedelta(seconds=s["seconds"] * (train - 1))).isoformat(),
            "val_last": (dt.datetime.fromisoformat(s["first"]) + dt.timedelta(seconds=s["seconds"] * (val - 1))).isoformat(),
            "window_counts": {str(H): {"train": train - 96 - H + 1, "val": val - train - H + 1, "test_structural_only": T - val - H + 1} for H in [96, 192, 336, 720]},
            "preparation_policy_sha256": policy_sha, "preparer_source_sha256": source_sha,
            "source_structural_receipt_sha256": config["structural_receipt_sha256"],
            "all_measured_zero_preserved_no_zero_as_missing": True,
            "test_measurement_array_materialized": False, "test_windows_constructed": False,
            "test_numerical_values_decoded": False, "test_scores_accessed": False, "model_scores_accessed": False,
            "internally_decoded_CSV_TSF_token_strings_not_claimed_cryptographic_seal": True,
            "clock_provenance": ("Author TSF start+index implied regular clock; derived synchronized panel, not independently verified physical wall-clock/DST/missing-hour chronology"
                                 if sid in ["AEMO5", "MelbournePedestrian17"] else "Published source clock plus predeclared metadata-only interpretations/aggregation, without score-based time selection"),
            "unchanged_runtime_scaler": "float64 TRAIN-only population mean/std of imputedfloat32prefix, castfloat32, std<1e-6->1",
            "missing_policy": "finite original aggregated TRAIN-only leading mean plus causal channel forward-fill, no backward/interpolative/future mean",
            **stats, **filling,
        }
        prepared.append((payload, metadata))
    # Only completed and checked all-five prefixes may be written, never placeholders.
    directory = scoped(base, "data")
    directory.mkdir(exist_ok=True)
    all_meta = {}
    for payload, meta in prepared:
        path = scoped(base, meta["path"])
        with path.open("xb") as file:
            np.savez_compressed(file, data=payload, split_counts=np.array(meta["bounds"][:2], dtype=np.int64))
        meta["sha256"] = file_sha(path)
        meta["bytes"] = path.stat().st_size
        json_new(scoped(base, f"data/{meta['dataset']}.metadata.json"), meta)
        all_meta[meta["dataset"]] = meta
    json_new(scoped(base, "data/metadata.json"), {"status": "ONLY_FIVE_FIXED_CANDIDATE_TRAINVAL_PREFIXES", "datasets": all_meta,
                                               "preparation_policy_sha256": policy_sha, "preparer_source_sha256": source_sha,
                                               "all_suffix_target_tokens_float_converted": 0, "actual_research_started": False})
    result = {"status": "COMPLETE_FIVE_FIXED_TRAINVAL_PREFIX_PREPARATION_NO_TEST_ARRAY_OR_SCORES", "candidate_order": IDS,
              "policy_sha256": policy_sha, "preparer_source_sha256": source_sha,
              "combined_metadata_sha256": file_sha(scoped(base, "data/metadata.json")),
              "payload_sha256": {k: v["sha256"] for k, v in all_meta.items()}, "test_arrays_prepared": False, "actual_research_started": False,
              "created_utc": dt.datetime.now(dt.timezone.utc).isoformat()}
    json_new(scoped(base, "FIVE_PREFIX_PREPARATION_RECEIPT.json"), result)
    return result

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare-prefixes", action="store_true")
    parser.add_argument("--policy-sha256")
    parser.add_argument("--local-root")
    args = parser.parse_args()
    if not args.prepare_prefixes:
        print(json.dumps({"status": "PENDING_ROOT_FIXED_FIVE_POLICY_AND_EXPLICIT_PREFIX_PREPARATION", "test_values_read": False,
                          "project_inputs_read": False, "files_written": False, "SSH": False}))
        return
    base = Path(args.local_root).absolute() if args.local_root else Path(__file__).absolute().parents[2] / "main_experiments/NEW5_PILOT_20261011"
    print(json.dumps(prepare(base, args.policy_sha256), ensure_ascii=False), flush=True)

if __name__ == "__main__":
    main()
