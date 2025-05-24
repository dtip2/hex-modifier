# streamlit_app.py
import streamlit as st
import binascii
import time
# import sys # Not strictly needed for core logic here
# import threading # We'll try to avoid explicit threading for initial Streamlit version for simplicity
# from queue import Queue # Will be replaced by a mock updater

# --- Configuration (Copied from V4) ---
CONTEXT_LENGTH = 8

# --- Helper Functions (Copied and kept identical from V4) ---
def load_file_to_bytes_streamlit(uploaded_file_object):
    """Loads an UploadedFile object from Streamlit into bytes."""
    if uploaded_file_object is not None:
        try:
            return uploaded_file_object.read()
        except Exception as e:
            st.error(f"Failed to load {uploaded_file_object.name}: {str(e)}")
            return None
    return None

def count_byte_differences(bytes1, bytes2):
    if bytes1 is None or bytes2 is None: return 0
    try:
        min_len = min(len(bytes1), len(bytes2))
        diff_count = sum(bytes1[i] != bytes2[i] for i in range(min_len))
        diff_count += abs(len(bytes1) - len(bytes2))
        return diff_count
    except Exception as e: st.error(f"Error counting differences: {str(e)}"); return -1

def find_differences_with_context(file1_bytes, file2_bytes, context_len=CONTEXT_LENGTH):
    if file1_bytes is None or file2_bytes is None: return []
    changes = []
    len1, len2 = len(file1_bytes), len(file2_bytes)
    idx1, idx2 = 0, 0
    while idx1 < len1 or idx2 < len2:
        while idx1 < len1 and idx2 < len2 and file1_bytes[idx1] == file2_bytes[idx2]: idx1 += 1; idx2 += 1
        is_diff = (idx1 < len1 and idx2 < len2 and file1_bytes[idx1] != file2_bytes[idx2])
        is_f1_remaining = (idx1 < len1 and idx2 == len2)
        is_f2_remaining = (idx1 == len1 and idx2 < len2)
        if is_diff or is_f1_remaining or is_f2_remaining:
            diff_start_idx1, diff_start_idx2 = idx1, idx2
            original_offset_f1 = diff_start_idx1
            actual_context_start1 = max(0, diff_start_idx1 - context_len)
            context_before = file1_bytes[actual_context_start1:diff_start_idx1]
            diff_end_idx1, diff_end_idx2 = idx1, idx2
            if is_diff:
                while diff_end_idx1 < len1 and diff_end_idx2 < len2 and \
                      file1_bytes[diff_end_idx1] != file2_bytes[diff_end_idx2]:
                    diff_end_idx1 += 1; diff_end_idx2 += 1
            elif is_f1_remaining: diff_end_idx1 = len1
            elif is_f2_remaining: diff_end_idx2 = len2
            if diff_end_idx1 == len1 and diff_end_idx2 < len2 and is_diff: diff_end_idx2 = len2
            elif diff_end_idx2 == len2 and diff_end_idx1 < len1 and is_diff: diff_end_idx1 = len1
            old_data = file1_bytes[diff_start_idx1:diff_end_idx1]
            new_data = file2_bytes[diff_start_idx2:diff_end_idx2]
            context_after_start1 = diff_end_idx1
            context_after_end1 = min(len1, context_after_start1 + context_len)
            context_after = file1_bytes[context_after_start1:context_after_end1]
            idx1 = diff_end_idx1; idx2 = diff_end_idx2
            changes.append({"original_offset": original_offset_f1,
                            "old_data_bytes": old_data,
                            "new_data_bytes": new_data,
                            "context_before": context_before,
                            "context_after": context_after,
                            "original_f2_offset_for_new_data": original_offset_f2,
                            "original_new_data_len": len(new_data)
                           })
        else: break
    return changes

# --- Streamlit GUI Updater (Mock Queue) ---
class StreamlitGuiUpdate:
    def __init__(self, progress_bar_placeholder, status_text_placeholder, log_area_placeholder=None):
        self.progress_bar_placeholder = progress_bar_placeholder
        self.status_text_placeholder = status_text_placeholder
        self.log_area_placeholder = log_area_placeholder
        self.log_messages = []

    def put(self, item_tuple):
        message_type, value = item_tuple
        if message_type == 'status':
            if self.status_text_placeholder:
                self.status_text_placeholder.info(f"{value}") # Using .info for status
                # Log important status messages for debugging in Streamlit if needed
                # print(value) # This will go to console where Streamlit runs
                self.log_messages.append(value)
                if self.log_area_placeholder:
                    self.log_area_placeholder.text_area("Live Log", "\n".join(self.log_messages[-15:]), height=200, key="live_log_text_area")


        elif message_type == 'progress':
            if self.progress_bar_placeholder:
                # Streamlit progress bar expects a value between 0 and 100 (or 0.0 and 1.0 if using float)
                progress_value = int(value)
                if 0 <= progress_value <= 100:
                    self.progress_bar_placeholder.progress(progress_value)
                elif progress_value > 100: # Cap at 100
                    self.progress_bar_placeholder.progress(100)

        elif value is None: # Orchestrator finished message
             if self.status_text_placeholder:
                self.status_text_placeholder.success("Orchestrator processing complete. Finalizing...")


# --- Patch Application Engine (Pass 1) ---
# (apply_patches_pure_python_search remains IDENTICAL to V4)
def apply_patches_pure_python_search(
    target_file_bytes_input, changes_to_attempt, gui_queue,
    strictness_level, pass_description="Pass",
    progress_start_percent=0, progress_scan_end_percent=40,
    progress_resolve_end_percent=50, progress_apply_end_percent=60
):
    def post_update(message_type, value):
        gui_queue.put((message_type, value))
    def post_status_update(message):
        full_message = f"Status: ({pass_description} - Strictness {strictness_level}) {message}"
        post_update('status', full_message)
        # print(full_message) # Keep console print for debugging if Streamlit UI updates are tricky
    if target_file_bytes_input is None:
        post_status_update("Error: Input target data is missing.")
        return None, 0, [{"original_offset": -1, "patch_index": -1, "patch_number": "N/A", "reason": "Error: Input target data is missing."}], set()
    if not changes_to_attempt:
        post_status_update("No changes to attempt in this pass.")
        post_update('progress', progress_apply_end_percent); return target_file_bytes_input, 0, [], set()
    post_status_update(f"Starting..."); post_update('progress', progress_start_percent)
    patch_details_list_this_pass = [{} for _ in changes_to_attempt]; pattern_to_local_patch_indices = {}; unique_patterns = set()
    total_patches_this_pass = len(changes_to_attempt)
    post_status_update(f"Building search patterns for {total_patches_this_pass} patches..."); build_start_time = time.time()
    for i, change_obj in enumerate(changes_to_attempt):
        original_idx = change_obj["original_patch_index"]; original_patch_num = change_obj["patch_num_original"]
        old_data = change_obj["old_data_bytes"]; new_data = change_obj["new_data_bytes"]
        context_before = change_obj["context_before"]; context_after = change_obj["context_after"]
        patch_details_list_this_pass[i] = {"local_patch_index_this_pass": i, "original_patch_index": original_idx,
            "patch_num_original": original_patch_num, "old_data": old_data, "new_data": new_data,
            "original_offset_in_file1": change_obj["original_offset"],
            "context_before_hex": context_before.hex(sep=' '), "context_after_hex": context_after.hex(sep=' '), "patterns": []}
        patterns_to_try = []; pri = 0
        p_ctx_both = context_before + old_data + context_after if context_before and old_data and context_after else None
        p_ctx_b = context_before + old_data if context_before and old_data else None
        p_ctx_a = old_data + context_after if old_data and context_after else None
        p_old = old_data if old_data else None
        p_ins_ctx_both = context_before + context_after if not old_data and new_data and context_before and context_after else None
        p_ins_ctx_b = context_before if not old_data and new_data and context_before else None
        if strictness_level <= 6:
             if p_ctx_both: patterns_to_try.append((p_ctx_both, pri, len(context_before), "CtxBefore+Old+CtxAfter")); pri+=1
             if p_ins_ctx_both: patterns_to_try.append((p_ins_ctx_both, pri+0.1, len(context_before), "CtxBefore+CtxAfter (Insert)")); pri+=1
             if p_ins_ctx_b: patterns_to_try.append((p_ins_ctx_b, pri+0.2, len(context_before), "CtxBefore (Insert)")); pri+=1
        if strictness_level <= 5:
             if p_ctx_b: patterns_to_try.append((p_ctx_b, pri, len(context_before), "CtxBefore+Old")); pri+=1
             if p_ctx_a: patterns_to_try.append((p_ctx_a, pri, 0, "Old+CtxAfter")); pri+=1
        if strictness_level <= 4:
             if p_old: patterns_to_try.append((p_old, pri, 0, "Old Data Only")); pri+=1
        for pattern_bytes, priority, ctx_b_len, desc in patterns_to_try:
            if not pattern_bytes: continue
            pattern_info = {"pattern": pattern_bytes, "priority": priority, "ctx_b_len": ctx_b_len, "desc": desc}
            patch_details_list_this_pass[i]["patterns"].append(pattern_info); unique_patterns.add(pattern_bytes)
            if pattern_bytes not in pattern_to_local_patch_indices: pattern_to_local_patch_indices[pattern_bytes] = []
            if i not in pattern_to_local_patch_indices[pattern_bytes]: pattern_to_local_patch_indices[pattern_bytes].append(i)
    patterns_list = list(unique_patterns); build_end_time = time.time()
    current_progress_after_build = progress_start_percent + (progress_scan_end_percent - progress_start_percent) * 0.05
    post_update('progress', int(current_progress_after_build))
    post_status_update(f"Pattern analysis complete ({len(patterns_list)} unique) in {build_end_time - build_start_time:.2f}s. Searching target...")
    if not patterns_list:
         post_status_update(f"No search patterns generated."); skipped_details_this_pass = []
         for dp in patch_details_list_this_pass:
              skipped_details_this_pass.append({ "original_patch_index": dp["original_patch_index"], "patch_number": dp["patch_num_original"],
                  "original_offset": dp["original_offset_in_file1"], "reason": f"P{dp['patch_num_original']}: No search patterns applicable.",
                  "old_data_snippet": dp["old_data"][:16].hex(sep=' '), "new_data_snippet": dp["new_data"][:16].hex(sep=' '),
                  "context_before": dp["context_before_hex"], "context_after": dp["context_after_hex"], })
         post_update('progress', progress_apply_end_percent); return target_file_bytes_input, 0, skipped_details_this_pass, set()
    found_matches_local = {}; search_start_time = time.time(); target_len = len(target_file_bytes_input)
    scan_progress_range = progress_scan_end_percent - current_progress_after_build; scan_count = 0
    for i_scan in range(target_len):
        scan_count += 1
        if scan_count % 250000 == 0:
            sp_done = (i_scan / target_len) if target_len > 0 else 1; cs_prog = current_progress_after_build + (scan_progress_range * sp_done)
            post_update('progress', int(cs_prog)); post_status_update(f"Scanning target at offset {i_scan}/{target_len}...")
        for pattern_bytes in patterns_list:
             pattern_len = len(pattern_bytes)
             if i_scan + pattern_len <= target_len and target_file_bytes_input[i_scan : i_scan + pattern_len] == pattern_bytes:
                 matched_local_indices = pattern_to_local_patch_indices.get(pattern_bytes, [])
                 for local_idx in matched_local_indices:
                     p_info = next((p for p in patch_details_list_this_pass[local_idx]["patterns"] if p["pattern"] == pattern_bytes), None)
                     if p_info:
                         m_info = {"start_index_in_target": i_scan, **p_info}
                         if local_idx not in found_matches_local: found_matches_local[local_idx] = []
                         found_matches_local[local_idx].append(m_info)
    search_end_time = time.time(); post_update('progress', progress_scan_end_percent)
    post_status_update(f"Scan complete ({scan_count}) in {search_end_time - search_start_time:.2f}s. Resolving matches...")
    patches_to_apply_this_pass = []; skipped_local_indices_map = {}; applied_local_indices_this_pass = set()
    resolve_start_time = time.time(); resolve_progress_range = progress_resolve_end_percent - progress_scan_end_percent
    for local_idx in range(total_patches_this_pass):
        if local_idx > 0 and local_idx % 250 == 0:
            rp_done = (local_idx / total_patches_this_pass) if total_patches_this_pass > 0 else 1
            cr_prog = progress_scan_end_percent + (resolve_progress_range * rp_done)
            post_update('progress', int(cr_prog)); post_status_update(f"Resolving patch {local_idx+1}/{total_patches_this_pass}...")
        patch_detail_pass = patch_details_list_this_pass[local_idx]; p_num_msg = patch_detail_pass["patch_num_original"]
        pot_matches = found_matches_local.get(local_idx, [])
        cs_info = {"original_patch_index": patch_detail_pass["original_patch_index"], "patch_number": p_num_msg,
            "original_offset": patch_detail_pass["original_offset_in_file1"], "old_data_snippet": patch_detail_pass["old_data"][:16].hex(sep=' '),
            "new_data_snippet": patch_detail_pass["new_data"][:16].hex(sep=' '), "context_before": patch_detail_pass["context_before_hex"],
            "context_after": patch_detail_pass["context_after_hex"],}
        if not pot_matches:
             if local_idx not in skipped_local_indices_map: skipped_local_indices_map[local_idx] = {**cs_info, "reason": f"P{p_num_msg}: No search patterns found."}
             continue
        pot_matches.sort(key=lambda m: (m["priority"], m["start_index_in_target"])); best_pri = pot_matches[0]["priority"]
        best_pri_matches = [m for m in pot_matches if m["priority"] == best_pri]; tdo_this_patch = {}
        for m in best_pri_matches:
             td_offset = m["start_index_in_target"] + m["ctx_b_len"]
             if td_offset not in tdo_this_patch: tdo_this_patch[td_offset] = []
             tdo_this_patch[td_offset].append(m)
        chosen_match = None; num_distinct_td_offsets = len(tdo_this_patch)
        if num_distinct_td_offsets == 1: chosen_match = list(tdo_this_patch.values())[0][0]
        elif num_distinct_td_offsets > 1:
            if strictness_level <= 2:
                 first_td_offset = min(tdo_this_patch.keys()); chosen_match = tdo_this_patch[first_td_offset][0]
                 # print(f"  P{p_num_msg}: Ambiguous ({num_distinct_td_offsets} targets) - Applying at 0x{first_td_offset:08X}") # Console print
                 post_status_update(f"P{p_num_msg}: Ambiguous ({num_distinct_td_offsets} targets) - Applying at 0x{first_td_offset:08X} (S{strictness_level})")

            else:
                 m_desc = best_pri_matches[0]["desc"]; fto_hex = [f"0x{off:08X}" for off in sorted(tdo_this_patch.keys())[:5]]
                 reason = (f"P{p_num_msg}: Ambiguous (S{strictness_level}+) - Pattern ('{m_desc}') targets {num_distinct_td_offsets} offsets: "
                           f"{', '.join(fto_hex)}{'...' if num_distinct_td_offsets > 5 else ''}. Skipped.")
                 if local_idx not in skipped_local_indices_map: skipped_local_indices_map[local_idx] = {**cs_info, "reason": reason}
                 continue
        if chosen_match:
            td_offset = chosen_match["start_index_in_target"] + chosen_match["ctx_b_len"]
            len_old = len(patch_detail_pass["old_data"]); tde_offset = td_offset + len_old
            perform_ver = (strictness_level >= 2); can_schedule = False; skip_reason_ver = ""
            if perform_ver:
                s_in_bounds = (td_offset >= 0 and tde_offset <= len(target_file_bytes_input))
                cd_target_slice = b""; dm_expected_old = False
                if not patch_detail_pass["old_data"]: dm_expected_old = True
                elif s_in_bounds:
                    cd_target_slice = target_file_bytes_input[td_offset:tde_offset]
                    dm_expected_old = (cd_target_slice == patch_detail_pass["old_data"])
                if dm_expected_old: can_schedule = True
                else: skip_reason_ver = (f"P{p_num_msg}: Verification Failed (S{strictness_level}+) - Target data at 0x{td_offset:08X} "
                               f"(len {len(cd_target_slice)}) != expected old_data (len {len_old}).")
            else: # print(f"  P{p_num_msg}: Skipping verification (S1)"); # Console print
                  post_status_update(f"P{p_num_msg}: Skipping verification (S{strictness_level})")
                  can_schedule = True
            if can_schedule:
                patches_to_apply_this_pass.append({"target_offset": td_offset, "len_old_data": len_old, "data_new": patch_detail_pass["new_data"],
                    "local_patch_index_this_pass": local_idx, "patch_num_original": p_num_msg, "pattern_desc": chosen_match["desc"]})
                applied_local_indices_this_pass.add(local_idx)
            elif skip_reason_ver:
                if local_idx not in skipped_local_indices_map: skipped_local_indices_map[local_idx] = {**cs_info, "reason": skip_reason_ver}
    resolve_end_time = time.time(); post_update('progress', progress_resolve_end_percent)
    post_status_update(f"Match resolution done in {resolve_end_time - resolve_start_time:.2f}s. Applying {len(patches_to_apply_this_pass)} patches...")
    patches_to_apply_this_pass.sort(key=lambda p: p["target_offset"], reverse=True)
    target_ba = bytearray(target_file_bytes_input); apply_start_time = time.time()
    apply_prog_range = progress_apply_end_percent - progress_resolve_end_percent
    for i_apply, p_info_apply in enumerate(patches_to_apply_this_pass):
        if i_apply > 0 and i_apply % 250 == 0:
            ap_done = (i_apply / len(patches_to_apply_this_pass)) if len(patches_to_apply_this_pass) > 0 else 1
            ca_prog = progress_resolve_end_percent + (apply_prog_range * ap_done)
            post_update('progress', int(ca_prog)); post_status_update(f"Applying patch {i_apply+1}/{len(patches_to_apply_this_pass)}...")
        offset = p_info_apply["target_offset"]; len_old = p_info_apply["len_old_data"]; data_new = p_info_apply["data_new"]
        l_idx_apply = p_info_apply["local_patch_index_this_pass"]; pd_pass_apply = patch_details_list_this_pass[l_idx_apply]
        p_num_apply = pd_pass_apply["patch_num_original"]
        cs_info_apply_fail = {"original_patch_index": pd_pass_apply["original_patch_index"], "patch_number": p_num_apply,
            "original_offset": pd_pass_apply["original_offset_in_file1"], "old_data_snippet": pd_pass_apply["old_data"][:16].hex(sep=' '),
            "new_data_snippet": pd_pass_apply["new_data"][:16].hex(sep=' '), "context_before": pd_pass_apply["context_before_hex"],
            "context_after": pd_pass_apply["context_after_hex"],}
        try: target_ba[offset : offset + len_old] = data_new
        except Exception as e:
            # print(f"CRITICAL ERROR applying P{p_num_apply} (LocalIdx {l_idx_apply}) at 0x{offset:08X}: {e}") # Console
            post_status_update(f"CRITICAL ERROR applying P{p_num_apply} (LocalIdx {l_idx_apply}) at 0x{offset:08X}: {e}")
            reason = f"P{p_num_apply}: CRITICAL Error during final application at 0x{offset:08X}: {e}"
            if l_idx_apply not in skipped_local_indices_map: skipped_local_indices_map[l_idx_apply] = {**cs_info_apply_fail, "reason": reason}
            if l_idx_apply in applied_local_indices_this_pass: applied_local_indices_this_pass.remove(l_idx_apply) # Should this be here or does it affect applied_orig_indices?
    apply_end_time = time.time(); final_modified_bytes = bytes(target_ba)
    skipped_details_list = list(skipped_local_indices_map.values()); applied_orig_indices = set()
    for l_idx_applied in applied_local_indices_this_pass: applied_orig_indices.add(patch_details_list_this_pass[l_idx_applied]["original_patch_index"])
    applied_count = len(applied_orig_indices)
    post_update('progress', progress_apply_end_percent)
    post_status_update(f"Application stage done in {apply_end_time - apply_start_time:.2f}s. Applied {applied_count}. Skipped {len(skipped_details_list)} in this pass.")
    return final_modified_bytes, applied_count, skipped_details_list, applied_orig_indices


# --- Patching Orchestrator (Test P1+P3 Only, Byte-Level Audit for Skip Log) ---
# (apply_patches_with_multiple_passes remains IDENTICAL to V4, just uses new gui_queue)
def apply_patches_with_multiple_passes(original_file3_bytes, all_diff_blocks_initial, initial_strictness, gui_queue):
    def post_orchestrator_update(message_type, value):
        gui_queue.put((message_type, value))
        # if message_type == 'status': print(value) # Console print
    def post_orchestrator_status(message):
        post_orchestrator_update('status', f"Status: Orchestrator - {message}")

    if not all_diff_blocks_initial:
        post_orchestrator_status("No differences to apply.")
        post_orchestrator_update('progress', 100)
        gui_queue.put(None)
        return original_file3_bytes, [] # Return empty list for original diffs for audit

    all_diff_blocks_augmented = []
    for i, block in enumerate(all_diff_blocks_initial):
        augmented_block = block.copy(); augmented_block["original_patch_index"] = i; augmented_block["patch_num_original"] = i + 1
        all_diff_blocks_augmented.append(augmented_block)

    current_target_bytes = original_file3_bytes
    p1_skipped_map = {};
    p1_applied_indices = set()

    P1_START, P1_SCAN_END, P1_RESOLVE_END, P1_APPLY_END = 0, 40, 60, 70
    P3_START, P3_APPLY_END = 70, 100

    post_orchestrator_status(f"Starting Pass 1 (User Strictness: {initial_strictness})...")
    pass1_start_time = time.time()
    modified_bytes_p1, _, skipped_details_p1, applied_indices_p1_run = \
        apply_patches_pure_python_search(
            current_target_bytes, all_diff_blocks_augmented, gui_queue,
            initial_strictness, pass_description="Pass 1",
            progress_start_percent=P1_START, progress_scan_end_percent=P1_SCAN_END,
            progress_resolve_end_percent=P1_RESOLVE_END, progress_apply_end_percent=P1_APPLY_END
        )
    p1_applied_indices.update(applied_indices_p1_run)
    current_target_bytes = modified_bytes_p1 if modified_bytes_p1 is not None else current_target_bytes

    for skip_info in skipped_details_p1:
        if skip_info["original_patch_index"] not in p1_applied_indices:
             p1_skipped_map[skip_info["original_patch_index"]] = skip_info

    pass1_end_time = time.time()
    post_orchestrator_status(f"Pass 1 finished in {pass1_end_time - pass1_start_time:.2f}s. P1 Applied: {len(p1_applied_indices)}, P1 Skipped: {len(p1_skipped_map)}")

    post_orchestrator_status(f"Pass 2 (Context Search) is SKIPPED in this test version.")
    post_orchestrator_update('progress', P3_START)

    changes_for_pass3 = [chg_obj for chg_obj in all_diff_blocks_augmented
                         if chg_obj["original_patch_index"] in p1_skipped_map]

    if changes_for_pass3:
        post_orchestrator_status(f"Starting Third Pass (Byte-wise Direct Offset) for {len(changes_for_pass3)} P1-skipped patches...")
        pass3_start_time = time.time()
        target_bytearray_p3 = bytearray(current_target_bytes)
        
        # post_orchestrator_status(f"DEBUG: Patches considered for Pass 3 Byte-wise ({len(changes_for_pass3)} total):") # To UI if needed

        for i_p3_block, change_obj_p3 in enumerate(changes_for_pass3):
            patch_num_p3 = change_obj_p3["patch_num_original"]
            original_f1_offset_block = change_obj_p3["original_offset"]
            old_data_f1_block = change_obj_p3["old_data_bytes"]
            new_data_f2_block = change_obj_p3["new_data_bytes"]

            current_p3_progress = P3_START + ((i_p3_block + 1) / len(changes_for_pass3)) * (P3_APPLY_END - P3_START)
            post_orchestrator_update('progress', int(current_p3_progress))

            # Detailed byte check logs can be overwhelming for Streamlit status. Console print or a separate debug log is better.
            # print(f"\n[Pass 3 Candidate Block Check] Patch Num: {patch_num_p3}, Orig.BlockOffset: 0x{original_f1_offset_block:08X}")
            # print(f"  Block Old (F1): {old_data_f1_block.hex()}, Block New (F2): {new_data_f2_block.hex()}")
            if i_p3_block % 50 == 0:
                 post_orchestrator_status(f"P3 Byte-wise - Checking block {patch_num_p3} ({i_p3_block+1}/{len(changes_for_pass3)})...")

            len_to_process = min(len(old_data_f1_block), len(new_data_f2_block))
            target_len_p3 = len(target_bytearray_p3)

            if len_to_process == 0:
                # ... (reasoning for skipping pure insertion/deletion in P3) ...
                continue

            for byte_idx_in_block in range(len_to_process):
                current_byte_original_f1_offset = original_f1_offset_block + byte_idx_in_block
                expected_old_byte = old_data_f1_block[byte_idx_in_block : byte_idx_in_block+1]
                intended_new_byte = new_data_f2_block[byte_idx_in_block : byte_idx_in_block+1]
                if current_byte_original_f1_offset < 0: continue
                if current_byte_original_f1_offset < target_len_p3:
                    current_byte_in_target = target_bytearray_p3[current_byte_original_f1_offset : current_byte_original_f1_offset+1]
                    if current_byte_in_target == expected_old_byte:
                        if current_byte_in_target != intended_new_byte:
                            try:
                                target_bytearray_p3[current_byte_original_f1_offset : current_byte_original_f1_offset+1] = intended_new_byte
                            except Exception as e: # print(f"    P3 Byte ERROR: {e}") # Console
                                 post_orchestrator_status(f"P3 Byte ERROR applying at 0x{current_byte_original_f1_offset:08X}: {e}")

        current_target_bytes = bytes(target_bytearray_p3)
        pass3_end_time = time.time()
        post_orchestrator_status(f"Pass 3 finished in {pass3_end_time - pass3_start_time:.2f}s.")
    else:
        post_orchestrator_status("No P1-skipped patches for Pass 3 attempt.")

    post_orchestrator_update('progress', P3_APPLY_END)
    post_orchestrator_status(f"All patching passes complete. Final audit for skip log will occur next.")
    gui_queue.put(None) # Signal end of orchestrated processing
    return current_target_bytes, all_diff_blocks_augmented


# --- Byte-Level Skip Report Generator ---
# (generate_byte_level_skip_report remains IDENTICAL to V4)
def generate_byte_level_skip_report(original_diff_blocks, final_patched_bytes, file2_bytes):
    byte_level_skips = []
    len_final_patched = len(final_patched_bytes)
    # len_file2 = len(file2_bytes) # Not directly used in this version of the audit logic

    # print("\n--- Generating Byte-Level Skip Report (Audit) ---") # Console

    for diff_block in original_diff_blocks:
        patch_num = diff_block["patch_num_original"]
        new_data_from_f2 = diff_block["new_data_bytes"]
        base_offset_in_final = diff_block["original_offset"]

        # print(f"Audit - Patch {patch_num}, Orig.F1 Offset: 0x{base_offset_in_final:X}, Expected New Data (from F2 diff): {new_data_from_f2.hex()}") # Console

        for i, expected_byte_from_f2 in enumerate(new_data_from_f2):
            expected_target_offset = base_offset_in_final + i
            if expected_target_offset < len_final_patched:
                actual_byte_in_final = final_patched_bytes[expected_target_offset]
                if actual_byte_in_final != expected_byte_from_f2:
                    byte_level_skips.append({
                        "patch_number": patch_num,
                        "absolute_offset_in_final": expected_target_offset,
                        "expected_byte_f2": expected_byte_from_f2,
                        "actual_byte_final": actual_byte_in_final,
                        "reason": "Byte mismatch after all passes."
                    })
                    # print(f"  SKIP @ 0x{expected_target_offset:X}: Expected {expected_byte_from_f2:02X}, Found {actual_byte_in_final:02X}") # Console
            else:
                byte_level_skips.append({
                    "patch_number": patch_num,
                    "absolute_offset_in_final": expected_target_offset,
                    "expected_byte_f2": expected_byte_from_f2,
                    "actual_byte_final": None,
                    "reason": "Expected byte location is beyond end of final patched file."
                })
                # print(f"  SKIP @ 0x{expected_target_offset:X}: Expected {expected_byte_from_f2:02X}, but offset is out of bounds (final_len={len_final_patched})") # Console
    # print(f"--- Byte-Level Skip Report Generation Complete. Found {len(byte_level_skips)} differing bytes. ---") # Console
    return byte_level_skips

# --- Streamlit UI ---
st.set_page_config(layout="wide")
st.title("Binary File Patcher V4 (Streamlit Edition)")
st.markdown("Compares File 1 & 2, applies differences to File 3 using a multi-pass strategy, and generates a byte-level audit log.")

# Initialize session state
if 'file1_bytes' not in st.session_state: st.session_state.file1_bytes = None
if 'file1_name' not in st.session_state: st.session_state.file1_name = "File1_Original"
if 'file2_bytes' not in st.session_state: st.session_state.file2_bytes = None
if 'file2_name' not in st.session_state: st.session_state.file2_name = "File2_Modified"
if 'original_file3_bytes' not in st.session_state: st.session_state.original_file3_bytes = None
if 'file3_name' not in st.session_state: st.session_state.file3_name = "File3_Target"
if 'diff_blocks' not in st.session_state: st.session_state.diff_blocks = []
if 'diff_count' not in st.session_state: st.session_state.diff_count = 0
if 'patched_file_bytes' not in st.session_state: st.session_state.patched_file_bytes = None
if 'byte_level_skips_report' not in st.session_state: st.session_state.byte_level_skips_report = []
if 'last_run_summary' not in st.session_state: st.session_state.last_run_summary = {}


# --- File Upload and Difference Calculation ---
col1, col2, col3 = st.columns(3)

with col1:
    st.header("File 1 (Original)")
    uploaded_file1 = st.file_uploader("Upload Original File (e.g., stock ECU map)", type=['bin', 'dat', None], key="file1_uploader")
    if uploaded_file1:
        st.session_state.file1_bytes = load_file_to_bytes_streamlit(uploaded_file1)
        st.session_state.file1_name = uploaded_file1.name
        if st.session_state.file1_bytes:
            st.success(f"Loaded: {st.session_state.file1_name} ({len(st.session_state.file1_bytes)} bytes)")
        # Reset dependent states if file1 changes
        st.session_state.diff_blocks = []
        st.session_state.diff_count = 0
        st.session_state.patched_file_bytes = None
        st.session_state.byte_level_skips_report = []


with col2:
    st.header("File 2 (Modified)")
    uploaded_file2 = st.file_uploader("Upload Modified File (e.g., Stage 1 ECU map)", type=['bin', 'dat', None], key="file2_uploader")
    if uploaded_file2:
        st.session_state.file2_bytes = load_file_to_bytes_streamlit(uploaded_file2)
        st.session_state.file2_name = uploaded_file2.name
        if st.session_state.file2_bytes:
            st.success(f"Loaded: {st.session_state.file2_name} ({len(st.session_state.file2_bytes)} bytes)")
        # Reset dependent states if file2 changes
        st.session_state.diff_blocks = []
        st.session_state.diff_count = 0
        st.session_state.patched_file_bytes = None
        st.session_state.byte_level_skips_report = []


with col3:
    st.header("File 3 (Target to Patch)")
    uploaded_file3 = st.file_uploader("Upload Target File (e.g., your current ECU map)", type=['bin', 'dat', None], key="file3_uploader")
    if uploaded_file3:
        st.session_state.original_file3_bytes = load_file_to_bytes_streamlit(uploaded_file3)
        st.session_state.file3_name = uploaded_file3.name
        if st.session_state.original_file3_bytes:
            st.success(f"Loaded: {st.session_state.file3_name} ({len(st.session_state.original_file3_bytes)} bytes)")
        st.session_state.patched_file_bytes = None # Reset if target changes
        st.session_state.byte_level_skips_report = []


# Calculate and display differences
if st.session_state.file1_bytes and st.session_state.file2_bytes and not st.session_state.diff_blocks:
    with st.spinner("Calculating differences between File 1 and File 2..."):
        start_time = time.time()
        st.session_state.diff_count = count_byte_differences(st.session_state.file1_bytes, st.session_state.file2_bytes)
        st.session_state.diff_blocks = find_differences_with_context(st.session_state.file1_bytes, st.session_state.file2_bytes)
        end_time = time.time()
        st.info(f"Differences calculated in {end_time - start_time:.2f}s.")

if st.session_state.file1_bytes and st.session_state.file2_bytes:
    st.subheader("Difference Summary (File 1 vs File 2)")
    st.write(f"Total Byte Differences: {st.session_state.diff_count}")
    st.write(f"Difference Blocks Found: {len(st.session_state.diff_blocks)}")
    if not st.session_state.diff_blocks and st.session_state.diff_count == 0:
        st.success("Files 1 and 2 are identical. No patches to generate.")

st.divider()

# --- Patching Controls and Execution ---
st.header("Patching Controls")
strictness_level = st.slider("Pass 1 Strictness Level (1=Risky/Aggressive, 4=Default, 6=Safest/Most Context)", 1, 6, 4, key="strictness_slider")

# Placeholders for live updates
patch_status_placeholder = st.empty()
patch_progress_placeholder = st.empty()
live_log_placeholder = st.empty() # For some live detailed logs

if st.button("Apply Differences to File 3", key="apply_button", type="primary"):
    patch_status_placeholder.info("Initiating patching process...")
    patch_progress_placeholder.progress(0)
    st.session_state.patched_file_bytes = None
    st.session_state.byte_level_skips_report = []
    st.session_state.last_run_summary = {}
    live_log_placeholder.text_area("Live Log", "", height=200, key="live_log_text_area_init") # Clear/Initialize

    if not st.session_state.file1_bytes or not st.session_state.file2_bytes or not st.session_state.original_file3_bytes:
        patch_status_placeholder.error("Error: Please load all three files (File 1, File 2, and File 3).")
    elif not st.session_state.diff_blocks and st.session_state.diff_count > 0 : # If diffs were somehow cleared but files loaded
        patch_status_placeholder.warning("Re-calculating differences before patching...")
        with st.spinner("Re-calculating differences..."):
             st.session_state.diff_blocks = find_differences_with_context(st.session_state.file1_bytes, st.session_state.file2_bytes)
        if not st.session_state.diff_blocks:
             patch_status_placeholder.info("No differences found between File 1 and File 2. Nothing to apply.")
        else:
             patch_status_placeholder.info(f"{len(st.session_state.diff_blocks)} difference blocks found. Proceeding to patch.")
    elif not st.session_state.diff_blocks and st.session_state.diff_count == 0:
        patch_status_placeholder.success("Files 1 and 2 are identical. No patches to apply to File 3.")
        st.session_state.patched_file_bytes = st.session_state.original_file3_bytes # No changes
        patch_progress_placeholder.progress(100)
    else:
        start_patch_time = time.time()
        # Create the mock GUI updater for Streamlit
        streamlit_updater = StreamlitGuiUpdate(patch_progress_placeholder, patch_status_placeholder, live_log_placeholder)

        with st.spinner(f"Applying {len(st.session_state.diff_blocks)} diff blocks... This may take some time."):
            final_bytes_result, all_original_diffs_for_audit = apply_patches_with_multiple_passes(
                st.session_state.original_file3_bytes,
                st.session_state.diff_blocks,
                strictness_level,
                streamlit_updater # Pass the Streamlit updater object
            )
            st.session_state.patched_file_bytes = final_bytes_result

            if final_bytes_result is not None and st.session_state.file2_bytes is not None and all_original_diffs_for_audit:
                patch_status_placeholder.info("Generating byte-level skip report (audit)...")
                patch_progress_placeholder.progress(95) # Before final audit
                st.session_state.byte_level_skips_report = generate_byte_level_skip_report(
                    all_original_diffs_for_audit,
                    final_bytes_result,
                    st.session_state.file2_bytes
                )
            else:
                st.session_state.byte_level_skips_report = []

        end_patch_time = time.time()
        total_patch_time = end_patch_time - start_patch_time
        patch_progress_placeholder.progress(100)

        # Prepare summary
        total_original_diff_bytes_f2 = sum(len(d["new_data_bytes"]) for d in all_original_diffs_for_audit) if all_original_diffs_for_audit else 0
        final_skipped_count_audit = len(st.session_state.byte_level_skips_report)
        successfully_matched_bytes_audit = total_original_diff_bytes_f2 - final_skipped_count_audit
        total_byte_diff_vs_f3_orig = count_byte_differences(st.session_state.original_file3_bytes, st.session_state.patched_file_bytes)

        st.session_state.last_run_summary = {
            "total_patch_time": total_patch_time,
            "total_original_diff_bytes_f2": total_original_diff_bytes_f2,
            "final_skipped_count_audit": final_skipped_count_audit,
            "successfully_matched_bytes_audit": successfully_matched_bytes_audit,
            "total_byte_diff_vs_f3_orig": total_byte_diff_vs_f3_orig
        }

        if st.session_state.patched_file_bytes is not None:
            if final_skipped_count_audit > 0:
                patch_status_placeholder.warning(
                    f"Patching complete with {final_skipped_count_audit} target bytes NOT matching File 2 data (see audit log). "
                    f"Total time: {total_patch_time:.2f}s."
                )
            elif total_original_diff_bytes_f2 > 0: # No skips and there were diffs
                 patch_status_placeholder.success(
                    f"Patching successfully completed! All {total_original_diff_bytes_f2} target bytes appear correct. "
                    f"Total time: {total_patch_time:.2f}s."
                )
            else: # No diffs to begin with
                 patch_status_placeholder.success(
                    f"Patching complete (no differences to apply). File 3 is unchanged. "
                    f"Total time: {total_patch_time:.2f}s."
                 )
            st.balloons()
        else:
            patch_status_placeholder.error("Patching process failed or resulted in no data.")

st.divider()

# --- Results and Download ---
if st.session_state.patched_file_bytes is not None:
    st.header("Patching Results")

    summary = st.session_state.last_run_summary
    if summary:
        st.subheader("Run Summary:")
        st.markdown(f"""
        - Patching Process Time: **{summary.get('total_patch_time', 0):.2f} seconds**
        - Total Target Bytes from File 2 (in diffs): **{summary.get('total_original_diff_bytes_f2', 0)}**
        - Bytes Correctly Matching File 2 in Patched File: **{summary.get('successfully_matched_bytes_audit', 0)}**
        - Individual Target Bytes NOT Matching (Audit): **<font color='red'>{summary.get('final_skipped_count_audit', 0)}</font>**
        - Overall Byte Difference (Patched File vs Original File 3): **{summary.get('total_byte_diff_vs_f3_orig', 'N/A')}**
        """, unsafe_allow_html=True)


    col_dl1, col_dl2 = st.columns(2)
    with col_dl1:
        patched_file_name = f"patched_{st.session_state.file3_name}" if st.session_state.file3_name else "patched_file.bin"
        st.download_button(
            label=f"Download Patched File ({patched_file_name})",
            data=st.session_state.patched_file_bytes,
            file_name=patched_file_name,
            mime="application/octet-stream",
            key="download_patched"
        )

    if st.session_state.byte_level_skips_report or (summary and summary.get('final_skipped_count_audit', 0) == 0 and summary.get('total_original_diff_bytes_f2',0) > 0):
        # Prepare skip log content for download/display
        log_content_lines = ["Byte-Level Skip Log (Audit of Final Patched File vs. Original File 2 Differences)\n"]
        if st.session_state.byte_level_skips_report:
            log_content_lines.append(f"The following {len(st.session_state.byte_level_skips_report)} individual bytes from original File1-File2 differences\n")
            log_content_lines.append(f"did not match the expected File 2 byte in the final patched file.\n")
            log_content_lines.append(f"Offsets are absolute in the *final patched file*.\n")
        elif summary and summary.get('final_skipped_count_audit', 0) == 0 and summary.get('total_original_diff_bytes_f2',0) > 0:
            log_content_lines.append("Audit complete: All target bytes from File 2 differences appear to be correctly in the final patched file.\n")
        else: # No diffs initially, or audit didn't run properly
            log_content_lines.append("No byte-level discrepancies found or no patches were applied that involved new data from File 2.\n")

        log_content_lines.append("-" * 40 + "\n\n")

        # Sort by offset for readability
        sorted_skips = sorted(st.session_state.byte_level_skips_report, key=lambda x: x.get("absolute_offset_in_final", -1))

        for skip_info in sorted_skips:
            patch_ref_num = skip_info.get("patch_number", "N/A")
            abs_offset = skip_info.get("absolute_offset_in_final", -1)
            expected_f2_byte = skip_info.get("expected_byte_f2", -1)
            actual_final_byte = skip_info.get("actual_byte_final", None)
            reason = skip_info.get("reason", "Mismatch")
            actual_str = f"{actual_final_byte:02X}" if actual_final_byte is not None else "N/A (OOB)"

            log_content_lines.append(f"Reference Original Patch #: {patch_ref_num}\n")
            log_content_lines.append(f"  Absolute Offset in Final File: 0x{abs_offset:08X}\n")
            log_content_lines.append(f"  Expected Byte (from File 2 diff): 0x{expected_f2_byte:02X}\n")
            log_content_lines.append(f"  Actual Byte in Final File:      {actual_str}\n") # Adjusted spacing
            log_content_lines.append(f"  Reason: {reason}\n")
            log_content_lines.append("-" * 20 + "\n")

        skip_log_full_text = "".join(log_content_lines)

        with col_dl2:
            st.download_button(
                label="Download Byte-Level Skip Log (.txt)",
                data=skip_log_full_text,
                file_name="byte_skip_log.txt",
                mime="text/plain",
                key="download_skiplog"
            )

        if st.session_state.byte_level_skips_report:
            st.subheader("Byte-Level Skip Log Preview (First 20 Mismatches)")
            preview_lines = skip_log_full_text.splitlines()
            # Find the start of actual entries (after header)
            entry_start_index = 0
            for i, line in enumerate(preview_lines):
                if line.startswith("Reference Original Patch #"):
                    entry_start_index = i
                    break
            
            # Count actual skip entries (each takes about 5 lines in the log format)
            num_preview_entries = 20 
            lines_per_entry_approx = 5 
            end_line_index = entry_start_index + (num_preview_entries * lines_per_entry_approx)

            st.text_area("Skip Log Preview", "\n".join(preview_lines[:end_line_index]), height=300, key="skiplog_preview_area")
            if len(sorted_skips) > num_preview_entries:
                 st.caption(f"... and {len(sorted_skips) - num_preview_entries} more differing bytes (see full downloaded log).")
        elif summary and summary.get('final_skipped_count_audit', 0) == 0 and summary.get('total_original_diff_bytes_f2',0) > 0:
            st.success("Audit Log: All target bytes appear correct.")


st.sidebar.header("About")
st.sidebar.info(
    "This is a Streamlit web application version of the V4 Binary File Patcher. "
    "It helps apply differences between two binary files (e.g., ECU maps) to a third target file."
)
st.sidebar.markdown("---")
st.sidebar.markdown("### How to Use:")
st.sidebar.markdown("1. **Upload File 1:** The original/stock binary file.")
st.sidebar.markdown("2. **Upload File 2:** The modified binary file (e.g., a tuned version of File 1).")
st.sidebar.markdown("   *(Differences between File 1 and 2 will be calculated.)*")
st.sidebar.markdown("3. **Upload File 3:** The target binary file you want to patch.")
st.sidebar.markdown("4. **Adjust Strictness:** Controls how carefully Pass 1 searches for patch locations.")
st.sidebar.markdown("5. **Click 'Apply Differences'.**")
st.sidebar.markdown("6. **Review Results & Download:** Download the patched file and the skip/audit log.")
