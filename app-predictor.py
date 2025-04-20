# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
from collections import Counter
import math
import datetime

# --- Helper Functions ---
def create_default_grid(rows, cols, default_value):
    return [[default_value for _ in range(cols)] for _ in range(rows)]

# --- Prediction Functions (Keep unchanged) ---
def predict_majority6(grid, current_col):
    if not grid or not grid[0]: return "Waiting for data"
    if current_col >= len(grid[0]): return "No prediction"
    current_column_data = [row[current_col] for row in grid if len(row) > current_col and row[current_col] is not None]
    len_data = len(current_column_data)
    if len_data == 0: return "Waiting for data"
    count_b = current_column_data.count('Banker'); count_p = current_column_data.count('Player')
    if count_b >= 4 or count_p >= 4: return "Waiting (4+ identical)"
    if len_data == 5:
        if count_b > count_p: return "Banker"
        if count_p > count_b: return "Player"
    if len_data >= 2:
        last_two = current_column_data[-2:]
        if last_two[0] == last_two[1]:
            if len_data >= 3 and current_column_data[-3] == last_two[0]: return "Waiting for pattern"
            return last_two[0]
    return "Waiting for pattern"

def predict_x_mark(grid, current_row, current_col):
    if not grid or not grid[0]: return "Waiting for data"
    num_rows = len(grid); num_cols = len(grid[0]) # Use actual grid dimensions
    pred_row, pred_col = current_row, current_col
    max_cols_setting = st.session_state.get('cols', 18) # Get max cols setting
    if pred_col >= max_cols_setting: return "No prediction" # Check against max setting

    matrix_start_row = (pred_row // 3) * 3; matrix_start_col = (pred_col // 3) * 3
    if not (0 <= matrix_start_row < num_rows and 0 <= matrix_start_col < max_cols_setting):
        return "Waiting (Matrix OOB)"
    if matrix_start_row >= len(grid) or matrix_start_col >= len(grid[0]):
        return "Waiting (Matrix Calc OOB)"
    first_value = grid[matrix_start_row][matrix_start_col]
    if first_value is None: return "Waiting (Need Matrix Top-Left)"

    relative_row = pred_row - matrix_start_row; relative_col = pred_col - matrix_start_col
    is_pred_cell_an_x_position = (relative_row == 0 and relative_col == 2) or \
                                 (relative_row == 1 and relative_col == 1) or \
                                 (relative_row == 2 and relative_col == 0) or \
                                 (relative_row == 2 and relative_col == 2)

    if matrix_start_row + 2 < num_rows and matrix_start_col + 2 < max_cols_setting:
        if matrix_start_row + 2 >= len(grid) or matrix_start_col + 2 >= len(grid[0]):
             return "Waiting (Matrix Incomplete)"
        if is_pred_cell_an_x_position: return first_value
        else: return "Waiting (Not X Position)"
    else:
        if is_pred_cell_an_x_position:
             return "Waiting (Matrix Incomplete)"
        else:
             return "Waiting (Matrix OOB / Incomplete)"

def predict_no_mirror(grid, current_row, current_col):
    if not grid or not grid[0]: return "Waiting for data"
    num_rows = len(grid); num_cols = len(grid[0])
    pred_row, pred_col = current_row, current_col
    max_cols_setting = st.session_state.get('cols', 18)
    if pred_col >= max_cols_setting: return "No prediction"
    if pred_col < 2: return "Waiting (Need Col 3+)"
    if not (0 <= pred_row < num_rows): return "Error: Invalid row index"
    if pred_col - 1 < 0 or pred_col - 2 < 0:
         return "Error: Invalid source column index calculation"
    if pred_row >= len(grid) or pred_col-1 >= len(grid[pred_row]) or pred_col-2 >= len(grid[pred_row]):
        return "Waiting (Need values in prev cells)"

    val_prev2 = grid[pred_row][pred_col - 2]
    val_prev1 = grid[pred_row][pred_col - 1]
    if val_prev1 is None or val_prev2 is None: return "Waiting (Need values in prev cells)"
    if val_prev1 == val_prev2: return 'Player' if val_prev1 == 'Banker' else 'Banker'
    else: return val_prev1

def predict_special89(special89_state, last_result_after_natural):
    if special89_state == "waiting_for_natural": return "Waiting for Natural"
    elif special89_state == "waiting_for_result_after_natural": return "Waiting for next result"
    elif special89_state == "ready_for_prediction": return last_result_after_natural if last_result_after_natural else "Error: No result stored"
    else: return "Waiting for Natural"

def predict_2_and_5(grid, current_row, current_col):
    if not grid or not grid[0]: return "Waiting for data"
    num_rows = len(grid); num_cols = len(grid[0])
    pred_row, pred_col = current_row, current_col
    max_cols_setting = st.session_state.get('cols', 18)
    if pred_col >= max_cols_setting: return "No prediction"
    if pred_col < 3: return "Waiting (Starts Col 4)"

    source_col = pred_col - 1
    if 0 <= pred_row <= 2: source_row = 1
    elif 3 <= pred_row <= 5: source_row = 4
    else: return "Error: Invalid pred row index"

    if not (0 <= source_row < len(grid) and 0 <= source_col < len(grid[source_row])):
        return f"Waiting (Src Cell [{source_row+1},{source_col+1}] OOB/Not created)"

    source_value = grid[source_row][source_col]
    if source_value is None: return f"Waiting (Src Cell [{source_row+1},{source_col+1}] empty)"
    else: return source_value

# --- Aggregate & Utility Functions ---
def calculate_prediction_percent(predictions_list):
    valid_predictions = [p for p in predictions_list if p in ['Banker', 'Player']]
    total = len(valid_predictions)
    if total == 0: return "Waiting..."
    b_count = valid_predictions.count('Banker'); p_count = total - b_count
    if b_count > p_count: return f"{math.ceil((b_count/total)*100)}% Banker"
    elif p_count > b_count: return f"{math.ceil((p_count/total)*100)}% Player"
    else: return "50% B / 50% P"

def get_final_prediction(predictions):
    valid_preds = [p for p in predictions if p in ['Banker', 'Player']]
    num_valid = len(valid_preds)
    if num_valid < 3 : return "No prediction (Need ≥3 signals)"
    pred_counts = Counter(valid_preds)
    most_common = pred_counts.most_common(1)
    if not most_common: return "No prediction"
    outcome, count = most_common[0]
    if count >= 3:
         return f"Predict <b>{outcome}</b> ({count}/{num_valid} agree)"
    return "No prediction (Weak Signal)"

def get_bead_road_stats(grid):
    if not grid or not grid[0]: return 0, 0
    flat_grid = [cell for row in grid for cell in row if cell is not None and cell in ['Banker', 'Player']]
    b_count = flat_grid.count('Banker'); p_count = flat_grid.count('Player')
    return b_count, p_count

# --- Progression Strategy Functions ---

def initialize_progression_state():
    """Initializes all progression and session state variables."""
    st.session_state.progression_sequence = [1, 1, 2, 4, 8]
    st.session_state.current_progression_index = 0

    if 'initial_bet_unit' not in st.session_state:
        st.session_state.initial_bet_unit = 40000.0
    if 'current_balance' not in st.session_state:
        st.session_state.current_balance = 2000000.0

    st.session_state.current_balance = float(st.session_state.get('current_balance', 10000000.0))
    st.session_state.initial_bet_unit = float(st.session_state.get('initial_bet_unit', 100000.0))

    st.session_state.session_start_balance = st.session_state.current_balance
    st.session_state.last_withdrawal_profit_level = 0.0
    st.session_state.session_start_time = datetime.datetime.now()
    st.session_state.session_wins = 0
    st.session_state.session_losses = 0
    st.session_state.bet_history = []

    # <<< State for Delayed Martingale >>>
    st.session_state.betting_mode = 'PROFIT' # 'PROFIT' or 'RECOVERY'
    st.session_state.current_recovery_multiplier = 1 # 1, 2, 4, 8...
    st.session_state.recovery_bets_at_this_level = 0 # 0 or 1: Bets made at current multiplier
    # <<< END NEW STATE >>>

    calculate_suggested_bet() # Calculate the first bet

def calculate_suggested_bet():
    """Calculates the next bet based on the current mode (PROFIT or RECOVERY)."""
    current_balance = float(st.session_state.get('current_balance', 0.0))
    session_start = float(st.session_state.get('session_start_balance', current_balance))
    unit = float(st.session_state.get('initial_bet_unit', 1.0))
    mode = st.session_state.get('betting_mode', 'PROFIT')

    # 1. Determine Current Mode (and handle transitions)
    if current_balance >= session_start:
        if mode == 'RECOVERY':
             # Only show toast if actually transitioning
             st.toast("🎉 Về bờ! Chuyển sang chế độ Lợi Nhuận (1-1-2-4-8).", icon="💰")
             st.session_state.betting_mode = 'PROFIT'
             st.session_state.current_progression_index = 0 # Reset standard progression
             st.session_state.current_recovery_multiplier = 1 # Reset recovery state
             st.session_state.recovery_bets_at_this_level = 0
        mode = 'PROFIT' # Ensure mode is set/kept
        st.session_state.betting_mode = 'PROFIT' # Update state explicitly
    else: # current_balance < session_start
        if mode == 'PROFIT':
             # Only show toast if actually transitioning
             st.toast("🚨 Vốn giảm! Chuyển sang chế độ Gỡ Lỗ (Delay Martingale).", icon="🛡️")
             st.session_state.betting_mode = 'RECOVERY'
             st.session_state.current_recovery_multiplier = 1 # Start recovery at 1x
             st.session_state.recovery_bets_at_this_level = 0 # Start fresh at this level
        mode = 'RECOVERY' # Ensure mode is set/kept
        st.session_state.betting_mode = 'RECOVERY' # Update state explicitly

    # 2. Calculate Bet Based on Mode
    suggested_bet = 0
    if mode == 'PROFIT':
        idx = st.session_state.get('current_progression_index', 0)
        sequence = st.session_state.get('progression_sequence', [1, 1, 2, 4, 8])
        if not (0 <= idx < len(sequence)):
            idx = 0
            st.session_state.current_progression_index = 0
        multiplier = sequence[idx]
        suggested_bet = unit * multiplier

    elif mode == 'RECOVERY':
        multiplier = st.session_state.get('current_recovery_multiplier', 1)
        suggested_bet = unit * multiplier

    st.session_state.suggested_bet_amount = float(suggested_bet)

def handle_progression_win(payout_ratio):
    """Handles a win, updates balance, history, and determines next bet state."""
    bet_amount = float(st.session_state.get('suggested_bet_amount', 0.0))
    current_balance = float(st.session_state.get('current_balance', 0.0))
    session_start = float(st.session_state.get('session_start_balance', current_balance))
    mode = st.session_state.get('betting_mode', 'PROFIT')

    winnings = bet_amount * payout_ratio
    new_balance = current_balance + winnings

    # Record Win in History
    if 'bet_history' not in st.session_state: st.session_state.bet_history = []
    st.session_state.bet_history.append({
        'outcome': 'Win', 'amount': bet_amount, 'profit': winnings, 'timestamp': datetime.datetime.now(), 'mode': mode
    })
    max_history = 50
    if len(st.session_state.bet_history) > max_history:
        st.session_state.bet_history = st.session_state.bet_history[-max_history:]

    # Update Balance and Session Win Count
    st.session_state.current_balance = new_balance
    st.session_state.session_wins = st.session_state.get('session_wins', 0) + 1

    # --- Update State for Next Bet ---
    # Check if balance reached start *before* updating progression state
    if new_balance >= session_start and mode == 'RECOVERY':
        # Transition to PROFIT handled by calculate_suggested_bet on next run
         pass # Calculate_suggested_bet will handle the mode switch toast and state reset
    elif mode == 'PROFIT':
        # Advance standard progression
        prog_idx = st.session_state.get('current_progression_index', 0) + 1
        sequence_len = len(st.session_state.get('progression_sequence', [1, 1, 2, 4, 8]))
        if prog_idx >= sequence_len:
            st.session_state.current_progression_index = 0 # Reset sequence
            st.toast("Chuỗi thắng PROFIT hoàn tất! 🎉 Reset về mức cược đầu.", icon="✅")
        else:
            st.session_state.current_progression_index = prog_idx
    elif mode == 'RECOVERY':
        # Win during recovery: Reset bets_at_level, keep multiplier (repeat bet logic)
        current_multiplier = st.session_state.get('current_recovery_multiplier', 1)
        st.toast(f"Thắng khi đang gỡ! 👍 Tiếp tục cược {current_multiplier} unit.", icon="✅")
        st.session_state.recovery_bets_at_this_level = 0 # Reset counter, next bet is first at this level

    # Check for overall session profit target
    last_notified_level = st.session_state.get('last_withdrawal_profit_level', 0.0)
    current_profit = new_balance - session_start
    profit_target = 800000.0
    if current_profit >= profit_target and last_notified_level < profit_target:
        st.toast(f"✅ Đạt mục tiêu phiên ({profit_target:,.0f} lợi nhuận)!", icon="🎯")
        st.balloons()
        st.session_state.last_withdrawal_profit_level = profit_target

    # Calculate next bet (will determine correct mode and state)
    calculate_suggested_bet()

def handle_progression_loss():
    """Handles a loss, updates balance, history, and determines next bet state."""
    bet_amount = float(st.session_state.get('suggested_bet_amount', 0.0))
    current_balance = float(st.session_state.get('current_balance', 0.0))
    unit = float(st.session_state.get('initial_bet_unit', 1.0))
    mode = st.session_state.get('betting_mode', 'PROFIT') # Get mode *before* updating balance

    new_balance = current_balance - bet_amount

    # Record Loss in History
    if 'bet_history' not in st.session_state: st.session_state.bet_history = []
    st.session_state.bet_history.append({
        'outcome': 'Loss', 'amount': bet_amount, 'profit': -bet_amount, 'timestamp': datetime.datetime.now(), 'mode': mode
    })
    max_history = 50
    if len(st.session_state.bet_history) > max_history:
        st.session_state.bet_history = st.session_state.bet_history[-max_history:]

    # Update Balance and Session Loss Count
    st.session_state.current_balance = new_balance
    st.session_state.session_losses = st.session_state.get('session_losses', 0) + 1

    # --- Update State for Next Bet ---
    if mode == 'PROFIT':
        # Loss during profit mode: Reset standard progression index
        st.session_state.current_progression_index = 0
        st.toast("Thua khi đang Lợi Nhuận! 😢 Reset về mức cược đầu.", icon="❌")
        # Check if this loss triggers recovery mode happens in calculate_suggested_bet

    elif mode == 'RECOVERY':
        # Loss during recovery: Apply the new 1-1-2-2-4-4 logic
        current_multiplier = st.session_state.get('current_recovery_multiplier', 1)
        bets_at_level = st.session_state.get('recovery_bets_at_this_level', 0)

        bets_at_level += 1 # Increment bets made (now lost) at this level

        if current_multiplier == 1:
            if bets_at_level == 1:
                # First loss at 1x, next bet is still 1x
                st.session_state.current_recovery_multiplier = 1
                st.session_state.recovery_bets_at_this_level = bets_at_level # Now 1
                st.toast("Thua lần 1 khi Gỡ! 😥 Tiếp tục cược 1 unit.", icon="❌")
            elif bets_at_level >= 2:
                # Second loss at 1x, double for next bet
                st.session_state.current_recovery_multiplier = 2
                st.session_state.recovery_bets_at_this_level = 0 # Reset counter for 2x level
                st.toast("Thua lần 2 khi Gỡ! 😥 Bắt đầu nhân đôi (2 units).", icon="❌")
        else: # current_multiplier > 1
            if bets_at_level == 1:
                # First loss at this >1x level, repeat the bet (keep multiplier)
                st.session_state.current_recovery_multiplier = current_multiplier
                st.session_state.recovery_bets_at_this_level = bets_at_level # Now 1
                st.toast(f"Thua mức {current_multiplier}x! 😥 Lặp lại cược {current_multiplier} units.", icon="❌")
            elif bets_at_level >= 2:
                # Second loss at this >1x level, double for next bet
                new_multiplier = current_multiplier * 2
                st.session_state.current_recovery_multiplier = new_multiplier
                st.session_state.recovery_bets_at_this_level = 0 # Reset counter for new level
                st.toast(f"Thua mức {current_multiplier}x lần 2! 😥 Nhân đôi lên {new_multiplier} units.", icon="❌")

    # Calculate next bet (will check balance vs start and potentially switch mode)
    calculate_suggested_bet()


def update_initial_bet_unit():
    if 'input_initial_bet_unit' in st.session_state:
        try:
            new_unit = float(st.session_state.input_initial_bet_unit)
            if new_unit > 0: st.session_state.initial_bet_unit = new_unit
            else: st.session_state.initial_bet_unit = 1.0
        except (ValueError, TypeError):
             st.session_state.initial_bet_unit = 100000.0
    calculate_suggested_bet()

def set_current_balance_from_input():
    """Callback to update current_balance AND reset session tracking and betting state."""
    if 'starting_balance_input' in st.session_state:
        try:
            new_balance = float(st.session_state.starting_balance_input)
            if new_balance >= 0:
                 st.session_state.current_balance = new_balance
                 # Reset session tracking AND betting strategy state
                 st.session_state.session_start_balance = st.session_state.current_balance
                 st.session_state.last_withdrawal_profit_level = 0.0
                 st.session_state.session_start_time = datetime.datetime.now()
                 st.session_state.session_wins = 0
                 st.session_state.session_losses = 0
                 st.session_state.bet_history = []
                 st.session_state.betting_mode = 'PROFIT' # Start fresh in profit mode
                 st.session_state.current_progression_index = 0
                 st.session_state.current_recovery_multiplier = 1 # Reset recovery state
                 st.session_state.recovery_bets_at_this_level = 0
                 st.toast("Số dư cập nhật. Phiên theo dõi & Chiến lược cược mới bắt đầu.", icon="💰")
                 calculate_suggested_bet()
            else:
                 st.warning("Số dư không thể là số âm.")
        except (ValueError, TypeError):
            st.error("Vui lòng nhập số dư hợp lệ.")
    else:
        st.error("Lỗi: Không tìm thấy trường nhập số dư ban đầu.")

def reset_session():
    """Resets only the current session's tracking, progression, and betting mode."""
    st.session_state.current_progression_index = 0
    st.session_state.betting_mode = 'PROFIT'
    st.session_state.current_recovery_multiplier = 1
    st.session_state.recovery_bets_at_this_level = 0

    st.session_state.session_start_balance = st.session_state.current_balance
    st.session_state.last_withdrawal_profit_level = 0.0
    st.session_state.session_start_time = datetime.datetime.now()
    st.session_state.session_wins = 0
    st.session_state.session_losses = 0
    st.session_state.bet_history = []
    st.toast("Phiên cược đã được reset.", icon="🔄")
    calculate_suggested_bet()

# --- Backend Functions ---
# ... (update_all_predictions, add_result, undo_last_result - Giữ nguyên) ...
def update_all_predictions():
    grid = st.session_state.get('bead_road_grid')
    row = st.session_state.get('current_bead_road_row')
    col = st.session_state.get('current_bead_road_col')
    s89_state = st.session_state.get('special89_state')
    s89_result = st.session_state.get('last_result_after_natural')

    if grid is None or row is None or col is None or s89_state is None:
         st.session_state.predictions = {
            'majority6': "Waiting...", 'xMark': "Waiting...", 'noMirror': "Waiting...",
            'special89': "Waiting...", '2and5': "Waiting...",
            'percentage': "Waiting...", 'final': "Waiting for data..."
        }
         return

    pred_maj6 = predict_majority6(grid, col)
    pred_x_mark = predict_x_mark(grid, row, col)
    pred_no_mirror = predict_no_mirror(grid, row, col)
    pred_s89 = predict_special89(s89_state, s89_result)
    pred_2and5 = predict_2_and_5(grid, row, col)

    predictions_list = [pred_maj6, pred_x_mark, pred_no_mirror, pred_s89, pred_2and5]
    pred_percent = calculate_prediction_percent(predictions_list)
    final_pred = get_final_prediction(predictions_list)

    st.session_state.predictions = {
        'majority6': pred_maj6, 'xMark': pred_x_mark, 'noMirror': pred_no_mirror,
        'special89': pred_s89, '2and5': pred_2and5,
        'percentage': pred_percent, 'final': final_pred
    }

def add_result(result, is_natural):
    required_keys = ['special89_state', 'last_result_after_natural', 'last_natural_pos',
                     'current_bead_road_row', 'current_bead_road_col', 'game_count',
                     'rows', 'cols', 'bead_road_grid', 'natural_marks_grid']
    if not all(key in st.session_state for key in required_keys):
        st.error("Lỗi: Trạng thái game chưa được khởi tạo đầy đủ.")
        return

    prev_s89_state = st.session_state.get('special89_state', 'waiting_for_natural')
    prev_s89_last_res = st.session_state.get('last_result_after_natural', None)
    prev_s89_last_nat_pos = st.session_state.get('last_natural_pos', None)
    prev_bead_row = st.session_state.get('current_bead_road_row')
    prev_bead_col = st.session_state.get('current_bead_road_col')

    game_id = st.session_state.get('game_count', 0) + 1
    new_game = {
        'id': game_id, 'result': result, 'is_natural': is_natural,
        'prev_s89_state': prev_s89_state, 'prev_s89_last_res': prev_s89_last_res,
        'prev_s89_last_nat_pos': prev_s89_last_nat_pos, 'prev_bead_row': prev_bead_row,
        'prev_bead_col': prev_bead_col, 'bead_row_filled': prev_bead_row, 'bead_col_filled': prev_bead_col,
    }

    current_row = st.session_state.get('current_bead_road_row', 0)
    current_col = st.session_state.get('current_bead_road_col', 0)
    rows = st.session_state.get('rows', 6)
    cols = st.session_state.get('cols', 18)
    grid = st.session_state.get('bead_road_grid'); nat_grid = st.session_state.get('natural_marks_grid')

    if grid is None or nat_grid is None:
        st.error("Lỗi: Grid dữ liệu không tồn tại.")
        return

    if len(grid) != rows or (len(grid) > 0 and len(grid[0]) != cols):
        st.warning(f"Grid dimensions mismatch ({len(grid)}x{len(grid[0]) if grid else 0} vs {rows}x{cols}), re-creating grid.")
        grid = create_default_grid(rows, cols, None)
        nat_grid = create_default_grid(rows, cols, False)
        current_row, current_col = 0, 0
        st.session_state.current_bead_road_row = 0
        st.session_state.current_bead_road_col = 0
        st.session_state.bead_road_grid = grid
        st.session_state.natural_marks_grid = nat_grid


    if 0 <= current_row < rows and 0 <= current_col < cols:
        grid[current_row][current_col] = result
        nat_grid[current_row][current_col] = is_natural

        st.session_state.bead_road_grid = grid
        st.session_state.natural_marks_grid = nat_grid
        next_row = current_row + 1; next_col = current_col
        if next_row >= rows: next_row = 0; next_col = current_col + 1

        st.session_state.current_bead_road_row = next_row
        st.session_state.current_bead_road_col = next_col

        if next_col >= cols:
            st.toast("Bead Road Grid is full.", icon="⚠️")
    else:
        st.toast(f"Current position ({current_row}, {current_col}) is outside grid bounds ({rows}x{cols}). Grid might be full.", icon="error")

    st.session_state.game_history = st.session_state.get('game_history', []) + [new_game]
    st.session_state.game_count = game_id

    current_s89_state = prev_s89_state; next_s89_state = current_s89_state
    next_nat_pos = prev_s89_last_nat_pos; next_res_after_nat = prev_s89_last_res
    if is_natural:
        if 0 <= current_row < rows and 0 <= current_col < cols:
             next_nat_pos = {'row': current_row, 'col': current_col}
        else:
             next_nat_pos = None
        next_s89_state = "waiting_for_result_after_natural"; next_res_after_nat = None
    elif current_s89_state == "waiting_for_result_after_natural":
        next_res_after_nat = result; next_s89_state = "ready_for_prediction"
    elif current_s89_state == "ready_for_prediction":
        if not is_natural:
             next_s89_state = "waiting_for_natural"
        else:
             if 0 <= current_row < rows and 0 <= current_col < cols:
                  next_nat_pos = {'row': current_row, 'col': current_col}
             else:
                  next_nat_pos = None
             next_s89_state = "waiting_for_result_after_natural"
             next_res_after_nat = None

    st.session_state.last_natural_pos = next_nat_pos; st.session_state.special89_state = next_s89_state
    st.session_state.last_result_after_natural = next_res_after_nat

    update_all_predictions()
    # st.rerun() # No-op

def undo_last_result():
    history = st.session_state.get('game_history', [])
    if not history: st.toast("Không có gì để hoàn tác.", icon="🤷‍♂️"); return

    undone_game = history.pop(); st.session_state.game_history = history
    st.session_state.game_count = st.session_state.get('game_count', 1) - 1

    # Note: Undo does not easily revert the betting state machine.
    # The suggested bet after undo might not be accurate if undoing across mode changes.

    row_filled = undone_game.get('bead_row_filled'); col_filled = undone_game.get('bead_col_filled')
    if row_filled is not None and col_filled is not None:
        rows = st.session_state.get('rows', 6)
        cols = st.session_state.get('cols', 18)
        grid = st.session_state.get('bead_road_grid'); nat_grid = st.session_state.get('natural_marks_grid')
        if grid is None or nat_grid is None:
            st.error("Lỗi: Grid dữ liệu không tồn tại khi hoàn tác.")
            st.session_state.current_bead_road_row = undone_game.get('prev_bead_row', 0)
            st.session_state.current_bead_road_col = undone_game.get('prev_bead_col', 0)
        elif 0 <= row_filled < rows and 0 <= col_filled < cols:
            if row_filled < len(grid) and col_filled < len(grid[0]):
                 grid[row_filled][col_filled] = None
            if row_filled < len(nat_grid) and col_filled < len(nat_grid[0]):
                 nat_grid[row_filled][col_filled] = False
            st.session_state.bead_road_grid = grid; st.session_state.natural_marks_grid = nat_grid
            st.session_state.current_bead_road_row = row_filled
            st.session_state.current_bead_road_col = col_filled
        else:
             st.session_state.current_bead_road_row = undone_game.get('prev_bead_row', 0)
             st.session_state.current_bead_road_col = undone_game.get('prev_bead_col', 0)
    else:
         st.session_state.current_bead_road_row = undone_game.get('prev_bead_row', 0)
         st.session_state.current_bead_road_col = undone_game.get('prev_bead_col', 0)

    st.session_state.special89_state = undone_game.get('prev_s89_state', 'waiting_for_natural')
    st.session_state.last_result_after_natural = undone_game.get('prev_s89_last_res', None)
    st.session_state.last_natural_pos = undone_game.get('prev_s89_last_nat_pos', None)

    update_all_predictions()
    calculate_suggested_bet() # Recalculate bet based on current state
    # st.rerun() # No-op


# <<< MODIFICATION: Ensure new states are reset >>>
def reset_game():
    keys_to_reset = [
        'initialized', 'game_history', 'game_count', 'bead_road_grid', 'natural_marks_grid',
        'current_bead_road_row', 'current_bead_road_col', 'special89_state',
        'last_natural_pos', 'last_result_after_natural', 'predictions',
        'progression_sequence', 'current_progression_index', 'initial_bet_unit',
        'current_balance', 'suggested_bet_amount',
        'starting_balance_input', 'input_initial_bet_unit',
        'session_start_balance', 'last_withdrawal_profit_level',
        'session_start_time', 'session_wins', 'session_losses',
        'bet_history',
        # Reset betting strategy state too
        'betting_mode', 'current_recovery_multiplier', 'recovery_bets_at_this_level'
    ]
    for key in keys_to_reset:
        if key in st.session_state:
            del st.session_state[key]

    st.toast("Game đã được reset hoàn toàn!", icon="🔄")
    # st.rerun() # No-op: App will rerun because state is cleared

# --- Main Application ---
def main():
    st.set_page_config(
        page_title="Baccarat Pro Predictor",
        layout="wide",
        initial_sidebar_state="collapsed"
    )
    st.markdown('<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.2/css/all.min.css">', unsafe_allow_html=True)

    default_cols = 18

    if 'initialized' not in st.session_state:
        st.session_state.initialized = True
        st.session_state.rows = 6
        st.session_state.cols = default_cols
        initialize_progression_state()
        st.session_state.game_history = []
        st.session_state.game_count = 0
        st.session_state.bead_road_grid = create_default_grid(st.session_state.rows, st.session_state.cols, None)
        st.session_state.natural_marks_grid = create_default_grid(st.session_state.rows, st.session_state.cols, False)
        st.session_state.current_bead_road_row = 0
        st.session_state.current_bead_road_col = 0
        st.session_state.special89_state = "waiting_for_natural"
        st.session_state.last_natural_pos = None
        st.session_state.last_result_after_natural = None
        update_all_predictions() # Initial prediction calculation
    # Ensure necessary states exist if already initialized
    elif 'cols' not in st.session_state:
         st.session_state.cols = default_cols
    if 'betting_mode' not in st.session_state:
        st.session_state.betting_mode = 'PROFIT'
    if 'current_recovery_multiplier' not in st.session_state:
        st.session_state.current_recovery_multiplier = 1
    if 'recovery_bets_at_this_level' not in st.session_state:
         st.session_state.recovery_bets_at_this_level = 0
    if 'session_start_balance' not in st.session_state:
         st.session_state.session_start_balance = st.session_state.current_balance
    # Calculate bet just in case state was missing and needed initialization
    if 'suggested_bet_amount' not in st.session_state:
        calculate_suggested_bet()


    # --- CSS (Giữ nguyên) ---
    st.markdown(f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;700&family=Playfair+Display:wght@700&display=swap');
        :root {{
            --primary-bg: #1a1d21; --secondary-bg: #2c3035; --tertiary-bg: #3a3e44;
            --primary-text: #e1e3e6; --secondary-text: #a0a4ab; --accent-gold: #d4af37;
            --accent-gold-darker: #b8860b; --player-blue: #0d6efd; --player-blue-darker: #0a58ca;
            --banker-red: #dc3545; --banker-red-darker: #b02a37;
            --font-header: 'Playfair Display', serif; --font-body: 'Roboto', sans-serif;
            --border-color: #4a4e54; --border-radius: 6px;
            --box-shadow: 0 3px 10px rgba(0, 0, 0, 0.25); --box-shadow-inset: inset 0 1px 2px rgba(0, 0, 0, 0.4);
            --win-green: #4caf50; --loss-red: #f44336;
            --bead-size: 36px; --bead-font-size: 16px; --bead-margin: 3px;
            --bead-natural-marker-size: 16px; --bead-natural-marker-font-size: 11px;
            --bead-natural-marker-offset: -4px; --sticky-top-offset: 15px;
        }}
        body {{ font-family: var(--font-body); color: var(--primary-text); background-color: var(--primary-bg); }}
        .main {{ background-color: var(--primary-bg); padding: 15px; border-radius: var(--border-radius); font-family: var(--font-body); }}
        .stApp > header {{ display: none; }}
        .main .block-container {{ padding: 5px 10px !important; margin: 0 !important; }}

        /* Căn chỉnh cột lên trên */
        div[data-testid="stHorizontalBlock"] > div {{ align-self: flex-start !important; }}

        .stButton>button {{
            font-family: var(--font-body); padding: 6px 12px; border-radius: 4px;
            font-size: 12px; font-weight: bold; color: var(--primary-text); border: 1px solid var(--border-color);
            transition: all 0.2s ease-in-out; width: 100%; margin: 3px 0;
            box-shadow: var(--box-shadow-inset); text-align: center; background: var(--tertiary-bg);
            height: 35px; box-sizing: border-box; display: inline-flex !important;
            align-items: center !important; justify-content: center !important;
            line-height: 1; position: relative; white-space: nowrap;
        }}
        .stButton>button:hover:not(:disabled) {{ transform: translateY(-1px); box-shadow: 0 4px 8px rgba(0, 0, 0, 0.3); filter: brightness(1.1); }}
        .stButton>button:active:not(:disabled) {{ transform: translateY(0px); box-shadow: var(--box-shadow-inset); }}
        .stButton>button:disabled {{ background-color: #555 !important; color: #888 !important; cursor: not-allowed; box-shadow: none; transform: none; filter: grayscale(50%); border-color: #666; opacity: 0.7; }}

        div.stButton[data-testid*="player_std_btn"] button,
        div.stButton[data-testid*="player_natural_btn"] button {{
            background: linear-gradient(145deg, var(--player-blue), var(--player-blue-darker)) !important;
            border-color: var(--player-blue-darker) !important; color: white !important;
        }}
        div.stButton[data-testid*="banker_std_btn"] button,
        div.stButton[data-testid*="banker_natural_btn"] button {{
            background: linear-gradient(145deg, var(--banker-red), var(--banker-red-darker)) !important;
            border-color: var(--banker-red-darker) !important; color: white !important;
        }}
        div.stButton[data-testid*="natural_btn"] button span[data-testid="stButtonIcon"] {{
             color: var(--accent-gold) !important; font-size: 1.1em !important; margin-left: 5px;
             line-height: 1; filter: drop-shadow(0 0 1px black); vertical-align: middle;
        }}
        div.stButton[data-testid*="undo_std"] button,
        div.stButton[data-testid*="prog_loss_std"] button {{ background: linear-gradient(145deg, #6c757d, #5a6268); border-color: #5a6268; }}
        div.stButton[data-testid*="reset_std"] button {{ background: linear-gradient(145deg, #f57f17, #e65100); border-color: #e65100; color: #fff; }}
        div.stButton[data-testid*="reset_session_std"] button {{ background: linear-gradient(145deg, #ffca28, #ffb300); border-color: #ffb300; color: #111; }}
        div.stButton[data-testid*="prog_win_p_std"] button {{ background: linear-gradient(145deg, var(--player-blue), var(--player-blue-darker)); border-color: var(--player-blue-darker); }}
        div.stButton[data-testid*="prog_win_b_std"] button {{ background: linear-gradient(145deg, var(--banker-red), var(--banker-red-darker)); border-color: var(--banker-red-darker); }}
        div.stButton[data-testid*="set_starting_balance_button"] button {{ background: linear-gradient(145deg, var(--accent-gold), var(--accent-gold-darker)); border-color: var(--accent-gold-darker); color: #111; }}

        .app-title {{ font-family: var(--font-header); color: var(--accent-gold); font-size: 26px; font-weight: 700; text-align: center; margin-bottom: 15px; text-shadow: 1px 1px 2px rgba(0,0,0,0.4); }}
        .card {{ background-color: var(--secondary-bg); border-radius: var(--border-radius); padding: 18px; margin-bottom: 18px; box-shadow: var(--box-shadow); border: 1px solid var(--border-color); }}
        h4 {{ font-family: var(--font-body); font-weight: 700; color: var(--accent-gold); margin-top: 0; margin-bottom: 10px; border-bottom: 1px solid var(--border-color); padding-bottom: 6px; font-size: 14px; text-transform: uppercase; letter-spacing: 0.5px; display: flex; align-items: center; }}
        h4 i {{ margin-right: 8px; font-size: 1em; color: var(--accent-gold-darker); }}
        h6 {{ font-family: var(--font-body); font-weight: bold; color: var(--secondary-text); margin-top: 5px; margin-bottom: 5px; font-size: 13px; text-transform: uppercase;}}
        p, .stMarkdown p {{ color: var(--secondary-text); font-size: 13px; line-height: 1.5; margin-bottom: 8px; }}

        .stNumberInput, .stTextInput {{ display: flex; flex-direction: column; margin-bottom: 8px; }}
        label, .stNumberInput label, .stTextInput label {{ font-size: 13px !important; color: var(--secondary-text) !important; margin-bottom: 3px !important; display: block; font-weight: bold; order: 1; }}
        .stNumberInput input, .stTextInput input {{ font-size: 13px; color: var(--primary-text); background-color: var(--tertiary-bg); border: 1px solid var(--border-color); border-radius: 4px; padding: 6px 8px; width: 100%; box-shadow: var(--box-shadow-inset); box-sizing: border-box; order: 2; height: 31px; }}
        .stNumberInput input:focus, .stTextInput input:focus {{ border-color: var(--accent-gold); box-shadow: 0 0 4px rgba(212, 175, 55, 0.4), var(--box-shadow-inset); outline: none; }}
         ::placeholder {{ color: var(--secondary-text); opacity: 0.7; }}
        div[data-testid="stHorizontalBlock"] > div:nth-child(1) .stNumberInput {{ margin-bottom: 0 !important; }}
        div[data-testid="stHorizontalBlock"] > div:nth-child(2) {{ display: flex; align-items: flex-end; }}
        div[data-testid="stHorizontalBlock"] > div:nth-child(2) .stButton {{ width: 100%; }}
        div[data-testid="stHorizontalBlock"] > div:nth-child(2) .stButton button {{ margin-bottom: 0 !important; }}

        .stMetric {{ text-align: center; background-color: var(--secondary-bg); padding: 10px; border-radius: 4px; margin-top: 10px; margin-bottom: 8px; border: 1px solid var(--border-color); box-sizing: border-box; display: flex; flex-direction: column; justify-content: center; height: 70px; }}
        .stMetric label {{ color: var(--secondary-text) !important; font-size: 11px !important; font-weight: 400; text-transform: uppercase; margin-bottom: 2px !important; line-height: 1.2; }}
        .stMetric p {{ font-size: 20px !important; color: var(--primary-text) !important; font-weight: 700; margin-top: 2px; line-height: 1.1; word-wrap: break-word; }}
        .stMetric .stMetricDelta {{ display: none; }}
        div[data-testid="stMetric"]:has(label:contains("Lợi Nhuận Phiên")) {{ margin-top: 0px; }}

        .prediction-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 5px 10px; margin-bottom: 5px;}}
        .prediction-box {{ background-color: var(--tertiary-bg); border-radius: 4px; padding: 6px 10px; margin: 0; font-size: 12px; color: var(--primary-text); border-left: 3px solid var(--accent-gold); display: flex; justify-content: space-between; align-items: center; min-height: 30px; }}
        .prediction-box b {{ color: var(--secondary-text); font-weight: normal; margin-right: 5px; flex-shrink: 0; }}
        .prediction-box span {{ text-align: right; font-weight: bold; }}
        .final-prediction {{ color: var(--accent-gold); font-size: 14px; font-weight: bold; text-align: center; margin-top: 10px; padding: 8px; background-color: var(--secondary-bg); border-radius: 4px; border: 1px solid var(--accent-gold-darker); box-shadow: 0 0 6px rgba(212, 175, 55, 0.2); }}
        .final-prediction b {{ font-weight: 700; }}
        .prediction-result-Banker {{ color: var(--banker-red); }}
        .prediction-result-Player {{ color: var(--player-blue); }}
        .final-prediction .prediction-result-Banker {{ color: var(--banker-red); }}
        .final-prediction .prediction-result-Player {{ color: var(--player-blue); }}

        .bead-road-sticky-container,
        .mid-col-sticky-container {{ position: sticky; top: var(--sticky-top-offset, 15px); z-index: 10; }}
        .mid-col-sticky-container {{ z-index: 9; }}

        .bead-road-card-container {{ display: flex; justify-content: center; }}
        .bead-road-card {{ width: fit-content; }}
        .bead-road-container {{
            line-height: 0; text-align: center; background-color: var(--tertiary-bg);
            padding: 15px; border-radius: var(--border-radius); border: 1px solid var(--border-color);
            margin-top: 8px; display: inline-block; overflow-x: auto; max-width: 100%;
        }}
        .bead-row {{ margin-bottom: var(--bead-margin); height: var(--bead-size); white-space: nowrap;}}
        .bead-cell-banker, .bead-cell-player, .bead-cell-current, .bead-cell-empty {{
            border-radius: 50%; text-align: center; width: var(--bead-size) !important;
            height: var(--bead-size) !important; line-height: var(--bead-size) !important;
            font-size: var(--bead-font-size) !important; font-weight: bold; display: inline-block;
            margin: 0 var(--bead-margin); border: 1px solid transparent; vertical-align: middle;
            box-shadow: var(--box-shadow-inset); position: relative; color: #fff;
        }}
        .bead-cell-banker {{ background-color: var(--banker-red); border-color: var(--banker-red-darker); }}
        .bead-cell-player {{ background-color: var(--player-blue); border-color: var(--player-blue-darker); }}
        .bead-cell-current {{
             background-color: transparent; border: 2px dashed var(--accent-gold);
             box-shadow: 0 0 8px rgba(212, 175, 55, 0.4);
             line-height: calc(var(--bead-size) - 4px) !important;
             width: calc(var(--bead-size) - 0px) !important; height: calc(var(--bead-size) - 0px) !important;
         }}
        .bead-cell-empty {{ background-color: var(--secondary-bg); border-color: var(--border-color); box-shadow: none; }}
        .bead-cell-banker.natural::after,
        .bead-cell-player.natural::after {{
            content: 'N'; position: absolute; top: var(--bead-natural-marker-offset);
            right: var(--bead-natural-marker-offset); width: var(--bead-natural-marker-size);
            height: var(--bead-natural-marker-size); line-height: var(--bead-natural-marker-size);
            border-radius: 50%; background-color: var(--accent-gold); color: #000;
            font-size: var(--bead-natural-marker-font-size); font-weight: bold; text-align: center;
            box-shadow: 0 0 2px rgba(0,0,0,0.4); z-index: 1; display: flex;
            align-items: center; justify-content: center;
        }}

        .progression-info {{ margin-top: 10px; margin-bottom: 10px; text-align: center; background-color: var(--tertiary-bg); padding: 8px; border-radius: 4px; border: 1px solid var(--border-color); }}
        .progression-mode {{ font-size: 11px; color: var(--accent-gold); font-weight: bold; margin-bottom: 4px; text-transform: uppercase; }}
        .progression-step {{ font-size: 12px; color: var(--secondary-text); margin-bottom: 4px; text-transform: uppercase; }}
        .suggested-bet {{ font-size: 16px; color: var(--accent-gold); font-weight: bold; margin-bottom: 0; display: flex; align-items: center; justify-content: center; }}
        .suggested-bet i {{ margin-right: 6px; font-size: 0.9em; }}
        .progression-buttons {{ margin-top: 5px; margin-bottom: 10px; }}
        div[data-testid="stVerticalBlock"] > div[data-testid="stButton"]:has(button span:contains("Reset Phiên Cược")) {{ margin-top: 15px !important; margin-bottom: 5px !important; }}
        div[data-testid="stVerticalBlock"] > div[data-testid="stButton"]:has(button span:contains("Reset Toàn Bộ Game")) {{ margin-top: 0px !important; }}
        hr {{ border-top: 1px solid var(--border-color); margin: 10px 0; }}
        .stDivider {{ margin: 10px 0;}}
        .stAlert {{ border-radius: 4px; font-size: 12px; background-color: var(--tertiary-bg); border: 1px solid var(--accent-gold-darker); padding: 8px 12px; margin-bottom: 10px;}}
        .stAlert p, .stAlert div, .stAlert li {{ font-size: 12px !important; color: var(--primary-text); }}
        .stToast {{ font-size: 13px; }}
        .session-stats-container {{ background-color: var(--secondary-bg); border-radius: var(--border-radius); padding: 15px; margin-bottom: 18px; border: 1px solid var(--border-color); box-shadow: var(--box-shadow); }}
        .session-stats-container h4 {{ margin-bottom: 8px; }}
        .session-stats-grid {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; text-align: center; }}
        .session-stat-item {{ background-color: var(--tertiary-bg); padding: 8px; border-radius: 4px; border: 1px solid var(--border-color); }}
        .session-stat-item .label {{ font-size: 11px; color: var(--secondary-text); text-transform: uppercase; margin-bottom: 3px; display: block;}}
        .session-stat-item .value {{ font-size: 16px; color: var(--primary-text); font-weight: bold; display: block; }}
        .session-stat-item .value .positive {{ color: var(--win-green); }}
        .session-stat-item .value .negative {{ color: var(--loss-red); }}
        .session-stat-item i {{ margin-right: 5px; }}
        .bet-history-container {{
            max-height: 150px; overflow-y: auto; background-color: var(--tertiary-bg);
            padding: 10px; border-radius: 4px; border: 1px solid var(--border-color);
            margin-top: 10px;
        }}
        .bet-history-item {{ font-size: 12px; padding: 3px 5px; margin-bottom: 4px; border-radius: 3px; display: flex; justify-content: space-between; border-bottom: 1px solid var(--secondary-bg); }}
        .bet-history-item:last-child {{ margin-bottom: 0; border-bottom: none; }}
        .bet-history-item span {{ vertical-align: middle; }}
        .bet-history-outcome {{ font-weight: bold; margin-right: 8px; }}
        .bet-history-outcome.win {{ color: var(--win-green); }}
        .bet-history-outcome.loss {{ color: var(--loss-red); }}
        .bet-history-amount {{ color: var(--secondary-text); margin-right: 8px; }}
        .bet-history-profit {{ font-weight: bold; }}
        .bet-history-profit.win {{ color: var(--win-green); }}
        .bet-history-profit.loss {{ color: var(--loss-red); }}
        .bet-history-container::-webkit-scrollbar {{ width: 6px; }}
        .bet-history-container::-webkit-scrollbar-track {{ background: var(--secondary-bg); border-radius: 3px;}}
        .bet-history-container::-webkit-scrollbar-thumb {{ background: var(--border-color); border-radius: 3px;}}
        .bet-history-container::-webkit-scrollbar-thumb:hover {{ background: var(--accent-gold); }}

        </style>
        """, unsafe_allow_html=True) # End of CSS block

    # --- Main App Layout (REARRANGED COLUMNS) ---
    st.markdown('<div class="main">', unsafe_allow_html=True)
    st.markdown('<div class="app-title"><i class="fas fa-crown"></i> Baccarat Pro Predictor <i class="fas fa-crown"></i></div>', unsafe_allow_html=True)

    left_col, mid_col, right_col = st.columns([1.3, 1.7, 2.0])

    # --- CỘT TRÁI: Chiến lược & Lịch sử ---
    with left_col:
        # ... (Code cột trái giữ nguyên với hiển thị mode mới) ...
        with st.container(border=False):
            st.markdown('<div class="card"><h4><i class="fas fa-chart-line"></i> Chiến Lược Cược</h4>', unsafe_allow_html=True)
            balance = float(st.session_state.get('current_balance', 0.0))
            initial_unit = float(st.session_state.get('initial_bet_unit', 100000.0))
            mode = st.session_state.get('betting_mode', 'PROFIT')

            st.markdown("<h6>Cài Đặt</h6>", unsafe_allow_html=True)
            col_start_bal_inp, col_start_bal_btn = st.columns([0.7, 0.3])
            with col_start_bal_inp:
                 st.number_input("Số Dư", min_value=0.0, step=1000.0, key="starting_balance_input", value=balance, format="%.0f", label_visibility="collapsed", placeholder="Nhập số dư...")
            with col_start_bal_btn:
                 st.button("Đặt Số Dư", key="set_starting_balance_button", on_click=set_current_balance_from_input, use_container_width=True)
            st.number_input("Unit", min_value=1.0, step=1000.0, key="input_initial_bet_unit", value=initial_unit, on_change=update_initial_bet_unit, format="%.0f", help="Mức cược cơ bản (unit).", label_visibility="collapsed", placeholder="Nhập unit cược...")
            st.divider()

            col_metric1, col_metric2 = st.columns(2)
            with col_metric1:
                 st.metric(label="Số Dư Hiện Tại", value=f"{st.session_state.get('current_balance', 0.0):,.0f} đ")
            with col_metric2:
                start_bal = float(st.session_state.get('session_start_balance', st.session_state.get('current_balance', 0.0)))
                current_bal = float(st.session_state.get('current_balance', 0.0))
                session_profit = current_bal - start_bal
                st.metric(label="Lợi Nhuận Phiên", value=f"{session_profit:,.0f} đ")

            # --- Hiển thị trạng thái Progression/Recovery ---
            prog_idx = st.session_state.get('current_progression_index', 0)
            prog_seq = st.session_state.get('progression_sequence', [1, 1, 2, 4, 8])
            suggested_bet = float(st.session_state.get('suggested_bet_amount', 0.0))
            current_balance_for_check = float(st.session_state.get('current_balance', 0.0))
            balance_ok = current_balance_for_check >= suggested_bet

            st.markdown('<div class="progression-info">', unsafe_allow_html=True)
            if mode == 'PROFIT':
                if not (0 <= prog_idx < len(prog_seq)): prog_idx = 0
                step_num = prog_idx + 1; total_steps = len(prog_seq)
                current_unit_multiplier = prog_seq[prog_idx]
                st.markdown(f'<div class="progression-mode">Mode: Lợi Nhuận (1-1-2-4-8)</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="progression-step">Bước: {step_num}/{total_steps} (x{current_unit_multiplier})</div>', unsafe_allow_html=True)
            else: # mode == 'RECOVERY'
                multiplier = st.session_state.get('current_recovery_multiplier', 1)
                bets_at_level = st.session_state.get('recovery_bets_at_this_level', 0)
                # Bet sắp tới là lần thứ bets_at_level + 1 tại mức này
                bet_order_text = f"(Lần {bets_at_level + 1})"
                st.markdown(f'<div class="progression-mode">Mode: Gỡ Lỗ (Delay Martingale)</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="progression-step">Mức cược: x{multiplier} {bet_order_text}</div>', unsafe_allow_html=True)

            st.markdown(f'<div class="suggested-bet"><i class="fas fa-coins"></i> Cược: {suggested_bet:,.0f} đ</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

            if not balance_ok:
                st.warning(f"Số dư không đủ ({current_balance_for_check:,.0f} đ)!", icon="⚠️")

            st.markdown('<div class="progression-buttons">', unsafe_allow_html=True)
            p_col1, p_col2, p_col3 = st.columns(3)
            with p_col1: st.button("Thắng (P)", key="prog_win_p_std", on_click=handle_progression_win, args=(1.0,), disabled=not balance_ok, use_container_width=True)
            with p_col2: st.button("Thắng (B)", key="prog_win_b_std", on_click=handle_progression_win, args=(0.95,), disabled=not balance_ok, use_container_width=True)
            with p_col3: st.button("Thua", key="prog_loss_std", on_click=handle_progression_loss, disabled=not balance_ok, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

            st.markdown("<h6>Lịch Sử Cược Phiên</h6>", unsafe_allow_html=True)
            bet_history = st.session_state.get('bet_history', [])
            st.markdown('<div class="bet-history-container">', unsafe_allow_html=True)
            if not bet_history:
                st.markdown('<p style="font-size: 12px; color: var(--secondary-text); text-align: center;">Chưa có lịch sử cược.</p>', unsafe_allow_html=True)
            else:
                for bet in reversed(bet_history[-15:]):
                    outcome_class = "win" if bet['outcome'] == "Win" else "loss"
                    profit_sign = "+" if bet['profit'] > 0 else ""
                    history_item_html = f"""
                    <div class="bet-history-item">
                        <span>
                            <span class="bet-history-outcome {outcome_class}">{bet['outcome']}</span>
                            <span class="bet-history-amount">({bet['amount']:,.0f} đ)</span>
                        </span>
                        <span class="bet-history-profit {outcome_class}">{profit_sign}{bet['profit']:,.0f} đ</span>
                    </div>
                    """
                    st.markdown(history_item_html, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

            st.button("Reset Phiên Cược", key="reset_session_std", on_click=reset_session, use_container_width=True)
            st.button("Reset Toàn Bộ Game", key="reset_std", on_click=reset_game, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True) # Close card


    # --- CỘT GIỮA: Nhập liệu & Dự đoán (Sticky) ---
    with mid_col:
        st.markdown('<div class="mid-col-sticky-container">', unsafe_allow_html=True) # Mở div sticky
        # ... (Code Input Card và Prediction Card giữ nguyên) ...
        with st.container(border=False):
            st.markdown('<div class="card"><h4><i class="fas fa-keyboard"></i> Nhập Kết Quả</h4>', unsafe_allow_html=True)
            col_in1, col_in2 = st.columns(2)
            with col_in1:
                st.button('Player', key="player_std_btn",
                           on_click=add_result, args=('Player', False),
                           use_container_width=True, help="Player Win (Standard)")
                st.button('Player', icon="⭐", key="player_natural_btn",
                           on_click=add_result, args=('Player', True),
                           use_container_width=True, help="Player Win (Natural 8 or 9)")
            with col_in2:
                 st.button('Banker', key="banker_std_btn",
                           on_click=add_result, args=('Banker', False),
                           use_container_width=True, help="Banker Win (Standard)")
                 st.button('Banker', icon="⭐", key="banker_natural_btn",
                           on_click=add_result, args=('Banker', True),
                           use_container_width=True, help="Banker Win (Natural 8 or 9)")
            st.button("Undo Last Result", key="undo_std", on_click=undo_last_result, disabled=not st.session_state.get('game_history', []), use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with st.container(border=False):
            st.markdown('<div class="card"><h4><i class="fas fa-brain"></i> Dự Đoán</h4>', unsafe_allow_html=True)
            pred = st.session_state.get('predictions', {})
            def format_prediction(label, value):
                result_class = ""; display_value = value
                if value == 'Banker': result_class = "prediction-result-Banker"
                elif value == 'Player': result_class = "prediction-result-Player"
                if isinstance(value, str): # Shorten messages
                    if value.startswith("Waiting (Src Cell"): display_value = "Waiting (2&5 Src)"
                    elif value.startswith("Waiting (Need values"): display_value = "Waiting (No Mirror Src)"
                    elif value.startswith("Waiting for Natural"): display_value = "Wait Natural"
                    elif value.startswith("Waiting for next result"): display_value = "Wait Next"
                    elif value.startswith("Waiting for pattern"): display_value = "Wait Pattern"
                    elif value.startswith("Waiting (Need Col 3+)"): display_value = "Wait Col 3+"
                    elif value.startswith("Waiting (Not X Position)"): display_value = "Wait X Pos"
                    elif value.startswith("Waiting (Need Matrix Top-Left)"): display_value = "Wait Matrix TL"
                    elif value.startswith("Waiting (4+ identical)"): display_value = "Wait (4+)"
                    elif value.startswith("Waiting (Matrix"): display_value = "Wait Matrix OOB"
                    elif value.startswith("Error:"): display_value = value.split(':')[0]
                return f'<div class="prediction-box"><b>{label}:</b> <span class="{result_class}">{display_value}</span></div>'
            st.markdown('<div class="prediction-grid">', unsafe_allow_html=True)
            st.markdown(format_prediction("Majority 6", pred.get("majority6", "...")), unsafe_allow_html=True)
            st.markdown(format_prediction("X Mark", pred.get("xMark", "...")), unsafe_allow_html=True)
            st.markdown(format_prediction("No Mirror", pred.get("noMirror", "...")), unsafe_allow_html=True)
            st.markdown(format_prediction("89 Special", pred.get("special89", "...")), unsafe_allow_html=True)
            st.markdown(format_prediction("2&5", pred.get("2and5", "...")), unsafe_allow_html=True)
            st.markdown('<div></div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="prediction-box overall" style="margin-top: 5px;"><b>Overall:</b> <span>{pred.get("percentage", "...")}</span></div>', unsafe_allow_html=True)
            final_pred_raw = pred.get("final", "No prediction")
            final_pred_html = final_pred_raw
            if "<b>Banker</b>" in final_pred_raw: final_pred_html = final_pred_raw.replace("<b>Banker</b>", '<b class="prediction-result-Banker">Banker</b>')
            elif "<b>Player</b>" in final_pred_raw: final_pred_html = final_pred_raw.replace("<b>Player</b>", '<b class="prediction-result-Player">Player</b>')
            st.markdown(f'<div class="final-prediction">{final_pred_html}</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True) # Close card

        st.markdown('</div>', unsafe_allow_html=True) # Đóng mid-col-sticky-container


    # --- CỘT PHẢI: Bead Road & Thống kê (Sticky) ---
    with right_col:
        # ... (Code cột phải giữ nguyên) ...
        st.markdown('<div class="bead-road-sticky-container">', unsafe_allow_html=True)
        st.markdown('<div class="bead-road-card-container">', unsafe_allow_html=True)
        with st.container(border=False):
            st.markdown('<div class="card bead-road-card"><h4><i class="fas fa-border-all"></i> Bead Road</h4>', unsafe_allow_html=True)
            bead_html = '<div class="bead-road-container">'
            rows = st.session_state.get('rows', 6)
            cols = st.session_state.get('cols', 18)
            grid = st.session_state.get('bead_road_grid', create_default_grid(rows, cols, None))
            nat_grid = st.session_state.get('natural_marks_grid', create_default_grid(rows, cols, False))
            current_row = st.session_state.get('current_bead_road_row', 0)
            current_col = st.session_state.get('current_bead_road_col', 0)
            next_pos_valid = (0 <= current_row < rows) and (0 <= current_col < cols)

            grid_valid = isinstance(grid, list) and len(grid) == rows and \
                         (rows == 0 or (len(grid) > 0 and isinstance(grid[0], list) and len(grid[0]) == cols))
            nat_grid_valid = isinstance(nat_grid, list) and len(nat_grid) == rows and \
                             (rows == 0 or (len(nat_grid) > 0 and isinstance(nat_grid[0], list) and len(nat_grid[0]) == cols))

            if not grid_valid or not nat_grid_valid:
                 bead_html += "<p style='color: red;'>Error: Bead Road data invalid or dimensions mismatch.</p>"
                 if 'bead_road_grid' in st.session_state: del st.session_state.bead_road_grid
                 if 'natural_marks_grid' in st.session_state: del st.session_state.natural_marks_grid
                 grid = create_default_grid(rows, cols, None)
                 nat_grid = create_default_grid(rows, cols, False)
                 st.session_state.bead_road_grid = grid
                 st.session_state.natural_marks_grid = nat_grid
                 st.warning("Recreated Bead Road grids due to invalid state.")

            if isinstance(grid, list) and len(grid) == rows and (rows == 0 or len(grid[0]) == cols):
                for i in range(rows):
                    bead_html += '<div class="bead-row">'
                    for j in range(cols):
                        cell = grid[i][j]
                        is_natural = nat_grid[i][j]
                        is_current_target = next_pos_valid and (i == current_row) and (j == current_col)
                        cell_class = "bead-cell-empty"; cell_content = ""
                        if is_current_target: cell_class = "bead-cell-current"
                        elif cell == 'Banker':
                            cell_class = 'bead-cell-banker'
                            cell_content = 'B'
                            if is_natural: cell_class += ' natural'
                        elif cell == 'Player':
                            cell_class = 'bead-cell-player'
                            cell_content = 'P'
                            if is_natural: cell_class += ' natural'
                        bead_html += f'<div class="{cell_class}">{cell_content}</div>'
                    bead_html += '</div>'
            bead_html += '</div>'
            st.markdown(bead_html, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True) # Đóng sticky container

        with st.container(border=False):
            st.markdown('<div class="session-stats-container card">', unsafe_allow_html=True)
            st.markdown('<h4><i class="fas fa-chart-pie"></i> Thống Kê Phiên</h4>', unsafe_allow_html=True)
            session_wins = st.session_state.get('session_wins', 0)
            session_losses = st.session_state.get('session_losses', 0)
            session_start_time = st.session_state.get('session_start_time', datetime.datetime.now())
            if isinstance(session_start_time, datetime.datetime):
                elapsed_time = datetime.datetime.now() - session_start_time
                total_seconds = int(elapsed_time.total_seconds())
                hours, remainder = divmod(total_seconds, 3600)
                minutes, seconds = divmod(remainder, 60)
                formatted_time = f"{hours:02}:{minutes:02}:{seconds:02}"
            else:
                formatted_time = "N/A"
            st.markdown('<div class="session-stats-grid">', unsafe_allow_html=True)
            st.markdown(f"""
                <div class="session-stat-item">
                    <span class="label"><i class="far fa-clock"></i> Thời Gian</span>
                    <span class="value">{formatted_time}</span>
                </div>""", unsafe_allow_html=True)
            st.markdown(f"""
                <div class="session-stat-item">
                    <span class="label"><i class="fas fa-trophy"></i> Thắng (Cược)</span>
                    <span class="value positive">{session_wins}</span>
                </div>""", unsafe_allow_html=True)
            st.markdown(f"""
                <div class="session-stat-item">
                    <span class="label"><i class="fas fa-heart-broken"></i> Thua (Cược)</span>
                    <span class="value negative">{session_losses}</span>
                </div>""", unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True) # Close main div

# --- Run the App ---
if __name__ == '__main__':
    main()
