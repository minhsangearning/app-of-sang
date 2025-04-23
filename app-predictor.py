# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
from collections import Counter
import math
import datetime
import copy # Needed for deep copy in undo


# --- Constants ---
ROWS = 6
# DEFAULT_COLS = 18 # Set later in main
MIN_TARGET_PROFIT_PER_SEQUENCE = 456000.0 # Minimum profit for sequence success (all Banker wins)
MAX_TARGET_PROFIT_PER_SEQUENCE = 480000.0 # Maximum profit for sequence success (all Player wins)
DAILY_TARGET_SEQUENCES = 3
MIN_DAILY_TARGET_PROFIT = MIN_TARGET_PROFIT_PER_SEQUENCE * DAILY_TARGET_SEQUENCES # 1,368,000
MAX_DAILY_TARGET_PROFIT = MAX_TARGET_PROFIT_PER_SEQUENCE * DAILY_TARGET_SEQUENCES # 1,440,000


# --- Helper Functions ---
def create_default_grid(rows, cols, default_value):
    """Creates a 2D list (grid) with default values."""
    rows = max(0, int(rows))
    cols = max(0, int(cols))
    return [[default_value for _ in range(cols)] for _ in range(rows)]

def get_column_data(grid, col_index, max_rows=6):
    """Safely extracts data from a specific column, up to max_rows."""
    if not grid or not grid[0] or col_index < 0 or col_index >= len(grid[0]):
        return []
    column_data = []
    num_rows = len(grid)
    for row_idx in range(min(num_rows, max_rows)):
        if row_idx < len(grid) and col_index < len(grid[row_idx]):
            value = grid[row_idx][col_index]
            if value is not None:
                column_data.append(value)
    return column_data

def get_linear_results(grid, current_row, current_col, rows, cols):
    """
    Flattens the grid into a linear sequence based on the input order
    (down the column, then move to the next column).
    Stops BEFORE the current cursor position.
    """
    linear_results = []
    last_filled_col = -1
    last_filled_row = -1

    if current_col == 0 and current_row == 0: return []
    elif current_row == 0:
        last_filled_col = current_col - 1
        last_filled_row = rows - 1
    else:
        last_filled_col = current_col
        last_filled_row = current_row - 1

    if last_filled_col < 0: return [] # No columns filled yet

    for j in range(cols):
        for i in range(rows):
            if j > last_filled_col or (j == last_filled_col and i > last_filled_row): break
            if i < len(grid) and j < len(grid[i]):
                cell_value = grid[i][j]
                if cell_value in ['Banker', 'Player']:
                    linear_results.append(cell_value)
            else: break # Grid boundary safety
        if j > last_filled_col or (j == last_filled_col and i > last_filled_row): break
    return linear_results


# --- Prediction Functions ---
def predict_majority6(grid, current_col): # Only needs col
    if not grid or not grid[0]: return "Waiting for data"
    if current_col < 0 or current_col >= len(grid[0]): return "No prediction (Col OOB)"
    current_column_data = get_column_data(grid, current_col, max_rows=ROWS)
    relevant_data = current_column_data[:ROWS]
    len_data = len(relevant_data)
    if len_data == 0: return "Waiting for data"
    count_b = relevant_data.count('Banker'); count_p = relevant_data.count('Player')
    if count_b >= 4 or count_p >= 4: return "Waiting (4+ identical)"
    if len_data == 5:
        if count_b > count_p: return "Banker"
        if count_p > count_b: return "Player"
        return "Waiting (5, Tie)"
    if len_data >= 2:
        last_two = relevant_data[-2:]
        if last_two[0] == last_two[1]:
            if len_data >= 3 and relevant_data[-3] == last_two[0]:
                return "Waiting (3 identical)"
            return last_two[0]
    return "Waiting for pattern"

def predict_x_mark(grid, current_row, current_col): # Needs row & col
    if not grid or not grid[0]: return "Waiting for data"
    num_rows = len(grid); num_cols = len(grid[0])
    pred_row, pred_col = current_row, current_col
    max_cols_setting = st.session_state.get('cols', num_cols)
    if pred_col >= max_cols_setting: return "No prediction (Max Col)"
    if pred_row < 0 or pred_row >= num_rows or pred_col < 0: return "Waiting (Cursor OOB)"
    matrix_start_row = (pred_row // 3) * 3; matrix_start_col = (pred_col // 3) * 3
    if not (0 <= matrix_start_row < num_rows and 0 <= matrix_start_col < num_cols): return "Waiting (Matrix Calc OOB)"
    first_value = None
    if matrix_start_row < len(grid) and matrix_start_col < len(grid[matrix_start_row]): first_value = grid[matrix_start_row][matrix_start_col]
    else: return "Waiting (Matrix Grid Access)"
    if first_value not in ['Banker', 'Player']: return "Waiting (Need Matrix TL B/P)"
    relative_row = pred_row - matrix_start_row; relative_col = pred_col - matrix_start_col
    is_pred_cell_an_x_position = (relative_row == 0 and relative_col == 2) or \
                                 (relative_row == 1 and relative_col == 1) or \
                                 (relative_row == 2 and relative_col == 0) or \
                                 (relative_row == 2 and relative_col == 2)
    if is_pred_cell_an_x_position: return first_value
    else: return "Waiting (Not X Pos)"

def predict_no_mirror(grid, current_row, current_col): # Needs row & col
    if not grid or not grid[0]: return "Waiting for data"
    num_rows = len(grid); num_cols = len(grid[0])
    pred_row, pred_col = current_row, current_col
    max_cols_setting = st.session_state.get('cols', num_cols)
    if pred_col >= max_cols_setting: return "No prediction (Max Col)"
    if pred_col < 2: return "Waiting (Need Col 3+)"
    if not (0 <= pred_row < num_rows): return "Error: Invalid row index"
    src_col_prev2 = pred_col - 2; src_col_prev1 = pred_col - 1
    val_prev2 = None; val_prev1 = None
    if pred_row < len(grid):
        if src_col_prev2 < len(grid[pred_row]): val_prev2 = grid[pred_row][src_col_prev2]
        if src_col_prev1 < len(grid[pred_row]): val_prev1 = grid[pred_row][src_col_prev1]
    if val_prev1 not in ['Banker', 'Player'] or val_prev2 not in ['Banker', 'Player']: return "Waiting (Need Prev B/P)"
    if val_prev1 == val_prev2: return 'Player' if val_prev1 == 'Banker' else 'Banker'
    else: return val_prev1

def predict_special89(special89_state, last_result_after_natural): # Needs state
    if special89_state == "waiting_for_natural": return "Wait Natural"
    elif special89_state == "waiting_for_result_after_natural": return "Wait Next"
    elif special89_state == "ready_for_prediction": return last_result_after_natural if last_result_after_natural else "Error: No S89 result"
    else: return "Wait Natural"

def predict_2_and_5(grid, current_row, current_col): # Needs row & col
    if not grid or not grid[0]: return "Waiting for data"
    num_rows = len(grid); num_cols = len(grid[0])
    pred_row, pred_col = current_row, current_col
    max_cols_setting = st.session_state.get('cols', num_cols)
    if pred_col >= max_cols_setting: return "No prediction (Max Col)"
    if pred_col < 3: return "Waiting (Starts Col 4)"
    source_col = pred_col - 1
    if 0 <= pred_row <= 2: source_row = 1
    elif 3 <= pred_row <= 5: source_row = 4
    else: return "Error: Invalid pred row"
    source_value = None
    if 0 <= source_row < num_rows and 0 <= source_col < num_cols and \
       source_row < len(grid) and source_col < len(grid[source_row]): source_value = grid[source_row][source_col]
    else: return f"Waiting (Src [{source_row+1},{source_col+1}] OOB)"
    if source_value not in ['Banker', 'Player']: return f"Waiting (Src [{source_row+1},{source_col+1}] empty/invalid)"
    else: return source_value

# --- MODIFIED: 3 Baloon predicts only for row 0 ---
def predict_3_baloon(grid, current_row, current_col):
    """Predicts Col N+1 Row 1 based on Col N Row 3. ONLY predicts if current_row is 0."""
    if current_row != 0:
        return "N/A (Row > 1)"

    if not grid or not grid[0]: return "Waiting for data"
    num_rows = len(grid); num_cols = len(grid[0])
    pred_col = current_col; source_col = pred_col - 1

    if source_col < 0: return "Waiting (Need Col 2+)"
    source_column_data = get_column_data(grid, source_col, max_rows=ROWS)
    if len(source_column_data) < ROWS: return f"Wait (Col {source_col+1} < {ROWS})"

    source_value_row3 = None; source_row_idx = 2
    if source_row_idx < len(grid) and source_col < len(grid[source_row_idx]):
         source_value_row3 = grid[source_row_idx][source_col]
    else: return "Waiting (Src Row 3 OOB)"

    if source_value_row3 in ['Banker', 'Player']: return source_value_row3
    else: return f"Waiting (Col {source_col+1} Row 3 empty)"

# --- MODIFIED: I Strategy predicts only for row 0 ---
def predict_i_strategy(grid, current_row, current_col):
    """Predicts Col N+1 Row 1 based on majority in Col N (first 6). ONLY predicts if current_row is 0."""
    if current_row != 0:
        return "N/A (Row > 1)"

    if not grid or not grid[0]: return "Waiting for data"
    num_rows = len(grid); num_cols = len(grid[0])
    pred_col = current_col; source_col = pred_col - 1

    if source_col < 0: return "Waiting (Need Col 2+)"
    source_column_data = get_column_data(grid, source_col, max_rows=ROWS)
    if len(source_column_data) != ROWS: return f"Wait (Col {source_col+1} != {ROWS})"

    count_b = source_column_data.count('Banker'); count_p = source_column_data.count('Player')
    if count_b > count_p: return 'Banker'
    elif count_p > count_b: return 'Player'
    else: # Tie (3-3)
        source_value_row1 = None; source_row_idx = 0
        if source_row_idx < len(grid) and source_col < len(grid[source_row_idx]):
             source_value_row1 = grid[source_row_idx][source_col]
        else: return "Waiting (Tie, Src Row 1 OOB)"
        if source_value_row1 in ['Banker', 'Player']: return source_value_row1
        else: return "Waiting (Tie, Src Row 1 empty)"

def predict_5s_streak(): # Uses state directly
    if st.session_state.get('streak_ready_to_predict', False):
        prediction = st.session_state.get('streak_prediction_value', None)
        if prediction in ['Banker', 'Player']: return prediction
        else: return "Error (Streak Predict Val)"
    else:
        if not st.session_state.get('streak5_seen', False): return "Wait (Need 5+ Streak)"
        elif not st.session_state.get('streak3_after_5_seen', False): return "Wait (Need 1st 3-Streak)"
        else: return "Wait (Need 2nd 3-Streak)"


# --- Streak State Management (Unchanged) ---
def update_streak_state():
    grid = st.session_state.get('bead_road_grid'); current_row = st.session_state.get('current_bead_road_row')
    current_col = st.session_state.get('current_bead_road_col'); rows = st.session_state.get('rows'); cols = st.session_state.get('cols')
    linear_results = get_linear_results(grid, current_row, current_col, rows, cols)
    if not linear_results:
        st.session_state.streak5_seen = False; st.session_state.streak3_after_5_seen = False
        st.session_state.streak_ready_to_predict = False; st.session_state.streak_prediction_value = None; return
    current_streak_len = 0; last_result = linear_results[-1]
    for result in reversed(linear_results):
        if result == last_result: current_streak_len += 1
        else: break
    was_ready_to_predict = st.session_state.get('streak_ready_to_predict', False)
    if not st.session_state.get('streak5_seen', False) and current_streak_len >= 5:
        st.session_state.streak5_seen = True; st.session_state.streak3_after_5_seen = False
        st.session_state.streak_ready_to_predict = False; st.session_state.streak_prediction_value = None
    if st.session_state.get('streak5_seen', False):
        if current_streak_len == 3:
            if not st.session_state.get('streak3_after_5_seen', False):
                st.session_state.streak3_after_5_seen = True; st.session_state.streak_ready_to_predict = False
                st.session_state.streak_prediction_value = None
            else:
                if not was_ready_to_predict:
                     st.session_state.streak_ready_to_predict = True; st.session_state.streak_prediction_value = last_result
        elif was_ready_to_predict:
             st.session_state.streak_ready_to_predict = False; st.session_state.streak_prediction_value = None

# --- Aggregate & Utility Functions (Unchanged) ---
def calculate_prediction_percent(predictions_list):
    valid_predictions = [p for p in predictions_list if p in ['Banker', 'Player']]
    total = len(valid_predictions); b_count = valid_predictions.count('Banker'); p_count = total - b_count
    if total == 0: return "Waiting..."
    if b_count > p_count: return f"{math.ceil((b_count/total)*100)}% Banker"
    elif p_count > b_count: return f"{math.ceil((p_count/total)*100)}% Player"
    else: return "50% B / 50% P"

def get_final_prediction(predictions):
    valid_preds = [p for p in predictions if p in ['Banker', 'Player']]; num_valid = len(valid_preds)
    if num_valid < 3 : return "No prediction (Need ≥3 B/P signals)"
    pred_counts = Counter(valid_preds); most_common = pred_counts.most_common(1)
    if not most_common: return "No prediction"
    outcome, count = most_common[0]
    if len(pred_counts) > 1 and len(pred_counts.most_common(2)) > 1 and pred_counts.most_common(2)[1][1] == count: return f"No prediction (Tie {count}-{count})"
    if count >= 3: return f"Predict <b>{outcome}</b> ({count}/{num_valid} agree)"
    else: return f"No prediction (Weak Signal - Max {count})"

def get_bead_road_stats(grid):
    if not grid or not grid[0]: return 0, 0
    flat_grid = [cell for row in grid for cell in row if cell in ['Banker', 'Player']]
    b_count = flat_grid.count('Banker'); p_count = flat_grid.count('Player')
    return b_count, p_count

# --- Progression Strategy Functions (Unchanged from previous fix) ---
def initialize_progression_state():
    st.session_state.progression_sequence = [1, 2, 3, 6]; st.session_state.current_progression_index = 0
    st.session_state.current_profit_sequence_profit = 0.0
    if 'initial_bet_unit' not in st.session_state: st.session_state.initial_bet_unit = 40000.0
    if 'current_balance' not in st.session_state: st.session_state.current_balance = 2000000.0
    st.session_state.current_balance = float(st.session_state.get('current_balance', 2000000.0))
    st.session_state.initial_bet_unit = float(st.session_state.get('initial_bet_unit', 40000.0))
    st.session_state.session_start_balance = st.session_state.current_balance
    st.session_state.last_withdrawal_profit_level = 0.0; st.session_state.session_start_time = datetime.datetime.now()
    st.session_state.session_wins = 0; st.session_state.session_losses = 0; st.session_state.bet_history = []
    st.session_state.session_sequences_completed = 0; st.session_state.betting_mode = 'PROFIT'
    st.session_state.current_recovery_multiplier = 1; st.session_state.recovery_bets_at_this_level = 0
    calculate_suggested_bet()

def calculate_suggested_bet():
    current_balance = float(st.session_state.get('current_balance', 0.0)); session_start = float(st.session_state.get('session_start_balance', current_balance))
    unit = float(st.session_state.get('initial_bet_unit', 1.0)); mode = st.session_state.get('betting_mode', 'PROFIT')
    if current_balance >= session_start:
        if mode == 'RECOVERY':
            st.toast("🎉 Về bờ! Chuyển sang chế độ Lợi Nhuận (1-2-3-6).", icon="💰"); st.session_state.betting_mode = 'PROFIT'
            st.session_state.current_progression_index = 0; st.session_state.current_profit_sequence_profit = 0.0
            st.session_state.current_recovery_multiplier = 1; st.session_state.recovery_bets_at_this_level = 0
        mode = 'PROFIT'; st.session_state.betting_mode = 'PROFIT'
    else:
        if mode == 'PROFIT':
            st.toast("🚨 Vốn giảm! Chuyển sang chế độ Gỡ Lỗ (Delay Martingale).", icon="🛡️"); st.session_state.betting_mode = 'RECOVERY'
            st.session_state.current_profit_sequence_profit = 0.0; st.session_state.current_recovery_multiplier = 1
            st.session_state.recovery_bets_at_this_level = 0
        mode = 'RECOVERY'; st.session_state.betting_mode = 'RECOVERY'
    suggested_bet = 0.0
    if mode == 'PROFIT':
        idx = st.session_state.get('current_progression_index', 0); sequence = st.session_state.get('progression_sequence', [1, 2, 3, 6])
        if not (0 <= idx < len(sequence)): idx = 0; st.session_state.current_progression_index = 0
        multiplier = sequence[idx]; suggested_bet = unit * multiplier
    elif mode == 'RECOVERY': multiplier = st.session_state.get('current_recovery_multiplier', 1); suggested_bet = unit * multiplier
    st.session_state.suggested_bet_amount = float(suggested_bet)

def handle_progression_win(payout_ratio):
    bet_amount = float(st.session_state.get('suggested_bet_amount', 0.0)); current_balance = float(st.session_state.get('current_balance', 0.0))
    session_start = float(st.session_state.get('session_start_balance', current_balance)); mode = st.session_state.get('betting_mode', 'PROFIT')
    winnings = bet_amount * payout_ratio; new_balance = current_balance + winnings
    if 'bet_history' not in st.session_state: st.session_state.bet_history = []
    st.session_state.bet_history.append({'outcome': 'Win', 'amount': bet_amount, 'profit': winnings, 'timestamp': datetime.datetime.now(), 'mode': mode})
    max_history = 50; st.session_state.bet_history = st.session_state.bet_history[-max_history:]
    st.session_state.current_balance = new_balance; st.session_state.session_wins = st.session_state.get('session_wins', 0) + 1
    if new_balance >= session_start and mode == 'RECOVERY': pass
    elif mode == 'PROFIT':
        st.session_state.current_profit_sequence_profit = st.session_state.get('current_profit_sequence_profit', 0.0) + winnings
        current_idx = st.session_state.get('current_progression_index', 0); prog_seq = st.session_state.get('progression_sequence', [1, 2, 3, 6])
        sequence_len = len(prog_seq); next_idx = current_idx + 1
        if next_idx >= sequence_len: # Sequence complete (win on last step)
            current_sequence_profit = st.session_state.get('current_profit_sequence_profit', 0.0)
            if current_sequence_profit >= MIN_TARGET_PROFIT_PER_SEQUENCE: # Check >= 456k
                st.session_state.session_sequences_completed = st.session_state.get('session_sequences_completed', 0) + 1
                st.toast(f"🎉 Đạt mục tiêu chuỗi PROFIT ({current_sequence_profit:,.0f} ≥ {MIN_TARGET_PROFIT_PER_SEQUENCE:,.0f})! Chuỗi #{st.session_state.session_sequences_completed}", icon="🎯")
                st.balloons()
            else:
                 st.toast(f"⚠️ Chuỗi PROFIT hoàn tất nhưng chưa đạt mục tiêu lợi nhuận ({current_sequence_profit:,.0f} < {MIN_TARGET_PROFIT_PER_SEQUENCE:,.0f}). Reset.", icon="📉")
            st.session_state.current_progression_index = 0; st.session_state.current_profit_sequence_profit = 0.0
            st.toast("Chuỗi thắng PROFIT (1-2-3-6) hoàn tất! Reset về mức cược đầu.", icon="✅")
        else: st.session_state.current_progression_index = next_idx
    elif mode == 'RECOVERY':
        current_multiplier = st.session_state.get('current_recovery_multiplier', 1)
        st.toast(f"Thắng khi đang gỡ! 👍 Tiếp tục cược {current_multiplier} unit.", icon="✅"); st.session_state.recovery_bets_at_this_level = 0
    calculate_suggested_bet()

def handle_progression_loss():
    bet_amount = float(st.session_state.get('suggested_bet_amount', 0.0)); current_balance = float(st.session_state.get('current_balance', 0.0))
    unit = float(st.session_state.get('initial_bet_unit', 1.0)); mode = st.session_state.get('betting_mode', 'PROFIT')
    new_balance = current_balance - bet_amount
    if 'bet_history' not in st.session_state: st.session_state.bet_history = []
    st.session_state.bet_history.append({'outcome': 'Loss', 'amount': bet_amount, 'profit': -bet_amount, 'timestamp': datetime.datetime.now(), 'mode': mode})
    max_history = 50; st.session_state.bet_history = st.session_state.bet_history[-max_history:]
    st.session_state.current_balance = new_balance; st.session_state.session_losses = st.session_state.get('session_losses', 0) + 1
    if mode == 'PROFIT':
        st.session_state.current_progression_index = 0; st.session_state.current_profit_sequence_profit = 0.0
        st.toast("Thua khi đang Lợi Nhuận! 😢 Reset chuỗi (1-2-3-6), về mức cược đầu.", icon="❌")
    elif mode == 'RECOVERY':
        current_multiplier = st.session_state.get('current_recovery_multiplier', 1); bets_at_level = st.session_state.get('recovery_bets_at_this_level', 0) + 1
        if current_multiplier == 1:
            if bets_at_level == 1:
                st.session_state.current_recovery_multiplier = 1; st.session_state.recovery_bets_at_this_level = bets_at_level
                st.toast("Thua lần 1 khi Gỡ! 😥 Tiếp tục cược 1 unit.", icon="❌")
            elif bets_at_level >= 2:
                st.session_state.current_recovery_multiplier = 2; st.session_state.recovery_bets_at_this_level = 0
                st.toast("Thua lần 2 khi Gỡ! 😥 Bắt đầu nhân đôi (2 units).", icon="❌")
        else:
            if bets_at_level == 1:
                st.session_state.current_recovery_multiplier = current_multiplier; st.session_state.recovery_bets_at_this_level = bets_at_level
                st.toast(f"Thua mức {current_multiplier}x! 😥 Lặp lại cược {current_multiplier} units.", icon="❌")
            elif bets_at_level >= 2:
                new_multiplier = current_multiplier * 2; st.session_state.current_recovery_multiplier = new_multiplier
                st.session_state.recovery_bets_at_this_level = 0; st.toast(f"Thua mức {current_multiplier}x lần 2! 😥 Nhân đôi lên {new_multiplier} units.", icon="❌")
    calculate_suggested_bet()

# --- Utility & Reset Functions (Unchanged) ---
def update_initial_bet_unit():
    if 'input_initial_bet_unit' in st.session_state:
        try:
            new_unit = float(st.session_state.input_initial_bet_unit);
            if new_unit > 0: st.session_state.initial_bet_unit = new_unit
            else: st.session_state.initial_bet_unit = 1.0; st.warning("Bet unit must be positive.")
            calculate_suggested_bet()
        except (ValueError, TypeError): st.error("Invalid input for Bet Unit.")

def set_current_balance_from_input():
    if 'starting_balance_input' in st.session_state:
        try:
            new_balance = float(st.session_state.starting_balance_input)
            if new_balance >= 0:
                st.session_state.current_balance = new_balance; st.session_state.session_start_balance = st.session_state.current_balance
                st.session_state.last_withdrawal_profit_level = 0.0; st.session_state.session_start_time = datetime.datetime.now()
                st.session_state.session_wins = 0; st.session_state.session_losses = 0; st.session_state.bet_history = []
                st.session_state.session_sequences_completed = 0; st.session_state.betting_mode = 'PROFIT'
                st.session_state.current_progression_index = 0; st.session_state.current_profit_sequence_profit = 0.0
                st.session_state.current_recovery_multiplier = 1; st.session_state.recovery_bets_at_this_level = 0
                st.toast("Số dư cập nhật. Phiên mới bắt đầu.", icon="💰"); calculate_suggested_bet()
            else: st.warning("Số dư không thể âm.")
        except (ValueError, TypeError): st.error("Vui lòng nhập số dư hợp lệ.")
    else: st.error("Lỗi: Field 'starting_balance_input' missing.")

def reset_session():
    st.session_state.current_progression_index = 0; st.session_state.current_profit_sequence_profit = 0.0
    st.session_state.betting_mode = 'PROFIT'; st.session_state.current_recovery_multiplier = 1
    st.session_state.recovery_bets_at_this_level = 0; st.session_state.session_start_balance = st.session_state.current_balance
    st.session_state.last_withdrawal_profit_level = 0.0; st.session_state.session_start_time = datetime.datetime.now()
    st.session_state.session_wins = 0; st.session_state.session_losses = 0; st.session_state.bet_history = []
    st.session_state.session_sequences_completed = 0
    st.toast("Phiên cược đã reset. Số dư & Bead Road giữ nguyên.", icon="🔄"); calculate_suggested_bet()

def reset_bead_road():
    rows = st.session_state.get('rows', ROWS); cols = st.session_state.get('cols', 18)
    st.session_state.game_history = []; st.session_state.game_count = 0
    st.session_state.bead_road_grid = create_default_grid(rows, cols, None)
    st.session_state.natural_marks_grid = create_default_grid(rows, cols, False)
    st.session_state.current_bead_road_row = 0; st.session_state.current_bead_road_col = 0
    st.session_state.special89_state = "waiting_for_natural"; st.session_state.last_natural_pos = None
    st.session_state.last_result_after_natural = None; st.session_state.streak5_seen = False
    st.session_state.streak3_after_5_seen = False; st.session_state.streak_ready_to_predict = False
    st.session_state.streak_prediction_value = None
    st.toast("Bead Road & Lịch sử game reset. Cược giữ nguyên.", icon="🎲"); update_all_predictions()

def reset_all():
    keys_to_reset = list(st.session_state.keys());
    for key in keys_to_reset: del st.session_state[key]
    st.toast("Reset toàn bộ ứng dụng!", icon="🔥")

# --- Backend Functions ---
def initialize_game_state():
     rows = st.session_state.get('rows', ROWS); cols = st.session_state.get('cols', 18)
     st.session_state.game_history = []; st.session_state.game_count = 0
     st.session_state.bead_road_grid = create_default_grid(rows, cols, None)
     st.session_state.natural_marks_grid = create_default_grid(rows, cols, False)
     st.session_state.current_bead_road_row = 0; st.session_state.current_bead_road_col = 0
     st.session_state.special89_state = "waiting_for_natural"; st.session_state.last_natural_pos = None
     st.session_state.last_result_after_natural = None; st.session_state.streak5_seen = False
     st.session_state.streak3_after_5_seen = False; st.session_state.streak_ready_to_predict = False
     st.session_state.streak_prediction_value = None; st.session_state.predictions = {}

# --- MODIFIED: update_all_predictions passes row to new functions ---
def update_all_predictions():
    """Calculates and updates all prediction types in the session state."""
    grid = st.session_state.get('bead_road_grid')
    row = st.session_state.get('current_bead_road_row') # Get current row
    col = st.session_state.get('current_bead_road_col') # Get current col
    s89_state = st.session_state.get('special89_state')
    s89_result = st.session_state.get('last_result_after_natural')

    if grid is None or row is None or col is None or s89_state is None:
        st.session_state.predictions = {'majority6': "Waiting...", 'xMark': "Waiting...", 'noMirror': "Waiting...", 'special89': "Waiting...", '2and5': "Waiting...", '3baloon': "Waiting...", 'iStrategy': "Waiting...", '5sStreak': "Waiting...", 'percentage': "Waiting...", 'final': "Waiting..."}
        return

    pred_maj6=predict_majority6(grid, col); pred_x_mark=predict_x_mark(grid, row, col)
    pred_no_mirror=predict_no_mirror(grid, row, col); pred_s89=predict_special89(s89_state, s89_result)
    pred_2and5=predict_2_and_5(grid, row, col)
    pred_3baloon=predict_3_baloon(grid, row, col) # Pass row
    pred_i_strategy=predict_i_strategy(grid, row, col) # Pass row
    pred_5s_streak=predict_5s_streak()

    predictions_list = [pred_maj6, pred_x_mark, pred_no_mirror, pred_s89, pred_2and5, pred_3baloon, pred_i_strategy, pred_5s_streak]
    pred_percent = calculate_prediction_percent(predictions_list); final_pred = get_final_prediction(predictions_list)
    st.session_state.predictions = {'majority6': pred_maj6, 'xMark': pred_x_mark, 'noMirror': pred_no_mirror, 'special89': pred_s89, '2and5': pred_2and5, '3baloon': pred_3baloon, 'iStrategy': pred_i_strategy, '5sStreak': pred_5s_streak, 'percentage': pred_percent, 'final': final_pred}


def add_result(result, is_natural):
    required_keys = ['special89_state', 'last_result_after_natural', 'last_natural_pos', 'current_bead_road_row', 'current_bead_road_col', 'game_count', 'rows', 'cols', 'bead_road_grid', 'natural_marks_grid', 'streak5_seen', 'streak3_after_5_seen', 'streak_ready_to_predict']
    if not all(key in st.session_state for key in required_keys): st.error("Lỗi: State chưa init. Thử Reset."); return
    prev_state_snapshot = {'s89_state': st.session_state.special89_state, 's89_last_res': st.session_state.last_result_after_natural, 's89_last_nat_pos': copy.deepcopy(st.session_state.last_natural_pos), 'bead_row': st.session_state.current_bead_road_row, 'bead_col': st.session_state.current_bead_road_col, 'streak5_seen': st.session_state.streak5_seen, 'streak3_after_5_seen': st.session_state.streak3_after_5_seen, 'streak_ready_to_predict': st.session_state.streak_ready_to_predict, 'streak_prediction_value': st.session_state.streak_prediction_value, 'grid_cell_value_before': None, 'nat_grid_cell_value_before': None}
    current_row=st.session_state.current_bead_road_row; current_col=st.session_state.current_bead_road_col; rows=st.session_state.rows; cols=st.session_state.cols; grid=st.session_state.bead_road_grid; nat_grid=st.session_state.natural_marks_grid
    if not (isinstance(grid, list) and len(grid) == rows and (rows == 0 or len(grid[0]) == cols)): st.error("Lỗi Grid."); return
    if not (isinstance(nat_grid, list) and len(nat_grid) == rows and (rows == 0 or len(nat_grid[0]) == cols)): st.error("Lỗi Grid Nat."); return
    if not (0 <= current_row < rows and 0 <= current_col < cols): st.toast(f"Pos ({current_row+1}, {current_col+1}) invalid.", icon="error"); return
    prev_state_snapshot['grid_cell_value_before'] = grid[current_row][current_col]; prev_state_snapshot['nat_grid_cell_value_before'] = nat_grid[current_row][current_col]
    grid[current_row][current_col] = result; nat_grid[current_row][current_col] = is_natural
    next_row = current_row + 1; next_col = current_col
    if next_row >= rows: next_row = 0; next_col = current_col + 1
    st.session_state.current_bead_road_row = next_row; st.session_state.current_bead_road_col = next_col
    if next_col >= cols: st.toast("Grid full.", icon="⚠️")
    game_id = st.session_state.game_count + 1; new_game = {'id': game_id, 'result': result, 'is_natural': is_natural, 'prev_state': prev_state_snapshot, 'bead_row_filled': current_row, 'bead_col_filled': current_col}
    st.session_state.game_history = st.session_state.get('game_history', []) + [new_game]; st.session_state.game_count = game_id
    current_s89_state = prev_state_snapshot['s89_state']; natural_pos_this_turn = {'row': current_row, 'col': current_col} if is_natural else None
    next_s89_state = current_s89_state; next_nat_pos = prev_state_snapshot['s89_last_nat_pos']; next_res_after_nat = prev_state_snapshot['s89_last_res']
    if is_natural: next_nat_pos = natural_pos_this_turn; next_s89_state = "waiting_for_result_after_natural"; next_res_after_nat = None
    elif current_s89_state == "waiting_for_result_after_natural": next_res_after_nat = result; next_s89_state = "ready_for_prediction"
    elif current_s89_state == "ready_for_prediction": next_s89_state = "waiting_for_natural"; next_res_after_nat = None; next_nat_pos = None
    st.session_state.last_natural_pos = next_nat_pos; st.session_state.special89_state = next_s89_state; st.session_state.last_result_after_natural = next_res_after_nat
    update_streak_state(); update_all_predictions()

def undo_last_result():
    history = st.session_state.get('game_history', []);
    if not history: st.toast("Không có gì để hoàn tác.", icon="🤷‍♂️"); return
    undone_game = history.pop(); st.session_state.game_history = history; st.session_state.game_count = st.session_state.get('game_count', 1) - 1
    row_filled = undone_game.get('bead_row_filled'); col_filled = undone_game.get('bead_col_filled'); prev_state = undone_game.get('prev_state')
    grid = st.session_state.get('bead_road_grid'); nat_grid = st.session_state.get('natural_marks_grid'); rows = st.session_state.get('rows', ROWS); cols = st.session_state.get('cols', 18)
    if grid is not None and nat_grid is not None and prev_state is not None and row_filled is not None and col_filled is not None and (0 <= row_filled < rows) and (0 <= col_filled < cols) and (row_filled < len(grid)) and (col_filled < len(grid[row_filled])):
        grid[row_filled][col_filled] = prev_state['grid_cell_value_before']; nat_grid[row_filled][col_filled] = prev_state['nat_grid_cell_value_before']
        st.session_state.current_bead_road_row = row_filled; st.session_state.current_bead_road_col = col_filled
    else: st.warning("Lỗi undo grid."); st.session_state.current_bead_road_row = prev_state.get('bead_row', 0) if prev_state else 0; st.session_state.current_bead_road_col = prev_state.get('bead_col', 0) if prev_state else 0
    if prev_state:
        st.session_state.special89_state = prev_state['s89_state']; st.session_state.last_result_after_natural = prev_state['s89_last_res']; st.session_state.last_natural_pos = prev_state['s89_last_nat_pos']
        st.session_state.streak5_seen = prev_state['streak5_seen']; st.session_state.streak3_after_5_seen = prev_state['streak3_after_5_seen']; st.session_state.streak_ready_to_predict = prev_state['streak_ready_to_predict']; st.session_state.streak_prediction_value = prev_state['streak_prediction_value']
    else: st.error("Lỗi Undo: Mất state.")
    st.warning("Lưu ý: Undo KHÔNG hoàn tác Thắng/Thua Cược.")
    update_all_predictions(); calculate_suggested_bet()


# --- Main Application ---
def main():
    st.set_page_config(page_title="Baccarat Pro Predictor", layout="wide", initial_sidebar_state="collapsed")
    st.markdown('<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.2/css/all.min.css">', unsafe_allow_html=True)
    default_cols = 18

    if 'initialized' not in st.session_state:
        st.session_state.initialized = True; st.session_state.rows = ROWS; st.session_state.cols = default_cols
        initialize_progression_state(); initialize_game_state(); update_all_predictions()
        st.toast("Khởi tạo thành công!", icon="🚀")

    # --- CSS Styling (Same as previous version) ---
    st.markdown(f"""
        <style>
        /* PASTE FULL CSS FROM PREVIOUS VERSION HERE */
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
            --bp-count-icon-size: 24px; /* Size for B/P count icons */
         }}
         body {{ font-family: var(--font-body); color: var(--primary-text); background-color: var(--primary-bg); }}
         .main {{ background-color: var(--primary-bg); padding: 15px; border-radius: var(--border-radius); font-family: var(--font-body); }}
         .stApp > header {{ display: none; }}
         .main .block-container {{ padding: 5px 10px !important; margin: 0 !important; max-width: 100% !important; }}
         div[data-testid="stHorizontalBlock"] > div {{ align-self: flex-start !important; }}
         .stButton>button {{ font-family: var(--font-body); padding: 6px 12px; border-radius: 4px; font-size: 12px; font-weight: bold; color: var(--primary-text); border: 1px solid var(--border-color); transition: all 0.2s ease-in-out; width: 100%; margin: 3px 0; box-shadow: var(--box-shadow-inset); text-align: center; background: var(--tertiary-bg); height: 35px; box-sizing: border-box; display: inline-flex !important; align-items: center !important; justify-content: center !important; line-height: 1; position: relative; white-space: nowrap; }}
         .stButton>button:hover:not(:disabled) {{ transform: translateY(-1px); box-shadow: 0 4px 8px rgba(0, 0, 0, 0.3); filter: brightness(1.1); }}
         .stButton>button:active:not(:disabled) {{ transform: translateY(0px); box-shadow: var(--box-shadow-inset); }}
         .stButton>button:disabled {{ background-color: #555 !important; color: #888 !important; cursor: not-allowed; box-shadow: none; transform: none; filter: grayscale(50%); border-color: #666; opacity: 0.7; }}
         div.stButton[data-testid*="player_std_btn"] button, div.stButton[data-testid*="player_natural_btn"] button {{ background: linear-gradient(145deg, var(--player-blue), var(--player-blue-darker)) !important; border-color: var(--player-blue-darker) !important; color: white !important; }}
         div.stButton[data-testid*="banker_std_btn"] button, div.stButton[data-testid*="banker_natural_btn"] button {{ background: linear-gradient(145deg, var(--banker-red), var(--banker-red-darker)) !important; border-color: var(--banker-red-darker) !important; color: white !important; }}
         div.stButton[data-testid*="natural_btn"] button span[data-testid="stButtonIcon"] {{ color: var(--accent-gold) !important; font-size: 1.1em !important; margin-left: 5px; line-height: 1; filter: drop-shadow(0 0 1px black); vertical-align: middle; }}
         div.stButton[data-testid*="undo_std"] button, div.stButton[data-testid*="prog_loss_std"] button {{ background: linear-gradient(145deg, #6c757d, #5a6268); border-color: #5a6268; }}
         div.stButton[data-testid*="reset_all_std"] button {{ background: linear-gradient(145deg, #f57f17, #e65100); border-color: #e65100; color: #fff; }}
         div.stButton[data-testid*="reset_session_std"] button {{ background: linear-gradient(145deg, #ffca28, #ffb300); border-color: #ffb300; color: #111; }}
         div.stButton[data-testid*="reset_bead_road_std"] button {{ background: linear-gradient(145deg, #17a2b8, #117a8b); border-color: #117a8b; color: #fff; }}
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
         .stNumberInput label, .stTextInput label {{ font-size: 13px !important; color: var(--secondary-text) !important; margin-bottom: 3px !important; display: block; font-weight: bold; order: 1; }}
         .stNumberInput label[data-testid="stWidgetLabel"], .stTextInput label[data-testid="stWidgetLabel"] {{ margin-bottom: 0 !important; }}
         .stNumberInput input, .stTextInput input {{ font-size: 13px; color: var(--primary-text); background-color: var(--tertiary-bg); border: 1px solid var(--border-color); border-radius: 4px; padding: 6px 8px; width: 100%; box-shadow: var(--box-shadow-inset); box-sizing: border-box; order: 2; height: 31px; }}
         .stNumberInput input:focus, .stTextInput input:focus {{ border-color: var(--accent-gold); box-shadow: 0 0 4px rgba(212, 175, 55, 0.4), var(--box-shadow-inset); outline: none; }}
         ::placeholder {{ color: var(--secondary-text); opacity: 0.7; }}
         div[data-testid="stHorizontalBlock"]:has(div .stNumberInput[key*="starting_balance_input"]) > div:nth-child(1) .stNumberInput {{ margin-bottom: 0 !important; }}
         div[data-testid="stHorizontalBlock"]:has(div .stNumberInput[key*="starting_balance_input"]) > div:nth-child(2) {{ display: flex; align-items: flex-end; }}
         div[data-testid="stHorizontalBlock"]:has(div .stNumberInput[key*="starting_balance_input"]) > div:nth-child(2) .stButton {{ width: 100%; }}
         div[data-testid="stHorizontalBlock"]:has(div .stNumberInput[key*="starting_balance_input"]) > div:nth-child(2) .stButton button {{ margin-bottom: 0 !important; }}
         .stMetric {{ text-align: center; background-color: var(--secondary-bg); padding: 10px; border-radius: 4px; margin-top: 10px; margin-bottom: 8px; border: 1px solid var(--border-color); box-sizing: border-box; display: flex; flex-direction: column; justify-content: center; height: 70px; }}
         .stMetric label {{ color: var(--secondary-text) !important; font-size: 11px !important; font-weight: 400; text-transform: uppercase; margin-bottom: 2px !important; line-height: 1.2; }}
         .stMetric p {{ font-size: 20px !important; color: var(--primary-text) !important; font-weight: 700; margin-top: 2px; line-height: 1.1; word-wrap: break-word; }}
         .stMetric .stMetricDelta {{ display: none; }}
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
         .bead-road-sticky-container, .mid-col-sticky-container {{ position: -webkit-sticky; position: sticky; top: var(--sticky-top-offset, 15px); z-index: 10; }}
         .mid-col-sticky-container {{ z-index: 9; }}
         .bead-road-card-container {{ display: flex; flex-direction: column; align-items: center; }}
         .bead-road-card {{ width: fit-content; margin-top: 0; }}
         .bead-road-container {{ line-height: 0; text-align: center; background-color: var(--tertiary-bg); padding: 15px; border-radius: var(--border-radius); border: 1px solid var(--border-color); display: inline-block; overflow-x: auto; max-width: 100%; white-space: nowrap; }}
         .bead-row {{ margin-bottom: var(--bead-margin); height: var(--bead-size); white-space: nowrap; }}
         .bead-cell-banker, .bead-cell-player, .bead-cell-current, .bead-cell-empty {{ border-radius: 50%; text-align: center; width: var(--bead-size) !important; height: var(--bead-size) !important; line-height: var(--bead-size) !important; font-size: var(--bead-font-size) !important; font-weight: bold; display: inline-block; margin: 0 var(--bead-margin); border: 1px solid transparent; vertical-align: middle; box-shadow: var(--box-shadow-inset); position: relative; color: #fff; }}
         .bead-cell-banker {{ background-color: var(--banker-red); border-color: var(--banker-red-darker); }}
         .bead-cell-player {{ background-color: var(--player-blue); border-color: var(--player-blue-darker); }}
         .bead-cell-current {{ background-color: transparent; border: 2px dashed var(--accent-gold); box-shadow: 0 0 8px rgba(212, 175, 55, 0.4); line-height: calc(var(--bead-size) - 4px) !important; width: calc(var(--bead-size) - 0px) !important; height: calc(var(--bead-size) - 0px) !important; }}
         .bead-cell-empty {{ background-color: var(--secondary-bg); border-color: var(--border-color); box-shadow: none; }}
         .bead-cell-banker.natural::after, .bead-cell-player.natural::after {{ content: 'N'; position: absolute; top: var(--bead-natural-marker-offset); right: var(--bead-natural-marker-offset); width: var(--bead-natural-marker-size); height: var(--bead-natural-marker-size); line-height: var(--bead-natural-marker-size); border-radius: 50%; background-color: var(--accent-gold); color: #000; font-size: var(--bead-natural-marker-font-size); font-weight: bold; text-align: center; box-shadow: 0 0 2px rgba(0,0,0,0.4); z-index: 1; display: flex; align-items: center; justify-content: center; }}
         .bp-count-container {{ display: flex; justify-content: center; align-items: center; padding: 5px 10px; margin-bottom: 8px; background-color: var(--tertiary-bg); border-radius: var(--border-radius); border: 1px solid var(--border-color); width: fit-content; margin-left: auto; margin-right: auto; }}
         .bp-icon {{ width: var(--bp-count-icon-size); height: var(--bp-count-icon-size); border-radius: 50%; display: inline-flex; align-items: center; justify-content: center; font-weight: bold; color: white; font-size: 14px; margin: 0 5px; box-shadow: var(--box-shadow-inset); }}
         .bp-icon-p {{ background-color: var(--player-blue); border: 1px solid var(--player-blue-darker); }}
         .bp-icon-b {{ background-color: var(--banker-red); border: 1px solid var(--banker-red-darker); }}
         .bp-count-text {{ font-size: 18px; font-weight: bold; color: var(--primary-text); margin: 0 10px; }}
         .progression-info {{ margin-top: 10px; margin-bottom: 10px; text-align: center; background-color: var(--tertiary-bg); padding: 8px; border-radius: 4px; border: 1px solid var(--border-color); }}
         .progression-mode {{ font-size: 11px; color: var(--accent-gold); font-weight: bold; margin-bottom: 4px; text-transform: uppercase; }}
         .progression-step {{ font-size: 12px; color: var(--secondary-text); margin-bottom: 4px; text-transform: uppercase; }}
         .suggested-bet {{ font-size: 16px; color: var(--accent-gold); font-weight: bold; margin-bottom: 0; display: flex; align-items: center; justify-content: center; }}
         .suggested-bet i {{ margin-right: 6px; font-size: 0.9em; }}
         .progression-buttons {{ margin-top: 5px; margin-bottom: 10px; }}
         div[data-testid="stVerticalBlock"] > div[data-testid="stButton"]:has(button span:contains("Reset Phiên Cược")) {{ margin-top: 15px !important; margin-bottom: 5px !important; }}
         div[data-testid="stVerticalBlock"] > div[data-testid="stButton"]:has(button span:contains("Reset Bead Road")) {{ margin-top: 0px !important; margin-bottom: 5px !important; }}
         div[data-testid="stVerticalBlock"] > div[data-testid="stButton"]:has(button span:contains("Reset Tất Cả")) {{ margin-top: 0px !important; }}
         hr {{ border-top: 1px solid var(--border-color); margin: 10px 0;}}
         .stDivider {{ margin: 10px 0;}}
         .stAlert {{ border-radius: 4px; font-size: 12px; background-color: var(--tertiary-bg); border: 1px solid var(--accent-gold-darker); padding: 8px 12px; margin-bottom: 10px;}}
         .stAlert p, .stAlert div, .stAlert li {{ font-size: 12px !important; color: var(--primary-text); }}
         .stToast {{ font-size: 13px; }}
         div[data-testid="stInfo"] {{ background-color: rgba(13, 110, 253, 0.1); border: 1px solid var(--player-blue); border-left-width: 5px; padding: 10px; margin-bottom: 10px; font-size: 12px; border-radius: 4px; }}
         div[data-testid="stInfo"] p {{ color: var(--primary-text); font-size: 12px !important; margin-bottom: 0; }}
         .session-stats-container {{ background-color: var(--secondary-bg); border-radius: var(--border-radius); padding: 15px; margin-bottom: 18px; border: 1px solid var(--border-color); box-shadow: var(--box-shadow); }}
         .session-stats-container h4 {{ margin-bottom: 8px; }}
         .session-stats-grid {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px; text-align: center; }}
         .session-stat-item {{ background-color: var(--tertiary-bg); padding: 8px; border-radius: 4px; border: 1px solid var(--border-color); }}
         .session-stat-item .label {{ font-size: 11px; color: var(--secondary-text); text-transform: uppercase; margin-bottom: 3px; display: block;}}
         .session-stat-item .value {{ font-size: 16px; color: var(--primary-text); font-weight: bold; display: block; }}
         .session-stat-item .value .positive {{ color: var(--win-green); }}
         .session-stat-item .value .negative {{ color: var(--loss-red); }}
         .session-stat-item i {{ margin-right: 5px; }}
         .bet-history-container {{ max-height: 150px; overflow-y: auto; background-color: var(--tertiary-bg); padding: 10px; border-radius: 4px; border: 1px solid var(--border-color); margin-top: 10px; }}
         .bet-history-item {{ font-size: 12px; padding: 3px 5px; margin-bottom: 4px; border-radius: 3px; display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid var(--secondary-bg); }}
         .bet-history-item:last-child {{ margin-bottom: 0; border-bottom: none; }}
         .bet-history-item span {{ vertical-align: middle; }}
         .bet-history-outcome {{ font-weight: bold; margin-right: 8px; }}
         .bet-history-outcome.win {{ color: var(--win-green); }}
         .bet-history-outcome.loss {{ color: var(--loss-red); }}
         .bet-history-amount {{ color: var(--secondary-text); margin-right: 8px; font-size: 11px;}}
         .bet-history-profit {{ font-weight: bold; }}
         .bet-history-profit.win {{ color: var(--win-green); }}
         .bet-history-profit.loss {{ color: var(--loss-red); }}
         .bet-history-container::-webkit-scrollbar {{ width: 6px; }}
         .bet-history-container::-webkit-scrollbar-track {{ background: var(--secondary-bg); border-radius: 3px;}}
         .bet-history-container::-webkit-scrollbar-thumb {{ background: var(--border-color); border-radius: 3px;}}
         .bet-history-container::-webkit-scrollbar-thumb:hover {{ background: var(--accent-gold); }}
        </style>
        """, unsafe_allow_html=True) # End of CSS block

    # --- Main App Layout ---
    st.markdown('<div class="main">', unsafe_allow_html=True)
    st.markdown('<div class="app-title"><i class="fas fa-crown"></i> Baccarat Pro Predictor <i class="fas fa-crown"></i></div>', unsafe_allow_html=True)

    left_col, mid_col, right_col = st.columns([1.3, 1.7, 2.0])

    # --- LEFT COLUMN ---
    with left_col:
        # (Left column code remains the same as the previous correct version)
        with st.container(border=False):
            st.markdown('<div class="card"><h4><i class="fas fa-chart-line"></i> Chiến Lược Cược</h4>', unsafe_allow_html=True)
            st.info(f"""
            **Mục tiêu:** {DAILY_TARGET_SEQUENCES} chuỗi thắng 1-2-3-6 mỗi ngày (Tổng lợi nhuận mục tiêu: {MIN_DAILY_TARGET_PROFIT:,.0f}đ - {MAX_DAILY_TARGET_PROFIT:,.0f}đ).
            * Đặt **'Unit' = 40,000 đ**.
            * Chuỗi Lợi Nhuận: 40k → 80k → 120k → 240k.
            * Hoàn thành chuỗi (thắng cược 240k) sẽ đạt mục tiêu lợi nhuận ~**{MIN_TARGET_PROFIT_PER_SEQUENCE:,.0f}đ - {MAX_TARGET_PROFIT_PER_SEQUENCE:,.0f}đ** / chuỗi (tùy thuộc vào thắng Banker hay Player).
            """, icon="💡")
            balance = float(st.session_state.get('current_balance', 0.0)); initial_unit = float(st.session_state.get('initial_bet_unit', 40000.0)); mode = st.session_state.get('betting_mode', 'PROFIT')
            st.markdown("<h6>Cài Đặt</h6>", unsafe_allow_html=True)
            col_start_bal_inp, col_start_bal_btn = st.columns([0.7, 0.3])
            with col_start_bal_inp: st.number_input("Số Dư", min_value=0.0, step=1000.0, key="starting_balance_input", value=balance, format="%.0f", label_visibility="collapsed", placeholder="Nhập số dư...")
            with col_start_bal_btn: st.button("Đặt Số Dư", key="set_starting_balance_button", on_click=set_current_balance_from_input, use_container_width=True, help="Đặt số dư và bắt đầu phiên mới.")
            st.number_input("Unit", min_value=1.0, step=1000.0, key="input_initial_bet_unit", value=initial_unit, on_change=update_initial_bet_unit, format="%.0f", help="Mức cược cơ bản. Đề xuất: 40000.", label_visibility="collapsed", placeholder="Nhập unit cược (vd: 40000)...")
            st.divider()
            col_metric1, col_metric2 = st.columns(2);
            with col_metric1: st.metric(label="Số Dư Hiện Tại", value=f"{st.session_state.get('current_balance', 0.0):,.0f} đ")
            with col_metric2: start_bal = float(st.session_state.get('session_start_balance', st.session_state.get('current_balance', 0.0))); current_bal = float(st.session_state.get('current_balance', 0.0)); session_profit = current_bal - start_bal; st.metric(label="Lợi Nhuận Phiên", value=f"{session_profit:,.0f} đ")
            prog_idx = st.session_state.get('current_progression_index', 0); prog_seq = st.session_state.get('progression_sequence', [1, 2, 3, 6]); suggested_bet = float(st.session_state.get('suggested_bet_amount', 0.0)); current_balance_for_check = float(st.session_state.get('current_balance', 0.0)); balance_ok = current_balance_for_check >= suggested_bet
            st.markdown('<div class="progression-info">', unsafe_allow_html=True)
            if mode == 'PROFIT':
                if not (0 <= prog_idx < len(prog_seq)): prog_idx = 0
                step_num = prog_idx + 1; total_steps = len(prog_seq); current_unit_multiplier = prog_seq[prog_idx]
                st.markdown(f'<div class="progression-mode">Mode: Lợi Nhuận (1-2-3-6)</div>', unsafe_allow_html=True); st.markdown(f'<div class="progression-step">Bước: {step_num}/{total_steps} (x{current_unit_multiplier})</div>', unsafe_allow_html=True)
            else: multiplier = st.session_state.get('current_recovery_multiplier', 1); bets_at_level = st.session_state.get('recovery_bets_at_this_level', 0); bet_order_text = f"(Lần {bets_at_level + 1})"; st.markdown(f'<div class="progression-mode">Mode: Gỡ Lỗ (Delay Martingale)</div>', unsafe_allow_html=True); st.markdown(f'<div class="progression-step">Mức cược: x{multiplier} {bet_order_text}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="suggested-bet"><i class="fas fa-coins"></i> Cược: {suggested_bet:,.0f} đ</div>', unsafe_allow_html=True); st.markdown('</div>', unsafe_allow_html=True)
            if not balance_ok: st.warning(f"Số dư không đủ ({current_balance_for_check:,.0f} đ)!", icon="⚠️")
            st.markdown('<div class="progression-buttons">', unsafe_allow_html=True)
            p_col1, p_col2, p_col3 = st.columns(3)
            with p_col1: st.button("Thắng (P)", key="prog_win_p_std", on_click=handle_progression_win, args=(1.0,), disabled=not balance_ok, use_container_width=True)
            with p_col2: st.button("Thắng (B)", key="prog_win_b_std", on_click=handle_progression_win, args=(0.95,), disabled=not balance_ok, use_container_width=True)
            with p_col3: st.button("Thua", key="prog_loss_std", on_click=handle_progression_loss, disabled=not balance_ok, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
            st.markdown("<h6>Lịch Sử Cược Phiên</h6>", unsafe_allow_html=True)
            bet_history = st.session_state.get('bet_history', [])
            st.markdown('<div class="bet-history-container">', unsafe_allow_html=True)
            if not bet_history: st.markdown('<p style="font-size: 12px; color: var(--secondary-text); text-align: center;">Chưa có lịch sử.</p>', unsafe_allow_html=True)
            else:
                for bet in reversed(bet_history[-15:]): outcome_class = "win" if bet['outcome'] == "Win" else "loss"; profit_sign = "+" if bet['profit'] > 0 else ""; mode_display = f"[{bet.get('mode','?')[:1]}]"; history_item_html = f"""<div class="bet-history-item"><span><span class="bet-history-outcome {outcome_class}">{mode_display} {bet['outcome']}</span> <span class="bet-history-amount">({bet['amount']:,.0f})</span></span> <span class="bet-history-profit {outcome_class}">{profit_sign}{bet['profit']:,.0f}</span></div>"""; st.markdown(history_item_html, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
            st.button("Reset Phiên Cược", key="reset_session_std", on_click=reset_session, use_container_width=True, help="Bắt đầu phiên mới với số dư hiện tại. Giữ nguyên Bead Road.")
            st.button("Reset Bead Road", key="reset_bead_road_std", on_click=reset_bead_road, use_container_width=True, help="Xóa bảng Bead Road, giữ số dư/trạng thái cược.")
            st.button("Reset Tất Cả", key="reset_all_std", on_click=reset_all, use_container_width=True, help="Reset toàn bộ ứng dụng.")
            st.markdown('</div>', unsafe_allow_html=True)

    # --- MIDDLE COLUMN ---
    with mid_col:
        st.markdown('<div class="mid-col-sticky-container">', unsafe_allow_html=True)
        # Input Card
        with st.container(border=False):
             st.markdown('<div class="card"><h4><i class="fas fa-keyboard"></i> Nhập Kết Quả</h4>', unsafe_allow_html=True)
             col_in1, col_in2 = st.columns(2)
             with col_in1: st.button('Player', key="player_std_btn", on_click=add_result, args=('Player', False), use_container_width=True); st.button('Player', icon="⭐", key="player_natural_btn", on_click=add_result, args=('Player', True), use_container_width=True)
             with col_in2: st.button('Banker', key="banker_std_btn", on_click=add_result, args=('Banker', False), use_container_width=True); st.button('Banker', icon="⭐", key="banker_natural_btn", on_click=add_result, args=('Banker', True), use_container_width=True)
             st.button("Undo Last Result", key="undo_std", on_click=undo_last_result, disabled=not st.session_state.get('game_history', []), use_container_width=True); st.markdown('</div>', unsafe_allow_html=True)
        # Prediction Card
        with st.container(border=False):
            st.markdown('<div class="card"><h4><i class="fas fa-brain"></i> Dự Đoán</h4>', unsafe_allow_html=True); pred = st.session_state.get('predictions', {})

            # --- format_prediction function with N/A(R>1) shortening ---
            def format_prediction(label, value):
                result_class = ""
                display_value = str(value) # Default display
                if value == 'Banker': result_class = "prediction-result-Banker"
                elif value == 'Player': result_class = "prediction-result-Player"
                if isinstance(value, str):
                    if value == "N/A (Row > 1)": display_value="N/A(R>1)" # Shortened
                    elif value.startswith("Wait (Col") and ("<" in value or "!=" in value): display_value="Wait(ColSize)"
                    elif value.startswith("Waiting (Src"): display_value="Wait(SrcEmpty/OOB)"
                    elif value.startswith("Wait (Need Col"): display_value="Wait(NeedCol)"
                    elif value.startswith("Waiting (Need Prev B/P)"): display_value="Wait(NoMirSrc)"
                    elif value.startswith("Wait Natural"): display_value="WaitNat"
                    elif value.startswith("Wait Next"): display_value="WaitNext"
                    elif value.startswith("Waiting for pattern"): display_value="WaitM6Ptn"
                    elif value.startswith("Waiting (Not X Pos)"): display_value="WaitXPos"
                    elif value.startswith("Waiting (Need Matrix TL B/P)"): display_value="WaitMatrixTL"
                    elif value.startswith("Waiting (4+ identical)"): display_value="WaitM6(4+)"
                    elif value.startswith("Waiting (3 identical)"): display_value="WaitM6(3)"
                    elif value.startswith("Waiting (Matrix"): display_value="WaitMatrixOOB"
                    elif value.startswith("Error: No S89 result"): display_value="ErrS89"
                    elif value.startswith("Error"): display_value="Error"
                    elif value.startswith("No prediction"): display_value="NoPredict"
                    elif value.startswith("Wait (Need 5+ Streak)"): display_value="Wait5+"
                    elif value.startswith("Wait (Need 1st 3-Streak)"): display_value="Wait1st3"
                    elif value.startswith("Wait (Need 2nd 3-Streak)"): display_value="Wait2nd3"
                    elif value.startswith("Error (Streak"): display_value="ErrStreak"
                elif value is None: display_value="N/A"
                return f'<div class="prediction-box"><b>{label}:</b> <span class="{result_class}">{display_value}</span></div>'
            # --- End format_prediction ---

            st.markdown('<div class="prediction-grid" style="grid-template-columns: repeat(2, 1fr);">', unsafe_allow_html=True)
            st.markdown(format_prediction("Majority 6", pred.get("majority6", "...")), unsafe_allow_html=True); st.markdown(format_prediction("X Mark", pred.get("xMark", "...")), unsafe_allow_html=True)
            st.markdown(format_prediction("No Mirror", pred.get("noMirror", "...")), unsafe_allow_html=True); st.markdown(format_prediction("89 Special", pred.get("special89", "...")), unsafe_allow_html=True)
            st.markdown(format_prediction("2&5", pred.get("2and5", "...")), unsafe_allow_html=True); st.markdown(format_prediction("3 Baloon", pred.get("3baloon", "...")), unsafe_allow_html=True)
            st.markdown(format_prediction("I Strategy", pred.get("iStrategy", "...")), unsafe_allow_html=True); st.markdown(format_prediction("5s Streak", pred.get("5sStreak", "...")), unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="prediction-box overall" style="margin-top: 5px;"><b>Overall:</b> <span>{pred.get("percentage", "...")}</span></div>', unsafe_allow_html=True)
            final_pred_raw = pred.get("final", "No prediction"); final_pred_html = final_pred_raw
            if "<b>Banker</b>" in final_pred_raw: final_pred_html = final_pred_raw.replace("<b>Banker</b>", '<b class="prediction-result-Banker">Banker</b>')
            elif "<b>Player</b>" in final_pred_raw: final_pred_html = final_pred_raw.replace("<b>Player</b>", '<b class="prediction-result-Player">Player</b>')
            st.markdown(f'<div class="final-prediction">{final_pred_html}</div>', unsafe_allow_html=True); st.markdown('</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # --- RIGHT COLUMN ---
    with right_col:
        st.markdown('<div class="bead-road-sticky-container">', unsafe_allow_html=True)
        st.markdown('<div class="bead-road-card-container">', unsafe_allow_html=True) # Centering container

        # B/P COUNT Display (Moved outside card)
        rows_disp = st.session_state.get('rows', ROWS); cols_disp = st.session_state.get('cols', default_cols)
        grid_disp = st.session_state.get('bead_road_grid', create_default_grid(rows_disp, cols_disp, None))
        b_count_disp, p_count_disp = get_bead_road_stats(grid_disp)
        st.markdown(f"""
            <div class="bp-count-container">
                <span class="bp-icon bp-icon-p">P</span>
                <span class="bp-count-text">{p_count_disp}</span>
                <span class="bp-icon bp-icon-b">B</span>
                <span class="bp-count-text">{b_count_disp}</span>
            </div>
        """, unsafe_allow_html=True)

        # Bead Road Card
        with st.container(border=False):
            st.markdown('<div class="card bead-road-card">', unsafe_allow_html=True) # Card starts here
            st.markdown('<h4><i class="fas fa-border-all"></i> Bead Road</h4>', unsafe_allow_html=True)
            bead_html = '<div class="bead-road-container">'
            nat_grid_disp = st.session_state.get('natural_marks_grid', create_default_grid(rows_disp, cols_disp, False))
            current_row_disp = st.session_state.get('current_bead_road_row', 0); current_col_disp = st.session_state.get('current_bead_road_col', 0)
            next_pos_valid_disp = (0 <= current_row_disp < rows_disp) and (0 <= current_col_disp < cols_disp)
            grid_valid_disp = isinstance(grid_disp, list) and len(grid_disp) == rows_disp and (rows_disp == 0 or (len(grid_disp) > 0 and isinstance(grid_disp[0], list) and len(grid_disp[0]) == cols_disp))
            nat_grid_valid_disp = isinstance(nat_grid_disp, list) and len(nat_grid_disp) == rows_disp and (rows_disp == 0 or (len(nat_grid_disp) > 0 and isinstance(nat_grid_disp[0], list) and len(nat_grid_disp[0]) == cols_disp))

            if not grid_valid_disp or not nat_grid_valid_disp:
                 bead_html += "<p style='color: red; text-align: center; font-size: 12px; white-space: normal;'>Lỗi: Bead Road. Thử Reset.</p>"
            else:
                for i in range(rows_disp):
                    bead_html += '<div class="bead-row">'
                    for j in range(cols_disp):
                        cell = grid_disp[i][j]; is_natural = nat_grid_disp[i][j]
                        is_current_target = next_pos_valid_disp and (i == current_row_disp) and (j == current_col_disp)
                        cell_class = "bead-cell-empty"; cell_content = ""
                        if is_current_target: cell_class = "bead-cell-current"
                        elif cell == 'Banker':
                            cell_class = 'bead-cell-banker'; cell_content = 'B'
                            if is_natural: cell_class += ' natural'
                        elif cell == 'Player':
                            cell_class = 'bead-cell-player'; cell_content = 'P'
                            if is_natural: cell_class += ' natural'
                        bead_html += f'<div class="{cell_class}">{cell_content}</div>'
                    bead_html += '</div>' # End bead-row
            bead_html += '</div>' # End bead-road-container
            st.markdown(bead_html, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True) # End bead-road-card

        st.markdown('</div>', unsafe_allow_html=True) # End bead-road-card-container
        st.markdown('</div>', unsafe_allow_html=True) # End bead-road-sticky-container

        # Session Statistics Card (Remains below sticky section)
        with st.container(border=False):
            st.markdown('<div class="session-stats-container card">', unsafe_allow_html=True)
            st.markdown('<h4><i class="fas fa-chart-pie"></i> Thống Kê Phiên</h4>', unsafe_allow_html=True)
            session_wins = st.session_state.get('session_wins', 0); session_losses = st.session_state.get('session_losses', 0)
            session_start_time = st.session_state.get('session_start_time', datetime.datetime.now()); sequences_completed = st.session_state.get('session_sequences_completed', 0)
            formatted_time = "N/A"
            if isinstance(session_start_time, datetime.datetime): elapsed_time = datetime.datetime.now() - session_start_time; total_seconds = int(elapsed_time.total_seconds()); hours, remainder = divmod(total_seconds, 3600); minutes, seconds = divmod(remainder, 60); formatted_time = f"{hours:02}:{minutes:02}:{seconds:02}"
            st.markdown('<div class="session-stats-grid">', unsafe_allow_html=True)
            st.markdown(f"""<div class="session-stat-item"><span class="label"><i class="far fa-clock"></i> Thời Gian</span><span class="value">{formatted_time}</span></div>""", unsafe_allow_html=True)
            st.markdown(f"""<div class="session-stat-item"><span class="label"><i class="fas fa-trophy"></i> Thắng (Cược)</span><span class="value positive">{session_wins}</span></div>""", unsafe_allow_html=True)
            st.markdown(f"""<div class="session-stat-item"><span class="label"><i class="fas fa-heart-broken"></i> Thua (Cược)</span><span class="value negative">{session_losses}</span></div>""", unsafe_allow_html=True)
            st.markdown(f"""<div class="session-stat-item"><span class="label"><i class="fas fa-medal"></i> Chuỗi HT</span><span class="value" style="color: var(--accent-gold);">{sequences_completed} / {DAILY_TARGET_SEQUENCES}</span></div>""", unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True); st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

# --- Run ---
if __name__ == '__main__':
    main()
