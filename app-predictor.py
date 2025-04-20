# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
from collections import Counter
import math
import datetime

# --- Helper Functions ---
def create_default_grid(rows, cols, default_value):
    """Creates a 2D list (grid) with default values."""
    return [[default_value for _ in range(cols)] for _ in range(rows)]

# --- Prediction Functions (Keep unchanged as requested) ---
def predict_majority6(grid, current_col):
    """Predicts based on majority in the current column (up to 6 results)."""
    if not grid or not grid[0]: return "Waiting for data"
    if current_col >= len(grid[0]): return "No prediction (Col OOB)" # Grid boundary check
    # Ensure column exists before accessing
    if current_col < 0: return "Error: Invalid column index"

    current_column_data = []
    # Safely access grid data
    for row_idx in range(len(grid)):
         if len(grid[row_idx]) > current_col and grid[row_idx][current_col] is not None:
              current_column_data.append(grid[row_idx][current_col])

    # Use only the first 6 available entries in the current column for prediction logic
    relevant_data = current_column_data[:6]
    len_data = len(relevant_data) # Use length of relevant data (max 6)

    if len_data == 0: return "Waiting for data"

    count_b = relevant_data.count('Banker')
    count_p = relevant_data.count('Player')

    # Rule: If 4 or more identical results exist in the first 6 cells, wait.
    if count_b >= 4 or count_p >= 4:
        return "Waiting (4+ identical)"

    # Rule: If exactly 5 results, predict the majority. Tie waits.
    if len_data == 5:
        if count_b > count_p: return "Banker"
        if count_p > count_b: return "Player"
        return "Waiting (5, Tie)" # Explicitly handle tie at 5

    # Rule: If 2 or more results, check the last two. If same, predict that outcome,
    # unless the last three are identical, then wait.
    if len_data >= 2:
        last_two = relevant_data[-2:]
        if last_two[0] == last_two[1]: # Check if last two are the same
            # Check if the last three are the same (only if len_data >= 3)
            if len_data >= 3 and relevant_data[-3] == last_two[0]:
                 return "Waiting (3 identical)" # Wait if last 3 are same
            return last_two[0] # Predict the repeating outcome if only last 2 are same

    # Default state if none of the above conditions are met
    return "Waiting for pattern"


def predict_x_mark(grid, current_row, current_col):
    """Predicts based on the 3x3 X-Mark pattern."""
    if not grid or not grid[0]: return "Waiting for data"
    num_rows = len(grid); num_cols = len(grid[0])
    pred_row, pred_col = current_row, current_col
    max_cols_setting = st.session_state.get('cols', 18) # Get max cols setting

    # Check if the prediction column is beyond the configured limit
    if pred_col >= max_cols_setting: return "No prediction (Max Col)"

    # Calculate the top-left corner of the 3x3 matrix containing the prediction cell
    matrix_start_row = (pred_row // 3) * 3
    matrix_start_col = (pred_col // 3) * 3

    # Check if the calculated matrix start is within the grid boundaries
    if not (0 <= matrix_start_row < num_rows and 0 <= matrix_start_col < num_cols):
        return "Waiting (Matrix OOB)" # Matrix start outside grid

    # Check if the required top-left cell for the pattern exists and has a value
    # Need safe access here
    if matrix_start_row >= len(grid) or matrix_start_col >= len(grid[matrix_start_row]):
         return "Waiting (Matrix Calc OOB)" # Cannot access potential top-left cell
    first_value = grid[matrix_start_row][matrix_start_col]
    if first_value is None: return "Waiting (Need Matrix TL)" # Top-left must have B/P

    # Determine the position of the prediction cell *within* its 3x3 matrix
    relative_row = pred_row - matrix_start_row
    relative_col = pred_col - matrix_start_col

    # Define the 'X' positions within a 3x3 matrix (0-indexed)
    # Top-Right (0, 2), Center (1, 1), Bottom-Left (2, 0), Bottom-Right (2, 2)
    is_pred_cell_an_x_position = (relative_row == 0 and relative_col == 2) or \
                                 (relative_row == 1 and relative_col == 1) or \
                                 (relative_row == 2 and relative_col == 0) or \
                                 (relative_row == 2 and relative_col == 2)

    # If the prediction cell is one of the 'X' positions, predict the value from the top-left (0,0) of the matrix
    if is_pred_cell_an_x_position:
        return first_value # Predict the top-left value
    else:
        # If the prediction cell is NOT an 'X' position, we don't predict based on this rule
        return "Waiting (Not X Pos)"


def predict_no_mirror(grid, current_row, current_col):
    """Predicts based on the 'No Mirror' rule (looks two cells back)."""
    if not grid or not grid[0]: return "Waiting for data"
    num_rows = len(grid); num_cols = len(grid[0])
    pred_row, pred_col = current_row, current_col
    max_cols_setting = st.session_state.get('cols', 18) # Get max cols setting

    # Check if the prediction column is beyond the configured limit
    if pred_col >= max_cols_setting: return "No prediction (Max Col)"

    # Rule requires looking back 2 columns, so prediction starts from column index 2 (the 3rd column)
    if pred_col < 2: return "Waiting (Need Col 3+)"

    # Validate prediction row index
    if not (0 <= pred_row < num_rows): return "Error: Invalid row index"

    # Calculate source column indices safely
    src_col_prev2 = pred_col - 2
    src_col_prev1 = pred_col - 1

    # Check if source columns are valid
    # This check is implicitly covered by pred_col < 2 check above, but good for clarity
    # if src_col_prev1 < 0 or src_col_prev2 < 0: return "Error: Invalid source col calc"

    # Safe access for source cells within the grid
    val_prev2 = None
    val_prev1 = None

    if pred_row < len(grid):
        if src_col_prev2 < len(grid[pred_row]):
            val_prev2 = grid[pred_row][src_col_prev2]
        if src_col_prev1 < len(grid[pred_row]):
            val_prev1 = grid[pred_row][src_col_prev1]

    # Check if we have the necessary values from the previous two cells in the same row
    if val_prev1 is None or val_prev2 is None:
        return "Waiting (Need Prev Cells)" # Need values in C-1 and C-2

    # The No Mirror Logic:
    if val_prev1 == val_prev2:
        # If the previous two are the same (mirror), predict the opposite
        return 'Player' if val_prev1 == 'Banker' else 'Banker'
    else:
        # If the previous two are different (no mirror), predict the most recent one
        return val_prev1

def predict_special89(special89_state, last_result_after_natural):
    """Predicts based on the result following a Natural 8/9."""
    if special89_state == "waiting_for_natural": return "Wait Natural"
    elif special89_state == "waiting_for_result_after_natural": return "Wait Next"
    elif special89_state == "ready_for_prediction":
        # Ensure we have a stored result to predict
        return last_result_after_natural if last_result_after_natural else "Error: No S89 result"
    else:
        # Default or unknown state
        return "Wait Natural" # Default back to waiting for a natural


def predict_2_and_5(grid, current_row, current_col):
    """Predicts based on the value in row 2 or 5 of the previous column."""
    if not grid or not grid[0]: return "Waiting for data"
    num_rows = len(grid); num_cols = len(grid[0])
    pred_row, pred_col = current_row, current_col
    max_cols_setting = st.session_state.get('cols', 18) # Get max cols setting

    # Check if the prediction column is beyond the configured limit
    if pred_col >= max_cols_setting: return "No prediction (Max Col)"

    # Rule applies starting from the 4th column (index 3)
    if pred_col < 3: return "Waiting (Starts Col 4)"

    # Source column is the one immediately preceding the prediction column
    source_col = pred_col - 1

    # Determine the source row based on the prediction row
    if 0 <= pred_row <= 2: # Prediction is in rows 1, 2, or 3 (0, 1, 2)
        source_row = 1 # Source is row 2 (index 1)
    elif 3 <= pred_row <= 5: # Prediction is in rows 4, 5, or 6 (3, 4, 5)
        source_row = 4 # Source is row 5 (index 4)
    else:
        # This case should ideally not happen if current_row is always 0-5
        return "Error: Invalid pred row"

    # Validate source cell coordinates before accessing grid
    if not (0 <= source_row < num_rows and 0 <= source_col < num_cols):
        return f"Waiting (Src [{source_row+1},{source_col+1}] OOB)" # Source cell outside grid

    # Safe access for the source cell value
    source_value = None
    if source_row < len(grid) and source_col < len(grid[source_row]):
         source_value = grid[source_row][source_col]
    else:
         # Should be caught by OOB check above, but for robustness:
         return f"Waiting (Src [{source_row+1},{source_col+1}] Grid Access)"


    # Check if the source cell has a value
    if source_value is None:
        return f"Waiting (Src [{source_row+1},{source_col+1}] empty)"
    else:
        # Predict the value found in the source cell
        return source_value

# --- Aggregate & Utility Functions ---
def calculate_prediction_percent(predictions_list):
    """Calculates the percentage agreement for Banker/Player predictions."""
    valid_predictions = [p for p in predictions_list if p in ['Banker', 'Player']]
    total = len(valid_predictions)
    if total == 0: return "Waiting..." # No valid B/P predictions yet
    b_count = valid_predictions.count('Banker'); p_count = total - b_count # Or use p_count = valid_predictions.count('Player')
    if b_count > p_count:
        # Use ceiling to round up the percentage
        return f"{math.ceil((b_count/total)*100)}% Banker"
    elif p_count > b_count:
        return f"{math.ceil((p_count/total)*100)}% Player"
    else: # Tie
        return "50% B / 50% P"

def get_final_prediction(predictions):
    """Determines the final prediction based on consensus (>= 3 signals)."""
    # Filter for only 'Banker' or 'Player' predictions
    valid_preds = [p for p in predictions if p in ['Banker', 'Player']]
    num_valid = len(valid_preds)

    # Require at least 3 valid signals to make a final prediction
    if num_valid < 3 : return "No prediction (Need ≥3 signals)"

    # Count the occurrences of 'Banker' and 'Player'
    pred_counts = Counter(valid_preds)
    most_common = pred_counts.most_common(1) # Get the most frequent prediction

    if not most_common: return "No prediction" # Should not happen if num_valid >= 3, but safe check

    outcome, count = most_common[0]

    # Check if there are multiple predictions with the same highest count (a tie)
    # If there's a second most common element and its count is the same as the first, it's a tie
    if len(pred_counts) > 1 and len(pred_counts.most_common(2)) > 1 and pred_counts.most_common(2)[1][1] == count:
        return f"No prediction (Tie {count}-{count})" # e.g., 3 B, 3 P -> Tie 3-3

    # Require at least 3 predictions agreeing on the outcome
    if count >= 3:
        # Format the prediction string with bold outcome and agreement count
        return f"Predict <b>{outcome}</b> ({count}/{num_valid} agree)"
    else:
        # If the highest count is less than 3, the signal is too weak
        return "No prediction (Weak Signal)" # e.g., 2 B, 1 P -> Weak Signal

def get_bead_road_stats(grid):
    """Calculates Banker and Player counts from the entire bead road grid."""
    if not grid or not grid[0]: return 0, 0
    # Flatten the grid and filter for valid Banker/Player entries
    flat_grid = [cell for row in grid for cell in row if cell in ['Banker', 'Player']]
    b_count = flat_grid.count('Banker')
    p_count = flat_grid.count('Player')
    return b_count, p_count

# --- Progression Strategy Functions (MODIFIED) ---

def initialize_progression_state():
    """Initializes all progression and session state variables."""
    # <<< MODIFIED: Update PROFIT sequence >>>
    st.session_state.progression_sequence = [1, 2] # New sequence: 1 unit, then 2 units
    # <<< END MODIFICATION >>>
    st.session_state.current_progression_index = 0

    # <<< NEW STATE: Track profit within the current PROFIT sequence run >>>
    st.session_state.current_profit_sequence_profit = 0.0
    # <<< END NEW STATE >>>

    if 'initial_bet_unit' not in st.session_state:
        # <<< NOTE: Set your desired default base bet here if needed, e.g., 40000.0 >>>
        # <<< Or instruct the user to set it to 40k via the UI for the 120k target >>>
        st.session_state.initial_bet_unit = 40000.0 # Keeping original default, user MUST set to 40k in UI
    if 'current_balance' not in st.session_state:
        st.session_state.current_balance = 2000000.0

    st.session_state.current_balance = float(st.session_state.get('current_balance', 10000000.0))
    st.session_state.initial_bet_unit = float(st.session_state.get('initial_bet_unit', 100000.0)) # Read the potentially user-set value

    # Initialize session tracking variables
    st.session_state.session_start_balance = st.session_state.current_balance
    st.session_state.last_withdrawal_profit_level = 0.0 # For overall session target
    st.session_state.session_start_time = datetime.datetime.now()
    st.session_state.session_wins = 0
    st.session_state.session_losses = 0
    st.session_state.bet_history = []

    # State for Delayed Martingale (RECOVERY mode)
    st.session_state.betting_mode = 'PROFIT' # Start in PROFIT mode
    st.session_state.current_recovery_multiplier = 1
    st.session_state.recovery_bets_at_this_level = 0

    calculate_suggested_bet() # Calculate the first bet amount

def calculate_suggested_bet():
    """Calculates the next bet based on the current mode (PROFIT or RECOVERY)."""
    current_balance = float(st.session_state.get('current_balance', 0.0))
    session_start = float(st.session_state.get('session_start_balance', current_balance))
    unit = float(st.session_state.get('initial_bet_unit', 1.0)) # Ensure unit is float
    mode = st.session_state.get('betting_mode', 'PROFIT')

    # 1. Determine Current Mode (and handle transitions)
    if current_balance >= session_start: # Back at or above starting balance
        if mode == 'RECOVERY': # Check if we are *transitioning* from RECOVERY to PROFIT
            # <<< MODIFIED: Update toast text for new sequence >>>
            st.toast("🎉 Về bờ! Chuyển sang chế độ Lợi Nhuận (1-2).", icon="💰")
            st.session_state.betting_mode = 'PROFIT'
            st.session_state.current_progression_index = 0 # Reset standard progression
            # <<< MODIFIED: Reset sequence profit tracker on transition to PROFIT >>>
            st.session_state.current_profit_sequence_profit = 0.0
            # <<< END MODIFICATION >>>
            st.session_state.current_recovery_multiplier = 1 # Reset recovery state
            st.session_state.recovery_bets_at_this_level = 0
        mode = 'PROFIT' # Ensure mode is set to PROFIT
        st.session_state.betting_mode = 'PROFIT' # Update state explicitly just in case
    else: # current_balance < session_start -> Need RECOVERY
        if mode == 'PROFIT': # Check if we are *transitioning* from PROFIT to RECOVERY
            st.toast("🚨 Vốn giảm! Chuyển sang chế độ Gỡ Lỗ (Delay Martingale).", icon="🛡️")
            st.session_state.betting_mode = 'RECOVERY'
            # <<< MODIFIED: Reset sequence profit when leaving PROFIT mode >>>
            st.session_state.current_profit_sequence_profit = 0.0
            # <<< END MODIFICATION >>>
            st.session_state.current_recovery_multiplier = 1 # Start recovery at 1x
            st.session_state.recovery_bets_at_this_level = 0 # Start fresh at this level
        mode = 'RECOVERY' # Ensure mode is set to RECOVERY
        st.session_state.betting_mode = 'RECOVERY' # Update state explicitly

    # 2. Calculate Bet Based on Mode
    suggested_bet = 0.0 # Initialize as float
    if mode == 'PROFIT':
        idx = st.session_state.get('current_progression_index', 0)
        # <<< MODIFIED: Use the updated sequence default >>>
        sequence = st.session_state.get('progression_sequence', [1, 2]) # Default to [1, 2]
        # <<< END MODIFICATION >>>
        # Ensure index is valid, reset if not
        if not (0 <= idx < len(sequence)):
            idx = 0
            st.session_state.current_progression_index = 0
        multiplier = sequence[idx]
        suggested_bet = unit * multiplier

    elif mode == 'RECOVERY':
        multiplier = st.session_state.get('current_recovery_multiplier', 1)
        suggested_bet = unit * multiplier

    st.session_state.suggested_bet_amount = float(suggested_bet) # Store as float


def handle_progression_win(payout_ratio):
    """Handles a win, updates balance, history, and determines next bet state."""
    bet_amount = float(st.session_state.get('suggested_bet_amount', 0.0))
    current_balance = float(st.session_state.get('current_balance', 0.0))
    session_start = float(st.session_state.get('session_start_balance', current_balance))
    mode = st.session_state.get('betting_mode', 'PROFIT')

    winnings = bet_amount * payout_ratio
    new_balance = current_balance + winnings

    # Record Win in History (Keep unchanged)
    if 'bet_history' not in st.session_state: st.session_state.bet_history = []
    st.session_state.bet_history.append({
        'outcome': 'Win', 'amount': bet_amount, 'profit': winnings, 'timestamp': datetime.datetime.now(), 'mode': mode
    })
    max_history = 50
    if len(st.session_state.bet_history) > max_history:
        st.session_state.bet_history = st.session_state.bet_history[-max_history:]

    # Update Balance and Session Win Count (Keep unchanged)
    st.session_state.current_balance = new_balance
    st.session_state.session_wins = st.session_state.get('session_wins', 0) + 1

    # --- Update State for Next Bet ---
    # Check if balance reached start *before* updating progression state (Keep unchanged)
    # If we reached start balance while in RECOVERY, calculate_suggested_bet will handle the switch
    if new_balance >= session_start and mode == 'RECOVERY':
        pass # Mode switch is handled by calculate_suggested_bet() called at the end

    elif mode == 'PROFIT':
        # <<< MODIFICATION: Implement new 1-2 sequence logic with profit target >>>
        # 1. Add profit to current sequence tracker
        st.session_state.current_profit_sequence_profit = st.session_state.get('current_profit_sequence_profit', 0.0) + winnings

        # 2. Get current state for checks
        current_idx = st.session_state.get('current_progression_index', 0)
        prog_seq = st.session_state.get('progression_sequence', [1, 2]) # Use [1, 2]
        sequence_len = len(prog_seq)

        # 3. Check if this win completes the sequence
        # The *next* index calculation determines if the sequence is finished
        next_idx = current_idx + 1
        if next_idx >= sequence_len: # Sequence complete (win on step 2)
            # Sequence complete! Check the profit target for this specific sequence.
            current_sequence_profit = st.session_state.get('current_profit_sequence_profit', 0.0)
            profit_target = 120000.0 # Your specific target for the 1-2 sequence
            if current_sequence_profit >= profit_target:
                st.toast(f"🎉 Đạt mục tiêu chuỗi PROFIT ({current_sequence_profit:,.0f} ≥ {profit_target:,.0f})! Rút tiền!", icon="💰")
                st.balloons() # Optional celebration

            # Always reset the PROFIT sequence index and its profit tracker after completion
            st.session_state.current_progression_index = 0
            st.session_state.current_profit_sequence_profit = 0.0 # Reset sequence profit
            # <<< MODIFIED: Update toast text for completed sequence >>>
            st.toast("Chuỗi thắng PROFIT (1-2) hoàn tất! Reset về mức cược đầu.", icon="✅")
        else:
            # Sequence not complete (win on step 1), just advance the index for the next bet (step 2)
            st.session_state.current_progression_index = next_idx
        # <<< END MODIFICATION >>>

    elif mode == 'RECOVERY':
        # Win during recovery: Reset bets_at_level, keep multiplier (Keep unchanged logic)
        current_multiplier = st.session_state.get('current_recovery_multiplier', 1)
        st.toast(f"Thắng khi đang gỡ! 👍 Tiếp tục cược {current_multiplier} unit.", icon="✅")
        st.session_state.recovery_bets_at_this_level = 0 # Reset counter, next bet is first at this level

    # Check for overall session profit target (Keep unchanged - this is separate from the sequence target)
    last_notified_level = st.session_state.get('last_withdrawal_profit_level', 0.0)
    current_profit_session = new_balance - session_start # Calculate overall session profit
    overall_profit_target = 800000.0 # Your overall session target
    if current_profit_session >= overall_profit_target and last_notified_level < overall_profit_target:
        st.toast(f"✅ Đạt mục tiêu TOÀN PHIÊN ({overall_profit_target:,.0f} lợi nhuận)!", icon="🎯")
        st.balloons()
        st.session_state.last_withdrawal_profit_level = overall_profit_target # Update the notification threshold

    # Calculate next bet (this will correctly check mode and set the next bet amount)
    calculate_suggested_bet()


def handle_progression_loss():
    """Handles a loss, updates balance, history, and determines next bet state."""
    bet_amount = float(st.session_state.get('suggested_bet_amount', 0.0))
    current_balance = float(st.session_state.get('current_balance', 0.0))
    unit = float(st.session_state.get('initial_bet_unit', 1.0))
    mode = st.session_state.get('betting_mode', 'PROFIT') # Get mode *before* potential switch

    new_balance = current_balance - bet_amount

    # Record Loss in History (Keep unchanged)
    if 'bet_history' not in st.session_state: st.session_state.bet_history = []
    st.session_state.bet_history.append({
        'outcome': 'Loss', 'amount': bet_amount, 'profit': -bet_amount, 'timestamp': datetime.datetime.now(), 'mode': mode
    })
    max_history = 50
    if len(st.session_state.bet_history) > max_history:
        st.session_state.bet_history = st.session_state.bet_history[-max_history:]

    # Update Balance and Session Loss Count (Keep unchanged)
    st.session_state.current_balance = new_balance
    st.session_state.session_losses = st.session_state.get('session_losses', 0) + 1

    # --- Update State for Next Bet ---
    if mode == 'PROFIT':
        # Loss during profit mode: Reset standard progression index AND the sequence profit tracker
        st.session_state.current_progression_index = 0
        # <<< MODIFIED: Reset sequence profit on loss in PROFIT mode >>>
        st.session_state.current_profit_sequence_profit = 0.0
        # <<< END MODIFICATION >>>
        # <<< MODIFIED: Update toast text >>>
        st.toast("Thua khi đang Lợi Nhuận! 😢 Reset về mức cược đầu (1 unit).", icon="❌")
        # Check if this loss triggers recovery mode happens in calculate_suggested_bet below

    elif mode == 'RECOVERY':
        # Loss during recovery: Apply the Delayed Martingale logic (Keep unchanged logic)
        current_multiplier = st.session_state.get('current_recovery_multiplier', 1)
        bets_at_level = st.session_state.get('recovery_bets_at_this_level', 0)
        bets_at_level += 1 # Increment bets made (now lost) at this level

        # Apply the 1-1-2-2-4-4... logic for recovery
        if current_multiplier == 1: # Handling the initial 1x level
            if bets_at_level == 1: # First loss at 1x
                st.session_state.current_recovery_multiplier = 1 # Stay at 1x
                st.session_state.recovery_bets_at_this_level = bets_at_level # Now 1
                st.toast("Thua lần 1 khi Gỡ! 😥 Tiếp tục cược 1 unit.", icon="❌")
            elif bets_at_level >= 2: # Second (or more) loss at 1x
                st.session_state.current_recovery_multiplier = 2 # Double to 2x
                st.session_state.recovery_bets_at_this_level = 0 # Reset counter for 2x level
                st.toast("Thua lần 2 khi Gỡ! 😥 Bắt đầu nhân đôi (2 units).", icon="❌")
        else: # Handling levels > 1x (2x, 4x, 8x...)
            if bets_at_level == 1: # First loss at this >1x level
                st.session_state.current_recovery_multiplier = current_multiplier # Repeat the bet (keep multiplier)
                st.session_state.recovery_bets_at_this_level = bets_at_level # Now 1
                st.toast(f"Thua mức {current_multiplier}x! 😥 Lặp lại cược {current_multiplier} units.", icon="❌")
            elif bets_at_level >= 2: # Second (or more) loss at this >1x level
                new_multiplier = current_multiplier * 2 # Double for the next bet
                st.session_state.current_recovery_multiplier = new_multiplier
                st.session_state.recovery_bets_at_this_level = 0 # Reset counter for new level
                st.toast(f"Thua mức {current_multiplier}x lần 2! 😥 Nhân đôi lên {new_multiplier} units.", icon="❌")

    # Calculate next bet (this will correctly check balance vs start and potentially switch mode)
    calculate_suggested_bet()


def update_initial_bet_unit():
    """Callback function to update the initial bet unit from input."""
    if 'input_initial_bet_unit' in st.session_state:
        try:
            new_unit = float(st.session_state.input_initial_bet_unit)
            if new_unit > 0:
                st.session_state.initial_bet_unit = new_unit
            else:
                # Handle non-positive input, maybe set to a default minimum
                st.session_state.initial_bet_unit = 1.0
                st.warning("Bet unit must be positive. Setting to 1.0")
            # Recalculate suggested bet after unit change
            calculate_suggested_bet()
        except (ValueError, TypeError):
            # Handle invalid input (non-numeric)
            st.error("Invalid input for Bet Unit. Please enter a number.")
            # Optionally revert to the previous valid value or a default
            # st.session_state.initial_bet_unit = st.session_state.get('initial_bet_unit', 100000.0) # Revert example

def set_current_balance_from_input():
    """Callback to update current_balance AND reset session tracking and betting state."""
    if 'starting_balance_input' in st.session_state:
        try:
            new_balance = float(st.session_state.starting_balance_input)
            if new_balance >= 0:
                st.session_state.current_balance = new_balance
                # Reset session tracking AND betting strategy state completely
                st.session_state.session_start_balance = st.session_state.current_balance # New session starts now
                st.session_state.last_withdrawal_profit_level = 0.0
                st.session_state.session_start_time = datetime.datetime.now()
                st.session_state.session_wins = 0
                st.session_state.session_losses = 0
                st.session_state.bet_history = []
                st.session_state.betting_mode = 'PROFIT' # Start fresh in profit mode
                st.session_state.current_progression_index = 0
                # <<< MODIFIED: Reset sequence profit >>>
                st.session_state.current_profit_sequence_profit = 0.0
                # <<< END MODIFICATION >>>
                st.session_state.current_recovery_multiplier = 1 # Reset recovery state
                st.session_state.recovery_bets_at_this_level = 0
                st.toast("Số dư cập nhật. Phiên theo dõi & Chiến lược cược mới bắt đầu.", icon="💰")
                calculate_suggested_bet() # Calculate bet based on new balance/state
            else:
                st.warning("Số dư không thể là số âm.")
        except (ValueError, TypeError):
            st.error("Vui lòng nhập số dư hợp lệ.")
    else:
        # This case indicates a potential issue with Streamlit's state or key naming
        st.error("Lỗi: Không tìm thấy trường nhập số dư ban đầu ('starting_balance_input').")


def reset_session():
    """Resets only the current session's tracking, progression, and betting mode, keeping current balance."""
    # Reset betting strategy state to initial PROFIT mode values
    st.session_state.current_progression_index = 0
    # <<< MODIFIED: Reset sequence profit >>>
    st.session_state.current_profit_sequence_profit = 0.0
    # <<< END MODIFICATION >>>
    st.session_state.betting_mode = 'PROFIT' # Force back to PROFIT mode
    st.session_state.current_recovery_multiplier = 1
    st.session_state.recovery_bets_at_this_level = 0

    # Reset session tracking variables, starting point is the *current* balance
    st.session_state.session_start_balance = st.session_state.current_balance
    st.session_state.last_withdrawal_profit_level = 0.0
    st.session_state.session_start_time = datetime.datetime.now()
    st.session_state.session_wins = 0
    st.session_state.session_losses = 0
    st.session_state.bet_history = [] # Clear history for the new session
    st.toast("Phiên cược đã được reset.", icon="🔄")
    calculate_suggested_bet() # Calculate the first bet for the new session


# --- Backend Functions ---
# (update_all_predictions, add_result, undo_last_result - Kept original logic, check safe access)

def update_all_predictions():
    """Calculates and updates all prediction types in the session state."""
    grid = st.session_state.get('bead_road_grid')
    row = st.session_state.get('current_bead_road_row')
    col = st.session_state.get('current_bead_road_col')
    s89_state = st.session_state.get('special89_state')
    s89_result = st.session_state.get('last_result_after_natural')

    # Initial check if necessary data is available
    if grid is None or row is None or col is None or s89_state is None:
        # Set default waiting messages if data is missing
        st.session_state.predictions = {
            'majority6': "Waiting...", 'xMark': "Waiting...", 'noMirror': "Waiting...",
            'special89': "Waiting...", '2and5': "Waiting...",
            'percentage': "Waiting...", 'final': "Waiting for data..."
        }
        return # Exit if essential data isn't ready

    # Calculate predictions from individual functions
    # Ensure prediction functions handle potential edge cases and grid boundaries
    pred_maj6 = predict_majority6(grid, col)
    pred_x_mark = predict_x_mark(grid, row, col)
    pred_no_mirror = predict_no_mirror(grid, row, col)
    pred_s89 = predict_special89(s89_state, s89_result)
    pred_2and5 = predict_2_and_5(grid, row, col)

    # Aggregate predictions for percentage and final decision
    predictions_list = [pred_maj6, pred_x_mark, pred_no_mirror, pred_s89, pred_2and5]
    pred_percent = calculate_prediction_percent(predictions_list)
    final_pred = get_final_prediction(predictions_list)

    # Store all predictions in session state
    st.session_state.predictions = {
        'majority6': pred_maj6, 'xMark': pred_x_mark, 'noMirror': pred_no_mirror,
        'special89': pred_s89, '2and5': pred_2and5,
        'percentage': pred_percent, 'final': final_pred
    }


def add_result(result, is_natural):
    """Adds a new game result to the grid and history, updates state."""
    # Check if all required state keys exist before proceeding
    required_keys = [
        'special89_state', 'last_result_after_natural', 'last_natural_pos',
        'current_bead_road_row', 'current_bead_road_col', 'game_count',
        'rows', 'cols', 'bead_road_grid', 'natural_marks_grid'
    ]
    if not all(key in st.session_state for key in required_keys):
        st.error("Lỗi: Trạng thái game chưa được khởi tạo đầy đủ. Thử Reset Toàn Bộ Game.")
        return

    # Store previous state for potential undo
    prev_s89_state = st.session_state.get('special89_state', 'waiting_for_natural')
    prev_s89_last_res = st.session_state.get('last_result_after_natural', None)
    prev_s89_last_nat_pos = st.session_state.get('last_natural_pos', None)
    prev_bead_row = st.session_state.get('current_bead_road_row')
    prev_bead_col = st.session_state.get('current_bead_road_col')

    # --- Prepare history entry ---
    game_id = st.session_state.get('game_count', 0) + 1
    new_game = {
        'id': game_id,
        'result': result,
        'is_natural': is_natural,
        # Store previous state values for undo
        'prev_s89_state': prev_s89_state,
        'prev_s89_last_res': prev_s89_last_res,
        'prev_s89_last_nat_pos': prev_s89_last_nat_pos,
        'prev_bead_row': prev_bead_row,
        'prev_bead_col': prev_bead_col,
        # Store the actual cell filled by *this* action
        'bead_row_filled': prev_bead_row, # The row index where the bead was placed
        'bead_col_filled': prev_bead_col, # The col index where the bead was placed
    }

    # --- Update Bead Road Grid ---
    current_row = st.session_state.get('current_bead_road_row', 0)
    current_col = st.session_state.get('current_bead_road_col', 0)
    rows = st.session_state.get('rows', 6)
    cols = st.session_state.get('cols', 18)
    grid = st.session_state.get('bead_road_grid')
    nat_grid = st.session_state.get('natural_marks_grid')

    # Validate grid existence
    if grid is None or nat_grid is None:
        st.error("Lỗi: Grid dữ liệu không tồn tại. Thử Reset Toàn Bộ Game.")
        return

    # Validate grid dimensions (optional but recommended)
    # This helps catch state corruption issues
    if len(grid) != rows or (len(grid) > 0 and len(grid[0]) != cols):
        st.warning(f"Grid dimensions mismatch detected! Expected {rows}x{cols}, got {len(grid)}x{len(grid[0]) if grid else 0}. Re-creating.")
        grid = create_default_grid(rows, cols, None)
        nat_grid = create_default_grid(rows, cols, False)
        current_row, current_col = 0, 0 # Reset position
        st.session_state.bead_road_grid = grid
        st.session_state.natural_marks_grid = nat_grid
        st.session_state.current_bead_road_row = 0
        st.session_state.current_bead_road_col = 0

    # Place the result in the current cell if within bounds
    next_row, next_col = -1, -1 # Initialize invalid values
    if 0 <= current_row < rows and 0 <= current_col < cols:
        grid[current_row][current_col] = result
        nat_grid[current_row][current_col] = is_natural
        st.session_state.bead_road_grid = grid # Update state
        st.session_state.natural_marks_grid = nat_grid # Update state

        # Calculate the next position for the cursor/target cell
        next_row = current_row + 1
        next_col = current_col
        if next_row >= rows: # If we reached the bottom row
            next_row = 0 # Wrap back to the top row
            next_col = current_col + 1 # Move to the next column

        # Update the cursor position in session state
        st.session_state.current_bead_road_row = next_row
        st.session_state.current_bead_road_col = next_col

        # Check if the *next* column is out of bounds (grid is full)
        if next_col >= cols:
            st.toast("Bead Road Grid is full.", icon="⚠️")
            # Optionally disable input buttons here or handle full grid state

    else:
        # This occurs if the calculated current_row/col was already invalid (e.g., grid full)
        st.toast(f"Cannot add result. Grid position ({current_row}, {current_col}) invalid or full.", icon="error")
        # Keep current row/col as they were, don't update further

    # --- Update Game History and Count ---
    st.session_state.game_history = st.session_state.get('game_history', []) + [new_game]
    st.session_state.game_count = game_id

    # --- Update Special 8/9 State Machine ---
    current_s89_state = prev_s89_state # State before this result was added
    next_s89_state = current_s89_state # Initialize next state as current
    next_nat_pos = prev_s89_last_nat_pos # Initialize next natural pos
    next_res_after_nat = prev_s89_last_res # Initialize next result after natural

    # Determine the position where the natural occurred *if* this result is natural
    natural_pos_this_turn = None
    if is_natural and (0 <= current_row < rows and 0 <= current_col < cols):
         natural_pos_this_turn = {'row': current_row, 'col': current_col}

    # State transitions based on current result and previous state
    if is_natural:
        # A natural just occurred
        next_nat_pos = natural_pos_this_turn # Store its position
        next_s89_state = "waiting_for_result_after_natural" # Need the next result
        next_res_after_nat = None # Clear any previously stored result
    elif current_s89_state == "waiting_for_result_after_natural":
        # This is the result immediately following a natural
        next_res_after_nat = result # Store this result for prediction
        next_s89_state = "ready_for_prediction" # Now we can predict
        # next_nat_pos remains the same (position of the *previous* natural)
    elif current_s89_state == "ready_for_prediction":
        # We were ready to predict, but a non-natural occurred. Reset.
        next_s89_state = "waiting_for_natural" # Go back to waiting for a new natural
        next_res_after_nat = None # Clear stored result
        next_nat_pos = None # Clear stored natural position
    # If current_s89_state is "waiting_for_natural" and is_natural is False, state doesn't change.

    # Update the session state with the new S89 values
    st.session_state.last_natural_pos = next_nat_pos
    st.session_state.special89_state = next_s89_state
    st.session_state.last_result_after_natural = next_res_after_nat

    # --- Final Updates ---
    update_all_predictions() # Recalculate predictions based on the new state
    # st.rerun() # Usually not needed, Streamlit handles rerun on state change/button click


def undo_last_result():
    """Reverts the last added result and restores the previous state."""
    history = st.session_state.get('game_history', [])
    if not history:
        st.toast("Không có gì để hoàn tác.", icon="🤷‍♂️")
        return

    # Retrieve the last game record and remove it from history
    undone_game = history.pop()
    st.session_state.game_history = history
    st.session_state.game_count = st.session_state.get('game_count', 1) - 1 # Decrement game count

    # --- Restore Bead Road ---
    # Get the position where the bead was actually placed in the undone step
    row_filled = undone_game.get('bead_row_filled')
    col_filled = undone_game.get('bead_col_filled')

    grid = st.session_state.get('bead_road_grid')
    nat_grid = st.session_state.get('natural_marks_grid')
    rows = st.session_state.get('rows', 6)
    cols = st.session_state.get('cols', 18)

    # Check if grid state exists and coordinates are valid before clearing the cell
    if grid is not None and nat_grid is not None and \
       row_filled is not None and col_filled is not None and \
       (0 <= row_filled < rows) and (0 <= col_filled < cols) and \
       (row_filled < len(grid)) and (col_filled < len(grid[0])): # Ensure safe access

        grid[row_filled][col_filled] = None # Clear the result
        nat_grid[row_filled][col_filled] = False # Clear the natural mark

        st.session_state.bead_road_grid = grid
        st.session_state.natural_marks_grid = nat_grid

        # Restore the cursor position to where the undone bead *was* placed
        st.session_state.current_bead_road_row = row_filled
        st.session_state.current_bead_road_col = col_filled
    else:
        # If state is inconsistent, try restoring cursor to previous *cursor* position
        st.warning("Lỗi khi hoàn tác grid, có thể trạng thái không nhất quán. Đặt lại vị trí con trỏ.")
        st.session_state.current_bead_road_row = undone_game.get('prev_bead_row', 0)
        st.session_state.current_bead_road_col = undone_game.get('prev_bead_col', 0)


    # --- Restore Special 8/9 State ---
    st.session_state.special89_state = undone_game.get('prev_s89_state', 'waiting_for_natural')
    st.session_state.last_result_after_natural = undone_game.get('prev_s89_last_res', None)
    st.session_state.last_natural_pos = undone_game.get('prev_s89_last_nat_pos', None)

    # --- Restore Betting State? ---
    # IMPORTANT: Undoing a game result DOES NOT automatically undo the corresponding bet win/loss.
    # This can lead to inconsistencies between game state and betting state.
    # A full bet history undo is complex. For now, we just recalculate the suggested bet based on the *current* (post-undo) state.
    st.warning("Lưu ý: Hoàn tác kết quả KHÔNG hoàn tác Thắng/Thua cược. Số dư & trạng thái cược có thể không khớp.")

    # --- Final Updates ---
    update_all_predictions() # Update predictions based on restored state
    calculate_suggested_bet() # Recalculate bet based on *current* balance and mode
    # st.rerun() # Usually not needed


def reset_game():
    """Resets MOST game state, attempting to keep Balance and Unit settings."""
    keys_to_reset = [
        'initialized', 'game_history', 'game_count', 'bead_road_grid', 'natural_marks_grid',
        'current_bead_road_row', 'current_bead_road_col', 'special89_state',
        'last_natural_pos', 'last_result_after_natural', 'predictions',
        'progression_sequence', 'current_progression_index', #'initial_bet_unit', # Keep user setting?
        #'current_balance', # Keep user setting?
        'suggested_bet_amount',
        #'starting_balance_input', 'input_initial_bet_unit', # Keep UI values?
        'session_start_balance', 'last_withdrawal_profit_level',
        'session_start_time', 'session_wins', 'session_losses',
        'bet_history',
        'betting_mode', 'current_recovery_multiplier', 'recovery_bets_at_this_level',
        # <<< MODIFIED: Add new key to reset >>>
        'current_profit_sequence_profit'
        # <<< END MODIFICATION >>>
    ]
    # Decide which fundamental settings to keep (like unit, balance) or reset fully
    # Let's reset almost everything except the current balance and unit settings if they exist
    try: keys_to_reset.remove('current_balance')
    except ValueError: pass
    try: keys_to_reset.remove('initial_bet_unit')
    except ValueError: pass
    # Keep UI input states too? Maybe not, let them re-sync with state.
    # try: keys_to_reset.remove('starting_balance_input')
    # except ValueError: pass
    # try: keys_to_reset.remove('input_initial_bet_unit')
    # except ValueError: pass


    for key in keys_to_reset:
        if key in st.session_state:
            del st.session_state[key]

    # Delete the initialized flag to trigger re-initialization block in main()
    if 'initialized' in st.session_state:
        del st.session_state['initialized']

    st.toast("Game đã được reset (giữ lại Số dư & Unit). Đang khởi tạo lại...", icon="🔄")
    # No explicit rerun needed, clearing 'initialized' will cause main() to re-init


# --- Main Application ---
def main():
    st.set_page_config(
        page_title="Baccarat Pro Predictor",
        layout="wide",
        initial_sidebar_state="collapsed" # Start with sidebar collapsed
    )
    # Include Font Awesome CSS
    st.markdown('<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.2/css/all.min.css">', unsafe_allow_html=True)

    default_cols = 18 # Default number of columns for the bead road

    # --- Initialization Block ---
    # This runs only once when the app starts or after a full reset_game()
    if 'initialized' not in st.session_state:
        st.session_state.initialized = True
        st.session_state.rows = 6
        st.session_state.cols = default_cols
        initialize_progression_state() # Sets up betting sequence, modes, balance, unit etc.
        st.session_state.game_history = []
        st.session_state.game_count = 0
        # Create the initial empty grids
        st.session_state.bead_road_grid = create_default_grid(st.session_state.rows, st.session_state.cols, None)
        st.session_state.natural_marks_grid = create_default_grid(st.session_state.rows, st.session_state.cols, False)
        # Set initial cursor position
        st.session_state.current_bead_road_row = 0
        st.session_state.current_bead_road_col = 0
        # Initialize prediction-related states
        st.session_state.special89_state = "waiting_for_natural"
        st.session_state.last_natural_pos = None
        st.session_state.last_result_after_natural = None
        update_all_predictions() # Calculate initial predictions (will be 'Waiting...')
        st.toast("Khởi tạo ứng dụng thành công!", icon="🚀")

    # --- State Validation (Optional but good practice) ---
    # Ensure essential states exist if the app was somehow re-run without full init
    required_betting_keys = [
        'progression_sequence', 'current_profit_sequence_profit', 'betting_mode',
        'current_recovery_multiplier', 'recovery_bets_at_this_level',
        'session_start_balance', 'suggested_bet_amount', 'initial_bet_unit', 'current_balance'
    ]
    # Simple check and potential re-init if core state is missing
    # A more robust check might compare types or values
    if not all(key in st.session_state for key in required_betting_keys):
         st.warning("Phát hiện thiếu trạng thái, thử khởi tạo lại...")
         initialize_progression_state() # Attempt re-initialization
         update_all_predictions()

    # --- CSS Styling ---
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
            --bead-natural-marker-offset: -4px; --sticky-top-offset: 15px; /* Adjust sticky offset */
        }}
        body {{ font-family: var(--font-body); color: var(--primary-text); background-color: var(--primary-bg); }}
        .main {{ background-color: var(--primary-bg); padding: 15px; border-radius: var(--border-radius); font-family: var(--font-body); }}
        .stApp > header {{ display: none; }} /* Hide Streamlit header */
        .main .block-container {{ padding: 5px 10px !important; margin: 0 !important; max-width: 100% !important; }}

        /* Align columns to top */
        div[data-testid="stHorizontalBlock"] > div {{ align-self: flex-start !important; }}

        /* General Button Styling */
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

        /* Specific Button Colors */
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
        /* Natural Star Icon */
        div.stButton[data-testid*="natural_btn"] button span[data-testid="stButtonIcon"] {{
             color: var(--accent-gold) !important; font-size: 1.1em !important; margin-left: 5px;
             line-height: 1; filter: drop-shadow(0 0 1px black); vertical-align: middle;
        }}
        div.stButton[data-testid*="undo_std"] button,
        div.stButton[data-testid*="prog_loss_std"] button {{ background: linear-gradient(145deg, #6c757d, #5a6268); border-color: #5a6268; }}
        div.stButton[data-testid*="reset_std"] button {{ background: linear-gradient(145deg, #f57f17, #e65100); border-color: #e65100; color: #fff; }}
        div.stButton[data-testid*="reset_session_std"] button {{ background: linear-gradient(145deg, #ffca28, #ffb300); border-color: #ffb300; color: #111; }}
        /* Win Buttons (Match Player/Banker Colors) */
        div.stButton[data-testid*="prog_win_p_std"] button {{ background: linear-gradient(145deg, var(--player-blue), var(--player-blue-darker)); border-color: var(--player-blue-darker); }}
        div.stButton[data-testid*="prog_win_b_std"] button {{ background: linear-gradient(145deg, var(--banker-red), var(--banker-red-darker)); border-color: var(--banker-red-darker); }}
        /* Set Balance Button */
        div.stButton[data-testid*="set_starting_balance_button"] button {{ background: linear-gradient(145deg, var(--accent-gold), var(--accent-gold-darker)); border-color: var(--accent-gold-darker); color: #111; }}

        /* App Title */
        .app-title {{ font-family: var(--font-header); color: var(--accent-gold); font-size: 26px; font-weight: 700; text-align: center; margin-bottom: 15px; text-shadow: 1px 1px 2px rgba(0,0,0,0.4); }}

        /* Card Styling */
        .card {{ background-color: var(--secondary-bg); border-radius: var(--border-radius); padding: 18px; margin-bottom: 18px; box-shadow: var(--box-shadow); border: 1px solid var(--border-color); }}

        /* Headings */
        h4 {{ font-family: var(--font-body); font-weight: 700; color: var(--accent-gold); margin-top: 0; margin-bottom: 10px; border-bottom: 1px solid var(--border-color); padding-bottom: 6px; font-size: 14px; text-transform: uppercase; letter-spacing: 0.5px; display: flex; align-items: center; }}
        h4 i {{ margin-right: 8px; font-size: 1em; color: var(--accent-gold-darker); }} /* Icon in h4 */
        h6 {{ font-family: var(--font-body); font-weight: bold; color: var(--secondary-text); margin-top: 5px; margin-bottom: 5px; font-size: 13px; text-transform: uppercase;}}

        /* Text */
        p, .stMarkdown p {{ color: var(--secondary-text); font-size: 13px; line-height: 1.5; margin-bottom: 8px; }}

        /* Input Fields (Number and Text) */
        .stNumberInput, .stTextInput {{ display: flex; flex-direction: column; margin-bottom: 8px; }}
        /* Target labels specifically within these inputs */
        .stNumberInput label, .stTextInput label {{
            font-size: 13px !important; color: var(--secondary-text) !important;
            margin-bottom: 3px !important; display: block; font-weight: bold; order: 1; /* Label on top */
        }}
         /* Collapse specific labels */
        .stNumberInput label[data-testid="stWidgetLabel"],
        .stTextInput label[data-testid="stWidgetLabel"] {{
             margin-bottom: 0 !important; /* Reduce space */
        }}
        .stNumberInput input, .stTextInput input {{
            font-size: 13px; color: var(--primary-text); background-color: var(--tertiary-bg);
            border: 1px solid var(--border-color); border-radius: 4px; padding: 6px 8px;
            width: 100%; box-shadow: var(--box-shadow-inset); box-sizing: border-box; order: 2; /* Input below label */
            height: 31px;
        }}
        .stNumberInput input:focus, .stTextInput input:focus {{
             border-color: var(--accent-gold); box-shadow: 0 0 4px rgba(212, 175, 55, 0.4), var(--box-shadow-inset); outline: none;
        }}
        ::placeholder {{ color: var(--secondary-text); opacity: 0.7; }}

        /* Input row adjustments */
        /* Target the horizontal block containing the balance input and button */
        div[data-testid="stHorizontalBlock"]:has(div .stNumberInput[key*="starting_balance_input"]) > div:nth-child(1) .stNumberInput {{ margin-bottom: 0 !important; }} /* Remove bottom margin on input */
        div[data-testid="stHorizontalBlock"]:has(div .stNumberInput[key*="starting_balance_input"]) > div:nth-child(2) {{ display: flex; align-items: flex-end; }} /* Align button container to bottom */
        div[data-testid="stHorizontalBlock"]:has(div .stNumberInput[key*="starting_balance_input"]) > div:nth-child(2) .stButton {{ width: 100%; }} /* Button full width */
        div[data-testid="stHorizontalBlock"]:has(div .stNumberInput[key*="starting_balance_input"]) > div:nth-child(2) .stButton button {{ margin-bottom: 0 !important; }} /* Remove bottom margin on button */


        /* Metric Display */
        .stMetric {{ text-align: center; background-color: var(--secondary-bg); padding: 10px; border-radius: 4px; margin-top: 10px; margin-bottom: 8px; border: 1px solid var(--border-color); box-sizing: border-box; display: flex; flex-direction: column; justify-content: center; height: 70px; }}
        .stMetric label {{ color: var(--secondary-text) !important; font-size: 11px !important; font-weight: 400; text-transform: uppercase; margin-bottom: 2px !important; line-height: 1.2; }}
        .stMetric p {{ font-size: 20px !important; color: var(--primary-text) !important; font-weight: 700; margin-top: 2px; line-height: 1.1; word-wrap: break-word; }}
        .stMetric .stMetricDelta {{ display: none; }} /* Hide default delta indicator */
        /* Adjust top margin for the second metric */
        div[data-testid="stMetric"]:has(label:contains("Lợi Nhuận Phiên")) {{ margin-top: 0px; }}


        /* Prediction Display */
        .prediction-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 5px 10px; margin-bottom: 5px;}}
        .prediction-box {{ background-color: var(--tertiary-bg); border-radius: 4px; padding: 6px 10px; margin: 0; font-size: 12px; color: var(--primary-text); border-left: 3px solid var(--accent-gold); display: flex; justify-content: space-between; align-items: center; min-height: 30px; }}
        .prediction-box b {{ color: var(--secondary-text); font-weight: normal; margin-right: 5px; flex-shrink: 0; }} /* Prediction label */
        .prediction-box span {{ text-align: right; font-weight: bold; }} /* Prediction value */
        /* Final Prediction Box */
        .final-prediction {{ color: var(--accent-gold); font-size: 14px; font-weight: bold; text-align: center; margin-top: 10px; padding: 8px; background-color: var(--secondary-bg); border-radius: 4px; border: 1px solid var(--accent-gold-darker); box-shadow: 0 0 6px rgba(212, 175, 55, 0.2); }}
        .final-prediction b {{ font-weight: 700; }} /* Make inner bold tags work */
        /* Prediction Result Colors */
        .prediction-result-Banker {{ color: var(--banker-red); }}
        .prediction-result-Player {{ color: var(--player-blue); }}
        /* Apply colors within the final prediction as well */
        .final-prediction .prediction-result-Banker {{ color: var(--banker-red); }}
        .final-prediction .prediction-result-Player {{ color: var(--player-blue); }}

        /* Sticky Containers for Columns */
        .bead-road-sticky-container,
        .mid-col-sticky-container {{
            position: -webkit-sticky; /* Safari */
            position: sticky;
            top: var(--sticky-top-offset, 15px);
            z-index: 10;
        }}
        .mid-col-sticky-container {{ z-index: 9; }} /* Ensure bead road is on top if overlap occurs */

        /* Bead Road Display */
        .bead-road-card-container {{ display: flex; justify-content: center; }} /* Center the card */
        .bead-road-card {{ width: fit-content; }} /* Allow card to size to content */
        .bead-road-container {{
            line-height: 0; /* Collapse vertical space */
            text-align: center; background-color: var(--tertiary-bg);
            padding: 15px; border-radius: var(--border-radius); border: 1px solid var(--border-color);
            margin-top: 8px; display: inline-block; /* Fit content width */
            overflow-x: auto; /* Enable horizontal scrolling */
            max-width: 100%; /* Prevent overflow of parent */
            white-space: nowrap; /* Keep beads in single line rows */
        }}
        .bead-row {{ margin-bottom: var(--bead-margin); height: var(--bead-size); white-space: nowrap; }} /* Row contains beads */
        .bead-cell-banker, .bead-cell-player, .bead-cell-current, .bead-cell-empty {{
            border-radius: 50%; text-align: center;
            width: var(--bead-size) !important; height: var(--bead-size) !important;
            line-height: var(--bead-size) !important; /* Vertically center text */
            font-size: var(--bead-font-size) !important; font-weight: bold;
            display: inline-block; /* Place beads side-by-side */
            margin: 0 var(--bead-margin); border: 1px solid transparent;
            vertical-align: middle; /* Align beads vertically */
            box-shadow: var(--box-shadow-inset); position: relative; color: #fff;
        }}
        .bead-cell-banker {{ background-color: var(--banker-red); border-color: var(--banker-red-darker); }}
        .bead-cell-player {{ background-color: var(--player-blue); border-color: var(--player-blue-darker); }}
        /* Current Target Cell (dashed outline) */
        .bead-cell-current {{
             background-color: transparent; border: 2px dashed var(--accent-gold);
             box-shadow: 0 0 8px rgba(212, 175, 55, 0.4);
             /* Adjust line height slightly for dashed border */
             line-height: calc(var(--bead-size) - 4px) !important;
             width: calc(var(--bead-size) - 0px) !important; /* Keep size consistent */
             height: calc(var(--bead-size) - 0px) !important;
        }}
        .bead-cell-empty {{ background-color: var(--secondary-bg); border-color: var(--border-color); box-shadow: none; }}
        /* Natural Marker ('N') */
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

        /* Betting Progression Info Box */
        .progression-info {{ margin-top: 10px; margin-bottom: 10px; text-align: center; background-color: var(--tertiary-bg); padding: 8px; border-radius: 4px; border: 1px solid var(--border-color); }}
        .progression-mode {{ font-size: 11px; color: var(--accent-gold); font-weight: bold; margin-bottom: 4px; text-transform: uppercase; }}
        .progression-step {{ font-size: 12px; color: var(--secondary-text); margin-bottom: 4px; text-transform: uppercase; }}
        .suggested-bet {{ font-size: 16px; color: var(--accent-gold); font-weight: bold; margin-bottom: 0; display: flex; align-items: center; justify-content: center; }}
        .suggested-bet i {{ margin-right: 6px; font-size: 0.9em; }} /* Icon in suggested bet */
        .progression-buttons {{ margin-top: 5px; margin-bottom: 10px; }} /* Container for Win/Loss buttons */

        /* Specific button margins (using :has for precision) */
        div[data-testid="stVerticalBlock"] > div[data-testid="stButton"]:has(button span:contains("Reset Phiên Cược")) {{ margin-top: 15px !important; margin-bottom: 5px !important; }}
        div[data-testid="stVerticalBlock"] > div[data-testid="stButton"]:has(button span:contains("Reset Toàn Bộ Game")) {{ margin-top: 0px !important; }}

        /* Dividers and Alerts */
        hr {{ border-top: 1px solid var(--border-color); margin: 10px 0;}}
        .stDivider {{ margin: 10px 0;}} /* Streamlit divider */
        .stAlert {{ border-radius: 4px; font-size: 12px; background-color: var(--tertiary-bg); border: 1px solid var(--accent-gold-darker); padding: 8px 12px; margin-bottom: 10px;}}
        .stAlert p, .stAlert div, .stAlert li {{ font-size: 12px !important; color: var(--primary-text); }}
        .stToast {{ font-size: 13px; }} /* Toast notifications */
        /* Info box styling (for the Unit reminder) */
        div[data-testid="stInfo"] {{
            background-color: rgba(13, 110, 253, 0.1); /* Light blue background */
            border: 1px solid var(--player-blue);
            border-left-width: 5px;
            padding: 10px; margin-bottom: 10px; font-size: 12px;
            border-radius: 4px;
        }}
         div[data-testid="stInfo"] p {{
             color: var(--primary-text);
             font-size: 12px !important;
             margin-bottom: 0;
         }}


        /* Session Statistics Box */
        .session-stats-container {{ background-color: var(--secondary-bg); border-radius: var(--border-radius); padding: 15px; margin-bottom: 18px; border: 1px solid var(--border-color); box-shadow: var(--box-shadow); }}
        .session-stats-container h4 {{ margin-bottom: 8px; }}
        .session-stats-grid {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; text-align: center; }}
        .session-stat-item {{ background-color: var(--tertiary-bg); padding: 8px; border-radius: 4px; border: 1px solid var(--border-color); }}
        .session-stat-item .label {{ font-size: 11px; color: var(--secondary-text); text-transform: uppercase; margin-bottom: 3px; display: block;}}
        .session-stat-item .value {{ font-size: 16px; color: var(--primary-text); font-weight: bold; display: block; }}
        .session-stat-item .value .positive {{ color: var(--win-green); }} /* Green for wins */
        .session-stat-item .value .negative {{ color: var(--loss-red); }} /* Red for losses */
        .session-stat-item i {{ margin-right: 5px; }} /* Icon spacing */

        /* Bet History Box */
        .bet-history-container {{
            max-height: 150px; /* Limit height and enable scroll */
            overflow-y: auto; background-color: var(--tertiary-bg);
            padding: 10px; border-radius: 4px; border: 1px solid var(--border-color);
            margin-top: 10px;
        }}
        .bet-history-item {{ font-size: 12px; padding: 3px 5px; margin-bottom: 4px; border-radius: 3px; display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid var(--secondary-bg); }}
        .bet-history-item:last-child {{ margin-bottom: 0; border-bottom: none; }}
        .bet-history-item span {{ vertical-align: middle; }}
        .bet-history-outcome {{ font-weight: bold; margin-right: 8px; }}
        .bet-history-outcome.win {{ color: var(--win-green); }}
        .bet-history-outcome.loss {{ color: var(--loss-red); }}
        .bet-history-amount {{ color: var(--secondary-text); margin-right: 8px; font-size: 11px;}} /* Smaller font for amount */
        .bet-history-profit {{ font-weight: bold; }}
        .bet-history-profit.win {{ color: var(--win-green); }}
        .bet-history-profit.loss {{ color: var(--loss-red); }}
        /* Scrollbar Styling for Bet History */
        .bet-history-container::-webkit-scrollbar {{ width: 6px; }}
        .bet-history-container::-webkit-scrollbar-track {{ background: var(--secondary-bg); border-radius: 3px;}}
        .bet-history-container::-webkit-scrollbar-thumb {{ background: var(--border-color); border-radius: 3px;}}
        .bet-history-container::-webkit-scrollbar-thumb:hover {{ background: var(--accent-gold); }}
        </style>
        """, unsafe_allow_html=True) # End of CSS block

    # --- Main App Layout ---
    st.markdown('<div class="main">', unsafe_allow_html=True)
    st.markdown('<div class="app-title"><i class="fas fa-crown"></i> Baccarat Pro Predictor <i class="fas fa-crown"></i></div>', unsafe_allow_html=True)

    # Define columns for layout
    left_col, mid_col, right_col = st.columns([1.3, 1.7, 2.0]) # Adjust ratios as needed

    # --- LEFT COLUMN: Betting Strategy & History ---
    with left_col:
        with st.container(border=False): # Use border=False for cleaner look if desired
            st.markdown('<div class="card"><h4><i class="fas fa-chart-line"></i> Chiến Lược Cược</h4>', unsafe_allow_html=True)

            # Reminder for setting the base bet unit
            # <<< MODIFIED: Added st.info reminder >>>
            st.info("Lưu ý: Đặt 'Unit' thành 40,000 để chuỗi 1-2 tương ứng 40k-80k và mục tiêu 120k hoạt động đúng.", icon="💡")

            # Read current values from state for display/input defaults
            balance = float(st.session_state.get('current_balance', 0.0))
            initial_unit = float(st.session_state.get('initial_bet_unit', 40000.0)) # Suggest 40k default visually
            mode = st.session_state.get('betting_mode', 'PROFIT')

            st.markdown("<h6>Cài Đặt</h6>", unsafe_allow_html=True)
            # Balance Input Row
            col_start_bal_inp, col_start_bal_btn = st.columns([0.7, 0.3])
            with col_start_bal_inp:
                # Use value from state, format nicely, use callback on button press
                st.number_input("Số Dư", min_value=0.0, step=1000.0, key="starting_balance_input", value=balance, format="%.0f", label_visibility="collapsed", placeholder="Nhập số dư...")
            with col_start_bal_btn:
                st.button("Đặt Số Dư", key="set_starting_balance_button", on_click=set_current_balance_from_input, use_container_width=True)

            # Unit Input (on_change callback updates state and recalculates bet)
            st.number_input("Unit", min_value=1.0, step=1000.0, key="input_initial_bet_unit", value=initial_unit, on_change=update_initial_bet_unit, format="%.0f", help="Mức cược cơ bản (unit). Ví dụ: 40000.", label_visibility="collapsed", placeholder="Nhập unit cược (vd: 40000)...")
            st.divider() # Visual separator

            # Display Current Balance and Session Profit Metrics
            col_metric1, col_metric2 = st.columns(2)
            with col_metric1:
                st.metric(label="Số Dư Hiện Tại", value=f"{st.session_state.get('current_balance', 0.0):,.0f} đ")
            with col_metric2:
                start_bal = float(st.session_state.get('session_start_balance', st.session_state.get('current_balance', 0.0)))
                current_bal = float(st.session_state.get('current_balance', 0.0))
                session_profit = current_bal - start_bal
                profit_color_class = "positive" if session_profit > 0 else ("negative" if session_profit < 0 else "")
                # Use markdown for potential coloring
                st.metric(label="Lợi Nhuận Phiên", value=f"{session_profit:,.0f} đ") # Basic metric display

            # --- Display Progression/Recovery Status ---
            prog_idx = st.session_state.get('current_progression_index', 0)
            prog_seq = st.session_state.get('progression_sequence', [1, 2]) # Use new [1, 2] default
            suggested_bet = float(st.session_state.get('suggested_bet_amount', 0.0))
            current_balance_for_check = float(st.session_state.get('current_balance', 0.0))
            balance_ok = current_balance_for_check >= suggested_bet # Check if balance covers suggested bet

            st.markdown('<div class="progression-info">', unsafe_allow_html=True)
            if mode == 'PROFIT':
                # Validate index just in case
                if not (0 <= prog_idx < len(prog_seq)): prog_idx = 0
                step_num = prog_idx + 1; total_steps = len(prog_seq)
                current_unit_multiplier = prog_seq[prog_idx]
                # <<< MODIFIED: Update display text for sequence >>>
                st.markdown(f'<div class="progression-mode">Mode: Lợi Nhuận (1-2)</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="progression-step">Bước: {step_num}/{total_steps} (x{current_unit_multiplier})</div>', unsafe_allow_html=True)
            else: # mode == 'RECOVERY'
                multiplier = st.session_state.get('current_recovery_multiplier', 1)
                bets_at_level = st.session_state.get('recovery_bets_at_this_level', 0)
                # Text indicates which bet this *will be* at the current level
                bet_order_text = f"(Lần {bets_at_level + 1})"
                st.markdown(f'<div class="progression-mode">Mode: Gỡ Lỗ (Delay Martingale)</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="progression-step">Mức cược: x{multiplier} {bet_order_text}</div>', unsafe_allow_html=True)

            # Display the calculated suggested bet
            st.markdown(f'<div class="suggested-bet"><i class="fas fa-coins"></i> Cược: {suggested_bet:,.0f} đ</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True) # Close progression-info

            # Warn if balance is insufficient for the suggested bet
            if not balance_ok:
                st.warning(f"Số dư không đủ ({current_balance_for_check:,.0f} đ) cho cược {suggested_bet:,.0f} đ!", icon="⚠️")

            # --- Win/Loss Buttons ---
            st.markdown('<div class="progression-buttons">', unsafe_allow_html=True)
            p_col1, p_col2, p_col3 = st.columns(3)
            # Payout ratios: Player=1.0, Banker=0.95 (standard commission)
            with p_col1: st.button("Thắng (P)", key="prog_win_p_std", on_click=handle_progression_win, args=(1.0,), disabled=not balance_ok, use_container_width=True)
            with p_col2: st.button("Thắng (B)", key="prog_win_b_std", on_click=handle_progression_win, args=(0.95,), disabled=not balance_ok, use_container_width=True)
            with p_col3: st.button("Thua", key="prog_loss_std", on_click=handle_progression_loss, disabled=not balance_ok, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True) # Close progression-buttons

            # --- Bet History Display ---
            st.markdown("<h6>Lịch Sử Cược Phiên</h6>", unsafe_allow_html=True)
            bet_history = st.session_state.get('bet_history', [])
            st.markdown('<div class="bet-history-container">', unsafe_allow_html=True)
            if not bet_history:
                st.markdown('<p style="font-size: 12px; color: var(--secondary-text); text-align: center;">Chưa có lịch sử cược.</p>', unsafe_allow_html=True)
            else:
                # Display latest bets first (e.g., last 15)
                for bet in reversed(bet_history[-15:]):
                    outcome_class = "win" if bet['outcome'] == "Win" else "loss"
                    profit_sign = "+" if bet['profit'] > 0 else ""
                    # Display mode [P]rofit or [R]ecovery for context
                    mode_display = f"[{bet.get('mode','?')[:1]}]"
                    history_item_html = f"""
                    <div class="bet-history-item">
                        <span>
                            <span class="bet-history-outcome {outcome_class}">{mode_display} {bet['outcome']}</span>
                            <span class="bet-history-amount">({bet['amount']:,.0f})</span>
                        </span>
                        <span class="bet-history-profit {outcome_class}">{profit_sign}{bet['profit']:,.0f}</span>
                    </div>
                    """
                    st.markdown(history_item_html, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True) # Close bet-history-container

            # Session and Game Reset Buttons
            st.button("Reset Phiên Cược", key="reset_session_std", on_click=reset_session, use_container_width=True, help="Bắt đầu phiên theo dõi mới với số dư hiện tại, reset lịch sử cược.")
            st.button("Reset Toàn Bộ Game", key="reset_std", on_click=reset_game, use_container_width=True, help="Reset mọi thứ (trừ Số dư & Unit đã nhập), bao gồm cả Bead Road và lịch sử.")
            st.markdown('</div>', unsafe_allow_html=True) # Close card

    # --- MIDDLE COLUMN: Input & Predictions (Sticky) ---
    with mid_col:
        st.markdown('<div class="mid-col-sticky-container">', unsafe_allow_html=True) # Start sticky container

        # Input Card
        with st.container(border=False):
             st.markdown('<div class="card"><h4><i class="fas fa-keyboard"></i> Nhập Kết Quả</h4>', unsafe_allow_html=True)
             col_in1, col_in2 = st.columns(2)
             with col_in1:
                 # Player buttons (Standard and Natural)
                 st.button('Player', key="player_std_btn", on_click=add_result, args=('Player', False), use_container_width=True, help="Player Win (Standard)")
                 st.button('Player', icon="⭐", key="player_natural_btn", on_click=add_result, args=('Player', True), use_container_width=True, help="Player Win (Natural 8 or 9)")
             with col_in2:
                 # Banker buttons (Standard and Natural)
                 st.button('Banker', key="banker_std_btn", on_click=add_result, args=('Banker', False), use_container_width=True, help="Banker Win (Standard)")
                 st.button('Banker', icon="⭐", key="banker_natural_btn", on_click=add_result, args=('Banker', True), use_container_width=True, help="Banker Win (Natural 8 or 9)")
             # Undo Button (disabled if no history)
             st.button("Undo Last Result", key="undo_std", on_click=undo_last_result, disabled=not st.session_state.get('game_history', []), use_container_width=True)
             st.markdown('</div>', unsafe_allow_html=True) # Close card

        # Prediction Card
        with st.container(border=False):
            st.markdown('<div class="card"><h4><i class="fas fa-brain"></i> Dự Đoán</h4>', unsafe_allow_html=True)
            pred = st.session_state.get('predictions', {}) # Get predictions dictionary

            # Helper function to format prediction display with colors and short names
            def format_prediction(label, value):
                 result_class = ""; display_value = str(value) # Default to string
                 # Assign color class based on Banker/Player prediction
                 if value == 'Banker': result_class = "prediction-result-Banker"
                 elif value == 'Player': result_class = "prediction-result-Player"

                 # Shorten common 'Waiting...' or 'Error...' messages for brevity
                 if isinstance(value, str):
                     if value.startswith("Waiting (Src Cell"): display_value = "Wait (2&5 Src)"
                     elif value.startswith("Waiting (Need Prev Cells)"): display_value = "Wait (NoMirror Src)"
                     elif value.startswith("Waiting for Natural"): display_value = "Wait Natural"
                     elif value.startswith("Waiting for next result"): display_value = "Wait Next"
                     elif value.startswith("Waiting for pattern"): display_value = "Wait Pattern"
                     elif value.startswith("Waiting (Need Col 3+)"): display_value = "Wait Col 3+"
                     elif value.startswith("Waiting (Not X Pos)"): display_value = "Wait X Pos"
                     elif value.startswith("Waiting (Need Matrix TL)"): display_value = "Wait Matrix TL"
                     elif value.startswith("Waiting (4+ identical)"): display_value = "Wait (4+)"
                     elif value.startswith("Waiting (3 identical)"): display_value = "Wait (3)"
                     elif value.startswith("Waiting (Matrix Incomplete)"): display_value = "Wait Matrix Size"
                     elif value.startswith("Waiting (Matrix"): display_value = "Wait Matrix OOB" # Catches OOB/Calc OOB etc.
                     elif value.startswith("Error: No S89 result"): display_value = "Error S89"
                     elif value.startswith("Error:"): display_value = value.split(':')[0] # Show only Error type like "Error"
                     elif value.startswith("No prediction"): display_value = "No Predict" # Shorten this too
                 # Handle None or other types if necessary, though most should be strings
                 elif value is None: display_value = "N/A"

                 # Return formatted HTML for the prediction box
                 return f'<div class="prediction-box"><b>{label}:</b> <span class="{result_class}">{display_value}</span></div>'

            # Display individual predictions in a grid
            st.markdown('<div class="prediction-grid">', unsafe_allow_html=True)
            st.markdown(format_prediction("Majority 6", pred.get("majority6", "...")), unsafe_allow_html=True)
            st.markdown(format_prediction("X Mark", pred.get("xMark", "...")), unsafe_allow_html=True)
            st.markdown(format_prediction("No Mirror", pred.get("noMirror", "...")), unsafe_allow_html=True)
            st.markdown(format_prediction("89 Special", pred.get("special89", "...")), unsafe_allow_html=True)
            st.markdown(format_prediction("2&5", pred.get("2and5", "...")), unsafe_allow_html=True)
            st.markdown('<div></div>', unsafe_allow_html=True) # Placeholder for grid alignment if needed
            st.markdown('</div>', unsafe_allow_html=True) # Close prediction-grid

            # Display Overall Percentage
            st.markdown(f'<div class="prediction-box overall" style="margin-top: 5px;"><b>Overall:</b> <span>{pred.get("percentage", "...")}</span></div>', unsafe_allow_html=True)

            # Display Final Prediction (with styled Banker/Player)
            final_pred_raw = pred.get("final", "No prediction")
            final_pred_html = final_pred_raw # Start with raw text
            # Replace Banker/Player with styled spans if present
            if "<b>Banker</b>" in final_pred_raw:
                final_pred_html = final_pred_raw.replace("<b>Banker</b>", '<b class="prediction-result-Banker">Banker</b>')
            elif "<b>Player</b>" in final_pred_raw:
                final_pred_html = final_pred_raw.replace("<b>Player</b>", '<b class="prediction-result-Player">Player</b>')
            st.markdown(f'<div class="final-prediction">{final_pred_html}</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True) # Close card

        st.markdown('</div>', unsafe_allow_html=True) # Close mid-col-sticky-container

    # --- RIGHT COLUMN: Bead Road & Session Stats (Sticky) ---
    with right_col:
        st.markdown('<div class="bead-road-sticky-container">', unsafe_allow_html=True) # Start sticky container
        st.markdown('<div class="bead-road-card-container">', unsafe_allow_html=True) # Centering container
        with st.container(border=False):
            st.markdown('<div class="card bead-road-card"><h4><i class="fas fa-border-all"></i> Bead Road</h4>', unsafe_allow_html=True)
            # Bead Road HTML Generation
            bead_html = '<div class="bead-road-container">' # Container enables scrolling
            rows = st.session_state.get('rows', 6)
            cols = st.session_state.get('cols', 18)
            grid = st.session_state.get('bead_road_grid', create_default_grid(rows, cols, None))
            nat_grid = st.session_state.get('natural_marks_grid', create_default_grid(rows, cols, False))
            current_row = st.session_state.get('current_bead_road_row', 0)
            current_col = st.session_state.get('current_bead_road_col', 0)
            # Check if the *next* target position is valid within the grid dimensions
            next_pos_valid = (0 <= current_row < rows) and (0 <= current_col < cols)

            # --- Grid Validation before rendering (important!) ---
            grid_valid = isinstance(grid, list) and len(grid) == rows and \
                          (rows == 0 or (len(grid) > 0 and isinstance(grid[0], list) and len(grid[0]) == cols))
            nat_grid_valid = isinstance(nat_grid, list) and len(nat_grid) == rows and \
                               (rows == 0 or (len(nat_grid) > 0 and isinstance(nat_grid[0], list) and len(nat_grid[0]) == cols))

            if not grid_valid or not nat_grid_valid:
                 # Display an error message within the container if grid is invalid
                 bead_html += "<p style='color: red; text-align: center; font-size: 12px; white-space: normal;'>Lỗi: Dữ liệu Bead Road không hợp lệ hoặc sai kích thước. Thử 'Reset Toàn Bộ Game'.</p>"
                 # Attempt to reset grid state as a fallback (might need full reset)
                 # st.session_state.bead_road_grid = create_default_grid(rows, cols, None)
                 # st.session_state.natural_marks_grid = create_default_grid(rows, cols, False)
                 # st.session_state.current_bead_road_row = 0
                 # st.session_state.current_bead_road_col = 0
            else:
                 # --- Render Bead Road HTML if grids are valid ---
                 for i in range(rows):
                     bead_html += '<div class="bead-row">' # Start a row
                     for j in range(cols):
                         # Safely get cell value and natural status
                         cell = grid[i][j] if i < len(grid) and j < len(grid[i]) else None
                         is_natural = nat_grid[i][j] if i < len(nat_grid) and j < len(nat_grid[i]) else False

                         # Determine if this cell is the next target position
                         is_current_target = next_pos_valid and (i == current_row) and (j == current_col)

                         # Determine CSS class and content for the cell
                         cell_class = "bead-cell-empty"; cell_content = "" # Default: empty
                         if is_current_target:
                             cell_class = "bead-cell-current" # Dashed outline for target
                         elif cell == 'Banker':
                             cell_class = 'bead-cell-banker'; cell_content = 'B'
                             if is_natural: cell_class += ' natural' # Add 'N' marker if natural
                         elif cell == 'Player':
                             cell_class = 'bead-cell-player'; cell_content = 'P'
                             if is_natural: cell_class += ' natural' # Add 'N' marker if natural

                         # Append the HTML for this cell to the row
                         bead_html += f'<div class="{cell_class}">{cell_content}</div>'
                     bead_html += '</div>' # End bead-row

            bead_html += '</div>' # End bead-road-container
            st.markdown(bead_html, unsafe_allow_html=True) # Render the generated HTML
            st.markdown('</div>', unsafe_allow_html=True) # Close bead-road-card
        st.markdown('</div>', unsafe_allow_html=True) # Close bead-road-card-container
        st.markdown('</div>', unsafe_allow_html=True) # Close bead-road-sticky-container

        # --- Session Statistics Card ---
        # This part is outside the sticky container for the bead road
        with st.container(border=False):
            st.markdown('<div class="session-stats-container card">', unsafe_allow_html=True)
            st.markdown('<h4><i class="fas fa-chart-pie"></i> Thống Kê Phiên</h4>', unsafe_allow_html=True)
            # Get session stats from state
            session_wins = st.session_state.get('session_wins', 0)
            session_losses = st.session_state.get('session_losses', 0)
            session_start_time = st.session_state.get('session_start_time', datetime.datetime.now())

            # Calculate elapsed time
            formatted_time = "N/A"
            if isinstance(session_start_time, datetime.datetime):
                elapsed_time = datetime.datetime.now() - session_start_time
                total_seconds = int(elapsed_time.total_seconds())
                hours, remainder = divmod(total_seconds, 3600)
                minutes, seconds = divmod(remainder, 60)
                formatted_time = f"{hours:02}:{minutes:02}:{seconds:02}"

            # Display stats in a grid
            st.markdown('<div class="session-stats-grid">', unsafe_allow_html=True)
            # Time Elapsed
            st.markdown(f"""
                <div class="session-stat-item">
                    <span class="label"><i class="far fa-clock"></i> Thời Gian</span>
                    <span class="value">{formatted_time}</span>
                </div>""", unsafe_allow_html=True)
            # Bets Won
            st.markdown(f"""
                <div class="session-stat-item">
                    <span class="label"><i class="fas fa-trophy"></i> Thắng (Cược)</span>
                    <span class="value positive">{session_wins}</span>
                </div>""", unsafe_allow_html=True)
            # Bets Lost
            st.markdown(f"""
                <div class="session-stat-item">
                    <span class="label"><i class="fas fa-heart-broken"></i> Thua (Cược)</span>
                    <span class="value negative">{session_losses}</span>
                </div>""", unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True) # Close session-stats-grid
            st.markdown('</div>', unsafe_allow_html=True) # Close session-stats-container card

    st.markdown('</div>', unsafe_allow_html=True) # Close main div

# --- Run the App Entry Point ---
if __name__ == '__main__':
    main()
