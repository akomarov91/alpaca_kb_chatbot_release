import csv
import os
import datetime

# === Configuration ===

# Directory to store weekly log files
LOG_DIR = "chat_logs"

# Ensure the directory exists
os.makedirs(LOG_DIR, exist_ok=True)

# === Helper: Get Current Week's Log Filename ===

def get_current_log_filename():
    """
    Returns the log file path based on the current year and ISO week number.
    Example: chat_logs/chat_history_2025_week_31.csv
    """
    today = datetime.date.today()
    year, week, _ = today.isocalendar()  # ISO calendar: (year, week number, weekday)
    return os.path.join(LOG_DIR, f"chat_history_{year}_week_{week}.csv")

# === Main: Log One Chat to the File ===

def log_chat_to_csv(user_query, bot_response):
    """
    Logs a single chatbot conversation to the current week's log file.
    Creates the file with headers if it doesn't exist yet.
    """
    log_file = get_current_log_filename()
    file_exists = os.path.isfile(log_file)

    # Open file in append mode
    with open(log_file, "a", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)

        # If new file, write the header first
        if not file_exists:
            writer.writerow(["Timestamp", "User", "Bot"])

        # Write the current interaction
        writer.writerow([
            datetime.datetime.now().isoformat(),  # Timestamp in ISO format
            user_query,
            bot_response
        ])