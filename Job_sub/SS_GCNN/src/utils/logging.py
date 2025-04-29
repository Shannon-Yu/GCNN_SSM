import datetime
import pytz

def log_message(log_file, message):
    """记录带时间戳的消息"""
    tz = pytz.timezone("Asia/Shanghai")
    timestamp = datetime.datetime.now(tz).strftime("[%Y-%m-%d %H:%M:%S]")
    log_line = f"{timestamp} {message}"
    with open(log_file, "a") as f:
        f.write(log_line + "\n")
    print(log_line)
