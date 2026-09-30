import time
blink_start_time = 0
is_eyes_closed = False
counter = 0
total_blinks = 0
blink_timestamps = []

def count_blinks(ear, threshold=0.23, consecutive_frames=2):
    global counter, total_blinks
    
    if ear < threshold:
        counter += 1
    else:
        if counter >= consecutive_frames:
            total_blinks += 1
        counter = 0
        
    return total_blinks


def check_drowsiness_factors(ear, threshold=0.23, time_limit=0.5):
    global blink_start_time, is_eyes_closed, blink_timestamps
    current_time = time.time()
    
    is_micro_sleep = False
    
    # 1. Micro-sleep check (Aankh kitni der se band hai)
    if ear < threshold:
        if not is_eyes_closed:
            is_eyes_closed = True
            blink_start_time = current_time
        else:
            duration = current_time - blink_start_time
            if duration > time_limit:
                is_micro_sleep = True
    else:
        if is_eyes_closed:
            # Aankh khuli, matlab ek blink complete ho gaya
            blink_timestamps.append(current_time)
        is_eyes_closed = False
        
    # 2. Blink Rate calculation (Pichle 60 seconds me total blinks)
    # Purane timestamps hata do jo 60 seconds se zyada purane hain
    blink_timestamps = [t for t in blink_timestamps if current_time - t <= 60]
    blink_rate = len(blink_timestamps) # Blinks per minute
    
    return is_micro_sleep, blink_rate