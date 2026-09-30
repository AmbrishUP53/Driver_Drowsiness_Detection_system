import winsound

def play_alarm():
    # Frequency: 2500 Hz, Duration: 800 ms (Beep aawaz bajane ke liye)
    try:
        winsound.Beep(2500, 800)
    except Exception as e:
        print(f"Error playing sound: {e}")