import pyautogui
import time
import tkinter as tk
from tkinter import scrolledtext

def type_paste_text(text, interval):
    time.sleep(1)  # Allow user to switch windows
    pyautogui.write(text, interval=interval)


def start_typing():
    text = text_area.get("1.0", tk.END)
    interval = float(speed_entry.get())
    type_paste_text(text, interval)
# Setting up the UI
root = tk.Tk()
root.title("Fake Keyboard Input")
root.geometry("400x300")

# Label and input for typing speed
speed_label = tk.Label(root, text="Typing Speed (seconds per character):")
speed_label.pack(pady=5)
speed_entry = tk.Entry(root)
speed_entry.insert(0, "0.05")  # Default speed
speed_entry.pack(pady=5)

# Text area for input text
text_label = tk.Label(root, text="Text to Type:")
text_label.pack(pady=5)
text_area = scrolledtext.ScrolledText(root, width=40, height=10)
text_area.pack(pady=5)

# Button to start typing
start_button = tk.Button(root, text="Start Typing", command=start_typing)
start_button.pack(pady=10)

root.mainloop()
