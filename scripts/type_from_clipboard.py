import pyautogui
import pyperclip
import time


def type_from_clipboard():
    copied_text = pyperclip.paste().strip()
    for line in copied_text.split("\n"):
        pyautogui.typewrite(line)
        pyautogui.press("enter")
        pyautogui.keyDown("shift")
        pyautogui.press("home")
        pyautogui.keyUp("shift")
        pyautogui.press("delete")


time.sleep(5)
type_from_clipboard()
