#!/bin/bash

# Wait for 5 seconds
sleep 5

# Get text from clipboard using xclip
copied_text=$(xclip -o -selection clipboard | tr -d '\r' | sed ':a;N;$!ba;s/\n/ \n/g')

# Loop through each line in the copied text
IFS=$'\n' # Set Internal Field Separator to newline
for line in $copied_text; do
    # Type the line using xdotool
    xdotool type --delay 50 "$line"
    
    # Press Enter
    xdotool key Return
    
    # Select all text (Shift+Home)
    xdotool key Shift+Home
    
    # Delete the selected text
    xdotool key Delete
done
