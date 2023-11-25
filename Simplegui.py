import tkinter as tk
from tkinter import ttk
# Need to import the final script 

def execute_script():
    user_input = entry.get()
    output = final_script(user_input)
    output_label.config(text=output)

root = tk.Tk()
root.title("Papers search")

#Entry input
entry = ttk.Entry(root, width=40)
entry.grid(row=0, column=0, padx=10, pady=10)

# Run button
run_button = ttk.Button(root, text="Search papers", command=execute_script)
run_button.grid(row=0, column=1, padx=10, pady=10)

# Output 
output_label = ttk.Label(root, text="", wraplength=300)
output_label.grid(row=1, columnspan=2, padx=10, pady=10)

root.mainloop()