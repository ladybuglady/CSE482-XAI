import os
import tkinter as tk
from tkinter import filedialog
import subprocess

class Application(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Application GUI")

        # Input variables
        self.json_file = tk.StringVar()
        self.diag_model = tk.StringVar()
        self.xai_model = tk.StringVar()
        self.save_to = tk.StringVar()

        # Set default values
        self.diag_model.set("Attia")
        self.xai_model.set("SHAP")

        # JSON file selection
        self.json_label = tk.Label(self, text="Select JSON file:")
        self.json_label.pack()
        self.json_entry = tk.Entry(self, textvariable=self.json_file, width=50)
        self.json_entry.pack()
        self.json_button = tk.Button(self, text="Select File", command=self.select_json_file)
        self.json_button.pack()

        # Diagnostic Model selection
        self.diag_label = tk.Label(self, text="Select Diagnostic Model:")
        self.diag_label.pack()
        self.diag_optionmenu = tk.OptionMenu(self, self.diag_model, "Attia", "Option 2", "Option 3")
        self.diag_optionmenu.pack()

        # Explainable Model selection
        self.xai_label = tk.Label(self, text="Select Explainable Model:")
        self.xai_label.pack()
        self.xai_optionmenu = tk.OptionMenu(self, self.xai_model, "SHAP", "Option 2", "Option 3")
        self.xai_optionmenu.pack()

        # Save filename input
        self.save_label = tk.Label(self, text="Enter Save Filename:")
        self.save_label.pack()
        self.save_entry = tk.Entry(self, textvariable=self.save_to, width=50)
        self.save_entry.pack()

        # Run button
        self.run_button = tk.Button(self, text="Run", command=self.run_script)
        self.run_button.pack()

    def select_json_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("JSON files", "*.json")])
        if file_path:
            cwd = os.getcwd()
            relative_path = os.path.relpath(file_path, start=cwd)
            self.json_file.set(relative_path)

    def run_script(self):
        command = f"python run_attia_analysis.py --ecg {self.json_file.get()} --diag {self.diag_model.get()} --xai {self.xai_model.get()} --saveTo {self.save_to.get()}"
        os.system(command)
        # command = [
        #     "python",
        #     "run_attia_analysis.py",
        #     "--ecg",
        #     self.json_file.get(),
        #     "--diag",
        #     self.diag_model.get(),
        #     "--xai",
        #     self.xai_model.get(),
        #     "--saveTo",
        #     self.save_to.get()
        # ]
        # subprocess.run(command, shell=True)  # Run the command in the terminal

if __name__ == "__main__":
    app = Application()
    app.mainloop()
