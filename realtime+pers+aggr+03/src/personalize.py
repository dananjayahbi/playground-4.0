import tkinter as tk
from ttkbootstrap import Style

class PersonalizationWindow(tk.Toplevel):
    def __init__(self, master, callback):
        """
        master: Tkinter root.
        callback: Function to call when personalization is complete.
        """
        super().__init__(master)
        self.title("Personalizing")
        # Set a fixed window size instead of maximized.
        self.geometry("600x400")
        self.configure(bg="black")
        self.protocol("WM_DELETE_WINDOW", self.on_close)
        
        style = Style(theme="darkly")
        
        self.progress_var = tk.DoubleVar()
        self.progress_bar = tk.ttk.Progressbar(self, variable=self.progress_var, maximum=100, mode='determinate')
        self.progress_bar.pack(fill="x", padx=50, pady=30)
        
        self.label = tk.Label(self, text="Personalizing your app, please wait...", font=("Helvetica", 28), bg="black", fg="white")
        self.label.pack(pady=20)
        
        self.callback = callback
        self.after(100, self.update_progress)
    
    def on_close(self):
        self.destroy()
        self.master.destroy()

    def update_progress(self):
        current = self.progress_var.get()
        if current < 100:
            self.progress_var.set(current + 5)
            self.after(100, self.update_progress)
        else:
            self.destroy()
            self.callback()
