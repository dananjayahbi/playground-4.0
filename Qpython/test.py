import tkinter as tk
from tkinter import ttk
import ttkbootstrap as tb
import pandas as pd
import tkinter.font as tkFont

def main():
    # ---------------------------
    # 1. Setup Main Application
    # ---------------------------
    root = tb.Window(themename="flatly")
    root.title("Motivational Quotes Viewer")

    # Attempt to maximize the window (cross-platform attempt)
    try:
        root.state("zoomed")  # Works on Windows
    except:
        root.attributes("-zoomed", True)  # Often works on Linux

    # ---------------------------
    # 2. Read Excel Data
    # ---------------------------
    df = pd.read_excel("QTS_meaningful_tags.xlsx")

    # ---------------------------
    # 3. Create Scrollable Frame
    # ---------------------------
    container = tb.Frame(root)
    container.pack(fill="both", expand=True, padx=10, pady=10)

    canvas = tk.Canvas(container)
    scrollbar = tb.Scrollbar(container, orient="vertical", command=canvas.yview)
    scrollable_frame = tb.Frame(canvas)

    scrollable_frame.bind(
        "<Configure>",
        lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
    )

    canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
    canvas.configure(yscrollcommand=scrollbar.set)

    canvas.pack(side="left", fill="both", expand=True)
    scrollbar.pack(side="right", fill="y")

    # ---------------------------
    # 4. Mouse Wheel Scrolling
    # ---------------------------
    def _on_mousewheel(event):
        """Enable mouse wheel scrolling across platforms."""
        # For Linux systems (scroll up/down events)
        if event.num == 4:  
            canvas.yview_scroll(-1, "units")
        elif event.num == 5:  
            canvas.yview_scroll(1, "units")
        # For Windows / MacOS (delta approach)
        else:  
            canvas.yview_scroll(int(-1*(event.delta/120)), "units")

    # Bind for Windows/Mac
    root.bind_all("<MouseWheel>", _on_mousewheel)
    # Bind for Linux
    root.bind_all("<Button-4>", _on_mousewheel)
    root.bind_all("<Button-5>", _on_mousewheel)

    # ---------------------------
    # 5. Header Row
    # ---------------------------
    header_font = ("Helvetica", 12, "bold")
    body_font = ("Helvetica", 12)

    header_frame = tb.Frame(scrollable_frame, padding=5)
    header_frame.pack(fill="x", pady=(0, 5))

    ttk.Label(header_frame, text="No.", width=4, font=header_font).grid(row=0, column=0, padx=5)
    ttk.Label(header_frame, text="Quote", font=header_font).grid(row=0, column=1, padx=5)
    ttk.Label(header_frame, text="Tags", font=header_font).grid(row=0, column=2, padx=5)

    # ---------------------------
    # 6. Toggle Copy Function
    # ---------------------------
    def toggle_copy(label, text, state):
        """
        Toggle the copy state for a label. If not used, copy the text to clipboard,
        update the label's font to show an overstrike (strikethrough), and mark as used.
        If already used, revert to the normal font.
        """
        if not state["used"]:
            root.clipboard_clear()
            root.clipboard_append(text)
            label.configure(font=state["overstrike_font"])
            state["used"] = True
        else:
            label.configure(font=state["normal_font"])
            state["used"] = False

    # ---------------------------
    # 7. Populate Rows
    # ---------------------------
    for index, row in df.iterrows():
        row_frame = tb.Frame(scrollable_frame, padding=5, relief="ridge")
        row_frame.pack(fill="x", pady=2, padx=2)

        # Number label
        number_label = ttk.Label(row_frame, text=str(row["No."]), width=4, font=body_font)
        number_label.grid(row=0, column=0, sticky="nw", padx=5)

        # Quote label
        quote_label = ttk.Label(row_frame, text=row["The Quote"], wraplength=500, justify="left", font=body_font)
        quote_label.grid(row=0, column=1, sticky="w", padx=5)

        # Tags label
        tags_label = ttk.Label(row_frame, text=row["Tags"], wraplength=300, justify="left", font=body_font)
        tags_label.grid(row=0, column=2, sticky="w", padx=5)

        # Fonts for toggling (normal/overstrike)
        normal_quote_font = tkFont.Font(family="Helvetica", size=12)
        overstrike_quote_font = tkFont.Font(family="Helvetica", size=12)
        overstrike_quote_font.configure(overstrike=True)

        normal_tags_font = tkFont.Font(family="Helvetica", size=12)
        overstrike_tags_font = tkFont.Font(family="Helvetica", size=12)
        overstrike_tags_font.configure(overstrike=True)

        # States
        quote_state = {
            "used": False,
            "normal_font": normal_quote_font,
            "overstrike_font": overstrike_quote_font
        }
        tags_state = {
            "used": False,
            "normal_font": normal_tags_font,
            "overstrike_font": overstrike_tags_font
        }

        # Copy buttons
        copy_quote_button = tb.Button(
            row_frame,
            text="Copy Quote",
            command=lambda q=row["The Quote"], l=quote_label, s=quote_state: toggle_copy(l, q, s)
        )
        copy_quote_button.grid(row=0, column=3, padx=5)

        copy_tags_button = tb.Button(
            row_frame,
            text="Copy Tags",
            command=lambda t=row["Tags"], l=tags_label, s=tags_state: toggle_copy(l, t, s)
        )
        copy_tags_button.grid(row=0, column=4, padx=5)

    # Start the main loop
    root.mainloop()

if __name__ == "__main__":
    main()
