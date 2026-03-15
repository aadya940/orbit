"""Bottom-right toast UI for human-in-the-loop (approval / help)."""

import asyncio
from typing import Any


def run_toast_ui(kind: str, context: dict[str, Any]) -> dict[str, Any]:
    """Run a compact bottom-right toast (blocking). White background, black text."""
    import tkinter as tk
    from tkinter import font as tkfont

    result: dict[str, Any] = {}

    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    root.overrideredirect(True)
    root.configure(bg="#ffffff")

    # Cursive "Orbit" – try script fonts (Windows: Segoe Script, Lucida Handwriting)
    try:
        font_cursive = tkfont.Font(root=root, family="Segoe Script", size=15)
    except Exception:
        try:
            font_cursive = tkfont.Font(root=root, family="Lucida Handwriting", size=15)
        except Exception:
            font_cursive = tkfont.Font(
                root=root, family="Segoe UI", size=13, weight=tkfont.BOLD
            )

    font_heading = tkfont.Font(root=root, family="Segoe UI", size=9, weight=tkfont.BOLD)
    font_body = tkfont.Font(root=root, family="Segoe UI", size=8)
    font_mono = tkfont.Font(root=root, family="Consolas", size=8)
    font_btn_bold = tkfont.Font(
        root=root, family="Segoe UI", size=8, weight=tkfont.BOLD
    )
    font_btn = tkfont.Font(root=root, family="Segoe UI", size=8)

    card = tk.Frame(root, bg="#ffffff", padx=9, pady=7)
    card.pack(fill=tk.BOTH, expand=True, padx=1, pady=1)

    # Header: "Orbit" in cursive
    orbit_lbl = tk.Label(
        card, text="Orbit", fg="#000000", bg="#ffffff", font=font_cursive
    )
    orbit_lbl.pack(anchor="w", pady=(0, 3))

    subtitle = "Permission required" if kind == "approval" else "Human step"
    sub_lbl = tk.Label(
        card, text=subtitle, fg="#000000", bg="#ffffff", font=font_heading
    )
    sub_lbl.pack(anchor="w", pady=(0, 3))

    # Body
    body = tk.Text(
        card,
        wrap=tk.WORD,
        height=2,
        bg="#ffffff",
        fg="#000000",
        insertbackground="#000000",
        relief=tk.FLAT,
        padx=0,
        pady=0,
        font=font_body,
        cursor="arrow",
        highlightthickness=0,
        borderwidth=0,
        spacing1=0,
        spacing3=1,
    )
    body.tag_configure("heading", font=font_heading, foreground="#000000")
    body.tag_configure("body", font=font_body, foreground="#000000")
    body.tag_configure("path", font=font_mono, foreground="#333333")

    if kind == "approval":
        tool = context.get("tool", "?")
        body.insert(tk.END, "Allow ", "body")
        body.insert(tk.END, f"{tool}", "heading")
        body.insert(tk.END, " ", "body")
        if "path" in context:
            body.insert(tk.END, context["path"], "path")
        elif "src" in context:
            body.insert(tk.END, f"{context['src']} → {context.get('dst', '')}", "path")
        elif "directory" in context:
            body.insert(tk.END, context["directory"], "path")
        else:
            body.insert(tk.END, "", "body")
    else:
        desc = context.get("description", "Complete the requested step.")
        body.insert(tk.END, desc, "body")
    body.config(state=tk.DISABLED)
    body.pack(anchor="w", fill=tk.X, pady=(0, 6))

    btn_frame = tk.Frame(card, bg="#ffffff")
    btn_frame.pack(anchor="e")

    def approve() -> None:
        result["status"] = "approved"
        result["message"] = "Approved"
        root.quit()
        root.destroy()

    def reject() -> None:
        result["status"] = "rejected"
        result["message"] = "Rejected"
        root.quit()
        root.destroy()

    def done() -> None:
        result["status"] = "completed"
        result["message"] = "Done"
        root.quit()
        root.destroy()

    if kind == "approval":
        approve_btn = tk.Button(
            btn_frame,
            text="Approve",
            command=approve,
            bg="#222222",
            fg="#ffffff",
            activebackground="#444444",
            activeforeground="#ffffff",
            relief=tk.FLAT,
            padx=9,
            pady=3,
            font=font_btn_bold,
            cursor="hand2",
        )
        approve_btn.pack(side=tk.RIGHT, padx=(0, 3))
        reject_btn = tk.Button(
            btn_frame,
            text="Reject",
            command=reject,
            bg="#e0e0e0",
            fg="#000000",
            activebackground="#d0d0d0",
            activeforeground="#000000",
            relief=tk.FLAT,
            padx=9,
            pady=3,
            font=font_btn,
            cursor="hand2",
        )
        reject_btn.pack(side=tk.RIGHT)
    else:
        done_btn = tk.Button(
            btn_frame,
            text="Done",
            command=done,
            bg="#222222",
            fg="#ffffff",
            activebackground="#444444",
            activeforeground="#ffffff",
            relief=tk.FLAT,
            padx=9,
            pady=3,
            font=font_btn_bold,
            cursor="hand2",
        )
        done_btn.pack(side=tk.RIGHT)

    def on_escape(event: Any) -> None:
        if not result:
            result["status"] = "rejected" if kind == "approval" else "completed"
            result["message"] = "Cancelled"
        root.quit()
        root.destroy()

    root.bind("<Escape>", on_escape)
    root.update_idletasks()
    w, h = root.winfo_reqwidth(), root.winfo_reqheight()
    screen_w = root.winfo_screenwidth()
    screen_h = root.winfo_screenheight()
    margin = 15
    x = screen_w - w - margin
    y = screen_h - h - margin
    root.geometry(f"+{x}+{y}")
    root.deiconify()
    root.lift()
    root.focus_force()
    try:
        root.mainloop()
    except Exception:
        pass
    return (
        result
        if result
        else {
            "status": "rejected" if kind == "approval" else "completed",
            "message": "Closed",
        }
    )


async def default_human_in_the_loop(
    kind: str, context: dict[str, Any]
) -> dict[str, Any]:
    """Default handler: toast in bottom-right. Override with Agent(human_in_the_loop=...) for custom behavior."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, run_toast_ui, kind, context)
