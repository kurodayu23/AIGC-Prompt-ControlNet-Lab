"""提示词组合器的轻量桌面入口，不加载生成模型。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from src.prompt_engine import PromptComposer


class PromptWindow:
    def __init__(self, window):
        self.window = window
        self.composer = PromptComposer()
        window.title("AIGC Prompt Studio")
        window.geometry("860x650")
        window.minsize(720, 560)
        style = ttk.Style()
        style.theme_use("clam")
        style.configure(".", font=("Microsoft YaHei UI", 10), background="#f5f6f8")
        style.configure("TButton", padding=(14, 8))
        style.configure("Title.TLabel", font=("Microsoft YaHei UI", 20, "bold"), foreground="#182235")
        frame = ttk.Frame(window, padding=24)
        frame.pack(fill="both", expand=True)
        frame.columnconfigure(1, weight=1)
        frame.rowconfigure(7, weight=1)
        ttk.Label(frame, text="提示词工作台", style="Title.TLabel").grid(row=0, column=0, columnspan=2, sticky="w")
        ttk.Label(frame, text="选择模板，描述主体，生成可复制的提示词。此工具不直接生成图片。", wraplength=760).grid(row=1, column=0, columnspan=2, sticky="w", pady=(8, 24))
        self.subject = tk.StringVar(value="a small friendly robot")
        self.template = tk.StringVar(value="cinematic_portrait")
        self.platform = tk.StringVar(value="Stable Diffusion")
        self.preset = tk.StringVar(value="v6_photoreal")
        rows = [("主体描述", ttk.Entry(frame, textvariable=self.subject)),
                ("构图模板", ttk.Combobox(frame, textvariable=self.template, values=self.composer.list_templates(), state="readonly")),
                ("目标平台", ttk.Combobox(frame, textvariable=self.platform, values=["Stable Diffusion", "Midjourney"], state="readonly")),
                ("Midjourney 预设", ttk.Combobox(frame, textvariable=self.preset, values=list(self.composer._db["midjourney_params"]), state="readonly"))]
        for row, (label, widget) in enumerate(rows, 2):
            ttk.Label(frame, text=label).grid(row=row, column=0, sticky="w", padx=(0, 20), pady=6)
            widget.grid(row=row, column=1, sticky="ew", pady=6)
        actions = ttk.Frame(frame)
        actions.grid(row=6, column=0, columnspan=2, sticky="w", pady=(16, 12))
        ttk.Button(actions, text="生成提示词", command=self.generate).pack(side="left", padx=(0, 8))
        ttk.Button(actions, text="复制结果", command=self.copy).pack(side="left", padx=(0, 8))
        ttk.Button(actions, text="保存文本", command=self.save).pack(side="left")
        self.output = tk.Text(frame, wrap="word", font=("Consolas", 11), relief="solid", borderwidth=1, padx=12, pady=12)
        self.output.grid(row=7, column=0, columnspan=2, sticky="nsew")
        self.status = tk.StringVar(value="本地运行 · 无需 Python 安装或模型下载")
        ttk.Label(frame, textvariable=self.status).grid(row=8, column=0, columnspan=2, sticky="w", pady=(12, 0))
        self.generate()

    def generate(self):
        subject = self.subject.get().strip()
        if not subject:
            self.status.set("请输入主体描述。")
            return
        platform = "midjourney" if self.platform.get() == "Midjourney" else "sd"
        result = self.composer.compose(self.template.get(), subject, target_platform=platform,
                                       mj_param_preset=self.preset.get() if platform == "midjourney" else None)
        text = result.as_midjourney_string() if platform == "midjourney" else json.dumps(result.as_sd_payload(), ensure_ascii=False, indent=2)
        self.output.delete("1.0", "end")
        self.output.insert("1.0", text)
        self.status.set("已生成 · 可复制到对应平台；平台预设基于仓库的 V6 模板")

    def copy(self):
        self.window.clipboard_clear()
        self.window.clipboard_append(self.output.get("1.0", "end-1c"))
        self.status.set("结果已复制。")

    def save(self):
        path = filedialog.asksaveasfilename(defaultextension=".txt", filetypes=[("文本文件", "*.txt")])
        if path:
            try:
                Path(path).write_text(self.output.get("1.0", "end-1c"), encoding="utf-8")
                self.status.set("结果已保存。")
            except OSError as exc:
                messagebox.showerror("保存失败", str(exc))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-test", metavar="REPORT")
    args = parser.parse_args()
    window = tk.Tk()
    app = PromptWindow(window)
    if args.smoke_test:
        def verify():
            sd = json.loads(app.output.get("1.0", "end-1c"))
            app.platform.set("Midjourney")
            app.generate()
            mj = app.output.get("1.0", "end-1c")
            result = {"sd": bool(sd["prompt"]), "midjourney": mj.startswith("/imagine prompt:"),
                      "templates": len(app.composer.list_templates())}
            Path(args.smoke_test).write_text(json.dumps(result), encoding="utf-8")
            window.destroy()
        window.after(800, verify)
    window.mainloop()


if __name__ == "__main__":
    main()
