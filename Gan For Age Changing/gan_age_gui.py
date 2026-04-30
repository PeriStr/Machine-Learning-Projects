"""
Age Progression GAN - Γραφικό Περιβάλλον Χρήστη (GUI)
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import tkinter.scrolledtext as scrolledtext
import threading
import queue
import os
import sys
import io
import json
import numpy as np
import time
import contextlib

# ── Matplotlib (TkAgg backend FIRST, before gan_age import) ──────────────────
import matplotlib
matplotlib.use('TkAgg')
_original_mpl_use = matplotlib.use
matplotlib.use = lambda *a, **kw: None  # block backend switching during gan_age import

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

# ── PIL ───────────────────────────────────────────────────────────────────────
try:
    from PIL import Image, ImageTk
    PIL_OK = True
except ImportError:
    PIL_OK = False

# ── Restore matplotlib.use after GUI imports ──────────────────────────────────
matplotlib.use = _original_mpl_use

# ─────────────────────────────────────────────────────────────────────────────
# Theme constants
# ─────────────────────────────────────────────────────────────────────────────
BG     = '#1a1a2e'
PANEL  = '#16213e'
CARD   = '#0f3460'
ACCENT = '#e94560'
TEXT   = '#e2e8f0'
TEXT2  = '#a0aec0'
GREEN  = '#22c55e'
RED    = '#ef4444'
YELLOW = '#f59e0b'

AGE_GROUPS = {0: '0-20', 1: '21-35', 2: '36-55', 3: '56-65', 4: '65+'}
AGE_COLORS = ['#4CAF50', '#2196F3', '#FF9800', '#9C27B0', '#F44336']

DEFAULT_CONFIG = {
    'data_path':        './UTKFace',
    'image_size':       128,
    'num_age_classes':  5,
    'nz':               100,
    'ngf':              64,
    'ndf':              64,
    'nc':               3,
    'batch_size':       32,
    'num_epochs':       100,
    'lr_g':             0.0002,
    'lr_d':             0.0002,
    'beta1':            0.5,
    'lambda_L1':        10.0,
    'lambda_cls':       1.0,
    'save_interval':    10,
    'sample_interval':  5,
    'output_dir':       './gan_output',
    'checkpoint_dir':   './checkpoints',
    'max_samples':      10000,
}


# ─────────────────────────────────────────────────────────────────────────────
# Stdout → Queue bridge
# ─────────────────────────────────────────────────────────────────────────────
class QueueStdout:
    def __init__(self, q: queue.Queue):
        self._q = q

    def write(self, text):
        if text and text.strip():
            self._q.put(('log', text.rstrip()))

    def flush(self):
        pass


# ─────────────────────────────────────────────────────────────────────────────
# Main Application
# ─────────────────────────────────────────────────────────────────────────────
class GANApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Age Progression GAN")
        self.geometry("1300x860")
        self.minsize(1000, 700)
        self.configure(bg=BG)

        self.training_active  = False
        self.training_thread  = None
        self.log_queue        = queue.Queue()
        self.loss_history     = {'G': [], 'D': [], 'epochs': []}

        self._setup_style()
        self._build_header()
        self._build_notebook()
        self._build_statusbar()
        self._poll_queue()
        self.after(600, self._check_gpu)

    # ── Style ─────────────────────────────────────────────────────────────────
    def _setup_style(self):
        s = ttk.Style(self)
        s.theme_use('clam')
        s.configure('.', background=BG, foreground=TEXT, font=('Segoe UI', 9))
        s.configure('TFrame', background=BG)
        s.configure('TLabel', background=BG, foreground=TEXT)
        s.configure('TNotebook', background=PANEL, tabmargins=[2, 5, 2, 0])
        s.configure('TNotebook.Tab', background=PANEL, foreground=TEXT2,
                    padding=[14, 6], font=('Segoe UI', 10))
        s.map('TNotebook.Tab',
              background=[('selected', ACCENT)],
              foreground=[('selected', TEXT)])
        s.configure('TButton', background=ACCENT, foreground=TEXT,
                    font=('Segoe UI', 9, 'bold'), padding=[8, 4])
        s.map('TButton', background=[('active', '#c73650'),
                                     ('disabled', '#4a4a6a')])
        s.configure('Green.TButton', background='#16a34a', foreground='white',
                    font=('Segoe UI', 9, 'bold'))
        s.map('Green.TButton', background=[('active', '#15803d'),
                                           ('disabled', '#4a4a6a')])
        s.configure('Red.TButton', background='#dc2626', foreground='white',
                    font=('Segoe UI', 9, 'bold'))
        s.map('Red.TButton', background=[('active', '#b91c1c'),
                                         ('disabled', '#4a4a6a')])
        s.configure('TLabelframe', background=BG, bordercolor='#2d3748')
        s.configure('TLabelframe.Label', background=BG, foreground=TEXT2,
                    font=('Segoe UI', 9, 'bold'))
        s.configure('TProgressbar', troughcolor=PANEL, background=ACCENT,
                    lightcolor=ACCENT, darkcolor=ACCENT)
        s.configure('TScrollbar', background=PANEL, troughcolor=BG,
                    arrowcolor=TEXT2)

    # ── Header ────────────────────────────────────────────────────────────────
    def _build_header(self):
        hdr = tk.Frame(self, bg=PANEL, height=52)
        hdr.pack(fill='x')
        hdr.pack_propagate(False)
        tk.Label(hdr, text="Age Progression GAN",
                 bg=PANEL, fg=TEXT, font=('Segoe UI', 14, 'bold')
                 ).pack(side='left', padx=20, pady=12)
        tk.Label(hdr, text="Conditional GAN · UTKFace Dataset",
                 bg=PANEL, fg=TEXT2, font=('Segoe UI', 9)
                 ).pack(side='left', padx=0, pady=14)
        self.gpu_lbl = tk.Label(hdr, text="ελέγχεται GPU…",
                                bg=PANEL, fg=TEXT2, font=('Segoe UI', 9))
        self.gpu_lbl.pack(side='right', padx=20)

    # ── Notebook ──────────────────────────────────────────────────────────────
    def _build_notebook(self):
        nb = ttk.Notebook(self)
        nb.pack(fill='both', expand=True, padx=6, pady=6)
        self.tab_train = TrainingTab(nb, self)
        self.tab_infer = InferenceTab(nb, self)
        nb.add(self.tab_train, text='  Εκπαίδευση  ')
        nb.add(self.tab_infer, text='  Inference  ')

    # ── Status bar ────────────────────────────────────────────────────────────
    def _build_statusbar(self):
        bar = tk.Frame(self, bg=PANEL, height=26)
        bar.pack(fill='x', side='bottom')
        bar.pack_propagate(False)
        self._status_var = tk.StringVar(value='Έτοιμο')
        tk.Label(bar, textvariable=self._status_var,
                 bg=PANEL, fg=TEXT2, font=('Segoe UI', 8)
                 ).pack(side='left', padx=10, pady=4)

    def set_status(self, msg: str):
        self._status_var.set(msg)

    # ── GPU check ─────────────────────────────────────────────────────────────
    def _check_gpu(self):
        try:
            import torch
            if torch.cuda.is_available():
                name = torch.cuda.get_device_name(0)
                mem  = torch.cuda.get_device_properties(0).total_memory // (1024**2)
                self.gpu_lbl.config(text=f"GPU: {name}  ({mem} MB)", fg=GREEN)
            else:
                self.gpu_lbl.config(text="CPU mode  (CUDA μη διαθέσιμο)", fg=YELLOW)
        except ImportError:
            self.gpu_lbl.config(text="PyTorch μη εγκατεστημένο", fg=RED)

    # ── Queue polling ─────────────────────────────────────────────────────────
    def _poll_queue(self):
        try:
            while True:
                kind, data = self.log_queue.get_nowait()
                if kind == 'log':
                    self.tab_train.append_log(data)
                elif kind == 'progress':
                    ep, total = data
                    self.tab_train.update_progress(ep, total)
                elif kind == 'loss':
                    ep, g, d = data
                    self.loss_history['epochs'].append(ep)
                    self.loss_history['G'].append(g)
                    self.loss_history['D'].append(d)
                    self.tab_train.refresh_plot(self.loss_history)
                elif kind == 'done':
                    self.training_active = False
                    self.tab_train.on_done()
                    self.set_status('Εκπαίδευση ολοκληρώθηκε')
                elif kind == 'error':
                    self.training_active = False
                    self.tab_train.on_done()
                    messagebox.showerror('Σφάλμα εκπαίδευσης', str(data))
        except queue.Empty:
            pass
        self.after(120, self._poll_queue)


# ─────────────────────────────────────────────────────────────────────────────
# Training Tab
# ─────────────────────────────────────────────────────────────────────────────
class TrainingTab(ttk.Frame):
    def __init__(self, parent, app: GANApp):
        super().__init__(parent)
        self.app = app
        self._vars: dict[str, tk.StringVar] = {}
        self._build()

    def _build(self):
        self.columnconfigure(0, weight=0, minsize=300)
        self.columnconfigure(1, weight=1)
        self.rowconfigure(0, weight=1)
        self._build_left()
        self._build_right()

    # ── Left: config panel ────────────────────────────────────────────────────
    def _build_left(self):
        left = tk.Frame(self, bg=PANEL, width=300)
        left.grid(row=0, column=0, sticky='nsew', padx=(4, 2), pady=4)
        left.pack_propagate(False)

        tk.Label(left, text="Ρυθμίσεις Εκπαίδευσης",
                 bg=PANEL, fg=TEXT, font=('Segoe UI', 11, 'bold')
                 ).pack(anchor='w', padx=12, pady=(12, 4))

        # Scrollable area for config fields
        outer = tk.Frame(left, bg=PANEL)
        outer.pack(fill='both', expand=True)
        canvas = tk.Canvas(outer, bg=PANEL, highlightthickness=0, bd=0)
        vsb = ttk.Scrollbar(outer, orient='vertical', command=canvas.yview)
        self._cfg_frame = tk.Frame(canvas, bg=PANEL)
        self._cfg_frame.bind('<Configure>',
            lambda e: canvas.configure(scrollregion=canvas.bbox('all')))
        canvas.create_window((0, 0), window=self._cfg_frame, anchor='nw')
        canvas.configure(yscrollcommand=vsb.set)
        canvas.pack(side='left', fill='both', expand=True)
        vsb.pack(side='right', fill='y')

        self._add_fields()

        # Buttons + progress at bottom
        bot = tk.Frame(left, bg=PANEL)
        bot.pack(fill='x', padx=10, pady=8)

        self.btn_start = ttk.Button(bot, text="▶  Εκκίνηση Εκπαίδευσης",
                                     style='Green.TButton',
                                     command=self._start)
        self.btn_start.pack(fill='x', pady=3)

        self.btn_stop = ttk.Button(bot, text="■  Διακοπή",
                                    style='Red.TButton',
                                    command=self._stop,
                                    state='disabled')
        self.btn_stop.pack(fill='x', pady=3)

        ttk.Button(bot, text="📂  Άνοιγμα Checkpoint",
                   command=self._open_ckpt).pack(fill='x', pady=3)

        tk.Label(bot, text="Πρόοδος:", bg=PANEL, fg=TEXT2,
                 font=('Segoe UI', 8)).pack(anchor='w', pady=(6, 1))
        self._prog_var = tk.DoubleVar(value=0)
        ttk.Progressbar(bot, variable=self._prog_var,
                        maximum=100).pack(fill='x')
        self._ep_lbl = tk.Label(bot, text="Epoch: 0 / 0",
                                 bg=PANEL, fg=TEXT2, font=('Segoe UI', 8))
        self._ep_lbl.pack(anchor='w', pady=2)

    def _section(self, text):
        tk.Label(self._cfg_frame, text=text,
                 bg=PANEL, fg=TEXT2, font=('Segoe UI', 8, 'bold')
                 ).pack(anchor='w', padx=10, pady=(8, 1))

    def _field(self, label, key, default):
        row = tk.Frame(self._cfg_frame, bg=PANEL)
        row.pack(fill='x', padx=10, pady=1)
        tk.Label(row, text=label, bg=PANEL, fg=TEXT,
                 font=('Segoe UI', 9), width=20, anchor='w').pack(side='left')
        var = tk.StringVar(value=str(default))
        self._vars[key] = var
        tk.Entry(row, textvariable=var, bg=CARD, fg=TEXT,
                 insertbackground=TEXT, relief='flat',
                 font=('Segoe UI', 9), width=14).pack(side='left', padx=4)

    def _add_fields(self):
        self._section("─── Dataset ───")
        # data_path with browse
        row = tk.Frame(self._cfg_frame, bg=PANEL)
        row.pack(fill='x', padx=10, pady=1)
        tk.Label(row, text="Data Path", bg=PANEL, fg=TEXT,
                 font=('Segoe UI', 9), width=20, anchor='w').pack(side='left')
        dp_var = tk.StringVar(value=DEFAULT_CONFIG['data_path'])
        self._vars['data_path'] = dp_var
        tk.Entry(row, textvariable=dp_var, bg=CARD, fg=TEXT,
                 insertbackground=TEXT, relief='flat',
                 font=('Segoe UI', 9), width=14).pack(side='left', padx=4)
        tk.Button(row, text="📁", command=lambda: dp_var.set(
                      filedialog.askdirectory() or dp_var.get()),
                  bg=CARD, fg=TEXT, relief='flat',
                  font=('Segoe UI', 9)).pack(side='left')

        self._field("Image Size",       'image_size',   128)
        self._field("Max Samples",      'max_samples',  10000)

        self._section("─── Αρχιτεκτονική ───")
        self._field("Latent dim nz",    'nz',   100)
        self._field("Gen filters ngf",  'ngf',  64)
        self._field("Disc filters ndf", 'ndf',  64)

        self._section("─── Εκπαίδευση ───")
        self._field("Epochs",           'num_epochs',   100)
        self._field("Batch Size",       'batch_size',   32)
        self._field("LR Generator",     'lr_g',         0.0002)
        self._field("LR Discriminator", 'lr_d',         0.0002)
        self._field("Beta1",            'beta1',        0.5)
        self._field("λ L1",             'lambda_L1',    10.0)
        self._field("λ Class",          'lambda_cls',   1.0)

        self._section("─── Αποθήκευση ───")
        self._field("Save interval",    'save_interval',    10)
        self._field("Sample interval",  'sample_interval',  5)

        row2 = tk.Frame(self._cfg_frame, bg=PANEL)
        row2.pack(fill='x', padx=10, pady=1)
        tk.Label(row2, text="Output dir", bg=PANEL, fg=TEXT,
                 font=('Segoe UI', 9), width=20, anchor='w').pack(side='left')
        od_var = tk.StringVar(value=DEFAULT_CONFIG['output_dir'])
        self._vars['output_dir'] = od_var
        tk.Entry(row2, textvariable=od_var, bg=CARD, fg=TEXT,
                 insertbackground=TEXT, relief='flat',
                 font=('Segoe UI', 9), width=14).pack(side='left', padx=4)

        row3 = tk.Frame(self._cfg_frame, bg=PANEL)
        row3.pack(fill='x', padx=10, pady=1)
        tk.Label(row3, text="Checkpoint dir", bg=PANEL, fg=TEXT,
                 font=('Segoe UI', 9), width=20, anchor='w').pack(side='left')
        cd_var = tk.StringVar(value=DEFAULT_CONFIG['checkpoint_dir'])
        self._vars['checkpoint_dir'] = cd_var
        tk.Entry(row3, textvariable=cd_var, bg=CARD, fg=TEXT,
                 insertbackground=TEXT, relief='flat',
                 font=('Segoe UI', 9), width=14).pack(side='left', padx=4)

    # ── Right: log + plot ─────────────────────────────────────────────────────
    def _build_right(self):
        right = tk.Frame(self, bg=BG)
        right.grid(row=0, column=1, sticky='nsew', padx=(2, 4), pady=4)
        right.rowconfigure(0, weight=3)
        right.rowconfigure(1, weight=2)
        right.columnconfigure(0, weight=1)

        # Log
        log_lf = ttk.LabelFrame(right, text=" Αρχείο Καταγραφής ", padding=4)
        log_lf.grid(row=0, column=0, sticky='nsew', pady=(0, 4))

        self._log = scrolledtext.ScrolledText(
            log_lf, bg='#0d1117', fg='#7ee787',
            font=('Consolas', 8), relief='flat',
            state='disabled', wrap='word'
        )
        self._log.pack(fill='both', expand=True)

        btn_row = tk.Frame(log_lf, bg=BG)
        btn_row.pack(fill='x', pady=(3, 0))
        tk.Button(btn_row, text="Καθαρισμός", command=self._clear_log,
                  bg=CARD, fg=TEXT2, relief='flat',
                  font=('Segoe UI', 8)).pack(side='right', padx=2)

        # Plot
        plot_lf = ttk.LabelFrame(right, text=" Καμπύλες Loss ", padding=4)
        plot_lf.grid(row=1, column=0, sticky='nsew')

        self._fig = Figure(figsize=(6, 3), dpi=80, facecolor='#0d1117')
        self._ax  = self._fig.add_subplot(111)
        self._style_ax()
        self._plot_canvas = FigureCanvasTkAgg(self._fig, master=plot_lf)
        self._plot_canvas.get_tk_widget().pack(fill='both', expand=True)

    def _style_ax(self):
        ax = self._ax
        ax.set_facecolor('#161b22')
        ax.tick_params(colors=TEXT2, labelsize=7)
        for sp in ax.spines.values():
            sp.set_edgecolor('#30363d')
        ax.grid(True, color='#21262d', linestyle='--', linewidth=0.5)
        ax.set_title('G / D Loss', color=TEXT, fontsize=9)
        ax.set_xlabel('Epoch', color=TEXT2, fontsize=8)

    # ── Public API ────────────────────────────────────────────────────────────
    def append_log(self, text: str):
        self._log.configure(state='normal')
        self._log.insert('end', text + '\n')
        self._log.see('end')
        self._log.configure(state='disabled')

    def _clear_log(self):
        self._log.configure(state='normal')
        self._log.delete('1.0', 'end')
        self._log.configure(state='disabled')

    def update_progress(self, ep: int, total: int):
        self._prog_var.set(ep / total * 100)
        self._ep_lbl.config(text=f"Epoch: {ep} / {total}")
        self.app.set_status(f"Εκπαίδευση… Epoch {ep}/{total}")

    def refresh_plot(self, hist: dict):
        self._ax.clear()
        self._style_ax()
        ep = hist['epochs']
        if ep:
            self._ax.plot(ep, hist['G'], color='#7ee787',
                          linewidth=1.4, label='Generator')
            self._ax.plot(ep, hist['D'], color='#ff7b72',
                          linewidth=1.4, label='Discriminator')
            self._ax.legend(facecolor='#161b22', edgecolor='#30363d',
                            labelcolor=TEXT, fontsize=7)
        self._fig.tight_layout()
        self._plot_canvas.draw()

    def on_done(self):
        self.btn_start.state(['!disabled'])
        self.btn_stop.state(['disabled'])

    # ── Actions ───────────────────────────────────────────────────────────────
    def _start(self):
        if self.app.training_active:
            messagebox.showwarning('', 'Εκπαίδευση ήδη σε εξέλιξη.')
            return
        cfg = self._collect_config()
        if cfg is None:
            return
        self.app.training_active = True
        self.app.loss_history = {'G': [], 'D': [], 'epochs': []}
        self.btn_start.state(['disabled'])
        self.btn_stop.state(['!disabled'])
        self._prog_var.set(0)
        self.append_log('═' * 55)
        self.append_log('  Εκκίνηση εκπαίδευσης…')
        self.append_log('═' * 55)
        self.app.training_thread = threading.Thread(
            target=_training_worker,
            args=(cfg, self.app.log_queue, self.app),
            daemon=True
        )
        self.app.training_thread.start()

    def _stop(self):
        self.app.training_active = False
        self.append_log('[!] Σήμα διακοπής στάλθηκε…')

    def _open_ckpt(self):
        path = filedialog.askopenfilename(
            title='Φόρτωση checkpoint',
            filetypes=[('PyTorch', '*.pth'), ('Όλα', '*.*')]
        )
        if path:
            self.append_log(f'Checkpoint: {path}')
            self.app.tab_infer.set_model(path)

    def _collect_config(self) -> dict | None:
        try:
            return {
                'data_path':        self._vars['data_path'].get(),
                'image_size':       int(self._vars['image_size'].get()),
                'num_age_classes':  5,
                'nz':               int(self._vars['nz'].get()),
                'ngf':              int(self._vars['ngf'].get()),
                'ndf':              int(self._vars['ndf'].get()),
                'nc':               3,
                'batch_size':       int(self._vars['batch_size'].get()),
                'num_epochs':       int(self._vars['num_epochs'].get()),
                'lr_g':             float(self._vars['lr_g'].get()),
                'lr_d':             float(self._vars['lr_d'].get()),
                'beta1':            float(self._vars['beta1'].get()),
                'lambda_L1':        float(self._vars['lambda_L1'].get()),
                'lambda_cls':       float(self._vars['lambda_cls'].get()),
                'save_interval':    int(self._vars['save_interval'].get()),
                'sample_interval':  int(self._vars['sample_interval'].get()),
                'output_dir':       self._vars['output_dir'].get(),
                'checkpoint_dir':   self._vars['checkpoint_dir'].get(),
                'max_samples':      int(self._vars['max_samples'].get()),
            }
        except ValueError as e:
            messagebox.showerror('Μη έγκυρη τιμή', str(e))
            return None


# ─────────────────────────────────────────────────────────────────────────────
# Training worker (background thread)
# ─────────────────────────────────────────────────────────────────────────────
def _training_worker(config: dict, q: queue.Queue, app: GANApp):
    old_stdout = sys.stdout
    sys.stdout = QueueStdout(q)
    try:
        import torch
        import torch.nn as nn
        import torch.optim as optim
        from torch.utils.data import DataLoader

        _dir = os.path.dirname(os.path.abspath(__file__))
        if _dir not in sys.path:
            sys.path.insert(0, _dir)

        with contextlib.redirect_stdout(old_stdout):
            from gan_age import (
                UTKFaceDataset, get_transforms, Generator, Discriminator,
                weights_init, GANLoss, create_demo_dataset,
            )

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Device: {device}")

        # Dataset
        data_path = config['data_path']
        if not os.path.exists(data_path):
            print(f"Dataset δεν βρέθηκε στο '{data_path}' – δημιουργία demo…")
            data_path = create_demo_dataset('./demo_dataset', 500)
            config['data_path'] = data_path

        train_tf, _ = get_transforms(config['image_size'])
        dataset = UTKFaceDataset(
            root_dir=data_path,
            transform=train_tf,
            max_samples=config.get('max_samples', 10000)
        )
        if len(dataset) == 0:
            raise ValueError("Δεν βρέθηκαν εικόνες!")

        loader = DataLoader(
            dataset,
            batch_size=config['batch_size'],
            shuffle=True,
            num_workers=0,
            pin_memory=torch.cuda.is_available(),
            drop_last=True,
        )

        G = Generator(
            nc=config['nc'], ngf=config['ngf'], nz=config['nz'],
            num_age_classes=config['num_age_classes'],
            image_size=config['image_size']
        ).to(device)
        D = Discriminator(
            nc=config['nc'], ndf=config['ndf'],
            num_age_classes=config['num_age_classes'],
            image_size=config['image_size']
        ).to(device)
        G.apply(weights_init)
        D.apply(weights_init)

        opt_G = optim.Adam(G.parameters(), lr=config['lr_g'],
                           betas=(config['beta1'], 0.999))
        opt_D = optim.Adam(D.parameters(), lr=config['lr_d'],
                           betas=(config['beta1'], 0.999))
        sched_G = optim.lr_scheduler.LambdaLR(
            opt_G, lr_lambda=lambda ep: max(0.0, 1.0 - ep / config['num_epochs']))
        sched_D = optim.lr_scheduler.LambdaLR(
            opt_D, lr_lambda=lambda ep: max(0.0, 1.0 - ep / config['num_epochs']))

        crit_gan = GANLoss('lsgan').to(device)
        crit_L1  = nn.L1Loss()
        crit_cls = nn.CrossEntropyLoss()

        os.makedirs(config['output_dir'], exist_ok=True)
        os.makedirs(config['checkpoint_dir'], exist_ok=True)

        print(f"Generator:     {sum(p.numel() for p in G.parameters()):,} params")
        print(f"Discriminator: {sum(p.numel() for p in D.parameters()):,} params")
        print(f"Dataset: {len(dataset)} εικόνες | {len(loader)} batches")

        best_g = float('inf')

        for epoch in range(config['num_epochs']):
            if not app.training_active:
                print("Εκπαίδευση διακόπηκε από χρήστη.")
                break

            G.train(); D.train()
            sum_g = sum_d = 0.0
            n = 0
            t0 = time.time()

            for bi, batch in enumerate(loader):
                if not app.training_active:
                    break

                real_imgs = batch['image'].to(device)
                real_ages = batch['age_group'].to(device)
                bsz       = real_imgs.size(0)
                tgt_ages  = torch.randint(0, config['num_age_classes'], (bsz,)).to(device)
                z         = torch.randn(bsz, config['nz']).to(device)

                # D
                opt_D.zero_grad()
                rv, rc  = D(real_imgs, real_ages)
                d_real  = crit_gan(rv, True)
                d_cls   = crit_cls(rc, real_ages)
                with torch.no_grad():
                    fake = G(real_imgs, tgt_ages, z)
                fv, _   = D(fake.detach(), tgt_ages)
                d_fake  = crit_gan(fv, False)
                d_loss  = (d_real + d_fake) / 2 + config['lambda_cls'] * d_cls
                d_loss.backward(); opt_D.step()

                # G
                opt_G.zero_grad()
                fake     = G(real_imgs, tgt_ages, z)
                fv2, fc  = D(fake, tgt_ages)
                g_adv    = crit_gan(fv2, True)
                z2       = torch.randn(bsz, config['nz']).to(device)
                recon    = G(fake, real_ages, z2)
                g_l1     = crit_L1(recon, real_imgs)
                g_cls    = crit_cls(fc, tgt_ages)
                g_loss   = g_adv + config['lambda_L1'] * g_l1 + config['lambda_cls'] * g_cls
                g_loss.backward(); opt_G.step()

                sum_g += g_loss.item()
                sum_d += d_loss.item()
                n += 1

                if bi % 20 == 0:
                    print(f"  [{epoch+1}/{config['num_epochs']}] "
                          f"batch {bi}/{len(loader)}  "
                          f"G={g_loss.item():.4f}  D={d_loss.item():.4f}")

            sched_G.step(); sched_D.step()
            avg_g = sum_g / max(n, 1)
            avg_d = sum_d / max(n, 1)
            t_ep  = time.time() - t0

            print(f"\n★ Epoch [{epoch+1}/{config['num_epochs']}] "
                  f"({t_ep:.1f}s)  G={avg_g:.4f}  D={avg_d:.4f}")

            q.put(('progress', (epoch + 1, config['num_epochs'])))
            q.put(('loss',     (epoch + 1, avg_g, avg_d)))

            if (epoch + 1) % config['save_interval'] == 0:
                ckpt = os.path.join(config['checkpoint_dir'],
                                    f'checkpoint_epoch_{epoch+1:04d}.pth')
                torch.save({
                    'epoch': epoch + 1, 'G_state': G.state_dict(),
                    'D_state': D.state_dict(), 'config': config, 'g_loss': avg_g,
                }, ckpt)
                print(f"  ✓ Checkpoint: {ckpt}")

            if avg_g < best_g:
                best_g = avg_g
                torch.save({
                    'G_state': G.state_dict(), 'config': config,
                    'epoch': epoch + 1, 'g_loss': best_g,
                }, os.path.join(config['checkpoint_dir'], 'best_model.pth'))
                print(f"  ★ Νέο best model  G={best_g:.4f}")

        q.put(('done', None))

    except Exception:
        import traceback
        q.put(('log',   traceback.format_exc()))
        q.put(('error', traceback.format_exc().splitlines()[-1]))
    finally:
        sys.stdout = old_stdout


# ─────────────────────────────────────────────────────────────────────────────
# Inference Tab
# ─────────────────────────────────────────────────────────────────────────────
class InferenceTab(ttk.Frame):
    def __init__(self, parent, app: GANApp):
        super().__init__(parent)
        self.app = app
        self._model_path = tk.StringVar()
        self._input_img  = None
        self._result_pil = []
        self._tk_refs    = []
        self._build()

    def _build(self):
        self.columnconfigure(0, weight=0, minsize=240)
        self.columnconfigure(1, weight=1)
        self.rowconfigure(0, weight=1)
        self._build_left()
        self._build_right()

    # ── Left: controls ────────────────────────────────────────────────────────
    def _build_left(self):
        left = tk.Frame(self, bg=PANEL, width=240)
        left.grid(row=0, column=0, sticky='nsew', padx=(4, 2), pady=4)
        left.pack_propagate(False)

        tk.Label(left, text="Inference",
                 bg=PANEL, fg=TEXT, font=('Segoe UI', 11, 'bold')
                 ).pack(anchor='w', padx=12, pady=(12, 6))

        # Model
        mf = ttk.LabelFrame(left, text=" Μοντέλο (.pth) ", padding=8)
        mf.pack(fill='x', padx=8, pady=4)
        tk.Entry(mf, textvariable=self._model_path, bg=CARD, fg=TEXT,
                 insertbackground=TEXT, relief='flat',
                 font=('Segoe UI', 8)).pack(fill='x', pady=2)
        ttk.Button(mf, text="📂  Φόρτωση Μοντέλου",
                   command=self._load_model).pack(fill='x', pady=2)
        self._model_lbl = tk.Label(mf, text="—  δεν φορτώθηκε",
                                    bg=mf['background'] if False else '#0f3460',
                                    fg=YELLOW, font=('Segoe UI', 8),
                                    wraplength=210, justify='left')
        self._model_lbl.pack(anchor='w')

        # Image
        imgf = ttk.LabelFrame(left, text=" Εικόνα Εισόδου ", padding=8)
        imgf.pack(fill='x', padx=8, pady=4)
        ttk.Button(imgf, text="🖼  Άνοιγμα Εικόνας",
                   command=self._load_image).pack(fill='x', pady=2)
        self._preview = tk.Label(imgf, bg=CARD,
                                  text="Δεν επιλέχθηκε εικόνα",
                                  fg=TEXT2, font=('Segoe UI', 8),
                                  width=22, height=9, anchor='center')
        self._preview.pack(pady=4)

        # Generate
        self._btn_gen = ttk.Button(left, text="✨  Δημιουργία Age Progression",
                                    style='Green.TButton',
                                    command=self._generate,
                                    state='disabled')
        self._btn_gen.pack(fill='x', padx=8, pady=6)

        self._gen_lbl = tk.Label(left, text="",
                                  bg=PANEL, fg=TEXT2,
                                  font=('Segoe UI', 8), wraplength=220)
        self._gen_lbl.pack(padx=8)

        # Save
        self._btn_save = ttk.Button(left, text="💾  Αποθήκευση Αποτελεσμάτων",
                                     command=self._save, state='disabled')
        self._btn_save.pack(fill='x', padx=8, pady=4)

    # ── Right: results grid ───────────────────────────────────────────────────
    def _build_right(self):
        right = tk.Frame(self, bg=BG)
        right.grid(row=0, column=1, sticky='nsew', padx=(2, 4), pady=4)
        right.rowconfigure(1, weight=1)
        right.columnconfigure(0, weight=1)

        tk.Label(right, text="Αποτελέσματα Age Progression / Regression",
                 bg=BG, fg=TEXT, font=('Segoe UI', 12, 'bold')
                 ).grid(row=0, column=0, sticky='w', padx=10, pady=8)

        grid = tk.Frame(right, bg=BG)
        grid.grid(row=1, column=0, sticky='nsew', padx=8, pady=4)

        slot_info = [
            ("Πρωτότυπο",         'white'),
            (f"→ {AGE_GROUPS[0]}", AGE_COLORS[0]),
            (f"→ {AGE_GROUPS[1]}", AGE_COLORS[1]),
            (f"→ {AGE_GROUPS[2]}", AGE_COLORS[2]),
            (f"→ {AGE_GROUPS[3]}", AGE_COLORS[3]),
            (f"→ {AGE_GROUPS[4]}", AGE_COLORS[4]),
        ]

        self._slots = []
        for i, (title, color) in enumerate(slot_info):
            r, c = divmod(i, 3)
            cell = tk.Frame(grid, bg=CARD, bd=0)
            cell.grid(row=r, column=c, padx=6, pady=6, sticky='nsew')
            grid.rowconfigure(r, weight=1)
            grid.columnconfigure(c, weight=1)

            img_lbl = tk.Label(cell, bg=CARD,
                                text="─",
                                fg=TEXT2, font=('Segoe UI', 8))
            img_lbl.pack(expand=True, fill='both', padx=4, pady=(4, 2))

            tk.Label(cell, text=title,
                     bg=CARD, fg=color,
                     font=('Segoe UI', 9, 'bold')).pack(pady=(0, 4))

            self._slots.append(img_lbl)

    # ── API ───────────────────────────────────────────────────────────────────
    def set_model(self, path: str):
        self._model_path.set(path)
        self._model_lbl.config(text=f"✓  {os.path.basename(path)}", fg=GREEN)
        self._try_enable_gen()

    def _load_model(self):
        p = filedialog.askopenfilename(
            title='Επιλογή μοντέλου',
            filetypes=[('PyTorch', '*.pth'), ('Όλα', '*.*')]
        )
        if p:
            self.set_model(p)

    def _load_image(self):
        if not PIL_OK:
            messagebox.showerror('', 'Η βιβλιοθήκη Pillow δεν είναι εγκατεστημένη.')
            return
        p = filedialog.askopenfilename(
            title='Επιλογή εικόνας',
            filetypes=[('Εικόνες', '*.jpg *.jpeg *.png *.bmp'), ('Όλα', '*.*')]
        )
        if not p:
            return
        try:
            img = Image.open(p).convert('RGB')
            self._input_img = img
            preview = img.resize((170, 150), Image.LANCZOS)
            tk_img = ImageTk.PhotoImage(preview)
            self._preview.configure(image=tk_img, text='',
                                    width=170, height=150)
            self._preview._ref = tk_img
            self._try_enable_gen()
        except Exception as e:
            messagebox.showerror('Σφάλμα εικόνας', str(e))

    def _try_enable_gen(self):
        if self._model_path.get() and self._input_img is not None:
            self._btn_gen.state(['!disabled'])

    def _generate(self):
        mpath = self._model_path.get()
        if not os.path.exists(mpath):
            messagebox.showerror('Σφάλμα', 'Το αρχείο μοντέλου δεν βρέθηκε.')
            return
        if self._input_img is None:
            messagebox.showerror('Σφάλμα', 'Επιλέξτε εικόνα εισόδου.')
            return
        self._gen_lbl.config(text='Επεξεργασία…', fg=YELLOW)
        self._btn_gen.state(['disabled'])
        threading.Thread(
            target=self._gen_worker,
            args=(mpath, self._input_img.copy()),
            daemon=True
        ).start()

    def _gen_worker(self, model_path: str, pil_img):
        try:
            import torch
            import torchvision.transforms as T

            _dir = os.path.dirname(os.path.abspath(__file__))
            if _dir not in sys.path:
                sys.path.insert(0, _dir)
            with contextlib.redirect_stdout(io.StringIO()):
                from gan_age import Generator

            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            ckpt   = torch.load(model_path, map_location=device)
            cfg    = ckpt['config']

            G = Generator(
                nc=cfg['nc'], ngf=cfg['ngf'], nz=cfg['nz'],
                num_age_classes=cfg['num_age_classes'],
                image_size=cfg['image_size']
            ).to(device)
            G.load_state_dict(ckpt['G_state'])
            G.eval()

            tf = T.Compose([
                T.Resize((cfg['image_size'], cfg['image_size'])),
                T.ToTensor(),
                T.Normalize([0.5]*3, [0.5]*3),
            ])
            x = tf(pil_img).unsqueeze(0).to(device)

            results = [pil_img.resize((200, 200), Image.LANCZOS)]
            with torch.no_grad():
                for ag in range(cfg['num_age_classes']):
                    tgt  = torch.tensor([ag]).to(device)
                    z    = torch.randn(1, cfg['nz']).to(device)
                    fake = G(x, tgt, z)
                    arr  = fake[0].cpu().numpy().transpose(1, 2, 0)
                    arr  = (arr * 0.5 + 0.5).clip(0, 1)
                    pil  = Image.fromarray((arr * 255).astype(np.uint8))
                    results.append(pil.resize((200, 200), Image.LANCZOS))

            self._result_pil = results
            self.app.after(0, self._show_results)

        except Exception:
            import traceback
            err = traceback.format_exc()
            self.app.after(0, lambda: (
                messagebox.showerror('Σφάλμα Inference', err),
                self._gen_lbl.config(text='Σφάλμα!', fg=RED),
                self._btn_gen.state(['!disabled']),
            ))

    def _show_results(self):
        self._tk_refs.clear()
        for slot, pil_img in zip(self._slots, self._result_pil):
            tk_img = ImageTk.PhotoImage(pil_img)
            slot.configure(image=tk_img, text='')
            slot._ref = tk_img
            self._tk_refs.append(tk_img)
        self._gen_lbl.config(text='✓  Ολοκληρώθηκε!', fg=GREEN)
        self._btn_gen.state(['!disabled'])
        self._btn_save.state(['!disabled'])
        self.app.set_status('Inference ολοκληρώθηκε')

    def _save(self):
        if not self._result_pil:
            return
        d = filedialog.askdirectory(title='Επιλογή φακέλου αποθήκευσης')
        if not d:
            return
        names = ['original'] + [f'age_{AGE_GROUPS[i]}' for i in range(5)]
        for img, name in zip(self._result_pil, names):
            img.save(os.path.join(d, f"{name.replace('+','plus')}.png"))
        messagebox.showinfo('Αποθήκευση',
                            f'Αποθηκεύτηκαν {len(self._result_pil)} εικόνες\nστο: {d}')


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    app = GANApp()
    app.mainloop()
