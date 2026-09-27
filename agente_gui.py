from __future__ import annotations

from pathlib import Path
import queue
import sys
import threading

try:
    import tkinter as tk
    from tkinter import filedialog, messagebox, scrolledtext, ttk
except ModuleNotFoundError as error:
    if error.name != "_tkinter":
        raise
    raise SystemExit(
        "La GUI richiede un interprete Python compilato con Tkinter. "
        "In VS Code esegui 'Python: Select Interpreter' e seleziona "
        "un interprete che supera il test `python -c \"import tkinter\"`."
    ) from error

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from AI import LLMResponse
from agente import PROMPT_INIZIALE, crea_agente


class AgenteGUI:
    def __init__(self, window: tk.Tk):
        self.window = window
        self.window.title("Agente AI")
        self.window.geometry("900x720")
        self.window.minsize(700, 560)
        self.events: queue.Queue[tuple[str, object]] = queue.Queue()

        self.cartella_sandbox = tk.StringVar(
            value=str(PROJECT_ROOT / "sandbox")
        )
        self.stato = tk.StringVar(value="Pronto")

        self._crea_interfaccia()
        self.window.after(100, self._leggi_eventi)

    def _crea_interfaccia(self) -> None:
        frame = ttk.Frame(self.window, padding=12)
        frame.pack(fill=tk.BOTH, expand=True)
        frame.columnconfigure(0, weight=1)
        frame.rowconfigure(5, weight=1)

        ttk.Label(frame, text="Cartella sandbox").grid(
            row=0, column=0, sticky=tk.W
        )
        ttk.Entry(frame, textvariable=self.cartella_sandbox).grid(
            row=1, column=0, sticky=tk.EW, pady=(4, 8)
        )
        ttk.Button(frame, text="Seleziona cartella", command=self._seleziona_cartella).grid(
            row=1, column=1, padx=(8, 0), pady=(4, 8)
        )

        azioni_frame = ttk.LabelFrame(frame, text="Azioni disponibili all'agente", padding=8)
        azioni_frame.grid(row=2, column=0, columnspan=2, sticky=tk.EW, pady=(0, 8))
        azioni_frame.columnconfigure(1, weight=1)
        azioni = (
            ("Elencare file", "Mostra i file della sandbox."),
            ("Cercare testo", "Trova una frase nei documenti."),
            ("Leggere estratti", "Legge le righe richieste di un file."),
            ("Creare file", "Crea un file senza sovrascriverne uno esistente."),
            ("Aggiungere testo", "Accoda contenuto a un file."),
            ("Spostare o rinominare file", "Sposta un file senza sovrascrivere la destinazione."),
            ("Estrarre informazioni", "Trova le righe che contengono un testo."),
            ("Confrontare file", "Mostra le differenze tra due file di testo."),
            ("Preparare attività", "Ricava attività da checklist o appunti."),
        )
        for row, (nome, descrizione) in enumerate(azioni):
            ttk.Label(azioni_frame, text=nome, width=26).grid(
                row=row, column=0, sticky=tk.W, padx=(0, 8)
            )
            ttk.Label(azioni_frame, text=descrizione).grid(
                row=row, column=1, sticky=tk.W
            )

        ttk.Label(frame, text="Richiesta per l'agente").grid(
            row=3, column=0, columnspan=2, sticky=tk.W
        )
        self.prompt = tk.Text(frame, height=4, wrap=tk.WORD)
        self.prompt.grid(row=4, column=0, columnspan=2, sticky=tk.EW, pady=(4, 8))
        self.prompt.insert("1.0", PROMPT_INIZIALE)
        self.prompt.bind("<Control-Return>", self._avvia_da_tastiera)

        controls = ttk.Frame(frame)
        controls.grid(row=5, column=0, columnspan=2, sticky=tk.NSEW)
        controls.columnconfigure(0, weight=1)
        controls.rowconfigure(1, weight=1)
        self.pulsante_avvia = ttk.Button(
            controls, text="Invia richiesta al modello", command=self._avvia_agente
        )
        self.pulsante_avvia.grid(row=0, column=0, sticky=tk.W, pady=(0, 8))
        ttk.Label(
            controls, text="Premi Ctrl+Invio per inviare"
        ).grid(row=0, column=1, sticky=tk.E, pady=(0, 8))

        self.output = scrolledtext.ScrolledText(
            controls, wrap=tk.WORD, state=tk.DISABLED
        )
        self.output.grid(row=1, column=0, columnspan=2, sticky=tk.NSEW)
        ttk.Label(frame, textvariable=self.stato).grid(
            row=6, column=0, columnspan=2, sticky=tk.W, pady=(8, 0)
        )

    def _avvia_da_tastiera(self, _event: tk.Event) -> str:
        self._avvia_agente()
        return "break"

    def _seleziona_cartella(self) -> None:
        cartella = filedialog.askdirectory(
            title="Seleziona la cartella sandbox",
            initialdir=self.cartella_sandbox.get() or str(PROJECT_ROOT),
        )
        if cartella:
            self.cartella_sandbox.set(cartella)

    def _avvia_agente(self) -> None:
        percorso_cartella = self.cartella_sandbox.get().strip()
        prompt = self.prompt.get("1.0", tk.END).strip()
        if not percorso_cartella or not Path(percorso_cartella).expanduser().is_dir():
            messagebox.showerror("Cartella non valida", "Seleziona una cartella sandbox esistente.")
            return
        cartella = Path(percorso_cartella).expanduser()
        if not prompt:
            messagebox.showerror("Richiesta mancante", "Inserisci una richiesta per l'agente.")
            return

        self._aggiungi_output(f"> {prompt}\n\n")
        self.pulsante_avvia.configure(state=tk.DISABLED)
        self.stato.set("Agente in esecuzione...")
        self.window.update_idletasks()
        threading.Thread(
            target=self._esegui_agente,
            args=(cartella, prompt),
            daemon=True,
        ).start()

    def _esegui_agente(self, cartella: Path, prompt: str) -> None:
        try:
            agent = crea_agente(cartella)
            agent.send(prompt)
            risultato = agent.run(on_response=self._mostra_risposta)
            self.events.put(("risultato", risultato))
        except Exception as error:
            self.events.put(("errore", error))

    def _mostra_risposta(self, response: LLMResponse) -> None:
        if response.content:
            self.events.put(("output", f"Modello:\n{response.content}\n\n"))
        for call in response.tool_calls:
            self.events.put((
                "output",
                f"Azione selezionata dal modello: {call.name}({call.arguments})\n",
            ))

    def _leggi_eventi(self) -> None:
        try:
            while True:
                tipo, contenuto = self.events.get_nowait()
                if tipo == "output":
                    self._aggiungi_output(str(contenuto))
                elif tipo == "risultato":
                    risultato = contenuto if contenuto is not None else "(nessun risultato)"
                    self._aggiungi_output(f"Risultato finale:\n{risultato}\n\n")
                    self.stato.set("Completato")
                    self.pulsante_avvia.configure(state=tk.NORMAL)
                elif tipo == "errore":
                    self._aggiungi_output(f"Errore durante l'esecuzione: {contenuto}\n\n")
                    self.stato.set("Errore")
                    self.pulsante_avvia.configure(state=tk.NORMAL)
        except queue.Empty:
            pass
        self.window.after(100, self._leggi_eventi)

    def _aggiungi_output(self, testo: str) -> None:
        self.output.configure(state=tk.NORMAL)
        self.output.insert(tk.END, testo)
        self.output.see(tk.END)
        self.output.configure(state=tk.DISABLED)


if __name__ == "__main__":
    root = tk.Tk()
    AgenteGUI(root)
    root.mainloop()