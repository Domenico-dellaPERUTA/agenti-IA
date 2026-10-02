"""Interfaccia Tkinter dimostrativa per usare la libreria ``AI``.

La finestra raccoglie prompt e percorso sandbox; il lavoro di rete/modello
avviene in un thread secondario, mentre gli eventi vengono trasferiti alla
finestra tramite una coda così Tkinter resta aggiornato dal thread principale.
Per integrare la libreria in un'altra applicazione si può usare direttamente
``Agent`` e un provider, senza importare questo modulo GUI.
"""

from __future__ import annotations

from dataclasses import dataclass, field
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


@dataclass
class RichiestaApprovazioneScript:
    """Scambia una richiesta di approvazione tra worker e thread Tkinter.

    Il worker attende ``completato``; il thread grafico imposta ``decisione``
    prima di segnalare l'evento.
    """

    file_path: str
    script: str
    decisione: bool = False
    completato: threading.Event = field(default_factory=threading.Event)


class AgenteGUI:
    """App demo per conversazioni, sandbox, approvazione script e log live."""

    def __init__(self, window: tk.Tk):
        """Imposta la finestra principale, stato condiviso e polling eventi."""
        self.window = window
        self.window.title("Agente AI")
        self.window.geometry("900x720")
        self.window.minsize(700, 560)
        # I worker non manipolano direttamente Tkinter: inviano eventi tipizzati
        # in modo informale nella coda, consumati periodicamente dal thread GUI.
        self.events: queue.Queue[tuple[str, object]] = queue.Queue()

        self.cartella_sandbox = tk.StringVar(
            value=str(PROJECT_ROOT / "sandbox")
        )
        self.stato = tk.StringVar(value="Pronto")

        self._crea_interfaccia()
        self.window.after(100, self._leggi_eventi)

    def _crea_interfaccia(self) -> None:
        """Costruisce i controlli Tk e collega widget a callback dell'app."""
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
            ("Eseguire script Bash", "Approva il codice; script e temporanei vengono rimossi al termine."),
            ("Cercare sul web", "Ricerca pubblica; la query viene inviata a DuckDuckGo."),
            ("Leggere una pagina", "Solo testo HTTPS pubblico; niente login o moduli."),
        )
        for row, (nome, descrizione) in enumerate(azioni):
            ttk.Label(azioni_frame, text=nome, width=26).grid(
                row=row, column=0, sticky=tk.W, padx=(0, 8)
            )
            ttk.Label(azioni_frame, text=descrizione).grid(
                row=row, column=1, sticky=tk.W
            )
        ttk.Label(
            azioni_frame,
            text=(
                "La query di ricerca viene trasmessa a DuckDuckGo: non includere "
                "dati privati o contenuti della sandbox. Nessun file viene caricato; "
                "sono consentite solo pagine HTML pubbliche in lettura. Le fonti "
                "web sono aggiunte automaticamente alla risposta."
            ),
            wraplength=790,
        ).grid(row=len(azioni), column=0, columnspan=2, sticky=tk.W, pady=(6, 0))

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
        """Mappa Ctrl+Invio all'invio del prompt e blocca il newline nel widget."""
        self._avvia_agente()
        return "break"

    def _seleziona_cartella(self) -> None:
        """Apre il selettore di cartelle e aggiorna la sandbox scelta."""
        cartella = filedialog.askdirectory(
            title="Seleziona la cartella sandbox",
            initialdir=self.cartella_sandbox.get() or str(PROJECT_ROOT),
        )
        if cartella:
            self.cartella_sandbox.set(cartella)

    def _avvia_agente(self) -> None:
        """Valida input GUI e avvia il lavoro fuori dal thread grafico."""
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
        """Crea un agente per la richiesta e invia gli eventi del worker alla GUI."""
        try:
            agent = crea_agente(
                cartella,
                script_approval=self._richiedi_approvazione_script,
                script_output=lambda output: self.events.put(("output", output)),
            )
            agent.send(prompt)
            risultato = agent.run(
                on_response=self._mostra_risposta,
                on_tool_result=lambda name, result: self.events.put(
                    ("output", f"Risultato {name}: {result}\n")
                ),
            )
            self.events.put(("risultato", risultato))
        except Exception as error:
            self.events.put(("errore", error))

    def _richiedi_approvazione_script(self, file_path: str, script: str) -> bool:
        """Invia la richiesta di approvazione al thread Tk e attende la decisione."""
        richiesta = RichiestaApprovazioneScript(file_path, script)
        self.events.put(("approvazione_script", richiesta))
        richiesta.completato.wait()
        return richiesta.decisione

    def _mostra_dialog_script(self, richiesta: RichiestaApprovazioneScript) -> None:
        """Mostra codice non modificabile e termina con approva o annulla."""
        dialog = tk.Toplevel(self.window)
        dialog.title("Autorizza esecuzione script")
        dialog.geometry("760x560")
        dialog.minsize(560, 400)
        dialog.transient(self.window)
        dialog.grab_set()

        frame = ttk.Frame(dialog, padding=12)
        frame.pack(fill=tk.BOTH, expand=True)
        percorso = richiesta.file_path
        ttk.Label(
            frame,
            text=(
                f"Questo script temporaneo verrà eseguito con i privilegi dell'utente "
                f"corrente e poi eliminato: {percorso}."
            ),
            wraplength=720,
        ).pack(anchor=tk.W, pady=(0, 6))
        ttk.Label(
            frame,
            text=(
                "Gli script possono modificare o leggere file accessibili all'utente "
                "e usare la rete; processi avviati in background potrebbero continuare "
                "dopo la fine dello script. Verifica attentamente il codice."
            ),
            wraplength=720,
        ).pack(anchor=tk.W, pady=(0, 8))
        codice = scrolledtext.ScrolledText(frame, wrap=tk.NONE, state=tk.NORMAL)
        codice.pack(fill=tk.BOTH, expand=True)
        codice.insert("1.0", richiesta.script)
        codice.configure(state=tk.DISABLED)

        pulsanti = ttk.Frame(frame)
        pulsanti.pack(fill=tk.X, pady=(10, 0))

        def chiudi(approvato: bool) -> None:
            """Memorizza la scelta, chiude la finestra e sblocca il worker."""
            richiesta.decisione = approvato
            dialog.grab_release()
            dialog.destroy()
            richiesta.completato.set()

        ttk.Button(
            pulsanti, text="Annulla", command=lambda: chiudi(False)
        ).pack(side=tk.RIGHT)
        ttk.Button(
            pulsanti, text="Approva ed esegui", command=lambda: chiudi(True)
        ).pack(side=tk.RIGHT, padx=(0, 8))
        dialog.protocol("WM_DELETE_WINDOW", lambda: chiudi(False))
        dialog.bind("<Escape>", lambda _event: chiudi(False))

    def _mostra_risposta(self, response: LLMResponse) -> None:
        """Trasforma risposta del modello e tool call in eventi leggibili."""
        if response.content:
            self.events.put(("output", f"Modello:\n{response.content}\n\n"))
        for call in response.tool_calls:
            self.events.put((
                "output",
                f"Azione selezionata dal modello: {call.name}({call.arguments})\n",
            ))

    def _leggi_eventi(self) -> None:
        """Svuota la coda aggiornando widget e stato solo dal thread Tk."""
        try:
            while True:
                tipo, contenuto = self.events.get_nowait()
                if tipo == "output":
                    self._aggiungi_output(str(contenuto))
                elif tipo == "approvazione_script":
                    if not isinstance(contenuto, RichiestaApprovazioneScript):
                        raise TypeError("Richiesta di approvazione script non valida.")
                    self._mostra_dialog_script(contenuto)
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
        """Accoda testo al log e scorre automaticamente all'ultima riga."""
        self.output.configure(state=tk.NORMAL)
        self.output.insert(tk.END, testo)
        self.output.see(tk.END)
        self.output.configure(state=tk.DISABLED)


if __name__ == "__main__":
    root = tk.Tk()
    AgenteGUI(root)
    root.mainloop()