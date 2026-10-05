"""Interfaccia Tkinter dimostrativa per usare la libreria ``AI``.

La finestra raccoglie prompt e percorso sandbox; il lavoro di rete/modello
avviene in un thread secondario, mentre gli eventi vengono trasferiti alla
finestra tramite una coda così Tkinter resta aggiornato dal thread principale.
Per integrare la libreria in un'altra applicazione si può usare direttamente
``Agent`` e un provider, senza importare questo modulo GUI.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
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

from AI import OrchestrationResult
from agente import PROMPT_INIZIALE, crea_orchestratore


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
        self.window.geometry("1080x900")
        self.window.minsize(820, 680)
        # I worker non manipolano direttamente Tkinter: inviano eventi tipizzati
        # in modo informale nella coda, consumati periodicamente dal thread GUI.
        self.events: queue.Queue[tuple[str, object]] = queue.Queue()
        self.agent_rows: dict[str, str] = {}

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
        frame.rowconfigure(6, weight=1)

        ttk.Label(frame, text="Cartella sandbox").grid(
            row=0, column=0, sticky=tk.W
        )
        ttk.Entry(frame, textvariable=self.cartella_sandbox).grid(
            row=1, column=0, sticky=tk.EW, pady=(4, 8)
        )
        ttk.Button(frame, text="Seleziona cartella", command=self._seleziona_cartella).grid(
            row=1, column=1, padx=(8, 0), pady=(4, 8)
        )

        info_frame = ttk.Frame(frame)
        info_frame.grid(row=2, column=0, columnspan=2, sticky=tk.EW, pady=(0, 8))
        ttk.Label(
            info_frame,
            text="Azioni disponibili all'agente",
        ).pack(side=tk.LEFT)
        ttk.Button(
            info_frame,
            text="ⓘ Info strumenti",
            command=self._mostra_info_strumenti,
        ).pack(side=tk.LEFT, padx=(8, 0))

        agent_frame = ttk.LabelFrame(frame, text="Agenti della richiesta", padding=6)
        agent_frame.grid(row=3, column=0, columnspan=2, sticky=tk.EW, pady=(0, 8))
        agent_frame.columnconfigure(0, weight=1)
        self.agent_list = ttk.Treeview(
            agent_frame,
            columns=("semaforo", "agente", "stato"),
            show="headings",
            height=5,
        )
        self.agent_list.heading("semaforo", text="")
        self.agent_list.heading("agente", text="Agente / attività")
        self.agent_list.heading("stato", text="Stato")
        self.agent_list.column("semaforo", width=38, minwidth=38, stretch=False, anchor=tk.CENTER)
        self.agent_list.column("agente", width=360, minwidth=180, stretch=True)
        self.agent_list.column("stato", width=160, minwidth=130, stretch=False)
        self.agent_list.tag_configure("running", foreground="#9a6700")
        self.agent_list.tag_configure("failed", foreground="#c62828")
        self.agent_list.tag_configure("completed", foreground="#218838")
        self.agent_list.grid(row=0, column=0, sticky=tk.EW)
        agent_scroll = ttk.Scrollbar(
            agent_frame, orient=tk.VERTICAL, command=self.agent_list.yview
        )
        agent_scroll.grid(row=0, column=1, sticky=tk.NS)
        self.agent_list.configure(yscrollcommand=agent_scroll.set)
        legend = ttk.Frame(agent_frame)
        legend.grid(row=1, column=0, sticky=tk.W, pady=(4, 0))
        for label, color in (
            ("● In esecuzione", "#9a6700"),
            ("● Errore", "#c62828"),
            ("● Completato", "#218838"),
        ):
            ttk.Label(legend, text=label, foreground=color).pack(
                side=tk.LEFT, padx=(0, 12)
            )

        ttk.Label(frame, text="Richiesta per l'agente").grid(
            row=4, column=0, columnspan=2, sticky=tk.W
        )
        self.prompt = tk.Text(frame, height=8, wrap=tk.WORD)
        self.prompt.grid(row=5, column=0, columnspan=2, sticky=tk.EW, pady=(4, 8))
        self.prompt.insert("1.0", PROMPT_INIZIALE)
        self.prompt.bind("<Control-Return>", self._avvia_da_tastiera)

        controls = ttk.Frame(frame)
        controls.grid(row=6, column=0, columnspan=2, sticky=tk.NSEW)
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
            controls, wrap=tk.WORD, state=tk.DISABLED, height=18
        )
        self.output.grid(row=1, column=0, columnspan=2, sticky=tk.NSEW)
        ttk.Label(frame, textvariable=self.stato).grid(
            row=7, column=0, columnspan=2, sticky=tk.W, pady=(8, 0)
        )

    def _mostra_info_strumenti(self) -> None:
        """Mostra descrizione degli strumenti e delle relative cautele."""
        messagebox.showinfo(
            "Azioni disponibili all'agente",
            "File nella sandbox\n"
            "• Elencare file, cercare testo, leggere estratti, estrarre informazioni, "
            "confrontare file e preparare attività.\n"
            "• Creare, aggiornare o spostare file solo nella cartella sandbox; "
            "la creazione non sovrascrive file esistenti.\n\n"
            "Script\n"
            "• L'esecuzione Bash richiede una conferma esplicita. Uno script opera "
            "con i privilegi dell'utente e non è confinato alla sandbox.\n\n"
            "Web\n"
            "• Ricerca pubblica tramite DuckDuckGo e lettura di pagine HTTPS pubbliche "
            "in sola lettura. Le query vengono inviate a DuckDuckGo: non includere "
            "dati privati, credenziali o contenuti della sandbox.\n"
            "• Alcuni siti possono rifiutare l'accesso (per esempio HTTP 403); "
            "l'agente deve continuare con le altre fonti accessibili e segnalarlo.",
        )

    def _pulisci_lista_agenti(self) -> None:
        """Rimuove lo stato della richiesta precedente prima di avviarne una nuova."""
        self.agent_rows.clear()
        for item in self.agent_list.get_children():
            self.agent_list.delete(item)

    def _imposta_stato_agente(
        self,
        agent_id: str,
        title: str,
        status: str,
        error: str = "",
    ) -> None:
        """Aggiorna o aggiunge una riga con semaforo per un agente."""
        labels = {
            "running": ("In esecuzione", "●"),
            "failed": ("Errore", "●"),
            "completed": ("Completato", "●"),
        }
        state_label, light = labels[status]
        item_id = self.agent_rows.get(agent_id)
        values = (light, title, f"{state_label}: {error}" if error else state_label)
        if item_id is None:
            item_id = self.agent_list.insert(
                "",
                tk.END,
                values=values,
                tags=(status,),
            )
            self.agent_rows[agent_id] = item_id
        else:
            self.agent_list.item(item_id, values=values, tags=(status,))

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

        self._pulisci_lista_agenti()
        self._aggiungi_output_etichettato("Utente", f"> {prompt}\n\n")
        self.pulsante_avvia.configure(state=tk.DISABLED)
        self.stato.set("Orchestratore in esecuzione...")
        self.window.update_idletasks()
        threading.Thread(
            target=self._esegui_agente,
            args=(cartella, prompt),
            daemon=True,
        ).start()

    def _esegui_agente(self, cartella: Path, prompt: str) -> None:
        """Esegue l'orchestratore e trasferisce gli eventi alla coda Tk."""
        try:
            orchestrator = crea_orchestratore(
                cartella,
                script_approval=self._richiedi_approvazione_script,
                script_output=lambda output: self.events.put(("output", output)),
            )
            risultato = orchestrator.run(
                prompt,
                on_event=lambda event: self.events.put(
                    ("orchestrator_event", event)
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

    def _leggi_eventi(self) -> None:
        """Svuota la coda aggiornando widget e stato solo dal thread Tk."""
        try:
            while True:
                tipo, contenuto = self.events.get_nowait()
                if tipo == "output":
                    self._aggiungi_output_etichettato("Script", str(contenuto))
                elif tipo == "orchestrator_event":
                    if not isinstance(contenuto, dict):
                        raise TypeError("Evento dell'orchestratore non valido.")
                    self._gestisci_evento_orchestratore(contenuto)
                elif tipo == "approvazione_script":
                    if not isinstance(contenuto, RichiestaApprovazioneScript):
                        raise TypeError("Richiesta di approvazione script non valida.")
                    self._mostra_dialog_script(contenuto)
                elif tipo == "risultato":
                    if isinstance(contenuto, OrchestrationResult):
                        answer = contenuto.answer
                        self._aggiungi_output_etichettato(
                            "Orchestratore",
                            "Risultato finale:\n"
                            f"{answer if answer else '(nessuna sintesi disponibile)'}\n\n"
                        )
                        has_failures = any(
                            task.status == "failed" for task in contenuto.tasks
                        )
                        self.stato.set(
                            "Completato con errori" if has_failures else "Completato"
                        )
                    else:
                        self._aggiungi_output_etichettato(
                            "Orchestratore",
                            f"Risultato finale:\n{contenuto or '(nessun risultato)'}\n\n"
                        )
                        self.stato.set("Completato")
                    self.pulsante_avvia.configure(state=tk.NORMAL)
                elif tipo == "errore":
                    self._aggiungi_output_etichettato(
                        "Orchestratore",
                        f"Errore durante l'esecuzione: {contenuto}\n\n",
                    )
                    self._segna_agenti_in_corso_come_errore(str(contenuto))
                    self.stato.set("Errore")
                    self.pulsante_avvia.configure(state=tk.NORMAL)
        except queue.Empty:
            pass
        self.window.after(100, self._leggi_eventi)

    def _gestisci_evento_orchestratore(self, event: dict[str, str]) -> None:
        """Mostra stato e risultati senza aggiornare Tk dai thread worker."""
        event_type = event.get("type")
        title = event.get("title") or event.get("task_id") or "Attività"
        message = event.get("message", "")
        agent_id = event.get("agent_id", "")
        agent_names = {
            "planner": "Pianificatore",
            "direct": "Agente diretto",
            "researcher": "Agente di ricerca",
            "file_writer": "Agente creazione file",
            "synthesizer": "Agente sintetizzatore",
        }
        actor = title if event.get("task_id") else agent_names.get(agent_id, title)
        if event_type == "planning_started":
            self.stato.set("Pianificazione della richiesta...")
            self._aggiungi_output_etichettato("Orchestratore", message)
        elif event_type == "agent_started":
            self._imposta_stato_agente(
                event.get("agent_id", event.get("role", title)),
                title,
                "running",
            )
            self._aggiungi_output_etichettato(actor, "Avviato")
        elif event_type == "agent_completed":
            self._imposta_stato_agente(
                event.get("agent_id", event.get("role", title)),
                title,
                "completed",
            )
            self._aggiungi_output_etichettato(actor, "Completato")
        elif event_type == "agent_failed":
            self._imposta_stato_agente(
                event.get("agent_id", event.get("role", title)),
                title,
                "failed",
                event.get("error", "Errore non specificato"),
            )
            self._aggiungi_output_etichettato(
                actor,
                f"Errore: {event.get('error', 'Errore non specificato')}",
            )
        elif event_type == "plan_warning":
            self._aggiungi_output_etichettato(
                "Pianificatore",
                f"Nota sul piano: {message}\n\n",
            )
        elif event_type == "agent_retry":
            self._aggiungi_output_etichettato(actor, f"Nuovo tentativo: {message}\n\n")
        elif event_type == "mode_selected":
            self._aggiungi_output_etichettato("Orchestratore", f"{message}\n\n")
        elif event_type == "task_started":
            self._imposta_stato_agente(
                f"task:{event.get('task_id', title)}",
                title,
                "running",
            )
            self._aggiungi_output_etichettato(actor, f"Attività avviata: {title}\n")
        elif event_type == "task_completed":
            self._imposta_stato_agente(
                f"task:{event.get('task_id', title)}",
                title,
                "completed",
            )
            self._aggiungi_output_etichettato(
                actor,
                f"Attività completata: {title}\n{message}\n\n",
            )
        elif event_type == "task_failed":
            error = event.get("error", "Errore non specificato")
            self._imposta_stato_agente(
                f"task:{event.get('task_id', title)}",
                title,
                "failed",
                error,
            )
            self._aggiungi_output_etichettato(
                actor,
                f"Attività non riuscita: {title}\n"
                f"{error}\n\n"
            )
        elif event_type == "synthesis_started":
            self.stato.set("Sintesi dei risultati...")
            self._aggiungi_output_etichettato("Agente sintetizzatore", message)
        elif event_type == "model_response":
            self._aggiungi_output_etichettato(
                actor,
                f"Risposta del modello:\n{message}\n\n",
            )
        elif event_type == "tool_result":
            tool_actor = (
                f"{actor} · {event.get('tool_name', title)}"
                if agent_id
                else actor
            )
            self._aggiungi_output_etichettato(
                tool_actor,
                f"Risultato strumento: {message}\n",
            )
        elif event_type == "tool_selected":
            self._aggiungi_output_etichettato(
                actor,
                f"Azione selezionata: {message}\n",
            )
        else:
            self._aggiungi_output_etichettato(
                "Orchestratore",
                f"Evento non riconosciuto: {event}\n",
            )

    def _segna_agenti_in_corso_come_errore(self, error: str) -> None:
        """Rende visibile l'errore finale sugli agenti rimasti in esecuzione."""
        for agent_id, item_id in self.agent_rows.items():
            values = self.agent_list.item(item_id, "values")
            if values[2] == "In esecuzione":
                self._imposta_stato_agente(
                    agent_id,
                    values[1],
                    "failed",
                    error,
                )

    def _aggiungi_output(self, testo: str) -> None:
        """Accoda testo al log e scorre automaticamente all'ultima riga."""
        self.output.configure(state=tk.NORMAL)
        self.output.insert(tk.END, testo)
        self.output.see(tk.END)
        self.output.configure(state=tk.DISABLED)

    def _aggiungi_output_etichettato(self, agente: str, testo: str) -> None:
        """Aggiunge un blocco leggibile con icona, data, ora e autore."""
        icons = (
            ("Utente", "💬"),
            ("Pianificatore", "🧭"),
            ("Orchestratore", "⚙️"),
            ("Agente di ricerca", "🔎"),
            ("Agente creazione file", "📝"),
            ("Agente di creazione file", "📝"),
            ("Agente sintetizzatore", "🧩"),
            ("Script", "💻"),
        )
        icon = next(
            (symbol for name, symbol in icons if agente.startswith(name)),
            "🔹",
        )
        timestamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S")
        lines = testo.strip().splitlines() or [""]
        description = "\n".join(f"   {line}" for line in lines)
        self._aggiungi_output(
            f"{icon} {timestamp} · {agente}\n{description}\n\n"
        )


if __name__ == "__main__":
    root = tk.Tk()
    AgenteGUI(root)
    root.mainloop()