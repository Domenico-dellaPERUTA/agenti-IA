"""Marchio dell'applicazione, condiviso tra CLI e GUI.

Il logo è ASCII puro: si renderizza in qualsiasi terminale anche senza
supporto Unicode, e i caratteri non devono essere toccati a mano
(l'allineamento è parte del disegno).
"""

LOGO_ASCII = "\n".join(
    riga.rstrip()
    for riga in r"""
    ___                    __  _    _______
   /   | ____ ____  ____  / /_(_)  /  _/   |
  / /| |/ __ `/ _ \/ __ \/ __/ /   / // /| |
 / ___ / /_/ /  __/ / / / /_/ /  _/ // ___ |
/_/  |_\__, /\___/_/ /_/\__/_/  /___/_/  |_|
      /____/
""".strip("\n").split("\n")
)

TAGLINE = "orchestratore multi-agente · sandbox · web in sola lettura"
