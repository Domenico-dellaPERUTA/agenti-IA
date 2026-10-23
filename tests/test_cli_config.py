"""Test della configurazione guidata della CLI e di ``AI.provider_setup``."""

from io import StringIO
import json
import unittest
from unittest import mock

import agente
from AI import LLMProvider
from AI.branding import LOGO_ASCII, TAGLINE
from AI.provider_setup import (
    crea_provider,
    descrivi_provider,
    elenca_modelli,
    modelli_ollama,
)


class InputFinto:
    """Simula `input()`: consuma le risposte date, poi restituisce vuoto."""

    def __init__(self, *risposte: str):
        self.risposte = iter(risposte)

    def __call__(self, prompt: str) -> str:
        try:
            return next(self.risposte)
        except StopIteration:
            return ""


def _nessuna_stampa(*_args) -> None:
    """Scarta l'output testuale del wizard durante i test."""


class ProviderSetupTests(unittest.TestCase):
    def test_ollama_senza_host_e_con_host(self):
        provider = crea_provider("ollama", {"modello": "qwen3:4b", "base_url": " "})
        self.assertIsInstance(provider, LLMProvider)
        self.assertIsNone(provider.host)
        provider = crea_provider(
            "ollama", {"modello": "qwen3:4b", "base_url": "http://lan:11434"}
        )
        self.assertEqual(provider.host, "http://lan:11434")

    def test_lmstudio_ricade_sul_localhost(self):
        provider = crea_provider("lmstudio", {"modello": "x", "base_url": ""})
        self.assertIn("localhost:1234", str(provider.client.base_url))

    def test_openai_senza_chiave_e_rifiutato(self):
        with mock.patch.dict("os.environ", {"OPENAI_API_KEY": ""}):
            with self.assertRaisesRegex(ValueError, "OPENAI_API_KEY"):
                crea_provider(
                    "openai", {"modello": "gpt-4o-mini", "api_key": "", "base_url": ""}
                )

    def test_modello_vuoto_e_rifiutato(self):
        for scheda in ("lmstudio", "ollama", "openai"):
            with self.assertRaises(ValueError):
                crea_provider(scheda, {"modello": " "})

    def test_modelli_ollama_da_api_tags(self):
        corpo = json.dumps(
            {"models": [{"name": "qwen3:4b"}, {"name": "glm-5.3-flash:cloud"}]}
        ).encode()
        finto = mock.MagicMock()
        finto.read.return_value = corpo
        finto.__enter__.return_value = finto
        with mock.patch("urllib.request.urlopen", return_value=finto):
            nomi = modelli_ollama("")

        self.assertEqual(nomi, ["qwen3:4b", "glm-5.3-flash:cloud"])

    def test_modelli_ollama_errore_leggibile(self):
        import urllib.error

        with mock.patch(
            "urllib.request.urlopen",
            side_effect=urllib.error.URLError("Connection refused"),
        ):
            with self.assertRaisesRegex(ValueError, "non raggiungibile"):
                modelli_ollama("http://127.0.0.1:11434")

    def test_elenca_modelli_lmstudio_legge_il_client(self):
        finto_client = mock.MagicMock()
        voce = mock.MagicMock()
        voce.id = "modello-a"
        finto_client.models.list.return_value.data = [voce]
        finto_provider = mock.MagicMock(spec=["client"])
        finto_provider.client = finto_client
        with mock.patch(
            "AI.provider_setup.crea_provider", return_value=finto_provider
        ):
            nomi = elenca_modelli(
                "lmstudio", {"modello": "x", "base_url": "http://h:1/v1"}
            )

        self.assertEqual(nomi, ["modello-a"])

    def test_descrivi_provider(self):
        self.assertIn(
            "qwen3:4b",
            descrivi_provider("ollama", {"modello": "qwen3:4b", "base_url": ""}),
        )


class WizardCliTests(unittest.TestCase):
    def test_flusso_predefinito_con_soli_invii(self):
        leggi = InputFinto()  # tutto vuoto: provider 1, host e modello predefiniti
        stampato = []
        scheda, config = agente.configura_provider(
            leggi=leggi,
            stampa=lambda *a: stampato.append(" ".join(map(str, a))),
            elenca=lambda scheda, config: ["qwen3-4b-2507", "altro"],
        )

        self.assertEqual(scheda, "lmstudio")
        self.assertEqual(config["base_url"], "http://localhost:1234/v1")
        self.assertEqual(config["modello"], "ornith-1.5-9b-uncensored")
        self.assertTrue(
            any("Connessione riuscita" in riga for riga in stampato)
        )

    def test_ollama_con_selezione_numerata_del_modello(self):
        leggi = InputFinto("2", "", "1")  # provider 2, host predefinito, modello 1
        scheda, config = agente.configura_provider(
            leggi=leggi,
            stampa=_nessuna_stampa,
            elenca=lambda scheda, config: ["qwen3:4b", "glm-5.3-flash:cloud"],
        )

        self.assertEqual(scheda, "ollama")
        self.assertEqual(config["modello"], "qwen3:4b")

    def test_modello_digitato_fuori_elenco_con_avviso(self):
        leggi = InputFinto("2", "", "un-modello-speciale")
        stampato = []
        scheda, config = agente.configura_provider(
            leggi=leggi,
            stampa=lambda *a: stampato.append(" ".join(map(str, a))),
            elenca=lambda scheda, config: ["qwen3:4b"],
        )

        self.assertEqual(config["modello"], "un-modello-speciale")
        self.assertTrue(any("non compare" in riga for riga in stampato))

    def test_openai_chiede_chiave_mascherata_e_mantiene_predefinito(self):
        leggi = InputFinto("3", "", "")
        nascosto = InputFinto("sk-test-123")
        scheda, config = agente.configura_provider(
            leggi=leggi,
            leggi_nascosto=nascosto,
            stampa=_nessuna_stampa,
            elenca=lambda scheda, config: ["gpt-4o-mini"],
        )

        self.assertEqual(scheda, "openai")
        self.assertEqual(config["api_key"], "sk-test-123")
        self.assertEqual(config["modello"], "gpt-4o-mini")

    def test_errore_verifica_poi_cambio_provider(self):
        stato = {"chiamate": 0}

        def elenca_fallace(scheda, config):
            stato["chiamate"] += 1
            if stato["chiamate"] == 1:
                raise ValueError("server spento")
            return ["qwen3:4b"]

        # lmstudio fallisce → P (cambia provider) → ollama → successo
        leggi = InputFinto("1", "", "P", "2", "", "")
        scheda, config = agente.configura_provider(
            leggi=leggi, stampa=_nessuna_stampa, elenca=elenca_fallace
        )

        self.assertEqual(scheda, "ollama")
        self.assertEqual(config["modello"], "qwen3:4b")

    def test_errore_verifica_poi_continua_comunque(self):
        def elenca_sempre_fallace(scheda, config):
            raise ValueError("server spento")

        leggi = InputFinto("1", "", "C")
        scheda, config = agente.configura_provider(
            leggi=leggi, stampa=_nessuna_stampa, elenca=elenca_sempre_fallace
        )

        self.assertEqual(scheda, "lmstudio")
        self.assertEqual(config["modello"], "ornith-1.5-9b-uncensored")

    def test_scelta_provider_non_valida_ripropone_la_domanda(self):
        leggi = InputFinto("9", "paperino", "2", "", "")
        stampato = []
        scheda, config = agente.configura_provider(
            leggi=leggi,
            stampa=lambda *a: stampato.append(" ".join(map(str, a))),
            elenca=lambda scheda, config: ["qwen3:4b"],
        )

        self.assertEqual(scheda, "ollama")
        self.assertTrue(
            sum("Scelta non valida" in riga for riga in stampato) == 2
        )

    def test_riprova_dopo_errore_nella_stessa_scheda(self):
        stato = {"chiamate": 0}

        def elenca_riparabile(scheda, config):
            stato["chiamate"] += 1
            if stato["chiamate"] == 1:
                raise ValueError("server non pronto")
            return ["qwen3:4b"]

        leggi = InputFinto("2", "", "R", "")
        scheda, config = agente.configura_provider(
            leggi=leggi, stampa=_nessuna_stampa, elenca=elenca_riparabile
        )

        self.assertEqual(stato["chiamate"], 2)
        self.assertEqual(scheda, "ollama")


class LoopReplTests(unittest.TestCase):
    """Il `main()` della CLI resta vivo fino a 'chiudi'/'esci' (o EOF)."""

    def _esegui_main(self, risposte, run_result):
        finto_orchestratore = mock.MagicMock()
        finto_orchestratore.run.return_value = run_result
        finta_app = mock.MagicMock()
        finta_app.create_orchestrator.return_value = finto_orchestratore
        stampato = StringIO()
        side_effect = EOFError if risposte == "EOF" else list(risposte)
        with (
            mock.patch.object(
                agente,
                "configura_provider",
                return_value=("lmstudio", {"modello": "dummy", "base_url": ""}),
            ),
            mock.patch.object(agente, "crea_provider", return_value=object()),
            mock.patch.object(agente, "AI", return_value=finta_app),
            mock.patch("builtins.input", side_effect=side_effect),
            mock.patch("sys.stdout", stampato),
        ):
            agente.main()
        return finto_orchestratore, stampato.getvalue()

    def _risultato(self):
        risultato = mock.MagicMock()
        risultato.answer = "ok"
        risultato.failures = []
        risultato.tasks = []
        return risultato

    def test_piu_richieste_fino_a_esci(self):
        orchestratore, output = self._esegui_main(
            ["prima domanda", "seconda domanda", "esci"], self._risultato()
        )

        self.assertEqual(orchestratore.run.call_count, 2)
        self.assertIn("Chiusura dell'applicazione.", output)

    def test_chiudi_non_dipende_dalle_maiuscole(self):
        orchestratore, output = self._esegui_main(
            ["domanda", "CHIUDI"], self._risultato()
        )

        self.assertEqual(orchestratore.run.call_count, 1)

    def test_righe_vuote_saltate_e_poi_uscita_pulita(self):
        orchestratore, output = self._esegui_main(
            ["", "domanda", "", "esci"], self._risultato()
        )

        self.assertEqual(orchestratore.run.call_count, 1)
        self.assertIn("Chiusura dell'applicazione.", output)

    def test_eof_chiude_il_loop_senza_eccezioni(self):
        orchestratore, output = self._esegui_main("EOF", self._risultato())

        self.assertEqual(orchestratore.run.call_count, 0)
        self.assertIn("Fine dell'input", output)


class BrandingTests(unittest.TestCase):
    def test_logo_e_ascii_puro_multilinea_e_riconoscibile(self):
        from AI.branding import LOGO_ASCII, TAGLINE

        righe = LOGO_ASCII.split("\n")
        self.assertGreaterEqual(len(righe), 5)
        self.assertTrue(all(ord(c) < 128 for c in LOGO_ASCII))
        self.assertLessEqual(max(len(r) for r in righe), 80)
        self.assertTrue(any("/  _/" in riga for riga in righe))  # scritta IA
        self.assertIn("sandbox", TAGLINE)

    def test_cli_mostra_il_logo_all_avvio(self):
        _, output = LoopReplTests()._esegui_main(["esci"], self._risultato_repl())

        self.assertIn(LOGO_ASCII.split("\n")[0], output)
        self.assertIn("orchestratore multi-agente", output)

    @staticmethod
    def _risultato_repl():
        risultato = mock.MagicMock()
        risultato.answer = "ok"
        risultato.failures = []
        risultato.tasks = []
        return risultato


if __name__ == "__main__":
    unittest.main()
