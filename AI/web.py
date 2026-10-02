"""Ricerca e lettura controllata di testo HTML pubblico via HTTPS GET.

Questo modulo non è un browser: valida URL e DNS, rifiuta host non pubblici,
fissa la connessione all'indirizzo validato e scarica soltanto HTML con limiti
di dimensione, richieste e reindirizzamenti.
"""

from __future__ import annotations

from dataclasses import dataclass
from email.message import Message
from html.parser import HTMLParser
import http.client
import ipaddress
import json
import re
import socket
import ssl
from urllib.parse import parse_qs, quote_plus, unquote, urljoin, urlsplit, urlunsplit


# Endpoint e limiti centralizzati: rendono esplicite le risorse massime che
# una singola istanza può consumare durante una ricerca o lettura.
SEARCH_URL = "https://html.duckduckgo.com/html/"
LITE_SEARCH_URL = "https://lite.duckduckgo.com/lite/"
MAX_QUERY_LENGTH = 300
MAX_RESULTS = 5
MAX_PAGE_BYTES = 1_000_000
MAX_REDIRECTS = 3
MAX_TOTAL_REQUESTS = 10
REQUEST_TIMEOUT = 10
USER_AGENT = "AgentiIA/1.0 (read-only text research)"

_EMAIL_PATTERN = re.compile(r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b")
_TOKEN_PATTERN = re.compile(
    r"\b(?:sk-[A-Za-z0-9_-]{16,}|gh[pousr]_[A-Za-z0-9]{20,}|"
    r"github_pat_[A-Za-z0-9_]{20,}|AKIA[0-9A-Z]{16}|xox[baprs]-[A-Za-z0-9-]{10,})\b"
)
_LOCAL_PATH_PATTERN = re.compile(
    r"(?:/Users/[^/\s]+|/home/[^/\s]+|[A-Za-z]:\\Users\\[^\\\s]+)",
    re.IGNORECASE,
)
_SECRET_ASSIGNMENT_PATTERN = re.compile(
    r"\b(?:password|passwd|api[_ -]?key|access[_ -]?token|token|secret|"
    r"credential|session[_ -]?id)\s*[:=]\s*\S+",
    re.IGNORECASE,
)
_SENSITIVE_QUERY_KEY_PATTERN = re.compile(
    r"(?:password|passwd|api[_-]?key|access[_-]?token|token|secret|"
    r"credential|session[_-]?id|auth)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class WebSource:
    """Descrive una fonte web e, facoltativamente, il suo estratto testuale.

    L'oggetto è immutabile perché viene condiviso tra risultati di ricerca e
    raccolta delle citazioni della sessione.
    """

    title: str
    url: str
    snippet: str = ""


class _SearchResultsParser(HTMLParser):
    """Estrae titoli, URL e snippet dai due layout HTML di DuckDuckGo.

    Le liste di tag e livelli di annidamento servono a ignorare script e stili
    e a delimitare correttamente risultati composti da elementi HTML nidificati.
    """

    def __init__(self) -> None:
        """Inizializza lo stato usato dai callback incrementali di HTMLParser."""
        super().__init__(convert_charrefs=True)
        self.results: list[WebSource] = []
        # Stato per il formato HTML standard: profondità div e buffer del
        # risultato attualmente aperto consentono di chiuderlo al div giusto.
        self._inside_result = False
        self._result_div_depth = 0
        self._title_active = False
        self._snippet_stack: list[str] = []
        self._title: list[str] = []
        self._snippet: list[str] = []
        self._url = ""
        # Stato separato per il layout Lite, che usa link e snippet differenti.
        self._lite_title_active = False
        self._lite_snippet_stack: list[str] = []
        self._lite_title: list[str] = []
        self._lite_snippet: list[str] = []
        self._lite_url = ""
        # Ignora annidamenti dentro contenuti eseguibili o non visibili.
        self._suppress_depth = 0
        self._suppressed_tags: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        """Aggiorna i marcatori di parsing all'apertura di ogni tag HTML."""
        attributes = dict(attrs)
        classes = set((attributes.get("class") or "").split())

        if tag in {"script", "style", "svg", "noscript"}:
            self._suppressed_tags.append(tag)
            self._suppress_depth += 1
            return
        if self._suppress_depth:
            return

        if tag == "a" and "result-link" in classes:
            self._lite_url = attributes.get("href") or ""
            self._lite_title = []
            self._lite_snippet = []
            self._lite_title_active = True
        if "result-snippet" in classes:
            self._lite_snippet_stack = [tag]
        elif self._lite_snippet_stack:
            self._lite_snippet_stack.append(tag)

        if tag == "div":
            if self._inside_result:
                self._result_div_depth += 1
            elif "result" in classes:
                self._inside_result = True
                self._result_div_depth = 1
                self._title = []
                self._snippet = []
                self._url = ""
        if self._inside_result and tag == "a" and "result__a" in classes:
            self._url = attributes.get("href") or ""
            self._title_active = True
        if self._inside_result and "result__snippet" in classes:
            self._snippet_stack = [tag]
        elif self._snippet_stack:
            self._snippet_stack.append(tag)

    def handle_endtag(self, tag: str) -> None:
        """Chiude le sezioni attive e materializza il risultato completo."""
        if self._suppressed_tags:
            if tag == self._suppressed_tags[-1]:
                self._suppressed_tags.pop()
                self._suppress_depth -= 1
            return

        if self._lite_title_active and tag == "a":
            self._lite_title_active = False
        if self._lite_snippet_stack:
            for index in range(len(self._lite_snippet_stack) - 1, -1, -1):
                if self._lite_snippet_stack[index] == tag:
                    del self._lite_snippet_stack[index:]
                    if not self._lite_snippet_stack:
                        url = _unwrap_search_url(self._lite_url)
                        title = " ".join(" ".join(self._lite_title).split())
                        snippet = " ".join(" ".join(self._lite_snippet).split())
                        if url and title:
                            self.results.append(
                                WebSource(title=title, url=url, snippet=snippet)
                            )
                        self._lite_url = ""
                        self._lite_title = []
                        self._lite_snippet = []
                    break

        if self._title_active and tag == "a":
            self._title_active = False
        if self._snippet_stack:
            for index in range(len(self._snippet_stack) - 1, -1, -1):
                if self._snippet_stack[index] == tag:
                    del self._snippet_stack[index:]
                    break
        if self._inside_result and tag == "div":
            self._result_div_depth -= 1
            if self._result_div_depth == 0:
                self._inside_result = False
                url = _unwrap_search_url(self._url)
                title = " ".join(" ".join(self._title).split())
                snippet = " ".join(" ".join(self._snippet).split())
                if url and title:
                    self.results.append(WebSource(title=title, url=url, snippet=snippet))

    def handle_data(self, data: str) -> None:
        """Accoda testo solo alla parte di risultato attualmente selezionata."""
        if self._suppress_depth:
            return
        if self._lite_title_active:
            self._lite_title.append(data)
        elif self._lite_snippet_stack:
            self._lite_snippet.append(data)
        elif self._title_active:
            self._title.append(data)
        elif self._snippet_stack:
            self._snippet.append(data)


class _VisibleTextParser(HTMLParser):
    """Converte una pagina HTML in testo, ignorando codice e contenuti nascosti."""

    def __init__(self) -> None:
        """Prepara il buffer testuale e la pila dei tag da escludere."""
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self._suppressed_tags: list[str] = []

    def handle_starttag(self, tag: str, _attrs: list[tuple[str, str | None]]) -> None:
        """Inizia a sopprimere il testo contenuto nei tag non visibili."""
        if tag in {"script", "style", "svg", "noscript", "template"}:
            self._suppressed_tags.append(tag)

    def handle_endtag(self, tag: str) -> None:
        """Termina la soppressione o inserisce separatori tra blocchi testuali."""
        if self._suppressed_tags and tag == self._suppressed_tags[-1]:
            self._suppressed_tags.pop()
        elif not self._suppressed_tags and tag in {
            "p", "div", "br", "li", "tr", "h1", "h2", "h3", "h4",
        }:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        """Conserva testo HTML solo quando non è dentro un tag soppresso."""
        if not self._suppressed_tags:
            self.parts.append(data)


class _PinnedHTTPSConnection(http.client.HTTPSConnection):
    """Connessione HTTPS che usa un IP già validato mantenendo TLS sul dominio."""

    def __init__(self, hostname: str, address: str, timeout: int):
        """Imposta hostname TLS originale e indirizzo IP fissato per il socket."""
        super().__init__(
            hostname,
            port=443,
            timeout=timeout,
            context=ssl.create_default_context(),
        )
        self._pinned_address = address

    def connect(self) -> None:
        """Apre il socket verso l'IP validato e verifica il certificato TLS host."""
        sock = socket.create_connection(
            (self._pinned_address, self.port),
            self.timeout,
            self.source_address,
        )
        self.sock = self._context.wrap_socket(sock, server_hostname=self.host)


class InternetAccess:
    """Ricerca e lettura web testuale con sole richieste HTTPS GET anonime.

    Le fonti consultate e il contatore delle richieste appartengono all'istanza,
    quindi la stessa istanza mantiene contesto e limiti durante una sessione.
    """

    def __init__(self) -> None:
        """Inizializza il registro delle fonti e il budget di rete della sessione."""
        self._sources: dict[str, WebSource] = {}
        self._request_count = 0

    def web_search(self, query: str) -> str:
        """Cerca testo pubblico e restituisce al modello titoli, snippet e URL.

        La query viene validata e codificata nell'URL di DuckDuckGo. Se il
        layout HTML principale non produce risultati si tenta il layout Lite.
        Le fonti trovate restano registrate per la citazione nella risposta.
        """
        query = self._validate_query(query)
        results = self._parse_search_results(
            f"{SEARCH_URL}?q={quote_plus(query)}"
        )
        if not results:
            results = self._parse_search_results(
                f"{LITE_SEARCH_URL}?q={quote_plus(query)}"
            )

        if not results:
            raise RuntimeError(
                "La ricerca non ha restituito risultati leggibili. "
                "Il motore potrebbe aver cambiato formato o temporaneamente limitato le richieste."
            )

        for source in results:
            self._sources[source.url] = source

        return json.dumps(
            [
                {
                    "title": source.title,
                    "url": source.url,
                    "snippet": source.snippet,
                }
                for source in results
            ],
            ensure_ascii=False,
            indent=2,
        )

    def _parse_search_results(self, url: str) -> list[WebSource]:
        """Scarica un motore di ricerca e normalizza un massimo di risultati unici."""
        body, content_type, _final_url = self._fetch_text(url)
        if content_type not in {"text/html", "application/xhtml+xml"}:
            raise ValueError("Il motore di ricerca ha restituito un formato non testuale.")

        parser = _SearchResultsParser()
        parser.feed(body)
        parser.close()
        results = []
        seen: set[str] = set()
        for source in parser.results:
            if source.url in seen:
                continue
            seen.add(source.url)
            safe_source = WebSource(
                title=_sanitize_text(source.title),
                url=source.url,
                snippet=_sanitize_text(source.snippet),
            )
            results.append(safe_source)
            if len(results) == MAX_RESULTS:
                break
        return results

    def read_webpage(self, url: str) -> str:
        """Legge il solo testo visibile di una pagina HTML HTTPS pubblica.

        Non interpreta JavaScript e non richiede immagini, fogli di stile,
        moduli o altre risorse. Salva la pagina letta tra le fonti della sessione.
        """
        body, content_type, final_url = self._fetch_text(url)
        if content_type not in {"text/html", "application/xhtml+xml"}:
            raise ValueError("Sono consentite solo pagine HTML pubbliche.")

        source = WebSource(title=urlsplit(final_url).hostname or final_url, url=final_url)
        self._sources[final_url] = source

        parser = _VisibleTextParser()
        parser.feed(body)
        text = "\n".join(
            line.strip()
            for line in "".join(parser.parts).splitlines()
            if line.strip()
        )

        if not text:
            raise ValueError("La pagina non contiene testo leggibile.")
        return f"Fonte: {final_url}\n\n{_sanitize_text(text)}"

    def sources(self) -> list[tuple[str, str]]:
        """Restituisce le coppie titolo/URL raccolte nell'ordine di inserimento."""
        return [(source.title, source.url) for source in self._sources.values()]

    @staticmethod
    def _validate_query(query: str) -> str:
        """Normalizza la query e rifiuta input vuoto, eccessivo o sensibile."""
        if not isinstance(query, str):
            raise TypeError("La ricerca web richiede una query testuale.")
        query = query.strip()
        if not query or len(query) > MAX_QUERY_LENGTH:
            raise ValueError(f"La query deve contenere da 1 a {MAX_QUERY_LENGTH} caratteri.")
        if any(pattern.search(query) for pattern in (
            _EMAIL_PATTERN,
            _TOKEN_PATTERN,
            _LOCAL_PATH_PATTERN,
            _SECRET_ASSIGNMENT_PATTERN,
        )):
            raise ValueError(
                "La query sembra contenere dati personali o credenziali. "
                "Rimuovili prima di inviare una ricerca web."
            )
        return query

    def _fetch_text(self, url: str) -> tuple[str, str, str]:
        """Segue solo redirect limitati e restituisce testo HTML decodificato.

        Ogni destinazione viene sottoposta nuovamente alla validazione URL
        prima della connessione, prevenendo redirect verso host locali.
        """
        current_url = url
        for redirect_number in range(MAX_REDIRECTS + 1):
            status, headers, body = self._request_once(current_url)
            if status in {301, 302, 303, 307, 308}:
                location = headers.get("location")
                if not location:
                    raise RuntimeError("Il reindirizzamento non specifica una destinazione.")
                if redirect_number == MAX_REDIRECTS:
                    raise RuntimeError("La pagina ha superato il limite di reindirizzamenti.")
                current_url = urljoin(current_url, location)
                continue
            if status < 200 or status >= 300:
                raise RuntimeError(f"La richiesta web è fallita con stato HTTP {status}.")

            message = Message()
            message["content-type"] = headers.get("content-type", "")
            content_type = message.get_content_type().lower()
            if content_type not in {"text/html", "application/xhtml+xml"}:
                raise ValueError(
                    f"Tipo di contenuto non consentito: {content_type or 'sconosciuto'}."
                )

            charset = message.get_content_charset() or "utf-8"
            try:
                decoded_body = body.decode(charset, errors="replace")
            except LookupError:
                decoded_body = body.decode("utf-8", errors="replace")
            return decoded_body, content_type, current_url

        raise RuntimeError("Impossibile completare la richiesta web.")

    def _request_once(self, url: str) -> tuple[int, dict[str, str], bytes]:
        """Esegue un singolo GET con host validato e limiti sulla risposta."""
        parsed, hostname, address = self._validate_public_url(url)
        if self._request_count >= MAX_TOTAL_REQUESTS:
            raise RuntimeError(
                f"La ricerca web ha raggiunto il limite di {MAX_TOTAL_REQUESTS} richieste."
            )
        path = urlunsplit(("", "", parsed.path or "/", parsed.query, ""))
        connection = _PinnedHTTPSConnection(hostname, address, REQUEST_TIMEOUT)
        try:
            self._request_count += 1
            connection.request(
                "GET",
                path,
                headers={
                    "Accept": "text/html, application/xhtml+xml",
                    "User-Agent": USER_AGENT,
                    "Connection": "close",
                },
            )
            response = connection.getresponse()
            headers = {
                key.lower(): value for key, value in response.getheaders()
            }
            status = response.status
            if status in {301, 302, 303, 307, 308} or not 200 <= status < 300:
                return status, headers, b""

            message = Message()
            message["content-type"] = headers.get("content-type", "")
            content_type = message.get_content_type().lower()
            if content_type not in {"text/html", "application/xhtml+xml"}:
                raise ValueError(
                    f"Tipo di contenuto non consentito: {content_type or 'sconosciuto'}."
                )
            if "attachment" in headers.get("content-disposition", "").lower():
                raise ValueError("I download di file sono disabilitati.")

            content_length = response.getheader("Content-Length")
            if content_length is not None:
                try:
                    declared_length = int(content_length)
                except ValueError as error:
                    raise ValueError(
                        "La risposta web dichiara una dimensione non valida."
                    ) from error
                if declared_length < 0:
                    raise ValueError("La risposta web dichiara una dimensione non valida.")
                if declared_length > MAX_PAGE_BYTES:
                    raise ValueError("La risposta web supera il limite di dimensione.")

            body = response.read(MAX_PAGE_BYTES + 1)
            if len(body) > MAX_PAGE_BYTES:
                raise ValueError("La risposta web supera il limite di dimensione.")
            return status, headers, body
        finally:
            connection.close()

    @staticmethod
    def _validate_public_url(url: str):
        """Convalida schema, credenziali, porta e tutti gli IP risolti.

        Restituisce URL normalizzato, hostname e primo IP pubblico. Fissare
        l'IP evita che una seconda risoluzione DNS cambi destinazione al connect.
        """
        if not isinstance(url, str) or not url or len(url) > 2048:
            raise ValueError("L'URL deve essere valido e non superare 2048 caratteri.")
        parsed = urlsplit(url)
        if parsed.scheme.lower() != "https":
            raise ValueError("Sono consentiti esclusivamente URL HTTPS pubblici.")
        if parsed.username is not None or parsed.password is not None:
            raise ValueError("Gli URL con credenziali incorporate non sono consentiti.")
        if _contains_sensitive_data(url):
            raise ValueError(
                "L'URL sembra contenere dati personali o credenziali e non verrà richiesto."
            )
        if parsed.port not in (None, 443):
            raise ValueError("Sono consentite solo connessioni HTTPS sulla porta standard.")
        if not parsed.hostname:
            raise ValueError("L'URL deve contenere un nome host.")

        hostname = parsed.hostname.encode("idna").decode("ascii").lower()
        try:
            address = ipaddress.ip_address(hostname)
        except ValueError:
            try:
                records = socket.getaddrinfo(
                    hostname,
                    443,
                    type=socket.SOCK_STREAM,
                )
            except OSError as error:
                raise RuntimeError(f"Impossibile risolvere l'host '{hostname}'.") from error
            addresses = list(dict.fromkeys(record[4][0] for record in records))
        else:
            addresses = [str(address)]

        if not addresses:
            raise RuntimeError(f"L'host '{hostname}' non ha indirizzi pubblici.")
        for candidate in addresses:
            try:
                ip = ipaddress.ip_address(candidate)
            except ValueError as error:
                raise ValueError(
                    "La risoluzione DNS ha restituito un indirizzo non valido."
                ) from error
            if not ip.is_global:
                raise PermissionError("Sono vietati gli indirizzi locali o non pubblici.")

        normalized_host = f"[{hostname}]" if ":" in hostname else hostname
        normalized = urlunsplit(
            ("https", normalized_host, parsed.path or "/", parsed.query, "")
        )
        return urlsplit(normalized), hostname, addresses[0]


def _contains_sensitive_data(value: str) -> bool:
    """Individua segreti nel testo decodificato o nelle chiavi dei parametri URL."""
    decoded_value = unquote(value)
    parsed = urlsplit(decoded_value)
    sensitive_query_key = any(
        _SENSITIVE_QUERY_KEY_PATTERN.search(key)
        for key in parse_qs(parsed.query, keep_blank_values=True)
    )
    return sensitive_query_key or any(
        pattern.search(decoded_value)
        for pattern in (
            _EMAIL_PATTERN,
            _TOKEN_PATTERN,
            _LOCAL_PATH_PATTERN,
            _SECRET_ASSIGNMENT_PATTERN,
        )
    )


def _sanitize_text(value: str) -> str:
    """Oscura credenziali, percorsi personali ed email prima di esporre testo."""
    value = _SECRET_ASSIGNMENT_PATTERN.sub("[dato sensibile rimosso]", value)
    value = _TOKEN_PATTERN.sub("[credenziale rimossa]", value)
    value = _LOCAL_PATH_PATTERN.sub("[percorso locale rimosso]", value)
    return _EMAIL_PATTERN.sub("[indirizzo email rimosso]", value)


def _unwrap_search_url(url: str) -> str:
    """Rimuove il redirect interno di DuckDuckGo e accetta soltanto URL HTTPS."""
    if not url:
        return ""
    if url.startswith("//"):
        url = f"https:{url}"
    parsed = urlsplit(url)
    if parsed.hostname and (
        parsed.hostname == "duckduckgo.com"
        or parsed.hostname.endswith(".duckduckgo.com")
    ):
        target = parse_qs(parsed.query).get("uddg", [""])[0]
        if target:
            url = target
    parsed = urlsplit(url)
    if parsed.scheme.lower() != "https" or not parsed.hostname:
        return ""
    if parsed.username is not None or parsed.password is not None:
        return ""
    if _contains_sensitive_data(url):
        return ""
    return urlunsplit(("https", parsed.netloc, parsed.path or "/", parsed.query, ""))
