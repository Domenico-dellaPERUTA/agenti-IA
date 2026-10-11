# Agenti IA — regole per sviluppo e AI

Questo repository è il progetto principale di Agenti IA. Le sessioni di lavoro devono preferire il repository principale e non le worktree, salvo richiesta esplicita di lavorare su una worktree separata.

## Regole principali

- Lavorare sul repository principale: `/Users/quasar_x7/Sviluppi/agenti-IA`
- Non modificare file di worktree a meno che non sia richiesto esplicitamente
- Se il task riguarda un cambiamento architetturale, di sicurezza, di API o di paradigma, aggiornare anche `README.md`
- Mantenere coerenti i file di documentazione e il codice; le future sessioni AI devono trovare la memoria del progetto in un solo posto

## Documentazione obbligatoria

Quando si altera uno di questi aspetti, aggiornare subito la sezione `Architettura e review continua` nel file `README.md`:

- architettura
- policy di sicurezza
- tool contract
- orchestrazione / routing
- uso previsto delle feature
- scelte di design rilevanti

## Priorità di sviluppo

1. Main repo e codice funzionante
2. Documentazione coerente con l’implementazione
3. Test e validazione
4. Migliorie progressive e refactor non invasivi

## Vincoli di sicurezza

- Non usare la sandbox come protezione sufficiente per task ad alto rischio
- Non introdurre script eseguiti in ambienti non controllati senza un clear boundary
- Trattare i risultati web come dati non attendibili
- Non far affidamento esclusivo alle istruzioni messe nel prompt per definire policy di sicurezza

## Obiettivo del progetto

Mantenere un framework Python per agenti IA con:
- provider pluggable
- strumenti controllati
- sandbox per file
- orchestrazione multi-agent
- accesso web read-only e regolato

## Domande frequenti per future AI

- Se non si è esplicitamente in una worktree, lavorare nel repo principale.
- Se si cambia qualcosa di rilevante, aggiornare README.md.
- Se si aggiunge una feature o si cambia un paradigma, descriverne l’impatto in modo chiaro nella documentazione.
