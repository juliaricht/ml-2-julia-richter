# Portfolio-Bericht — Datenexperiment: Anomalieerkennung auf CFRP-Sensordaten

**Modul:** Internet of Production / KI und Anwendungen
**Hochschule:** Ostfalia Hochschule für angewandte Wissenschaften
**Dozent:** Prof. Dr. U. Triltsch
**Semester:** SoSe 26
**Studierende:r:** *(bitte ergänzen — Name, Matrikelnummer)*
**Datum der Einreichung:** *(bitte ergänzen)*

---

## Zusammenfassung

In diesem Datenexperiment wird ein Workflow zur **unüberwachten Anomalieerkennung** auf realen Produktions-Sensordaten aus der Carbon-Fiber-Reinforced-Polymer-Fertigung (CFRP) entwickelt, implementiert und reproduzierbar dokumentiert. Datengrundlage ist der Datensatz `5_cfrp.npz` aus dem öffentlichen [SmartManuAD-Repository](https://github.com/SmartManuAD/Smart-Manufacturing-AD) mit 52 268 Zeitpunkten und 49 Sensorkanälen, von denen lediglich 0,50 % als Anomalie gekennzeichnet sind. Vier Modelle aus der Bibliothek **PyOD** werden vergleichend trainiert: Isolation Forest, Local Outlier Factor (LOF), ECOD und ein **AutoEncoder** (PyTorch, CUDA). Die Entscheidungsschwelle wird ausschließlich auf dem Validierungs-Split anhand des F1-Werts bestimmt und anschließend auf dem unberührten Test-Split angewandt. Der AutoEncoder erzielt die beste Leistung (Test-ROC-AUC 0,990, Test-F1 0,642) und erkennt 17 von 26 echten Defekten bei 10 Fehlalarmen. Sämtliche Experimente werden mit **MLflow** versioniert; die vollständige Pipeline ist über plattformneutrale `.bat`-Skripte reproduzierbar.

---

## 1. Anwendungsfall im Kontext des Internet of Production (Lernziel 1)

### 1.1 Motivation

Die Fertigung von **carbonfaserverstärkten Kunststoffen (CFRP)** ist energie- und materialintensiv: Eine einzelne fehlerhafte Bauteilfertigung verursacht Materialverlust im hohen zwei- bis vierstelligen Euro-Bereich, da Faserprepregs nicht recycelbar sind und der Fertigungsprozess (Fiber Placement) eine hohe Prozessenergie bindet. Defekte sind dabei selten und werden oft erst nach Abschluss des Lagenaufbaus sichtbar — also genau dann, wenn jede Korrektur am teuersten ist.

Das **Internet of Production (IoP)** verspricht genau hier den größten Hebel: Wenn die ~49 Prozesssensoren der Anlage (Kraft, Temperatur, Position, Laser-Messungen) **während** der Fertigung als Datenstrom in ein KI-Modell laufen, kann eine Anomalie in dem Moment erkannt werden, in dem sie entsteht — nicht erst bei der Qualitätsprüfung des fertigen Teils. Damit schließt sich der klassische IoP-Regelkreis **Sensor → Daten → Modell → Entscheidung** in einer industriell relevanten Domäne.

### 1.2 Auswahl des Datensatzes

Der Datensatz wurde aus den auf den Vorlesungsfolien empfohlenen Quellen ausgewählt, konkret aus der Sammlung **SmartManuAD** (GitHub). Drei Eigenschaften qualifizieren ihn besonders für diesen Anwendungsfall:

1. **Reale Produktionsdaten** (keine synthetischen Sensorwerte), gekennzeichnet mit echten Anomalielabels.
2. **Hohe Sensordichte** (49 Kanäle) bei moderater Datenmenge (~20 MB) — das Projekt ist auf einem Studierenden-Laptop reproduzierbar, ohne auf Cloud-Ressourcen angewiesen zu sein.
3. **Starke Klassenungleichverteilung** (0,5 % Anomalien) — methodisch ehrlicher Stresstest für die in der Vorlesung behandelten Metriken (Accuracy ist hier irreführend; PR-AUC und F1 sind die richtigen Größen).

Der Datensatz ist damit ein authentischer Industriedaten-Showcase und gleichzeitig didaktisch wertvoll, weil er den Studierenden zwingt, die Metrik-Auswahl an die Problemstellung anzupassen, statt blind Accuracy zu berichten.

### 1.3 Forschungsfrage

> *Können klassische und neuronale Anomalie-Detektoren auf reinen Sensordaten der CFRP-Fiberplacement-Fertigung den Anteil seltener Defekte (0,5 %) hinreichend zuverlässig identifizieren, um eine prozessbegleitende Qualitätsregelung zu rechtfertigen?*

---

## 2. Beschreibung des Datensatzes (Lernziel 1, 2)

### 2.1 Quelle und Lizenz

| Eigenschaft | Wert |
|---|---|
| Name | CFRP (Carbon-Fiber Reinforced Polymer, Fiber-Placement-Prozess) |
| Quelle | <https://github.com/SmartManuAD/Smart-Manufacturing-AD> |
| Dateiname | `5_cfrp.npz` |
| Format | komprimiertes NumPy-Archiv mit den Arrays `X` (Merkmale) und `y` (Labels) |
| Lokaler Pfad | `data/raw/5_cfrp.npz` (20,0 MB) |

### 2.2 Statistische Eigenschaften

| Eigenschaft | Wert |
|---|---|
| Anzahl Samples | 52 268 |
| Anzahl Merkmale | 49 |
| Datentyp Merkmale | `float64`, ganzzahlige Werte, Wertebereich 0 – 614 |
| Bedeutung der Merkmale | anonymisierte Sensorkanäle (Kraft, Temperatur, Position, Laser); konkrete Einheiten in der SmartManuAD-Veröffentlichung **nicht** dokumentiert |
| Zielvariable | binäres Anomalielabel (0 = nominal, 1 = Anomalie) |
| Klassenbalance | 52 006 nominal (99,50 %) / 262 Anomalien (**0,50 %**) |
| Fehlende Werte | keine (kein NaN, kein ±∞) |
| Konstante / Null-Spalten | keine |
| Verteilungsform | schiefe, schwerschwänzige Verteilungen (Median ~15, Maximum bis ~614) |

### 2.3 Einschränkungen und Anmerkungen zur Maßeinheit

Die zugrunde liegende Publikation des SmartManuAD-Repositoriums anonymisiert die Sensorkanäle — eine 1-zu-1-Zuordnung von Spalten zu physikalischen Größen (N, °C, mm, V …) wird nicht veröffentlicht. Diese Einschränkung wird im Bericht offen ausgewiesen und beeinflusst die Wahl der Vorverarbeitung: ohne bekannte Einheiten ist eine featurespezifische Normalisierung nicht sinnvoll, weshalb global ein robuster Skalierer (s. Abschnitt 3) eingesetzt wird.

---

## 3. Datenvorverarbeitung und Aufteilung (Lernziel 2)

### 3.1 Aufteilung 70 / 20 / 10

Die Aufteilung folgt dem in der Aufgabenstellung empfohlenen Verhältnis von **70 % Training / 20 % Validierung / 10 % Test** und ist **stratifiziert** auf dem Anomalielabel. Stratifikation ist hier nicht optional, sondern methodisch zwingend: Bei einer Anomalierate von 0,5 % würde eine rein zufällige Aufteilung mit nennenswerter Wahrscheinlichkeit Splits erzeugen, die im Test-Set kaum Positivklassen enthalten — die Test-Metriken wären dann nicht aussagekräftig.

Die Aufteilung ist als zwei verkettete `train_test_split`-Aufrufe realisiert: zuerst werden 30 % als Holdout abgetrennt, danach das Holdout 2:1 in Validierung und Test geteilt. Die absoluten Anteile ergeben damit exakt 70 / 20 / 10. Der Zufallsseed (`42`) ist in [config.yaml](../config.yaml) zentral festgehalten.

```python
# src/data_loader.py
def split_70_20_10(X, y, seed=42, stratify=True):
    strat_first = y if stratify else None
    X_train, X_hold, y_train, y_hold = train_test_split(
        X, y, test_size=0.30, random_state=seed, stratify=strat_first,
    )
    strat_second = y_hold if stratify else None
    X_val, X_test, y_val, y_test = train_test_split(
        X_hold, y_hold, test_size=1.0/3.0, random_state=seed, stratify=strat_second,
    )
    return X_train, y_train, X_val, y_val, X_test, y_test
```

| Split | Anzahl Samples | Anzahl Anomalien | Anomalierate |
|---|---:|---:|---:|
| Training | 36 587 | 183 | 0,500 % |
| Validierung | 10 454 | 53 | 0,507 % |
| Test | 5 227 | 26 | 0,497 % |

### 3.2 Imputation fehlender Werte

`SimpleImputer` mit der Strategie **Median**, gefittet ausschließlich auf den Trainings-Split. In CFRP existieren keine fehlenden Werte; die Imputation bleibt als defensive Maßnahme im Code, falls die Pipeline später auf einen anderen SmartManuAD-Datensatz angewandt wird. Median (statt Mean) ist robuster gegenüber Ausreißern — was bei industriellen Sensordaten mit gelegentlichen Spikes wichtig ist.

### 3.3 Skalierung der Merkmale

`RobustScaler` (zentriert auf Median, skaliert auf Interquartilsabstand), gefittet ausschließlich auf den Trainings-Split und anschließend auf Validierung und Test angewandt. **Begründung:** Die Merkmale sind heavy-tailed (Median ~15, Maximum ~614 in derselben Spalte). Ein `StandardScaler` würde die Mittelwerte und Standardabweichungen massiv von einigen wenigen Ausreißern dominieren lassen — und genau diese Ausreißer sind potenziell die Anomalien, die wir erkennen wollen. `RobustScaler` immunisiert die Skalierungsparameter gegen die Zielklasse.

### 3.4 Leak-Vermeidung (zentrale Invariante)

Das gesamte Preprocessing wird **ausschließlich auf dem Trainings-Split gefittet** und dann mit den eingefrorenen Parametern auf Validierung und Test transformiert. Ein Verstoß gegen diese Regel würde Leakage erzeugen und die Testmetriken systematisch zu optimistisch ausweisen.

Konkret heißt das im Code: auf `X_train` wird `fit_transform` aufgerufen, auf `X_val` und `X_test` ausschließlich `transform`:

```python
# src/preprocessing.py
scaler = StandardScaler() if kind == "standard" else RobustScaler()
X_train_s = scaler.fit_transform(X_train)
X_val_s   = scaler.transform(X_val)
X_test_s  = scaler.transform(X_test)
```

Damit ist mechanisch ausgeschlossen, dass Statistiken aus Validierung oder Test in die Skalierungsparameter einfließen.

---

## 4. Modellpipeline (Lernziel 3)

### 4.1 Aufgabenfamilie

Die Aufgabe ist eine **unüberwachte Anomalieerkennung**. Die Modelle erhalten beim Training keinen Zugriff auf das Label `y`; sie lernen ausschließlich die geometrische bzw. statistische Form der Trainingsdaten und vergeben jedem Sample einen Anomalie-Score. Erst die nachgelagerte Schwellwertbestimmung übersetzt diesen Score in eine binäre Entscheidung.

### 4.2 Verwendete Modelle

Vier Detektoren aus der Bibliothek [PyOD](https://pyod.readthedocs.io/) wurden vergleichend evaluiert, um drei methodisch unterschiedliche Familien zuzüglich eines neuronalen Modells abzudecken:

| Modell | Methodische Familie | Hyperparameter (Default) |
|---|---|---|
| **Isolation Forest** | baumbasiert (zufällige Isolation seltener Punkte) | `n_estimators=200`, `contamination≈0,005` |
| **LOF** (Local Outlier Factor) | dichtebasiert (lokale Nachbarschaftsdichte) | `n_neighbors=20` |
| **ECOD** | statistisch (kumulative Verteilungen, parameterfrei) | — |
| **AutoEncoder** | neuronal (Rekonstruktionsfehler als Score) | 49→32→16→32→49, 30 Epochen, Dropout 0,2, PyTorch/CUDA |

### 4.3 Pipeline-Diagramm

```
data/raw/5_cfrp.npz
        │
        ▼  data_loader.load_cfrp + split_70_20_10
   stratifizierter Split 70 / 20 / 10
        │
        ▼  preprocessing.prepare    ← Imputer + Scaler nur auf TRAIN gefittet
   PreparedData (Dataclass)
        │
        ▼  modeling.run_baseline    ← für jeden PyOD-Detektor
   Fit auf TRAIN → Score auf VAL und TEST
        │
        ▼  evaluation.best_f1_threshold (auf VAL)
   eingefrorene Entscheidungsschwelle
        │
        ▼  evaluation.evaluate (auf TEST)
   Metrics → MLflow-Run (Parameter, Metriken, Plots, Joblib)
```

### 4.4 Wahl der Entscheidungsschwelle

Der Anomalie-Score ist eine reelle Zahl; für eine binäre Entscheidung wird eine Schwelle benötigt. Die Schwelle wird ausschließlich auf dem **Validierungs-Split** bestimmt, indem über die Precision-Recall-Kurve alle realisierten Schwellenwerte aufgezählt und derjenige mit maximalem F1-Wert ausgewählt wird. Anschließend wird diese Schwelle unverändert auf den Test-Split angewandt:

```python
# src/modeling.py — Kern des Modell-Runners
threshold, _ = best_f1_threshold(data.y_val, val_scores)
val_metrics  = evaluate(data.y_val,  val_scores,  threshold)
test_metrics = evaluate(data.y_test, test_scores, threshold)
```

```python
# src/evaluation.py — Schwellwertsuche auf der PR-Kurve
def best_f1_threshold(y_true, scores):
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    f1 = 2 * precision[:-1] * recall[:-1] / np.maximum(
        precision[:-1] + recall[:-1], 1e-12,
    )
    best = int(np.argmax(f1))
    return float(thresholds[best]), float(f1[best])
```

**Methodischer Hinweis:** Eine Bestimmung der Schwelle auf dem Test-Split wäre eine klassische Form des Data Leakage und ist deshalb durch die Aufrufstruktur oben mechanisch ausgeschlossen — der Test-Split sieht den Schwellwert nur als bereits eingefrorene Konstante.

### 4.5 Hyperparameter-Suche (Isolation Forest)

Für Isolation Forest wurde zusätzlich eine kleine Rastersuche (`sweep=iforest_v1`, 12 Kombinationen aus `n_estimators ∈ {100, 200, 400}`, `max_samples ∈ {256, 1024}`, `max_features ∈ {0,5, 1,0}`) durchgeführt. Jede Kombination ist ein eigenständiger MLflow-Run innerhalb desselben Experiments. Die Selektion des Gewinners erfolgte **nach Val-F1** (nicht Test-F1) — siehe Abschnitt 5.4 für die methodologisch ehrliche Interpretation des Ergebnisses.

---

## 5. Ergebnisse und Interpretation (Lernziel 4)

### 5.1 Vergleichstabelle (Test-Split, F1-getunter Schwellwert)

| Modell            | ROC-AUC | PR-AUC |    F1 | Precision | Recall |
|------------------|--------:|-------:|------:|----------:|-------:|
| **AutoEncoder**  | **0,990** | **0,684** | **0,642** | **0,630** | **0,654** |
| Isolation Forest |   0,915 |  0,448 | 0,439 |     0,600 |  0,346 |
| ECOD             |   0,846 |  0,286 | 0,314 |     0,320 |  0,308 |
| LOF              |   0,854 |  0,051 | 0,141 |     0,080 |  0,615 |

### 5.2 Wahl der Metriken (Lernziel 4 — Begründung)

**Accuracy** ist auf diesem Datensatz **nicht** aussagekräftig: ein triviales Modell, das stets „nominal" vorhersagt, erreicht eine Accuracy von 99,5 % und übersieht dennoch jeden einzelnen Defekt. Berichtet werden daher:

- **ROC-AUC** als schwellwert-unabhängiges Maß der Diskriminierungsfähigkeit (0,5 = Zufall, 1,0 = perfekt).
- **PR-AUC** als schwellwert-unabhängiges Maß, das speziell bei stark unbalancierten Klassen die Aussagekraft von ROC-AUC ergänzt; Zufallsniveau hier ≈ 0,005 (Prävalenz der Positivklasse).
- **Precision / Recall / F1** am gewählten F1-optimalen Schwellwert auf VAL.
- **Verwechslungsmatrix** zur Sichtbarmachung der vier Zellen (TP, FP, FN, TN) — als PNG-Artefakt im MLflow-Run abgelegt.

### 5.3 Detail-Interpretation: AutoEncoder

Die Verwechslungsmatrix des AutoEncoders auf dem Test-Split lautet **TN = 5 191, FP = 10, FN = 9, TP = 17**:

- Von 26 realen Defekten im Test-Set werden **17 erkannt** (Recall 65,4 %).
- Von 27 abgegebenen Alarmen sind **17 echte Defekte** (Precision 63,0 %).
- **10 Fehlalarme** und **9 übersehene Defekte** — ein in der Größenordnung ausgewogenes Bild.

Die Lücke zwischen Validierungs- und Test-F1 (0,685 vs. 0,642) ist gering. Das spricht dafür, dass die Schwellwertauswahl auf VAL **nicht** zu einer übermäßigen Anpassung an die Validierungsdaten geführt hat — ein häufiger Fehlschluss bei kleiner Anzahl an Positivbeispielen.

### 5.4 Detail-Interpretation: Isolation-Forest-Sweep — ehrliche Lesart

Die Rastersuche brachte als Val-F1-Gewinner die Konfiguration `n_estimators=400, max_samples=256, max_features=1,0` mit **Val-F1 = 0,471, Test-F1 = 0,410**. Damit hat der Sweep-Gewinner einen **leicht schlechteren** Test-F1 als der Standard-Isolation-Forest (Test-F1 0,439). Auf den ersten Blick erscheint dies wie ein gescheiterter Sweep.

Die schwellwert-unabhängigen Metriken erzählen jedoch eine andere Geschichte: Mehrere Sweep-Konfigurationen erreichen Test-ROC-AUC ≈ 0,937 und Test-PR-AUC ≈ 0,55 — beide deutlich über dem Baseline-Niveau (0,915 / 0,448). Bei nur 26 echten Defekten im Test-Set ist F1 in der dritten Nachkommastelle stark rauschbehaftet; ROC-AUC und PR-AUC sind hier die robusteren Vergleichsgrößen.

Die methodologisch wichtigste Beobachtung: Die dritt-platzierte Konfiguration (`n_estimators=200, max_samples=1024, max_features=0,5`) hat mit **Test-F1 = 0,550** den höchsten Test-F1 — hätte man also nach Test-F1 selektiert, wäre genau diese das gemeldete Ergebnis gewesen. Eine solche Selektion wäre eine klassische Form des Data Leakage und genau der Grund, warum die Validierungsmenge existiert.

### 5.5 Übergreifende Interpretation und Schlüsselgrößen

- **AutoEncoder dominiert auf allen Metriken.** Der Vorteil ist nicht marginal: ein PR-AUC-Sprung von 0,448 (Isolation Forest) auf 0,684 entspricht in der Praxis einer deutlich besseren Trennung von Anomalien und Normalsamples über alle Schwellwerte hinweg.
- **LOF ist mit Vorsicht zu lesen.** Sein hoher Recall (0,615) bei winziger Precision (0,080) zeigt, dass es viele Anomalien findet, aber zugleich extrem viele Fehlalarme produziert. In einer industriellen Steuerschleife wäre das unbrauchbar.
- **ECOD ist parameterfrei** und erreicht trotzdem solide ROC-AUC (0,846) — eine vernünftige Baseline, wenn keine Tuning-Zeit zur Verfügung steht.
- **Wichtigster Leistungsfaktor:** die **Wahl der Schwelle**. Selbst der AutoEncoder mit ROC-AUC 0,99 produziert mit der wirklich Default-Schwelle der Bibliothek bei 0,5 % Kontamination unterirdische Werte; erst das gezielte F1-Tuning auf VAL hebt das Modell auf das berichtete Niveau.

---

## 6. Reproduzierbarkeit und Dokumentation (Lernziel 5)

Lernziel 5 verlangt eine *strukturierte und reproduzierbare* Dokumentation. Dieses Projekt setzt das auf vier Ebenen um:

### 6.1 Konfiguration an einem Ort

Alle reproduktionskritischen Werte stehen in [config.yaml](../config.yaml): Pfade, Split-Verhältnisse, Random-Seed, MLflow-Tracking-URI, Experiment-Name. Es gibt **keine hartkodierten Hyperparameter** in den Notebooks oder Skripten — wer eine Größe ändern will, ändert sie genau hier.

### 6.2 Modulare Codebasis

Das Verzeichnis [src/](../src/) trennt die Pipeline-Phasen sauber:

- `data_loader.py` — Laden und Splitten,
- `preprocessing.py` — Imputation und Skalierung,
- `evaluation.py` — Metriken und Schwellwertwahl,
- `modeling.py` — Modell-Runner mit MLflow-Logging,
- `mlflow_utils.py` — Glue zwischen Konfiguration und MLflow.

Die Notebooks im Verzeichnis [notebooks/](../notebooks/) sind dünn: sie rufen die `src/`-Funktionen auf und zeigen Ergebnisse an — sie enthalten keine eigene Logik. Das bedeutet, dass ein Bugfix oder eine Methodenänderung an genau einer Stelle vorgenommen wird.

### 6.3 MLflow als Versuchsprotokoll

Jeder einzelne Modellrun wird in `mlruns/` als eigener MLflow-Run abgelegt. Pro Run werden festgehalten:

- **Parameter** (Modellname, Kontaminationsrate, Skalierer, Seed, Hyperparameter)
- **Metriken** (alle Test- und Validierungsmetriken inklusive der vier Zellen der Verwechslungsmatrix)
- **Artefakte** (Verwechslungsmatrix-PNG, ROC-Kurven-PNG, das gefittete Modell als `joblib`, eine Kopie des Dataset-Cards)
- **Tags** (z. B. `sweep=iforest_v1`, `model_family=iforest`)

Damit ist jede Tabellenzeile dieses Berichts ein MLflow-Run, in dem alle erzeugenden Parameter dokumentiert sind. Damit Notebooks aus dem `notebooks/`-Unterverzeichnis dieselbe Run-Datenbank schreiben wie skriptgestützte Aufrufe, wird der Tracking-URI auf das Projektwurzelverzeichnis verankert:

```python
# src/mlflow_utils.py
def _resolve_tracking_uri(uri, project_root):
    if uri.startswith("file:"):
        raw = uri[len("file:"):]
        p = Path(raw)
        if not p.is_absolute():
            return (project_root / p).resolve().as_uri()
    return uri
```

Ohne diese Auflösung würde `file:./mlruns` relativ zum aktuellen Arbeitsverzeichnis interpretiert — und ein aus dem `notebooks/`-Ordner gestarteter Lauf würde versehentlich nach `notebooks/mlruns/` schreiben statt in das gemeinsame Versuchsprotokoll.

### 6.4 Reproduktion in drei Schritten

Auf einer frischen Windows-Maschine mit Python 3.11+:

1. `scripts\setup.bat` — legt ein virtuelles Python-Environment `.venv\` an und installiert alle Abhängigkeiten aus [requirements.txt](../requirements.txt).
2. Den Datensatz `5_cfrp.npz` aus dem SmartManuAD-Repository in `data/raw/` ablegen.
3. `scripts\run-modeling.bat` — führt das gesamte Modellierungs-Notebook nichtinteraktiv aus und schreibt neue Runs nach `mlruns/`.

`scripts\mlflow-ui.bat` öffnet anschließend die MLflow-Weboberfläche unter `http://127.0.0.1:5000`, in der alle Runs einsehbar sind.

---

## 7. Kritische Reflexion: Die Rolle von KI im Internet of Production (Lernziel 6)

Lernziel 6 fragt nach einer **kritischen Reflexion** der Rolle von KI und Datenanalyse im Internet of Production — entlang der drei Achsen Transparenz, Rückverfolgbarkeit und Nachhaltigkeit. Die Reflexion ist im Folgenden bewusst **allgemein** gehalten; das eigene Projekt taucht nur an wenigen Stellen als Illustration auf.

### 7.1 Transparenz

Das Internet of Production verlagert Entscheidungen von menschlichen Bedienern hin zu Modellen, die auf Sensordatenströmen operieren. Das macht die Frage nach **Transparenz** zu einer der zentralen Voraussetzungen für die Akzeptanz solcher Systeme in der industriellen Praxis. Drei Ebenen lassen sich unterscheiden:

**Modelltransparenz.** Klassische, regelbasierte oder statistische Verfahren (z. B. Schwellwertüberwachung, Regressionsmodelle, baum- oder distanzbasierte Anomaliedetektoren) sind in der Regel inspizierbar — eine Domänenexpertin kann die getroffene Entscheidung an einer überschaubaren Anzahl von Merkmalen festmachen. Tiefe neuronale Architekturen, die im IoP-Diskurs zunehmend an Bedeutung gewinnen, sind hingegen nichtlineare Funktionen über tausende bis Millionen Parameter, deren Entscheidung sich nicht ohne Hilfsmittel rekonstruieren lässt. Hier hat sich eine eigene Disziplin der **Explainable AI** (LIME, SHAP, integrierte Gradienten, Attention-Visualisierung) etabliert; im IoP-Kontext ist sie kein Luxus, sondern Voraussetzung für Auditierbarkeit, Haftungsklarheit und nicht zuletzt für die Einhaltung des EU AI Act, der Hochrisiko-Systeme — und industrielle Steuerungs- und Qualitätsentscheidungen können dazu zählen — explizit verpflichtet, ihre Entscheidungen erklärbar zu machen.

**Datentransparenz.** Mindestens ebenso wichtig wie die Modellseite ist die Frage, was der Trainings-Datensatz eigentlich enthält: welche Sensoren mit welcher physikalischen Bedeutung, welche Erfassungslücken, welche Vorverarbeitungs- und Anonymisierungsschritte schon vor der ML-Pipeline. Viele öffentlich verfügbare Industriedaten — auch der für dieses Projekt verwendete CFRP-Datensatz — sind aus Wettbewerbsgründen anonymisiert; selbst eine perfekt erklärende Methode kann dann nicht aussagen, *welcher physikalischen Größe* ein als verdächtig erkannter Sensorkanal entspricht. Für Forschungszwecke ist das tolerabel, für den produktiven IoP-Einsatz nicht.

**Prozesstransparenz.** Schließlich verlangt Transparenz auch, dass die *Grenzen* eines Modells offengelegt werden: für welche Betriebszustände wurde es validiert, wo ist es nicht zuständig, welche Eingaben treiben es in einen extrapolierenden Bereich? Ein Modell, das nur unter „normalen" Produktionsbedingungen gesehen wurde, gibt bei einem Anlagenanlauf oder einer Materialwechsel-Situation möglicherweise Scores ab, die statistisch sinnvoll aussehen, aber operativ irreführend sind. Das offene Kommunizieren dieser Grenzen ist Teil seriöser IoP-Praxis.

**Spannungsfeld.** In der Praxis erzeugt die Leistungs-Transparenz-Achse einen echten Zielkonflikt: häufig liefern die am wenigsten interpretierbaren Modelle die besten Metriken. Das Spannungsfeld lässt sich nicht wegargumentieren; es muss bewusst gestaltet werden — etwa durch Hybrid-Setups, in denen ein leistungsstarkes Modell die Entscheidung trägt und ein zweites, transparenteres Modell die Begründung liefert.

### 7.2 Rückverfolgbarkeit (Traceability)

Rückverfolgbarkeit ist im IoP die Frage, ob man **vom Modellresultat ausgehend rückwärts durch den gesamten Verarbeitungsweg laufen** kann — vom binären Anomaliealarm zurück über den kontinuierlichen Score, durch die Modellinferenz, durch die Vorverarbeitungsschritte, bis zum ursprünglichen Sensorrohwert. Erst diese Möglichkeit, jeden Schritt zwischen Rohdaten und Entscheidung sichtbar und kausal nachvollziehbar zu machen, schafft die Grundlage dafür, dem Verfahren als Ganzem zu vertrauen.

**Warum dieses Vertrauen so wichtig ist.** Eine industrielle KI-Entscheidung steht nicht für sich allein: sie löst eine Folgehandlung aus — eine Maschine wird angehalten, ein Bauteil wird aussortiert, ein Bedienerin wird alarmiert. Wenn auch nur ein einziger Schritt der Verarbeitungskette intransparent ist, wird das gesamte System zu einer Black Box, und die Folgehandlung wird zu einem Akt des Glaubens. Genau das untergräbt die in der Industrie zentralen Werte: Sicherheit, Haftbarkeit und das schrittweise Lernen aus Fehlentscheidungen.

**Die Kette, die zurückverfolgbar sein muss.** Ein typischer IoP-Anwendungsfall hat fünf Glieder, an denen die Rückverfolgung jeweils brechen kann:

1. **Rohdatenebene.** Ist der konkrete Sensorrohwert zum Zeitpunkt der Entscheidung archiviert? Welche Sensoreinheit, welche Kalibrierung, welches Zeitfenster?
2. **Vorverarbeitungsebene.** Mit welchen exakten Parametern (Median, Skalierungsfaktor, Imputationsregel) wurde der Rohwert transformiert? Sind diese Parameter eingefroren oder driften sie zwischen den Läufen?
3. **Modellebene.** Welches Modell hat den Score erzeugt — welche Architektur, welche Gewichte, welcher Trainingsstand?
4. **Schwellenebene.** Wie wurde die binäre Entscheidung aus dem Score erzeugt? Auf welcher Datengrundlage wurde der Schwellwert gewählt?
5. **Anwendungsebene.** Welche konkrete Folgehandlung wurde aus der Entscheidung abgeleitet, und wer hat sie autorisiert?

Wenn alle fünf Glieder dokumentiert sind, kann eine Anomaliemeldung im Nachhinein **erklärt** werden — und zwar nicht nur mit „das Modell hat es gesagt", sondern mit einer kausalen Geschichte: *Sensor 17 hat den Wert 412 gemessen, dieser Wert liegt nach Skalierung 3,2 Standard-IQRs außerhalb der Trainingsverteilung, das Modell vergibt dafür einen Rekonstruktionsfehler von 0,84, dieser liegt oberhalb der auf VAL bestimmten Schwelle von 0,62, deshalb Alarm.* Erst diese Erzählbarkeit macht den Prozess auditierbar.

**Wo die Kette in der Praxis bricht.** Häufig fehlen die mittleren Glieder. Rohdaten werden archiviert, Modellresultate werden archiviert — aber die Vorverarbeitungs-Parameter, die zwischen beiden vermitteln, sind nur „im Code" und ändern sich mit jeder neuen Pipeline-Version, ohne dass alte Runs nachgespielt werden könnten. Ebenso problematisch: nicht-deterministische Modelle (zufällige Initialisierung ohne Seed, nichtdeterministische GPU-Operationen) erzeugen bei jeder Wiederholung leicht andere Scores, sodass selbst bei identischen Eingaben die Erklärung „warum dieser Score?" nur näherungsweise rekonstruierbar ist.

**Vertrauen entsteht in mehreren Schichten.** Rückverfolgbarkeit ist deshalb kein einzelnes technisches Feature, sondern eine **Disziplin** entlang der gesamten Pipeline: deterministische Seeds, eingefrorene Vorverarbeitungs-Parameter (das fit-on-train-only-Prinzip aus Abschnitt 3.4 ist nichts anderes als Rückverfolgbarkeit über die Splitgrenze hinweg), versionierte Modellartefakte, dokumentierte Schwellenwerte. Werkzeuge wie MLflow institutionalisieren diese Disziplin, indem sie pro Versuchslauf eine vollständige Parameterkarte, Metriken und Modellartefakte ablegen — sie verändern aber nichts daran, dass die eigentliche Arbeit in der konsequent durchgehaltenen Praxis liegt.

**Industrielle Konsequenz.** In sicherheitsrelevanten oder zertifizierten Produktionsumgebungen (Luftfahrt, Medizintechnik, Automotive) ist diese durchgängige Rückverfolgbarkeit nicht optional, sondern regulatorisch gefordert; der EU AI Act etwa verlangt von Hochrisiko-Systemen ausdrücklich „automatisierte Aufzeichnungen" über den Lebenszyklus. Aber auch jenseits der Compliance lohnt sich der Aufwand: ein nachvollziehbarer Prozess **lernt schneller**, weil Fehlentscheidungen analysiert werden können statt im Nebel zu verschwinden.

### 7.3 Nachhaltigkeit

Nachhaltigkeit im IoP ist ein zweischneidiges Konzept. Es lässt sich entlang zweier Achsen denken:

**Nachhaltigkeit *durch* KI.** Datengetriebene Verfahren ermöglichen in der Produktion eine Reihe von Effizienzhebeln: vorausschauende Instandhaltung statt zeitbasierter Wartung, prozessbegleitende Qualitätsregelung statt End-of-Line-Ausschuss, Materialfluss-Optimierung gegen Überproduktion. In energie- und materialintensiven Branchen — Metallverarbeitung, Composite-Fertigung, Halbleiter, chemische Prozesse — sind hier signifikante Reduktionen von Material-, Energie- und CO₂-Verbrauch realistisch. Anomalieerkennung, wie sie in diesem Projekt am CFRP-Datensatz exemplarisch durchgeführt wurde, ist ein typischer Vertreter dieser Klasse: ein erkannter Defekt während des Lagenaufbaus erspart das spätere Verschrotten eines hochgradig energetisch eingebundenen Bauteils.

**Nachhaltigkeit *des* KI-Systems selbst.** Demgegenüber stehen die ökologischen Kosten der KI-Entwicklung und des laufenden Betriebs. Bei klassischen Modellen (lineare Modelle, Bäume, klassische Anomaliedetektoren) sind diese Kosten meist vernachlässigbar — Training und Inferenz laufen auf CPU im Sekunden- bis Minutenbereich. Bei tiefen neuronalen Netzen und insbesondere bei der jüngsten Generation großer Foundation-Modelle steigen die Trainingskosten dramatisch (Bereich kWh bis MWh pro Trainingslauf), und der Inferenzbetrieb in einer 24/7-Produktionslinie summiert sich ebenfalls. Das oft zitierte Beispiel der GPT-Modelle ist nur die sichtbarste Spitze eines allgemeineren Trends, in dem Modellleistung mit Energieverbrauch erkauft wird.

**Konsequenz für die Modellwahl.** Eine seriöse Nachhaltigkeitsabwägung im IoP muss beide Achsen gemeinsam betrachten: ein großes Modell, das eine kleine Ersparnis bringt, kann unterm Strich eine schlechtere CO₂-Bilanz haben als ein einfaches Modell, das dieselbe Wirkung mit einem Bruchteil der Rechenkosten erzielt. Die Standardfrage „können wir es noch genauer machen?" ist im Nachhaltigkeitskontext mit „zu welchem energetischen Preis?" zu ergänzen. **Right-sizing** der Modellkomplexität ist hier kein technisches Detail, sondern eine ethische Pflicht.

**Nachhaltigkeit *durch* Reproduzierbarkeit.** Ein oft übersehener Aspekt: jeder nicht reproduzierbare Trainingslauf wird in der Praxis irgendwann wiederholt. Eine saubere MLflow- und Git-Disziplin spart auf Sicht von Jahren reale Rechenzeit und damit Energie ein. Reproduzierbarkeit (vgl. 7.2) ist insofern nicht nur ein Compliance-Thema, sondern auch ein Nachhaltigkeits-Thema.

### 7.4 Persönliches Fazit

*(Bitte vor der Einreichung persönlich ausformulieren — als Anregung:)*

- Welche IoP-Idee hat sich für mich durch das Projekt am stärksten konkretisiert, die vorher abstrakt war?
- An welcher Stelle habe ich im eigenen Workflow gemerkt, dass methodische Disziplin (kein Tuning auf Test, Schwellwert nur auf Validierung) keine akademische Pose ist, sondern eine reale Sicherung gegen Selbsttäuschung?
- Welche der drei Achsen — Transparenz, Rückverfolgbarkeit, Nachhaltigkeit — halte ich für die in der industriellen Realität am häufigsten vernachlässigte, und warum?

---

## Anhang A — Zentrale Architektur-Invarianten

Vier Invarianten tragen die methodische Integrität der Pipeline. Sie sind im Code mechanisch verankert und an den entsprechenden Stellen dieses Berichts mit Snippets belegt:

1. **Fit nur auf TRAIN** (Abschnitt 3.4). Imputer und Scaler werden ausschließlich auf dem Trainings-Split gefittet; Validierung und Test sehen nur die `transform`-Phase.
2. **Schwellwertbestimmung nur auf VAL** (Abschnitt 4.4). Der Schwellwert wird auf der Validierungsmenge gewählt und auf TEST als eingefrorene Konstante übergeben.
3. **Definierte 6-Tupel-Reihenfolge des Splits** (Abschnitt 3.1). Die Reihenfolge `(X_train, y_train, X_val, y_val, X_test, y_test)` ist über die gesamte Pipeline stabil und in Tests gegen Vertauschung abgesichert.
4. **MLflow-Tracking-URI auf das Projektwurzelverzeichnis verankert** (Abschnitt 6.3). Damit schreibt jeder Lauf — egal aus welchem Unterverzeichnis gestartet — in dasselbe Versuchsprotokoll.

## Anhang B — Liste der verwendeten Bibliotheken

Aus [requirements.txt](../requirements.txt): NumPy, pandas, scikit-learn, PyOD, MLflow, Matplotlib, Seaborn, JupyterLab, joblib, PyYAML, PyTorch (CUDA optional). Konkrete Versionen sind dort und in den auf Festplatte gespeicherten MLflow-Runs eingefroren.

## Anhang C — Reproduktion auf einen Blick

```text
scripts\setup.bat            # einmalig: venv + Abhängigkeiten installieren
scripts\run-modeling.bat     # vollständige Pipeline ausführen
scripts\mlflow-ui.bat        # Ergebnisse im Browser unter 127.0.0.1:5000 ansehen
```
