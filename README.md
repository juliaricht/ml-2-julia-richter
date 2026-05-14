# ml-2-julia-richter

Portfolio-Projekt für *KI und maschinelles Lernen in der Produktion* (Ostfalia, SoSe 26). Unüberwachte Anomalieerkennung auf dem CFRP-Datensatz ([SmartManuAD](https://github.com/SmartManuAD/Smart-Manufacturing-AD)). Python 3.11+, scikit-learn, PyOD, MLflow.

Dieses Repository enthält ausschließlich die ausführbare Pipeline.

## Quickstart (Windows)

1. `scripts\setup.bat` — legt `.venv\` an und installiert die Abhängigkeiten aus `requirements.txt`.
2. `scripts\run-modeling.bat` — führt `notebooks/02_modeling.ipynb` nichtinteraktiv aus und schreibt Versuchsläufe nach `mlruns/`.
3. `scripts\mlflow-ui.bat` — startet die MLflow-Oberfläche unter <http://127.0.0.1:5000>.

Der Datensatz liegt bereits unter `data/raw/5_cfrp.npz`. Der Zufallsseed ist in `config.yaml` festgehalten. Der AutoEncoder nutzt die GPU, falls PyTorch mit CUDA installiert ist; ansonsten läuft er auf CPU.

## Struktur

- `notebooks/02_modeling.ipynb` — die ausführbare Pipeline (Aufruf via `run-modeling.bat`).
- `notebooks/01_data_exploration.ipynb` — explorative Datenanalyse, nicht Teil der Ergebnisreproduktion.
- `src/` — wiederverwendbare Module: Datenladen, Vorverarbeitung, Evaluation, MLflow-Logging.
- `scripts/` — Windows-`.bat`-Wrapper für die wichtigsten Befehle.
- `config.yaml` — Pfade, Splitverhältnisse, Random-Seed, MLflow-Einstellungen.
