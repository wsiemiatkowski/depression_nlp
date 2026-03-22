# Depression NLP
### Praca magisterska PJATK Data Science
## Biblioteki
### uv
Projekt korzysta z narzędzia uv. Aby uruchamiać skrypty należy 
[zainstalować uv](https://docs.astral.sh/uv/getting-started/installation/) 
oraz odpowiednie biblioteki poprzez:
````
uv sync
````
## Dane
### Dataset
Suicide and Depression Detection powstał przy użyciu Pushshift API na bazie subredditów SuicideWatch oraz
teen.
### Źródło
Dataset dostępny jest na platformie Kaggle, gdzie cieszy się sporą popularnością oraz kiloma updatami.
### Opis
Dataset przygotowano w języku angielskim. Zawiera dwie klasy: suicide oraz non-suicide. Użyciu subreddita
teen do przykładów non-suicide wydaje się trafnym wyborem, jako że język naturalny ewoluuje, czego pierwsze
oznaki często widzimy u młodych ludzi, stąd powinno zapobiegać to klasyfikacji slangu jako treść świadcząca o
depresji.

## Katalogi
### Katalog `data`
#### 01_raw
Dataset zawiera setki tysięcy przykładów, stąd nie znajduje się w tym repozytorium ani nie został w całości
użyty do pracy z przedstawionymi tutaj modelami. Aby uzyskać odpowiednie dane należy pobrać plik `.csv` z [Kaggle](https://www.kaggle.com/datasets/nikhileswarkomati/suicide-watch?resource=download)
do folderu `data/01_raw`.
#### 02_training_data
Aby uzyskać dane treningowe należy przygotować plik opisany powyżej a następnie uruchomić skrypt `create_dataset.py` poprzez uv:
```
uv run src/utils/create_dataset.py
```
#### 03_metrics
Folder zawiera metryki powstające w wyniku treningu oraz inferencji prezentowanych modeli.
#### 04_synthetic_data
Zawiera dane syntetyczne stworzone ekserymentalnie w celu inferencji modeli do porównania wyników. Dane tworzone są 
poprzez skrypty zawarte w `src/synthetic_data_generation`.

### Katalog `fine_tuned_models`
Zawiera trzy podfoldery z wytrenowanymi modelami, które powstają w ramach wykonania skryptów z `src/models_training`.
### Katalog `mobile_models`
Zawiera modele w formacie .onnx tworzone poprzez konwersje fine-tunowanych modeli z `fine_tuned_models` skryptem `src/utils/model_to_onnx.py`.
Katalog zawiera również plik `model_size_comparison.txt` porównujący zmiany wielkości modeli z transformers na onnx w skrypcie `src/utils/compare_models_size.py`.
### Katalog `src`
Główny katalog zawierający wszystkie skrypty, na który składają się katalogi:
1. `models_inference` a w nim:
+ `test_mobile_models.py` - skrypt testujący prawidłowe działanie modeli w formacie onnx
+ `test_synthetic_data.py` - skrypt testujący modele po fine-tuningu na danych syntetycznych
2. `models_training` a w nim:
+ `albert_training.py` - skrypt trenujący model `albert-base-v2`
+ `mobilebert_training.py` - skrypt trenujący model `google/mobilebert-uncased`
+ `tinybert_training.py` - skrypt trenujący model `huawei-noah/TinyBERT_General_4L_312D`
3. `synthetic_data_generation` który:
+ wymaga narzędzia [Ollama](https://ollama.com/download/mac)
+ wymaga pobrania modeli `gpt-oss:20b` oraz `gemma3:27b` poprzez `ollama pul <model_name>`
+ wymaga działającego serwera ollama, serwer można wywołać poprzez `ollama serve` jeśli ollama nie jest włączona
+ oba modele wymagają około 20GB pamięci RAM, skrypty wykonywano wraz z modelami działającymi na komputerze Mac Studio M1 Max z 32GB pamięci RAM
+ zawiera `generate_synthetic_data.py`- skrypt generujący dane syntetyczne przy użyciu `Ollama`, modelu `gemma3:27b` oraz biblioteki `dspy`
+ zawiera `llm_as_a_judge.py` - skrypt oceniający dane syntetyczne przy użyciu `Ollama`, modelu `gpt-oss:20b` oraz biblioteki `dspy`

4. `utils` a w nim:
+ `compare_models_size.py` - skrypt pozwalający na utworzenie pliku, który porównuje rozmiary modeli transofmers oraz onnx
+ `create_dataset.py` - skrypt tworzący dataset do fine-tuningu modeli
+ `model_to_onnx.py` - skrypt pozwalający na zmianę modeli z transofmers na onnx
+ `training_utils.py` - skrypt z metodami wspólnymi dla wszystkich skryptów z `models_training`

Wszystkie skrypty po za skryptami zawartymi w `models_training` należy wykonywać z poziomu głównego katalogu repozytorium
używając 
```
uv run pyth/to/script.py
``` 
Skrypty z `models_training` należy wywoływać z tego samego poziomu poprzez:
```
uv run -m src.models_training.script
```