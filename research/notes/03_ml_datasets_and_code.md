# ML/DL подходы, датасеты и открытый код для акустического обнаружения дронов

> Дата: 2026-02-14

---

## 1. Публичные датасеты

### Специализированные датасеты дронов

| Датасет | Классы | Объём | Частота | Доступность |
|---------|--------|-------|---------|-------------|
| **AIRA-UAS** | 4+ дронов + шум | ~10 ГБ | 44.1–48 kHz | Zenodo (открытый) |
| **DroneAudioDataset** (Al-Emadi) | 4 дрона | ~3 ГБ | 44.1 kHz | По запросу |
| **DroneDetect** (Cardiff Univ.) | 3 дрона (мультимод.) | ~15 ГБ | 44.1 kHz | IEEE DataPort |
| **MUADD** | 5+ дронов + шум | ~5 ГБ | 48 kHz | Zenodo |
| **Kaggle Drone Audio** | drone/no-drone | ~2 ГБ | 16–44.1 kHz | Kaggle (открытый) |

### Датасеты для негативного класса (фоновый шум)

| Датасет | Описание | Полезные классы |
|---------|----------|-----------------|
| **ESC-50** | 2000 клипов, 50 классов | birds, wind, rain, engine |
| **UrbanSound8K** | 8732 клипа, 10 классов | car_horn, engine, siren |
| **AudioSet** (Google) | 2+ млн клипов, 632 класса | Aircraft, Vehicle, Wind, Bird |
| **FSD50K** | 51,197 клипов, 200 классов | Предобучение и аугментация |

---

## 2. Архитектуры ML/DL

### 2.1 CNN на спектрограммах

| Архитектура | Параметры | Точность (бинарная) | Inference (Jetson Nano) | Для edge |
|---|---|---|---|---|
| Custom CNN (3 слоя) | ~200K | 88–91% | ~3 мс | Отлично |
| **MobileNetV2** | 3.4M | 91–94% | ~8 мс | **Лучший выбор** |
| EfficientNet-B0 | 5.3M | 93–95% | ~12 мс | Хорошо |
| ResNet-18 | 11.7M | 94–96% | ~15 мс | Средне |
| VGGish | 138M | 92–96% | Медленно | Плохо |
| YAMNet | 3.7M | 92–95% | ~6 мс | Хорошо (feature ext.) |

### 2.2 RNN/LSTM

- **BiLSTM**: 2–3 слоя (128–256 units), 89–93% точности
- **CRNN (CNN + LSTM)** — **один из лучших подходов**: 93–96% точности
- **GRU**: на 25% легче LSTM при сопоставимой точности, лучше для edge

### 2.3 Transformers

- **AST** (Audio Spectrogram Transformer, MIT): state-of-the-art, ~87M параметров, тяжёлый для edge
- **HTS-AT**: ~30M параметров, превосходит AST
- **BEATs** (Microsoft): self-supervised, отлично при ограниченных данных
- **DeiT-Tiny**: ~5.7M, подходит для edge с GPU

### 2.4 Оптимизации для edge

- **Квантизация INT8**: размер в 4x меньше, ускорение в 2–3x, потеря ~1–2%
- **Pruning**: удаление 50–70% весов при потере <2%
- **Knowledge Distillation**: teacher (ResNet-50) -> student (MobileNet)
- **Runtime**: TFLite, ONNX Runtime, TensorRT

---

## 3. Извлечение признаков

### 3.1 Сравнение

| Признак | Размерность | Вычисл. стоимость | Точность | Для edge |
|---------|-------------|---------------------|----------|----------|
| MFCC (20) | 20 | Низкая | 88–93% | Отлично |
| MFCC (40) + delta | 120 | Низкая | 90–94% | Хорошо |
| **Мел-спектрограмма (128)** | 128 | Средняя | **93–97%** | Средне |
| Raw waveform | 16000/сек | Высокая | 90–95% | Плохо |

**Мел-спектрограммы** — лучший выбор для CNN. Параметры: 128 мел-полос, dB-шкала, per-channel нормализация.

### 3.2 Признаки, специфичные для дронов

- **Blade Passing Frequency (BPF)**: f = N_blades * RPM / 60
- **Spectral Flatness** — один из лучших дискриминативных признаков (дроны: низкая, ветер: высокая)
- **Harmonic-to-Noise Ratio (HNR)** — высокий у дронов
- **Spectral Centroid** — дроны: 500–2000 Гц
- **Pitch Tracking / F0** — стабильный при hover, изменяется при маневрах

### 3.3 Аугментация данных

1. Time Stretching (0.8–1.2x)
2. Pitch Shifting (+-2 полутона, имитация разных RPM)
3. Добавление шума (белый, розовый, реальный окружающий)
4. **SpecAugment** (маскирование полос частот и времени)
5. **Mixup** (смешивание образцов)
6. Gain Augmentation (имитация расстояний)
7. Room Impulse Response (свёртка с импульсными характеристиками)

---

## 4. Интеграция beamforming + ML

### 4.1 Pipeline-подход (рекомендуется для начала)

```
[Микрофонная решётка]
    -> [DOA estimation (SRP-PHAT)] -> Azimuth, Elevation
    -> [Звукоусиление направления] -> Enhanced audio
    -> [Feature extraction (Mel-spectrogram)]
    -> [CNN/CRNN классификатор] -> Drone type, Confidence
    -> [Трекинг (Kalman filter)] -> Trajectory
```

### 4.2 End-to-End подход (следующий шаг)

```
[Многоканальное аудио (N mic)]
    -> [Multi-channel feature extraction]
    -> [SELDnet / CRNN]
    -> [Joint DOA + Classification output]
```

**SELDnet** (Adavanne et al., 2018): CRNN (CNN + BiGRU), вход: многоканальные мел-спектрограммы + фазовая разность, выход: класс + DOA.

**Входные признаки для многоканальных моделей:**
- Мел-спектрограмма каждого канала
- Inter-channel Phase Difference (IPD)
- Inter-channel Level Difference (ILD)
- GCC-PHAT features

---

## 5. Бенчмарки и метрики

### 5.1 Бинарная классификация (Drone / No-Drone)

| Метод | Accuracy | F1 | Расстояние | Условия |
|-------|----------|-----|------------|---------|
| CNN (мел-спектр.) | 96.2% | 0.95 | <50 м | Тихая среда |
| CRNN | 97.1% | 0.96 | <50 м | Тихая среда |
| SVM + MFCC | 91.3% | 0.90 | <30 м | Тихая среда |
| ResNet-18 (TL) | 95.8% | 0.95 | <50 м | Город |
| MobileNetV2 | 93.5% | 0.93 | <50 м | Город |

### 5.2 Влияние расстояния

| Расстояние | Accuracy | SNR |
|---|---|---|
| 0–20 м | 97–99% | >20 dB |
| 20–50 м | 93–97% | 10–20 dB |
| 50–100 м | 85–93% | 5–10 dB |
| 100–200 м | 70–85% | 0–5 dB |
| 200–500 м | 50–75% | <0 dB |

### 5.3 Метрики
- Accuracy, Precision, Recall, **F1** — стандартные
- **AUC-ROC** — качество ранжирования
- **FAR (False Alarm Rate)** — критична для практического развёртывания
- **Detection Range** — максимальная дистанция
- **Latency** — задержка до срабатывания

---

## 6. GitHub-репозитории

### Обнаружение дронов
- `saraalemadi/DroneAudioDetection` — CNN + SVM (TensorFlow/Keras)
- `ashT95/drone-sound-detection` — CNN (PyTorch)

### Аудио-классификация (для transfer learning)
- `qiuqiangkong/audioset_tagging_cnn` — CNN14, ResNet38, MobileNetV2 на AudioSet
- `qiuqiangkong/panns_inference` — Pretrained Audio Neural Networks
- `YuanGongND/ast` — Audio Spectrogram Transformer
- `microsoft/unilm/tree/master/beats` — BEATs self-supervised
- `tensorflow/models/.../yamnet` — YAMNet от Google

### Звуковая локализация (SELD)
- `sharathadavanne/seld-net` — SELDnet (joint localization + detection)
- `Jinbo-Hu/EINV2-SELD` — улучшенная модель SELD

### Библиотеки
- `librosa/librosa` — MFCC, мел-спектрограммы
- `pytorch/audio` (torchaudio) — дифференцируемые признаки
- `LCAV/pyroomacoustics` — beamforming, DOA (SRP-PHAT, MUSIC)
- `speechbrain/speechbrain` — готовые пайплайны
- `microsoft/onnxruntime` — оптимизированный inference для edge

---

## 7. Рекомендуемый план экспериментов

1. **Baseline**: SVM/Random Forest + MFCC (быстрый старт)
2. **CNN**: MobileNetV2 + мел-спектрограммы (основная модель)
3. **Transfer Learning**: YAMNet/PANNs -> fine-tune на дронах
4. **CRNN**: CNN + GRU для временных зависимостей
5. **Edge**: Квантизация лучшей модели -> TFLite/ONNX
6. **SELD**: SELDnet для совместной локализации и классификации

### Рекомендуемый стек
```
Python 3.9+
PyTorch + torchaudio
librosa (feature extraction)
pyroomacoustics (beamforming, DOA)
TFLite / ONNX Runtime (edge deployment)
Weights & Biases / MLflow (эксперименты)
Датасет: AIRA-UAS + ESC-50 + AudioSet embeddings
```
