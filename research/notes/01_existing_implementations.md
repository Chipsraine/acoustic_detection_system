# Акустические системы обнаружения и идентификации звукоизлучающих объектов: существующие реализации и научная база

> Дата: 2026-02-14

---

## 1. Коммерческие системы

### DroneShield (Австралия)
- Продукты: DroneShield RfOne, DroneSentry, DroneSentinel
- Комбинированная система (RF + акустика + радар + оптика)
- Акустическая подсистема: массив MEMS-микрофонов, библиотека сигнатур 200+ моделей дронов
- Дальность акустического обнаружения: до ~500 м
- Клиенты: армия Австралии, армия США, правоохранительные органы

### Dedrone (Германия/США)
- Продукты: DedroneTracker, DroneTracker
- Мультисенсорная платформа с ML-классификаторами
- Облачная аналитика, партнёрство с Airbus
- Клиенты: аэропорт Франкфурта, правительство Германии, тюрьмы США

### Squarehead Technology (Норвегия)
- Продукт: Discovair
- Сферическая микрофонная решётка из 300+ микрофонов
- Точность пеленгации < 1 градуса
- Изначально для локализации снайперов (система HALO)

### Другие компании
- **Thales** (Франция) — акустические сенсоры в мультисенсорном C-UAS комплексе
- **HENSOLDT** (Германия) — Xpeller: мультисенсорная C-UAS система с акустическим модулем
- **Orelia** (Франция) — DroneWatch: специализированная акустическая система
- **Rohde & Schwarz** (Германия) — R&S ARDRONIS: RF с опциональным акустическим модулем
- **MyDefence** (Дания) — WINGMAN: компактный акустический датчик

---

## 2. Гражданские и государственные применения

- **Охрана аэропортов:** Обнаружение несанкционированных БПЛА в зонах воздушного движения (ИКАО, Eurocontrol)
- **Пограничная служба:** Акустический мониторинг государственных границ — обнаружение нарушителей, транспорта, БПЛА
- **Экологический мониторинг:** Биоакустика — идентификация животных по крикам, обнаружение браконьеров (выстрелы, моторные лодки)
- **Охрана периметра:** Промышленные объекты, заповедники, склады — пассивный акустический сенсор без RF-излучения
- **Транспортный мониторинг:** Классификация транспортных средств по акустической сигнатуре
- **Малая авиация:** Раннее оповещение аэродромов о приближающихся воздушных судах

---

## 3. Академические исследования

### Зарубежные
- **KU Leuven** (Бельгия) — адаптивный beamforming и late fusion DNN
- **University of Oklahoma** (США) — CNN-классификация дронов по спектрограммам
- **DTU** (Дания) — beamforming для трекинга
- **University of Erlangen-Nuremberg** (Германия) — робастное обнаружение в шуме
- **Cranfield University** (Великобритания) — акустические сигнатуры БПЛА
- **KAIST** (Южная Корея) — deep learning для классификации по звуку
- **Technion** (Израиль) — массивы микрофонов для малоразмерных воздушных целей

### Российские
- **ЦАГИ** — аэроакустика винтов БПЛА, моделирование шума мультироторных систем
- **МАИ** — шум БПЛА, кафедра 610 (Конструкция и проектирование двигателей)
- **КАИ** — обработка сигналов, системы технического зрения, массивы датчиков
- **МФТИ** — адаптивная обработка сигналов и массивы антенн
- **МГТУ им. Баумана** — системы обнаружения и противодействия БПЛА
- **СПбПУ** — цифровая обработка сигналов и акустика
- **ИПФ РАН** (Нижний Новгород) — фундаментальная акустика

---

## 4. Открытые проекты и датасеты

### GitHub-репозитории
- `drone-audio-detection` — классификаторы на Python (PyTorch, TensorFlow)
- `DroneAudioDataset` — открытые датасеты звуков дронов
- `acoustic-drone-detection` — MFCC + SVM/CNN
- `sound-source-localization` — GCC-PHAT, MUSIC, SRP-PHAT
- `pyroomacoustics` (EPFL) — моделирование акустики и обработка сигналов

### Открытые датасеты
- **DREGON** (Trinity College Dublin) — записи звуков дронов
- **AudioSet** (Google) — содержит класс "Aircraft/Drone"
- **AIRA-UAS** (Zenodo) — записи 4+ типов БПЛА, ~10 ГБ
- **DroneDetect** (Cardiff University) — мультимодальный (RF + аудио + видео)
- **MUADD** — мультиклассовый акустический датасет дронов

---

## 5. Ключевые статьи

### Основополагающие
1. Mezei, Fiaska, Molnar — "Drone Sound Detection" (IEEE, 2015)
2. Shi et al. — "Acoustic-based UAV detection using late fusion of DNN" (Drones, 2020)
3. Anwar, Kaleem, Jamalipour — "ML Inspired Sound-Based Amateur Drone Detection" (IEEE TVT, 2019)
4. Kim et al. — "Real-time UAV sound detection and analysis system" (IEEE Sensors, 2017)
5. Al-Emadi et al. — "Audio Based Drone Detection using Deep Learning" (IWCMC, 2019)
6. Bernardini et al. — "Drone detection by acoustic signature identification" (2017)
7. Jeon et al. — "Empirical study of drone sound detection with DNN" (EUSIPCO, 2017)
8. Svanstrom et al. — "Real-time drone detection with visible, thermal and acoustic sensors" (ICPR, 2020)

### Обзоры
9. Taha, Shoufan — "ML-Based Drone Detection and Classification: State-of-the-Art" (IEEE Access, 2019)
10. Park et al. — "Survey on Anti-Drone Systems" (IEEE Access, 2021)

### Учебники по массивам и beamforming
11. Van Trees — "Optimum Array Processing" (Wiley, 2002)
12. Johnson, Dudgeon — "Array Signal Processing" (Prentice Hall, 1993)
13. Benesty, Chen, Huang — "Microphone Array Signal Processing" (Springer, 2008)
14. DiBiase — PhD Thesis по SRP-PHAT (Brown University, 2000)

### ML для аудио
15. Gong et al. — "AST: Audio Spectrogram Transformer" (Interspeech, 2021)
16. Kong et al. — "PANNs: Large-Scale Pretrained Audio Neural Networks" (IEEE/ACM TASLP, 2020)
17. Hershey et al. — "CNN Architectures for Large-Scale Audio Classification" (ICASSP, 2017)

### Инструменты
18. pyroomacoustics (EPFL) — github.com/LCAV/pyroomacoustics
19. librosa — librosa.org
20. ACOULAR — acoular.org

### Российские источники
21. Журнал "Акустический журнал" (РАН)
22. Журнал "Радиотехника и электроника" (РАН)
23. Работы ЦАГИ по аэроакустике БПЛА
24. Конференция ИТНТ (Самара)

---

## 6. Направления научной новизны для магистерской

### Высокоперспективные

**1. Self-supervised / contrastive learning для обнаружения дронов в городском шуме**
- Contrastive Learning (SimCLR, MoCo) на аудио-фрагментах, fine-tuning с малым количеством меток
- Мало исследовано для акустического обнаружения БПЛА

**2. Edge-optimized модели для встраиваемых систем**
- Knowledge Distillation, NAS под Jetson Nano/RPi, TinyML
- Комплексная оптимизация pipeline под edge-device

**3. Joint localization & classification (end-to-end)**
- Multi-task learning: единая модель = DOA + расстояние + тип дрона
- End-to-end подход на сырых многоканальных данных мало исследован

**4. Оптимизация конфигурации микрофонной решётки**
- Генетические алгоритмы / градиентная оптимизация расположения для максимизации P(detection)
- Task-specific оптимизация геометрии

**5. Few-shot classification для новых типов дронов**
- Prototypical Networks, Siamese Networks для адаптации по нескольким примерам

### Рекомендуемая тема для КАИ

> **"Разработка лёгкой нейросетевой модели совместной локализации и идентификации звукоизлучающих объектов по акустическим данным микрофонной решётки для систем реального времени на встраиваемых платформах"**

Обоснование:
- End-to-end multi-task learning на edge мало исследован
- Универсальность: система работает с любым классом объектов (БПЛА, самолёты, авто, животные)
- Практическая ценность (компактное устройство $300-500)
- Связь с авиационной и акустической тематикой КАИ
- Высокий стартап-потенциал: аэропорты, пограничная служба, экомониторинг
