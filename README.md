# Vocal Pitch Monitor v2.7

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![PyQt6](https://img.shields.io/badge/PyQt6-6.x-green.svg)](https://www.riverbankcomputing.com/software/pyqt/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Профессиональный инструмент для мониторинга вокального питча в реальном времени с поддержкой разделения аудио на стемы (Demucs), тренировками по эталонным трекам и расширенной аналитикой.

## 🎯 Возможности

### Основные функции
- **Мониторинг питча в реальном времени** — отображение ноты и отклонения в центах
- **Разделение аудио на стемы** — выделение вокала из любой песни через Demucs (HTDemucs, MDX Extra)
- **Тренировка с эталонным треком** — сравнение вашего голоса с оригинальным исполнением
- **Автоматическое определение вокального диапазона (Tessitura)** — отслеживание мин/макс нот за сессию
- **Экспорт результатов сессии** — сохранение графика и статистики в PNG
- **Поддержка MP3/WAV** — асинхронная загрузка без блокировки интерфейса
- **Лира с авто-скроллом** — синхронизированный текст песни с паузой при ручном скролле

### Технические особенности
- **Асинхронное декодирование аудио** — QThread worker предотвращает зависания GUI
- **Оптимизированный алгоритм YIN** — ограничение диапазона 30-88 MIDI (50-1000 Гц) для вокала
- **Rust-сглаживание питча** — быстрый медианный фильтр через `vpm_core` (с фоллбэком на Python)
- **NumPy-векторизация** — обработка аудио-буферов без Python-циклов
- **GPU-ускорение** — поддержка CUDA/ROCm для Demucs (с авторефолом на CPU при нехватке VRAM)

---

## 📦 Установка

### Требования
- Python 3.8+
- Windows/Linux/macOS
- GPU с поддержкой CUDA/ROCm (опционально, для ускорения Demucs)

### Быстрая установка

```bash
# Клонируйте репозиторий
git clone https://github.com/ZaVoZ/Vocal-Monitor-Pitch-FREE.git
cd Vocal-Monitor-Pitch-FREE

# Установите зависимости
pip install -r requirements.txt

# Для поддержки GPU (опционально)
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118  # CUDA 11.8
pip install demucs
```

### Зависимости

| Пакет | Назначение |
|-------|-----------|
| `numpy` | Векторные вычисления |
| `soundfile` | Чтение/запись WAV |
| `sounddevice` | Работа с микрофоном |
| `aubio` | Детекция питча (YIN) |
| `pyqtgraph` | Графики в реальном времени |
| `PyQt6` | Графический интерфейс |
| `pydub` | Конвертация MP3 → WAV |
| `torch` + `demucs` | Разделение на стемы (GPU/CPU) |
| `librosa` | Питч-шифт (опционально) |

---

## 🚀 Использование

### Запуск
```bash
python main.py
```

### Горячие клавиши
| Клавиша | Действие |
|---------|---------|
| `Space` | Старт/Стоп воспроизведение |
| `R` | Перезапуск трека |
| `M` | Включить/выключить микрофон |
| `H` | Включить/выключить монитор |
| `C` | Очистить график |
| `Ctrl+O` | Загрузить трек |

### Рабочий процесс

1. **Загрузка трека**: Нажмите `Load Track`, выберите MP3/WAV файл
2. **Разделение на вокал** (опционально): Выберите модель Demucs для извлечения вокала
3. **Настройка чувствительности**: Отрегулируйте ползунок Confidence Threshold (0.3–0.5)
4. **Включите микрофон**: Нажмите `Mic` для захвата голоса
5. **Поем!**: Следуйте за графиком оригинала, наблюдайте за точностью попадания в ноты

### Сборка для Windows

Для создания standalone .exe файла:

**Вариант 1: PyInstaller (рекомендуется)**
```bash
pip install pyinstaller
pyinstaller --onefile --windowed --icon=assets\logo.ico --name="Vocal Pitch Monitor" --add-data "assets;assets" main.py
```

**Вариант 2: Автоматическая сборка**
```bash
# На Windows запустите:
build_windows.bat

# Или вручную:
python -m venv vpm_env
vpm_env\Scripts\activate
pip install -r requirements.txt pyinstaller
pyinstaller --onefile --windowed --icon=assets\logo.ico --name="Vocal Pitch Monitor" --add-data "assets;assets" main.py
```

Готовый exe-файл появится в папке `dist/`.

**Вариант 3: cx_Freeze**
```bash
pip install cx_Freeze
python setup.py build
```

---

## 🏗️ Архитектура

```
Vocal Pitch Monitor
├── Audio Pipeline
│   ├── Async Decoder (QThread) — MP3/WAV → PCM
│   ├── Demucs Separator — вокал/минус (GPU/CPU)
│   └── Pitch Detector (Aubio YIN) — 30-88 MIDI range
│
├── Real-time Processing
│   ├── Rust Smooth Filter (vpm_core) — медианный фильтр
│   ├── Octave Correction — авто-коррекция октавных скачков
│   └── Tessitura Tracker — мин/макс ноты сессии
│
├── Visualization
│   ├── PyQtGraph — 60 FPS графики питча
│   ├── Auto-scroll Lyrics — синхронизированный текст
│   └── Session Exporter — PNG со статистикой
│
└── UX Features
    ├── Keyboard Shortcuts — горячие клавиши
    ├── Dark Theme — современная тёмная тема
    └── Cache Manager — кэш разделённых файлов
```

---

## 📈 Roadmap (План развития)

### ✅ Этап 1: Оптимизация ядра и стабильность (Завершён)
- [x] **Асинхронное декодирование аудио** — QThread worker для MP3/WAV
- [x] **Ограничение диапазона YIN** — 30-88 MIDI (вокальный диапазон)
- [x] **Rust-сглаживание** — `vpm_core` для быстрого медианного фильтра
- [x] **NumPy-векторизация** — обработка буферов без Python-циклов

### ✅ Этап 2: Расширение функционала тренировок (Завершён)
- [x] **Tessitura Diagnostic** — авто-определение рабочего диапазона (мин/макс ноты)
- [x] **Session Snapshot Export** — экспорт графика + статистика в PNG
- [ ] **MIDI Import** — загрузка партитур `.mid` для тренировки по эталону *(в работе)*

### 🔜 Этап 3: Архитектурные изменения (Долгосрочный план)
- [ ] **WebAssembly порт** — браузерная версия на Rust + WebAudio API
- [ ] **Нативный C++/Qt6 интерфейс** — 120+ FPS на слабом железе
- [ ] **ML-аналитика** — оценка тембра, вибрато, динамики через нейросеть

---

## ⚙️ Производительность

### Оптимизации v2.7

| Компонент | Было | Стало | Улучшение |
|-----------|------|-------|-----------|
| Загрузка MP3 (3 мин) | ~2 сек (блокировка) | ~0.5 сек (асинхронно) | 4× быстрее |
| Детекция питча | Полный диапазон 0-127 MIDI | 30-88 MIDI (вокал) | 1.5-2× меньше CPU |
| Сглаживание | Python-циклы | Rust `vpm_core` | 10× быстрее |
| Обработка буфера | Циклы `for` | NumPy векторизация | 3-5× быстрее |

### Рекомендации по железу

| Конфигурация | Demucs | FPS графика |
|--------------|--------|-------------|
| CPU (Xeon i5) | 30-60 сек | 60 FPS |
| GPU (GTX 1060 6GB) | 5-10 сек | 60 FPS |
| GPU (RTX 3060 12GB) | 2-5 сек | 60 FPS |
| GPU (RX 580 ROCm) | 10-20 сек | 60 FPS |

---

## 🛠️ Настройка

### Переменные окружения

```bash
# Для AMD GPU (Polaris)
export HSA_OVERRIDE_GFX_VERSION=10.3.0

# Для кэша разделённых файлов
export VPM_CACHE_DIR=~/.vpm_cache  # по умолчанию ~/.vpm_cache
```

### Модели Demucs

| Модель | Качество | VRAM | Скорость |
|--------|---------|------|----------|
| `htdemucs_ft` | Лучшее | ~4 GB | Медленно |
| `htdemucs` | Хорошее | ~2.5 GB | Быстро |
| `mdx_extra_q` | Отличное | ~4 GB | Средне |
| `mdx_extra` | Высокое | ~6 GB | Медленно |

---

## 📸 Скриншоты

![Интерфейс Vocal Pitch Monitor](screenshots/main_interface.png)
*Основной интерфейс мониторинга питча*

![Разделение вокала](screenshots/vocal_separation.png)
*Разделение трека на вокал и музыку через Demucs*

![Экспорт сессии](screenshots/session_export.png)
*Экспорт результатов тренировки в PNG*

> **Примечание**: Скриншоты расположены в папке `screenshots/`. Для добавления своих скриншотов поместите файлы PNG в эту папку.

---

## 🤝 Вклад в проект

1. Fork репозиторий
2. Создайте ветку (`git checkout -b feature/YourFeature`)
3. Закоммитьте изменения (`git commit -m 'Add YourFeature'`)
4. Push в ветку (`git push origin feature/YourFeature`)
5. Откройте Pull Request

---

## 📄 Лицензия

MIT License — см. файл [LICENSE](LICENSE)

---

## 🙏 Благодарности

- **Aubio** — библиотека для детекции питча
- **Demucs** (Meta Research) — разделение аудио на стемы
- **PyQtGraph** — быстрые графики для Python
- **Rust** — высокопроизводительные вычисления

---

## 📬 Контакты

- **Автор**: ZaVoZ
- **Issues**: [GitHub Issues](https://github.com/TheFirstPy/Vocal-Monitor-Pitch-FREE/issues)

---

*Vocal Pitch Monitor v2.7 — Пойте точно, тренируйтесь эффективно!*
