# Long document summarization with Chain of Density prompting

[English](README.md) | [Русский](README_RU.md)

[![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)](https://python.org)
[![LangChain](https://img.shields.io/badge/Langcahin-00a67e?logo=langchain)](https://langchain.com)
[![HuggingFace](https://img.shields.io/badge/Hugging%20Face-yellow?logo=huggingface&logoColor=white)](https://huggingface.co/)
[![Gradio](https://img.shields.io/badge/Gradio-white?logo=gradio)](https://www.gradio.app/)

## Обзор

Этот проект реализует комплексную систему автоматиеческой генерации краткой и точной аннотации для длинных текстовых документов.
Система использует языковую модель **Phi-3-mini-128k-instruct** и последовательное применение **Map-Reduce + Chain-of-Density** для обработки документов любой длины и сохранения релевантности и фактической целостности аннотации.

## Ключевые особенности

- **Точный парсинг:**
    Используется **PyMuPDF** для надёжного извлечения текста из сложных макетов (многостолбцовый, многостраничный).

- **Масштабирование суммаризации с помощью Map-Reduce:**
    Обрабатывает документы любой длинны, разбивая текст на фрагменты, создавая аннотацию каждого из них, после чего объединяя результат в единую связную аннотацию.

- **Повышенние качества за счёт использование Chain-of-Density (CoD) промтинга:**
    Итеративно сжимает резюме, чтобы максимально увеличить плотность информации, удаляя избыточность и повысить фактическую точность.

## Структура проекта

```Text
Long-document-CoD-summarizer/
├── src/
|   └── summarizer/
│       ├── app.py              # Gradio UI
│       ├── config.py           # Конфигурация модели
│       ├── ingestion.py        # Парсинг документа
│       └── summarizer.py       # Логика работы LLM
├── tests/                  
├── Dockerfile
├── .env.example                # Пример файла .env для конфигурации
├── pyproject.toml          
└── requirements.txt        
```

## 🚀 Getting started

### 1. Клонируйте репозиторий
```bash
https://github.com/IvanEfremovW/Long-document-CoD-summarizer.git
cd Long-document-CoD-summarizer
```

### 2.1 Запуск с Docker (рекомендуется)

#### Сборка образа
```bash
docker build -t Long-document-CoD-summarizer .
```
#### Запуск на GPU
```bash
docker run -it --rm --gpus all -p 7860:7860 Long-document-CoD-summarizer
```
#### Запуск на CPU (медленно, только для теста)
```bash
docker run -it --rm -p 7860:7860 Long-document-CoD-summarizer
```

### 2.2 Альтернативно запуск без Docker
>⚠️ Требуется Python 3.10+, CUDA 12.1 и совместимая версия PyTorch для запуска на GPU. 
```bash
# 1. Установите зависимости
pip install -r requirements.txt

# 2. Запустите app.py
python -m src.summarizer.app
```

### 3. Откройте веб-интерфейс
После запуска контейнера откройте в браузере:
👉 http://127.0.0.1:7860

Вы увидите интерфейс Gradio:

1. Нажмите “Choose File”
2. Загрузите документ
3. Дождитесь генерации аннотации
