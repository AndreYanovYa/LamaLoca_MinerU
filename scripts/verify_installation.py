# scripts/verify_installation.py
import sys
import sqlite3
import requests

def check_python():
    version = sys.version.split()[0]
    print(f"[✓] Python версия: {version}")
    if sys.version_info < (3, 11): # Смягчил до 3.11, так как в логах видно Python 3.11
        print("[!] Внимание: рекомендуется Python 3.12, но 3.11 тоже будет работать.")
    else:
        print("[✓] Версия Python подходит.")

def check_sqlite_fts5():
    conn = sqlite3.connect(":memory:")
    cursor = conn.cursor()
    try:
        cursor.execute("CREATE VIRTUAL TABLE test_fts USING fts5(content)")
        print("[✓] SQLite FTS5 поддерживается данной сборкой Python.")
    except sqlite3.OperationalError:
        print("[✗] ОШИБКА: SQLite FTS5 не доступен! Проверьте сборку Python для Windows.")
        sys.exit(1)
    finally:
        conn.close()

def check_ollama():
    try:
        response = requests.get("http://127.0.0.1:11434/api/tags", timeout=5)
        if response.status_code == 200:
            models = [m['name'] for m in response.json().get('models', [])]
            print(f"[✓] Ollama запущена. Доступные модели: {', '.join(models) if models else 'нет загруженных моделей'}")
        else:
            print("[✗] Ollama вернула неожиданный ответ.")
    except requests.exceptions.ConnectionError:
        print("[✗] ОШИБКА: Ollama не запущена или недоступна по адресу http://127.0.0.1:11434")
        print("    1. Запустите приложение Ollama.")
        print("    2. В командной строке выполните: ollama pull nomic-embed-text")

if __name__ == "__main__":
    print("=== Проверка окружения lama-loca v0.1 ===")
    check_python()
    check_sqlite_fts5()
    check_ollama()
    print("=== Проверка завершена ===")