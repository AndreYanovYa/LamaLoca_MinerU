# scripts/test_rag.py
"""Тест RAG-пайплайна: задаём вопрос и получаем ответ с цитатами."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.core.settings import Settings
from src.llm.rag_pipeline import RAGPipeline
from src.core.exceptions import LamaLocaError


def main():
    if len(sys.argv) < 2:
        print("Использование: python test_rag.py <вопрос>")
        print('Пример: python test_rag.py "Что такое предел функции?"')
        sys.exit(1)
    
    question = " ".join(sys.argv[1:])
    
    print("\n" + "=" * 60)
    print(" 🤖 ТЕСТ RAG-ПАЙПЛАЙНА (lama-loca v0.1)")
    print("=" * 60 + "\n")
    
    print(f"📝 Вопрос: {question}\n")
    print("⏳ Поиск в библиотеке и генерация ответа...\n")
    
    try:
        settings = Settings.load(Path("config.yaml"))
        rag = RAGPipeline(settings)
        result = rag.answer(question)
        
        print("=" * 60)
        print("💬 ОТВЕТ:")
        print("=" * 60)
        print(result["answer"])
        print()
        
        if result["sources"]:
            print("=" * 60)
            print("📚 ИСТОЧНИКИ:")
            print("=" * 60)
            for i, source in enumerate(result["sources"], start=1):
                print(f"[{i}] {source['title']}", end="")
                if source.get("page_number"):
                    print(f", стр. {source['page_number']}")
                elif source.get("chapter"):
                    print(f", {source['chapter']}")
                else:
                    print()
        else:
            print("⚠️ Источники не указаны в ответе.")
        
        print("=" * 60)
        print(f"📦 Использовано чанков: {len(result['context_chunks'])}")
        print("=" * 60 + "\n")
        
    except LamaLocaError as e:
        print(f"\n❌ Ошибка приложения: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Неожиданная ошибка: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()