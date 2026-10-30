# scripts/setup_gpu.py
"""Диагностика и ускорение GPU-стека lama-loca (Windows / NVIDIA RTX).

Проблема, которую решает скрипт: обычный `pip install -r requirements.txt`
ставит torch в CPU-режиме, а старые сборки Ollama не видят GPU Blackwell
(RTX 50xx, sm_120) — всё считается на процессоре («работает, но очень
медленно»). Скрипт проверяет все три слоя и печатает точные команды починки.

Запуск:  python scripts/setup_gpu.py
"""
from __future__ import annotations

import subprocess


def _run(cmd: str) -> str:
    try:
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True,
                           timeout=60)
        return (r.stdout or r.stderr).strip()
    except Exception as e:  # noqa: BLE001
        return f"<не выполнено: {e}>"


def check_nvidia_smi():
    print("=== 1. Драйвер NVIDIA ===")
    out = _run("nvidia-smi --query-gpu=name,driver_version,memory.total "
               "--format=csv,noheader")
    # Команда могла «выполниться», но напечатать ошибку в stdout/stderr.
    bad_markers = ("not found", "не найдена", "Система не может",
                   "is not recognized", "<не выполнено")
    if not out or any(m in out for m in bad_markers):
        print("  ✗ nvidia-smi недоступен — установите драйвер NVIDIA (>= 560 "
              "для RTX 5060): https://www.nvidia.com/drivers")
        return False
    print(f"  ✓ {out}")
    try:
        drv = int(out.split(",")[1].split(".")[0])
        if drv < 560:
            print("  ⚠ Драйвер старше 560 — он не поддерживает CUDA 12.8 "
                  "(нужен для чипов Blackwell sm_120). Обновите драйвер.")
    except (IndexError, ValueError):
        pass
    return True


def check_torch():
    print("\n=== 2. PyTorch (используется pix2tex/texify для формул) ===")
    try:
        import torch
    except ImportError:
        print("  – torch не установлен (некритично: движки формул "
              "деградируют до lazy-PNG)")
        return
    print(f"  Версия torch: {torch.__version__}")
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        cap = torch.cuda.get_device_capability(0)
        print(f"  ✓ CUDA доступна: {name} (sm_{cap[0]}{cap[1]}), "
              f"torch built with {torch.version.cuda}")
        if cap[0] >= 12 and tuple(map(int, torch.version.cuda.split("."))) < (12, 8):
            print("  ⚠ Torch собран с CUDA < 12.8 — чипы Blackwell (RTX 50xx) "
                  "могут не поддерживаться. Переустановите:")
            print("      pip install torch torchvision --index-url "
                  "https://download.pytorch.org/whl/cu128")
    else:
        print("  ✗ torch в CPU-РЕЖИМЕ — распознавание формул будет медленным!")
        print("    Починка (Windows, RTX 5060):")
        print("      pip uninstall -y torch torchvision torchaudio")
        print("      pip install torch torchvision --index-url "
              "https://download.pytorch.org/whl/cu128")


OLLAMA_BASE = "http://127.0.0.1:11434"


def _ollama_fix_steps(reason: str):
    """Пошаговая починка Ollama->GPU для Windows (RTX 50xx/Blackwell)."""
    print(f"    Вероятная причина: {reason}")
    print("    Починка (по шагам):")
    print('    1) Полностью завершите Ollama: правый клик по иконке в трее -> Quit')
    print('       (или: taskkill /F /IM ollama.exe /T)')
    print("    2) Установите свежий Ollama-windows_amd64.exe (для RTX 50xx нужны")
    print("       сборки последних месяцев; старые молча считают на CPU):")
    print("       https://ollama.com/download/windows")
    print("    3) Проверьте переменные окружения — они могут глушить GPU:")
    print("       OLLAMA_NUM_GPU=0 или CUDA_VISIBLE_DEVICES=CPU отключают видеокарту.")
    print("       (Параметры системы -> Переменные среды; затем перезапуск Ollama)")
    print("    4) Запустите Ollama и проверьте лог сервера:")
    print(r'       type %LOCALAPPDATA%\Ollama\server.log | findstr -i "gpu cuda detect"')
    print("       Ищите 'inference compute ... NVIDIA GeForce RTX 5060'. Если написано")
    print("       'no compatible GPUs were discovered' — драйвер (у вас 617 — ок) ок,")
    print("       но сборка Ollama не поддерживает CUDA 12.8/sm_120: переустановите её.")
    print("    5) Прогрейте модель и проверьте размещение:")
    print('       ollama run qwen3.5:4b "/no_think hi"   (подождать ~10 c)')
    print("       ollama ps   -> PROCESSOR должен быть 100% GPU")
    print("    6) Перезапустите python scripts/setup_gpu.py — секция 3 покажет вердикт.")


def check_ollama():
    print("\n=== 3. Ollama (LLM + эмбеддинги — главный источник скорости) ===")
    try:
        import requests
    except ImportError:
        print("  – requests не установлен")
        return

    # 3a. /api/version — видит ли СЕРВЕР CUDA при старте.
    ver_ok = False
    try:
        data = requests.get(f"{OLLAMA_BASE}/api/version", timeout=5).json()
    except Exception:
        print("  – Ollama не отвечает на 127.0.0.1:11434 — запустите приложение Ollama.")
        return
    ver = data.get("version", "?")
    has_cuda = data.get("cuda")
    if has_cuda is True:
        print(f"  ✓ Ollama {ver}: флаг cuda=true — сервер обнаружил CUDA-устройства,")
        print("     инференс пойдёт на GPU.")
        ver_ok = True
    elif has_cuda is False:
        print(f"  ✗ Ollama {ver}: cuda=false — сервер НЕ обнаружил ни одного CUDA-")
        print("     устройства (ваш случай «RTX 5060 не видна»). Весь инференс на CPU:")
        print("     медленная генерация, таймауты, обрыв контекста и несмыкаемые $$/**.")
        _ollama_fix_steps("старая сборка Ollama без поддержки Blackwell (sm_120/CUDA 12.8)")
    else:
        print(f"  ? Ollama {ver}: /api/version не сообщает флаг CUDA — проверяем")
        print("     фактическое размещение загруженных моделей через /api/ps ...")

    # 3b. /api/ps — где РЕАЛЬНО считаются загруженные модели.
    # ВАЖНО: пустой /api/ps — НЕ признак CPU-режима: так выглядит сервер в
    # простое (модель выгружена по keep_alive). Вердикт о режиме даёт либо
    # флаг cuda из 3a, либо прогретая модель здесь.
    try:
        ps = requests.get(f"{OLLAMA_BASE}/api/ps", timeout=5).json()
    except Exception:
        print("  – /api/ps недоступен.")
        return
    models = ps.get("models") or []
    if not models:
        print("  – Сейчас ни одна модель не загружена в память (/api/ps пуст) —")
        print("    это состояние простоя, а НЕ доказательство CPU-режима.")
        if ver_ok:
            print("    Сборка с CUDA подтверждена флагом выше: после прогрева будет 100% GPU.")
            print("    Держите ollama.keep_alive_seconds: -1 (config.yaml; код сам")
            print("    подменит -1 на безопасные \"1h\") — модель не")
            print("    выгружается и всегда готова на VRAM.")
        else:
            print("    Прогрев для точного вердикта: ollama run qwen3.5:4b \"/no_think hi\",")
            print("    затем ollama ps (PROCESSOR: 100% GPU = OK).")
        return
    for m in models:
        name = m.get("name", "?")
        devices = m.get("details", {}).get("devices") or []
        size, vram = m.get("size") or 0, m.get("size_vram") or 0
        gpu_devs = [d for d in devices if str(d.get("type", "")).lower() == "gpu"]
        cpu_devs = [d for d in devices if str(d.get("type", "")).lower() != "gpu"]
        if gpu_devs and not cpu_devs:
            ids = ", ".join(str(d.get("id")) for d in gpu_devs)
            print(f"  ✓ '{name}': целиком на GPU ({ids}) — режим OK.")
        elif gpu_devs and cpu_devs:
            gid = ", ".join(str(d.get("id")) for d in gpu_devs)
            cid = ", ".join(str(d.get("id")) for d in cpu_devs)
            print(f"  ⚠ '{name}': частично на GPU ({gid}), частично на CPU ({cid}) —")
            print("    VRAM не хватает под весь контекст. Уменьшите num_ctx в config.yaml,")
            print("    поставьте ollama.keep_alive_seconds: 3600 и включите OLLAMA_FLASH_ATTENTION=1.")
        elif cpu_devs:
            cid = ", ".join(str(d.get("id")) for d in cpu_devs)
            print(f"  ✗ '{name}': считается на {cid} — ЭТО CPU-РЕЖИМ, главная причина")
            print("    медлительности (генерация рвёт $$/**, грузит Ryzen, long-context виснет).")
            _ollama_fix_steps("сервер Ollama выбрал CPU-бэкенд вместо RTX 5060")
        elif size and vram < size * 0.9:
            print(f"  ⚠ '{name}': лишь ~{int(vram / size * 100)}% весов в VRAM — offload на CPU.")
            print("    Уменьшите num_ctx или используйте квант помельче (qwen3.5:4b-q4_K_M).")
        else:
            print(f"  ? '{name}': данных о размещении нет — проверьте 'ollama ps' вручную.")


def tips():
    print("\n=== 4. Дополнительные ускорители ===")
    print("  • В config.yaml задайте ollama.keep_alive_seconds: -1 — модель "
          "не будет выгружаться из VRAM каждые 5 минут простоя")
    print("    (исчезнут длинные «зависания» перед каждым ответом).")
    print("  • Формулы: engine texify точнее, но тяжелее; pix2tex на CUDA "
          "быстрее. Результат уже кэшируется по SHA-1 изображения.")
    print("  • Импорт большого PDF — фоновый воркер; UI не блокируется, "
          "но сам инференс ускоряется только переходом на CUDA.")


if __name__ == "__main__":
    ok = check_nvidia_smi()
    check_torch()
    check_ollama()
    if ok:
        tips()
