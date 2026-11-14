"""
تنزيل النماذج المطلوبة إلى مجلد backend/models
- inswapper_128.onnx  (~530MB)  تبديل الوجوه
- GFPGANv1.4.pth      (~340MB)  تحسين الوجه
- buffalo_l           يُنزَّل تلقائياً عبر InsightFace عند أول تشغيل
"""
import os
import sys
import urllib.request
from pathlib import Path

MODELS = Path(os.getenv("MODELS_DIR", Path(__file__).resolve().parent / "models"))
MODELS.mkdir(parents=True, exist_ok=True)

FILES = {
    "inswapper_128.onnx": [
        "https://huggingface.co/ezioruan/inswapper_128.onnx/resolve/main/inswapper_128.onnx",
        "https://huggingface.co/hacksider/deep-live-cam/resolve/main/inswapper_128.onnx",
    ],
    "GFPGANv1.4.pth": [
        "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth",
    ],
}


def _progress(count, block, total):
    if total > 0:
        pct = min(100, count * block * 100 // total)
        sys.stdout.write(f"\r  {pct:3d}%")
        sys.stdout.flush()


def main():
    for name, urls in FILES.items():
        dst = MODELS / name
        if dst.exists() and dst.stat().st_size > 1_000_000:
            print(f"✓ {name} موجود")
            continue
        for url in urls:
            try:
                print(f"↓ {name}  ←  {url}")
                urllib.request.urlretrieve(url, dst, _progress)
                print("\n  تم")
                break
            except Exception as e:
                print(f"\n  فشل: {e}")
                dst.unlink(missing_ok=True)
        else:
            print(f"✗ تعذّر تنزيل {name} — نزّله يدوياً وضعه في {MODELS}")

    print("\nتهيئة buffalo_l ...")
    try:
        from insightface.app import FaceAnalysis
        FaceAnalysis(name="buffalo_l", root=str(MODELS), providers=["CPUExecutionProvider"]).prepare(ctx_id=-1)
        print("✓ buffalo_l جاهز")
    except Exception as e:
        print("✗ buffalo_l:", e)


if __name__ == "__main__":
    main()
