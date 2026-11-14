"""
محرك تبديل الوجوه: InsightFace (buffalo_l + inswapper_128) + GFPGAN
"""
import sys
import threading
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

# ---- إصلاح توافق basicsr مع torchvision >= 0.17 (functional_tensor أُزيلت) ----
try:
    import torchvision.transforms.functional_tensor  # noqa: F401
except Exception:  # pragma: no cover
    try:
        import torchvision.transforms.functional as _F
        sys.modules["torchvision.transforms.functional_tensor"] = _F
    except Exception:
        pass


@dataclass
class SwapOptions:
    enhance: bool = True          # تحسين الوجه GFPGAN
    enhance_strength: float = 0.8 # 0..1 قوة التحسين
    blend: float = 1.0            # 0..1 قوة الدمج
    color_match: float = 0.5      # 0..1 تطابق لون البشرة
    all_faces: bool = False       # تبديل كل الوجوه أم الأكبر فقط
    watermark: bool = True        # علامة "AI Generated"


def _get_providers():
    try:
        import onnxruntime as ort
        avail = ort.get_available_providers()
    except Exception:
        avail = []
    if "CUDAExecutionProvider" in avail:
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


class FaceSwapEngine:
    """يُحمَّل مرة واحدة ويُعاد استخدامه لكل المهام."""

    def __init__(self, models_dir: Path):
        import insightface
        from insightface.app import FaceAnalysis

        self.models_dir = Path(models_dir)
        self.providers = _get_providers()
        self.use_gpu = "CUDAExecutionProvider" in self.providers

        swapper_path = self.models_dir / "inswapper_128.onnx"
        if not swapper_path.exists():
            raise FileNotFoundError(
                f"النموذج غير موجود: {swapper_path}\nشغّل: python download_models.py"
            )

        # buffalo_l يُنزَّل تلقائياً داخل models/models/buffalo_l
        self.analyzer = FaceAnalysis(
            name="buffalo_l", root=str(self.models_dir), providers=self.providers
        )
        self.analyzer.prepare(ctx_id=0 if self.use_gpu else -1, det_size=(640, 640))
        self.swapper = insightface.model_zoo.get_model(
            str(swapper_path), providers=self.providers
        )
        self._enhancer = None
        self._enh_lock = threading.Lock()

    # ------------------------------------------------------------------ GFPGAN
    def enhancer(self):
        with self._enh_lock:
            if self._enhancer is None:
                from gfpgan import GFPGANer

                path = self.models_dir / "GFPGANv1.4.pth"
                if not path.exists():
                    raise FileNotFoundError(f"نموذج GFPGAN غير موجود: {path}")
                self._enhancer = GFPGANer(
                    model_path=str(path),
                    upscale=1,
                    arch="clean",
                    channel_multiplier=2,
                    bg_upsampler=None,
                )
            return self._enhancer

    # ------------------------------------------------------------ source face
    def get_source_face(self, image_bgr: np.ndarray):
        faces = self.analyzer.get(image_bgr)
        if not faces:
            raise ValueError("لم يتم العثور على وجه في الصورة المرفوعة")
        return max(faces, key=lambda f: _area(f.bbox))

    # ------------------------------------------------------------ per frame
    def process_frame(self, frame: np.ndarray, source_face, opts: SwapOptions) -> np.ndarray:
        faces = self.analyzer.get(frame)
        if not faces:
            return _watermark(frame) if opts.watermark else frame
        if not opts.all_faces:
            faces = [max(faces, key=lambda f: _area(f.bbox))]

        out = frame.copy()
        for face in faces:
            swapped = self.swapper.get(out, face, source_face, paste_back=True)
            mask = _face_mask(face, frame.shape)
            if opts.color_match > 0:
                swapped = _color_transfer(swapped, out, mask, opts.color_match)
            alpha = (mask * float(np.clip(opts.blend, 0, 1)))[..., None]
            out = (swapped.astype(np.float32) * alpha + out.astype(np.float32) * (1 - alpha))
            out = np.clip(out, 0, 255).astype(np.uint8)

        if opts.enhance and opts.enhance_strength > 0:
            try:
                _, _, enhanced = self.enhancer().enhance(
                    out, has_aligned=False, only_center_face=not opts.all_faces, paste_back=True
                )
                if enhanced is not None:
                    s = float(np.clip(opts.enhance_strength, 0, 1))
                    out = cv2.addWeighted(enhanced, s, out, 1 - s, 0)
            except Exception as e:  # لا نوقف الفيديو بسبب إطار واحد
                print("[enhancer] frame skipped:", e)

        return _watermark(out) if opts.watermark else out


# ============================================================== helpers
def _area(b):
    return max(0, b[2] - b[0]) * max(0, b[3] - b[1])


def _face_mask(face, shape) -> np.ndarray:
    """قناع ناعم الحواف لمنطقة الوجه (0..1)."""
    h, w = shape[:2]
    mask = np.zeros((h, w), np.float32)
    x1, y1, x2, y2 = face.bbox.astype(int)
    fw, fh = x2 - x1, y2 - y1

    lm = getattr(face, "landmark_2d_106", None)
    if lm is not None and len(lm) > 0:
        pts = lm.astype(np.int32)
        hull = cv2.convexHull(pts)
        cv2.fillConvexPoly(mask, hull, 1.0)
        # توسيع بسيط نحو الجبهة
        k = max(3, int(fh * 0.08)) | 1
        mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    else:
        center = ((x1 + x2) // 2, (y1 + y2) // 2)
        cv2.ellipse(mask, center, (int(fw * 0.45), int(fh * 0.55)), 0, 0, 360, 1.0, -1)

    feather = max(5, int(min(fw, fh) * 0.15)) | 1
    mask = cv2.GaussianBlur(mask, (feather, feather), 0)
    return np.clip(mask, 0, 1)


def _color_transfer(src, ref, mask, strength) -> np.ndarray:
    """نقل إحصاءات اللون (LAB) من الوجه الأصلي إلى الوجه المبدَّل لمطابقة الإضاءة ولون البشرة."""
    sel = mask > 0.5
    if sel.sum() < 50:
        return src
    s_lab = cv2.cvtColor(src, cv2.COLOR_BGR2LAB).astype(np.float32)
    r_lab = cv2.cvtColor(ref, cv2.COLOR_BGR2LAB).astype(np.float32)
    out = s_lab.copy()
    for c in range(3):
        s_mu, s_sd = s_lab[..., c][sel].mean(), s_lab[..., c][sel].std() + 1e-6
        r_mu, r_sd = r_lab[..., c][sel].mean(), r_lab[..., c][sel].std() + 1e-6
        out[..., c] = (s_lab[..., c] - s_mu) * (r_sd / s_sd) + r_mu
    out = np.clip(out, 0, 255).astype(np.uint8)
    matched = cv2.cvtColor(out, cv2.COLOR_LAB2BGR)
    s = float(np.clip(strength, 0, 1))
    return cv2.addWeighted(matched, s, src, 1 - s, 0)


def _watermark(img):
    h, w = img.shape[:2]
    scale = max(0.4, w / 1600)
    txt = "AI Generated"
    (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)
    org = (w - tw - 12, h - 12)
    cv2.putText(img, txt, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(img, txt, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 2, cv2.LINE_AA)
    return img
