# FaceSwap AI — تبديل الوجوه في الفيديو

```
faceswap-app/
├── frontend/index.html        ← الواجهة (عربية، Dark Mode، سحب وإفلات، مقارنة، تحميل)
└── backend/
    ├── main.py                ← خادم FastAPI + طابور المهام + ffmpeg
    ├── engine.py              ← InsightFace + inswapper_128 + GFPGAN + مطابقة اللون
    ├── download_models.py     ← تنزيل النماذج
    └── requirements.txt
```

## كيف يعمل
1. `buffalo_l` (InsightFace) يكتشف الوجه ويستخرج 106 نقطة ملامح + بصمة الهوية من صورتك.
2. لكل إطار: يُكتشف الوجه → `inswapper_128.onnx` يولّد الوجه الجديد بزاوية وتعبير الوجه الأصلي.
3. **مطابقة اللون**: نقل متوسط وانحراف الألوان في فضاء LAB من الوجه الأصلي (الإضاءة/البشرة).
4. **الدمج**: قناع ناعم الحواف من نقاط الملامح × قوة الدمج.
5. **GFPGAN v1.4**: ترميم الوجه وإزالة الضبابية، بقوة قابلة للتحكم.
6. `ffmpeg`: ترميز H.264 وإعادة الصوت الأصلي → MP4 جاهز للمتصفح.

## التشغيل محلياً (يُفضّل كرت NVIDIA + CUDA 12)

```bash
# 1) المتطلبات النظامية
#   Python 3.10 أو 3.11 ،  ffmpeg ،  (Windows: Visual C++ Build Tools لبناء insightface)
sudo apt-get install -y ffmpeg

# 2) بيئة افتراضية
cd faceswap-app/backend
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate

# 3) PyTorch المناسب لكرتك أولاً (مثال CUDA 12.1)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 4) باقي المكتبات
pip install -r requirements.txt
#   بدون GPU:  pip uninstall -y onnxruntime-gpu && pip install onnxruntime

# 5) النماذج (~900MB)
python download_models.py

# 6) تشغيل الخادم
uvicorn main:app --host 0.0.0.0 --port 8000
```
افتح: http://localhost:8000 — الخادم يقدّم الواجهة تلقائياً.

## الواجهة على استضافة منفصلة
يمكنك فتح `frontend/index.html` من أي مكان (أو رفعه على GitHub Pages)، ثم كتابة عنوان الخادم
(مثل رابط ngrok) في الحقل أعلى الصفحة والضغط على "اتصال". CORS مفعّل.

## Google Colab (T4 مجاناً) — باختصار
```python
!git clone <مستودعك> app || true          # أو ارفع المجلد
%cd app/backend
!apt-get -qq install -y ffmpeg
!pip -q install -r requirements.txt
!python download_models.py
!pip -q install pyngrok
from pyngrok import ngrok; ngrok.set_auth_token("YOUR_TOKEN")
print(ngrok.connect(8000).public_url)
!uvicorn main:app --host 0.0.0.0 --port 8000
```

## API
| الطريقة | المسار | الوصف |
|---|---|---|
| GET | `/api/health` | حالة الخادم وGPU |
| POST | `/api/jobs` | `video`, `face`, `enhance`, `enhance_strength`, `blend`, `color_match`, `all_faces`, `watermark`, `consent` |
| GET | `/api/jobs/{id}` | التقدم: `progress`, `frames_done`, `frames_total`, `fps_proc`, `eta_sec` |
| GET | `/api/jobs/{id}/result?download=1` | الفيديو النهائي |
| DELETE | `/api/jobs/{id}` | حذف الملفات |

## الأداء المتوقع (T4)
- بدون تحسين: ~8–15 إطار/ث عند 720p
- مع GFPGAN: ~3–6 إطار/ث → فيديو 30 ثانية (900 إطار) ≈ 3–5 دقائق

## مشاكل شائعة
- **`No module named torchvision.transforms.functional_tensor`**: مُعالج تلقائياً في `engine.py`.
- **insightface لا يُبنى على Windows**: ثبّت "Desktop development with C++" من Visual Studio Build Tools.
- **يعمل على CPU رغم وجود GPU**: تأكد أن `onnxruntime` العادي غير مثبت مع `onnxruntime-gpu` معاً.
- **لا يُكتشف وجه في الصورة**: استخدم صورة أمامية واضحة بإضاءة جيدة ودقة ≥ 512px.

## الاستخدام المسؤول
الخادم يرفض المعالجة دون تأكيد الموافقة، ويضيف علامة "AI Generated" افتراضياً.
استخدم الأداة فقط على وجوه لديك إذن صريح من أصحابها؛ إنشاء محتوى مضلّل أو جنسي أو انتحال
لأشخاص حقيقيين دون موافقتهم مخالف للقانون في كثير من الدول. ترخيص `inswapper_128` للاستخدام غير التجاري.
