# تشخیص سبک‌وزن ناهنجاری در سیستم‌های انرژی سایبرفیزیکی

[English Version](README.md)

## معرفی پروژه

این پروژه یک نمونه مفهومی بازتولیدپذیر در پایتون است که با استفاده از مدل‌های سبک‌وزن یادگیری ماشین، شرایط عادی و غیرعادی را در یک سیستم انرژی سایبرفیزیکی شبیه‌سازی‌شده تشخیص می‌دهد.

سه مدل اصلی در پروژه مقایسه شده‌اند:

- Logistic Regression
- Random Forest
- Isolation Forest

هدف فقط رسیدن به دقت بالا نیست. معیارهایی مانند Recall کلاس ناهنجاری، False-Negative Rate، زمان inference و اندازه مدل نیز بررسی می‌شوند تا مشخص شود هر مدل تا چه اندازه برای استقرار روی سامانه‌های محدود مناسب است.

> این پروژه یک Proof of Concept است و هنوز با داده واقعی، میکروکنترلر، FPGA یا Hardware-in-the-Loop اعتبارسنجی نشده است.

---

## انگیزه انجام پروژه

سامانه‌های انرژی مدرن مانند مبدل‌های قدرت، شارژرهای خودرو برقی، شبکه‌های هوشمند و زیرساخت‌های صنعتی به حسگرها، ارتباطات دیجیتال و کنترل‌کننده‌های نهفته وابسته‌اند.

خرابی حسگر، خطای اندازه‌گیری، شرایط کاری غیرعادی یا دست‌کاری عمدی داده‌ها می‌تواند باعث تصمیم کنترلی اشتباه، کاهش قابلیت اطمینان و حتی آسیب فیزیکی شود.

بنابراین یک سامانه تشخیص مناسب باید بتواند:

- شرایط عادی و غیرعادی را جدا کند؛
- خطاها و حملات ساده را تشخیص دهد؛
- False Negative را کاهش دهد؛
- با تأخیر کم عمل کند؛
- حافظه کمی مصرف کند؛
- و در آینده قابلیت اجرا روی سخت‌افزارهای محدود را داشته باشد.

---

## محدوده فعلی پروژه

نسخه فعلی پروژه از داده‌های مصنوعی استفاده می‌کند. ویژگی‌های اصلی داده عبارت‌اند از:

- ولتاژ AC
- جریان
- فرکانس
- دما
- ولتاژ لینک DC
- اعوجاج هارمونیکی کل یا THD
- توان اکتیو

هر رکورد یکی از دو برچسب زیر را دارد:

- `normal`
- `anomaly`

در نسخه فعلی، مسئله به‌صورت تشخیص دودویی تعریف شده است؛ یعنی مدل فقط مشخص می‌کند رکورد عادی است یا غیرعادی و نوع دقیق ناهنجاری را تعیین نمی‌کند.

---

## سناریوهای ناهنجاری شبیه‌سازی‌شده

### افت ولتاژ

ولتاژ کمتر از محدوده معمول قرار می‌گیرد. این وضعیت می‌تواند نماینده افت ولتاژ شبکه، بار سنگین یا اختلال سمت منبع باشد.

### اضافه‌ولتاژ

ولتاژ از محدوده عادی بالاتر می‌رود. این وضعیت می‌تواند به خطای کنترل یا تنظیم نامناسب مبدل مربوط باشد.

### جهش جریان

جریان به‌صورت ناگهانی افزایش پیدا می‌کند. این حالت می‌تواند نماینده اضافه‌بار، جریان هجومی یا خطای مبدل باشد.

### ناهنجاری دما

دمای تجهیز از محدوده معمول بالاتر می‌رود. این وضعیت می‌تواند به اضافه‌بار، افزایش تلفات یا تهویه نامناسب مربوط باشد.

### Sensor Bias Attack

در این سناریو مقدار یک حسگر با یک بایاس ثابت یا جهت‌دار تغییر می‌کند.

### False-Data Injection

در این حالت برخی اندازه‌گیری‌ها به‌صورت عمدی دست‌کاری می‌شوند. این سناریو یک نمایش ساده از حمله تزریق داده جعلی است و حملات پیچیده و پنهان‌کارانه را پوشش نمی‌دهد.

### داده‌های گمشده و نویزی

برخی نمونه‌ها می‌توانند شامل مقادیر گمشده یا نویز غیرعادی باشند تا مقاومت فرایند پیش‌پردازش بررسی شود.

---

## ساختار مخزن

```text
lightweight-energy-anomaly-detection/
├── README.md
├── README_FA.md
├── requirements.txt
├── LICENSE
├── .gitignore
├── data/
│   └── synthetic_energy_data.csv
├── models/
├── results/
└── src/
    ├── generate_data.py
    ├── preprocess.py
    ├── visualize.py
    ├── train.py
    ├── evaluate.py
    └── predict.py
```

### توضیح پوشه‌ها

- `data/`: داده مصنوعی پروژه
- `src/`: اسکریپت‌های اصلی
- `models/`: مدل‌های ذخیره‌شده
- `results/`: نمودارها، معیارها و خروجی‌های مقایسه

---

## پیش‌نیازها

- Python 3.10 یا نسخه سازگار
- pip
- Git
- محیط مجازی پایتون

کتابخانه‌های اصلی:

- NumPy
- pandas
- scikit-learn
- matplotlib
- joblib

---

## دریافت پروژه

```bash
git clone https://github.com/AshkanAld/lightweight-energy-anomaly-detection.git
cd lightweight-energy-anomaly-detection
```

---

## ساخت محیط مجازی

### ویندوز

```bash
python -m venv .venv
.venv\Scripts\activate
```

### لینوکس یا macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
```

نصب وابستگی‌ها:

```bash
pip install -r requirements.txt
```

---

## اجرای مرحله‌به‌مرحله پروژه

### مرحله ۱: تولید داده مصنوعی

```bash
python src/generate_data.py
```

در نسخه فعلی، مجموعه داده شامل ۵۰۰۰ رکورد است:

- ۴۰۰۰ رکورد عادی
- ۱۰۰۰ رکورد غیرعادی

فایل خروجی:

```text
data/synthetic_energy_data.csv
```

### مرحله ۲: پیش‌پردازش

```bash
python src/preprocess.py
```

این مرحله شامل موارد زیر است:

- خواندن داده
- بررسی مقادیر گمشده
- آماده‌سازی ویژگی‌ها
- تبدیل برچسب‌ها
- تقسیم داده به train و test
- استانداردسازی در صورت نیاز

### مرحله ۳: مصورسازی

```bash
python src/visualize.py
```

خروجی‌ها می‌توانند شامل موارد زیر باشند:

- توزیع ویژگی‌ها
- مقایسه داده عادی و غیرعادی
- نمودار هم‌بستگی
- توزیع برچسب‌ها
- تغییرات ولتاژ، جریان، دما و THD

### مرحله ۴: آموزش Logistic Regression

```bash
python src/train.py --model logistic
```

این مدل به‌عنوان baseline استفاده می‌شود.

مزایا:

- بسیار کوچک
- بسیار سریع
- تفسیرپذیر

محدودیت:

- توان کمتر در مدل‌سازی روابط غیرخطی

### مرحله ۵: آموزش Random Forest

```bash
python src/train.py --model random_forest
```

مزایا:

- قدرت بالا در مدل‌سازی روابط غیرخطی
- امکان استخراج Feature Importance
- عملکرد قوی روی داده مصنوعی

محدودیت:

- اندازه بزرگ‌تر
- inference کندتر

### مرحله ۶: آموزش Isolation Forest

```bash
python src/train.py --model isolation_forest
```

Isolation Forest یک روش بدون نظارت برای تشخیص ناهنجاری است.

در نسخه فعلی، مقدار contamination برابر با نسبت ناهنجاری در داده مصنوعی انتخاب شده است:

```text
contamination = 0.20
```

### مرحله ۷: ارزیابی مدل‌ها

```bash
python src/evaluate.py
```

معیارهای ارزیابی:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- False Positive
- False Negative
- False-Negative Rate
- Model Size
- Inference Time

### مرحله ۸: پیش‌بینی روی داده جدید

```bash
python src/predict.py
```

این اسکریپت مدل ذخیره‌شده را بارگذاری می‌کند و برای نمونه جدید، برچسب عادی یا ناهنجار تولید می‌کند.

---

## نتایج فعلی

| مدل | Accuracy | Recall ناهنجاری | F1-score | ROC-AUC | False Negative | False Positive | اندازه مدل |
|---|---:|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 89.4% | 77.5% | 74.5% | 0.909 | 45 | 61 | 1.876 KB |
| Random Forest | 100% | 100% | 100% | 1.000 | 0 | 0 | 1063.696 KB |
| Isolation Forest | 97.5% | 92.0% | 93.6% | 0.997 | 16 | 9 | 1478.094 KB |

### زمان inference

| مدل | زمان میانه inference برای هر نمونه |
|---|---:|
| Logistic Regression | 0.002739 ms |
| Random Forest | 0.101078 ms |
| Isolation Forest | 0.068670 ms |

> زمان‌های inference روی رایانه محلی اندازه‌گیری شده‌اند و برای مقایسه نسبی مناسب‌اند، نه به‌عنوان معیار مستقل از سخت‌افزار.

---

## تفسیر نتایج

### Logistic Regression

- کوچک‌ترین و سریع‌ترین مدل
- ۴۵ ناهنجاری را از دست داده است
- برای سخت‌افزار محدود جذاب است
- برای کاربردهای ایمنی نیاز به بهبود Recall دارد

### Random Forest

- روی داده مصنوعی عملکرد کامل داشته است
- هیچ False Positive یا False Negative ثبت نشده است
- این نتیجه احتمالاً به دلیل جداشدگی واضح داده مصنوعی است
- نباید انتظار داشت روی داده واقعی نیز الزاماً ۱۰۰ درصد عمل کند

### Isolation Forest

- بدون نیاز مستقیم به برچسب در مرحله آموزش
- Recall برابر ۹۲ درصد
- گزینه مناسب برای محیط‌هایی با برچسب‌گذاری محدود
- اندازه مدل آن نسبتاً زیاد است

---

## چرا Recall و False Negative مهم‌اند؟

در سامانه‌های ایمنی‌حساس، Accuracy به‌تنهایی کافی نیست.

False Negative یعنی یک خطا یا حمله واقعی وجود دارد اما مدل آن را عادی تشخیص می‌دهد.

این وضعیت ممکن است باعث شود:

- خطای تجهیز نادیده گرفته شود؛
- حمله سایبری شناسایی نشود؛
- کنترل‌کننده بر اساس داده اشتباه تصمیم بگیرد؛
- خسارت فیزیکی یا خاموشی رخ دهد.

فرمول Recall:

```text
Recall = TP / (TP + FN)
```

هرچه False Negative کمتر باشد، Recall بیشتر است.

---

## ملاحظات استقرار سبک‌وزن

برای استقرار واقعی فقط اندازه فایل مدل کافی نیست. موارد زیر نیز باید بررسی شوند:

- RAM
- Flash memory
- تعداد عملیات
- توان مصرفی
- latency
- دقت عددی
- نوع پردازنده
- قابلیت اجرای بلادرنگ
- هزینه پیش‌پردازش

در این پروژه، Logistic Regression سبک‌ترین گزینه است، اما Recall پایین‌تری دارد. Random Forest عملکرد بهتری دارد، اما اندازه و latency آن بیشتر است.

---

## محدودیت‌ها

- داده‌ها مصنوعی هستند.
- داده واقعی یا آزمایشگاهی استفاده نشده است.
- حملات سایبری ساده‌سازی شده‌اند.
- مدل‌ها روی سخت‌افزار واقعی اجرا نشده‌اند.
- Hardware-in-the-Loop انجام نشده است.
- تشخیص فقط دودویی است.
- تحلیل زمانی و ترتیبی انجام نشده است.
- Cross-validation و تنظیم گسترده hyperparameterها محدود بوده است.
- مقاومت در برابر adversarial attacks بررسی نشده است.
- contamination در Isolation Forest از قبل معلوم فرض شده است.
- عملکرد کامل Random Forest احتمالاً ناشی از ساده‌بودن داده مصنوعی است.

---

## توسعه‌های آینده

### داده و اعتبارسنجی

- استفاده از داده واقعی
- استفاده از داده آزمایشگاهی
- استفاده از داده HIL
- استفاده از مجموعه‌داده‌های عمومی سیستم قدرت

### مدل‌های پیشرفته‌تر

- One-Class SVM
- Autoencoder
- LSTM
- GRU
- Temporal Convolutional Network
- مدل‌های کوچک عصبی

### تشخیص چندکلاسه

در نسخه بعدی، مدل می‌تواند نوع ناهنجاری را نیز تعیین کند:

- voltage sag
- overvoltage
- current spike
- temperature anomaly
- sensor bias
- false-data injection

### بهینه‌سازی

- Quantization
- Pruning
- Knowledge Distillation
- Feature Selection
- Threshold Optimization
- کاهش تعداد درخت‌های Random Forest

### استقرار سخت‌افزاری

- Raspberry Pi
- Microcontroller
- TinyML
- FPGA
- DSP
- Edge Device

### ارزیابی اعتمادپذیری

- uncertainty estimation
- calibration
- adversarial robustness
- explainability
- out-of-distribution detection
- secure inference

---

## نکات آموزشی و مرور شخصی

### دلیل استفاده از Logistic Regression

وجود یک baseline ساده مشخص می‌کند آیا مدل پیچیده‌تر واقعاً ارزش افزوده دارد یا خیر.

### دلیل عملکرد کامل Random Forest

ناهنجاری‌ها در داده مصنوعی نسبتاً واضح و جدا از داده عادی تولید شده‌اند. بنابراین نتیجه ۱۰۰ درصد نباید به داده واقعی تعمیم داده شود.

### اهمیت Isolation Forest

در سیستم واقعی ممکن است داده عادی زیاد باشد اما برچسب‌گذاری خطا و حمله دشوار باشد. Isolation Forest برای چنین شرایطی مهم است.

### موازنه اصلی پروژه

بهترین مدل لزوماً مدلی با بالاترین Accuracy نیست. باید بین این عوامل موازنه برقرار شود:

- Recall
- False Negative
- False Positive
- اندازه مدل
- زمان inference
- تفسیرپذیری
- قابلیت استقرار

---

## اجرای سریع

```bash
git clone https://github.com/AshkanAld/lightweight-energy-anomaly-detection.git
cd lightweight-energy-anomaly-detection

python -m venv .venv
```

ویندوز:

```bash
.venv\Scripts\activate
```

لینوکس یا macOS:

```bash
source .venv/bin/activate
```

سپس:

```bash
pip install -r requirements.txt
python src/generate_data.py
python src/preprocess.py
python src/visualize.py
python src/train.py --model logistic
python src/train.py --model random_forest
python src/train.py --model isolation_forest
python src/evaluate.py
```

---

## بازتولیدپذیری

برای بازتولید نتایج:

- از همان نسخه کتابخانه‌ها استفاده شود؛
- random seed ثابت باشد؛
- همه دستورها از پوشه اصلی اجرا شوند؛
- فایل‌های خروجی قبلی بررسی شوند؛
- تغییرات با Git ثبت شوند.

دستورهای معمول Git:

```bash
git status
git add README_FA.md
git commit -m "Add Persian README"
git push
```

---

## وضعیت فعلی پروژه

### تکمیل‌شده

- تولید داده مصنوعی
- پیش‌پردازش
- مصورسازی
- Logistic Regression
- Random Forest
- Isolation Forest
- محاسبه معیارها
- ماتریس اغتشاش
- Feature Importance
- Inference Time
- Model Size
- مقایسه مدل‌ها
- مستندسازی محدودیت‌ها

### انجام‌نشده

- داده واقعی
- استقرار سخت‌افزاری
- TinyML
- FPGA
- HIL
- تشخیص چندکلاسه
- مدل‌های زمانی
- تست adversarial robustness

---

## نتیجه‌گیری

این پروژه نشان می‌دهد که یک مسئله نظارت بر سیستم انرژی را می‌توان به یک مسئله یادگیری ماشین تبدیل کرد و مدل‌ها را علاوه بر معیارهای پیش‌بینی، از نظر ایمنی و استقرار نیز مقایسه کرد.

خلاصه نتیجه:

- Random Forest بهترین عملکرد را روی داده مصنوعی داشته است.
- Logistic Regression کوچک‌ترین و سریع‌ترین مدل است.
- Isolation Forest بدون نیاز مستقیم به برچسب، عملکرد قوی داشته است.
- نتایج هنوز برای نتیجه‌گیری درباره عملکرد واقعی کافی نیستند.
- مرحله بعدی مهم، استفاده از داده واقعی یا HIL و سپس استقرار روی سخت‌افزار محدود است.

---

## نویسنده

**Ashkan Alidousti Shahraki**

Electrical Power Engineer

Research interests:

- Power Systems
- Power Electronics
- Anomaly Detection
- Cyber-Physical Energy Systems
- Embedded AI

GitHub:

```text
https://github.com/AshkanAld
```

مخزن پروژه:

```text
https://github.com/AshkanAld/lightweight-energy-anomaly-detection
```

---

## مجوز

شرایط استفاده از پروژه مطابق فایل `LICENSE` است. اگر هنوز مجوزی اضافه نشده، استفاده از مجوز MIT برای یک پروژه آموزشی و متن‌باز گزینه مناسبی است.
