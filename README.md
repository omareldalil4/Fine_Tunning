# Fine_Tunning

هذا المشروع يقوم بتعديل نموذج CNN مدرّب مسبقاً (ResNet-18) لتطبيق عملية Fine-Tuning على مجموعة بيانات CIFAR-10 باستخدام مكتبة PyTorch.

## الملفات

- **train.py**: سكريبت التدريب الذي يقوم بتحميل البيانات، تعديل النموذج، وتنفيذ عملية التدريب.
- **.github/workflows/ci.yml**: ملف GitHub Actions لتشغيل سكريبت التدريب تلقائيًا عند كل push أو pull request.
- **requirements.txt**: يحتوي على المكتبات المطلوبة.
- **README.md**: هذا الملف.

## كيفية التشغيل

1. قم باستنساخ المشروع:
   ```bash
   git clone https://github.com/omareldalil4/Fine_Tunning.git
