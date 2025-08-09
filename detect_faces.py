import cv2
import sys, os

# 1) مسار الصورة (افتراضي test.jpg أو من سطر الأوامر)
img_path = sys.argv[1] if len(sys.argv) > 1 else "test.jpg"
if not os.path.exists(img_path):
    print(f"❌ Image not found: {img_path}")
    sys.exit(1)

# 2) قراءة الصورة وتحويلها لتدرّج رمادي
img = cv2.imread(img_path)
if img is None:
    print("❌ Could not read the image.")
    sys.exit(1)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 3) تحميل كاشف الوجوه (Haar Cascade) الموجود مع OpenCV
cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(cascade_path)
if face_cascade.empty():
    print("❌ Could not load Haar cascade.")
    sys.exit(1)

# 4) كشف الوجوه (يمكن تعديل القيم للتحسين)
faces = face_cascade.detectMultiScale(
    gray,
    scaleFactor=1.1,
    minNeighbors=5,
    minSize=(80, 80)
)

# 5) رسم مربعات حول الوجوه وكتابة العدد
for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

cv2.putText(img, f"Faces: {len(faces)}", (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

# 6) عرض وحفظ النتيجة
cv2.imshow("Face Detection (press any key to close)", img)
cv2.imwrite("output.jpg", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

print(f"✅ Done. Detected {len(faces)} face(s). Saved: output.jpg")