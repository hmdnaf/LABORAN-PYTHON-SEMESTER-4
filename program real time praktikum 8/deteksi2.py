import cv2
from ultralytics import YOLO

# Load model
model = YOLO("best.pt")  # Ganti dengan path ke best.pt milikmu

# Buka kamera (0 = default webcam)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Tidak bisa membuka webcam.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Gagal mengambil frame dari webcam.")
        break

    # Deteksi objek
    results = model(frame)[0]

    # Gambar kotak deteksi dan label
    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])  # koordinat bbox
        conf = box.conf.item()                 # confidence
        cls = int(box.cls.item())              # kelas
        label = model.names[cls]               # nama kelas

        # Gambar kotak dan label
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Tampilkan hasil
    cv2.imshow("Real-time Object Detection", frame)

    # Tekan 'q' untuk keluar
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Bersihkan
cap.release()
cv2.destroyAllWindows()