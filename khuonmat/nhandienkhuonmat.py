import cv2
import sys

# Khởi tạo đường dẫn đến ảnh và mô hình nhận diện đặc trưng haar cho mô hình cascade
imagePath = sys.argv[1]
cascPath = "haarcascade_frontalface_default.xml"

# Khởi tạo thuật toán Haar Cascade dùng hàm của thư viện opencv2
faceCascade = cv2.CascadeClassifier(cascPath)

# Đọc ảnh đầu vào và đổi thành ảnh gray
image = cv2.imread(imagePath)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Nhận diện khuôn mặt 
faces = faceCascade.detectMultiScale(
    gray,
    scaleFactor=1.1,  # bù đắp cho độ xa gần giữa các khuôn mặt 
    minNeighbors=5,   #số đối tượng gần đối tượng hiện tại
    minSize=(30, 30),
    flags = cv2.CASCADE_SCALE_IMAGE
)

print("Found {0} faces!".format(len(faces)))

# Vẽ hình chữ nhật lên ảnh.
for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

cv2.imshow("Faces found", image)
cv2.waitKey(0)
cv2.destroyAllWindows() 