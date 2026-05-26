import cv2
import numpy as np
from ultralytics import YOLO
from stable_baselines3 import PPO

# ==========================================
# 1. تحميل العقل (RL) والعيون (YOLO)
# ==========================================
print("Loading YOLOv8m Eyes...")
yolo_model = YOLO("yolov8m.pt") 

print("Loading RL Brain...")
rl_brain = PPO.load("ppo_traffic_brain")

# ==========================================
# 2. فتح الفيديو وسحب اللقطة الأولى للإعداد
# ==========================================
video_path = r"C:\Users\GTX\Videos\Porject Traffic AI\watermarked_preview (1).mp4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

ret, first_frame = cap.read()
if not ret:
    print("Error: Could not read the first frame.")
    exit()

# ==========================================
# 3. وضع الإعداد: رسم المسارات بالماوس
# ==========================================
polygons = {}
current_polygon = []
polygon_count = 0

def draw_polygon(event, x, y, flags, param):
    global current_polygon
    if event == cv2.EVENT_LBUTTONDOWN:
        current_polygon.append((x, y))

cv2.namedWindow("Setup: Draw Lanes")
cv2.setMouseCallback("Setup: Draw Lanes", draw_polygon)

print("\n--- تعليمات الإعداد ---")
print("1. اضغط بالزر الأيسر للماوس لرسم زوايا المسرب.")
print("2. اضغط (مسافة Space) لإغلاق المسرب والانتقال للذي يليه.")
print("3. اضغط (حرف r) لمسح النقاط الحالية وإعادة رسم المسرب.")
print("------------------------\n")

while polygon_count < 4:
    display_frame = first_frame.copy()
    
    # رسم المسارات التي تم حفظها مسبقاً
    for idx, poly in polygons.items():
        cv2.polylines(display_frame, [poly], isClosed=True, color=(0, 255, 0), thickness=2)
        cv2.putText(display_frame, f"Lane {idx} Saved", tuple(poly[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
        
    # رسم المسار الحالي أثناء النقرات
    if len(current_polygon) > 0:
        for i in range(len(current_polygon)):
            cv2.circle(display_frame, current_polygon[i], 5, (0, 0, 255), -1)
            if i > 0:
                cv2.line(display_frame, current_polygon[i-1], current_polygon[i], (0, 0, 255), 2)
                
    # التعليمات على الشاشة
    cv2.putText(display_frame, f"Draw Lane {polygon_count} (Space to save, 'r' to reset)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    
    cv2.imshow("Setup: Draw Lanes", display_frame)
    key = cv2.waitKey(1) & 0xFF
    
    if key == ord(' '): # حفظ المسرب
        if len(current_polygon) >= 3: # يجب أن يكون مضلع (3 نقاط على الأقل)
            polygons[polygon_count] = np.array(current_polygon, np.int32)
            print(f"Lane {polygon_count} Saved!")
            polygon_count += 1
            current_polygon = [] # تصفير للبدء بالمسرب التالي
        else:
            print("تحذير: يجب وضع 3 نقاط على الأقل لرسم مسرب!")
            
    elif key == ord('r'): # إعادة رسم المسرب الحالي
        current_polygon = []
        print(f"تم تصفير المسرب {polygon_count}. أعد الرسم.")
        
    elif key == ord('q'):
        print("تم الإلغاء.")
        exit()

# إغلاق نافذة الإعداد وإرجاع الفيديو للبداية
cv2.destroyWindow("Setup: Draw Lanes")
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

print("\nتم حفظ المسارات بنجاح! يتم الآن تشغيل الذكاء الاصطناعي...\n")

# ==========================================
# 4. المعالجة المباشرة (النظام يعمل)
# ==========================================
current_phase = 0
time_on_same_phase = 0
phases_names = ["Lane 0", "Lane 1", "Lane 2", "Lane 3"]

while True:
    ret, frame = cap.read()
    if not ret:
        print("انتهى الفيديو!")
        break 

    obs = np.zeros(6, dtype=np.int32)
    emergency_detected = -1

    # تشغيل YOLO
    results = yolo_model(frame, classes=[2, 3, 5, 7], verbose=False)[0]

    # رسم المضلعات على الشاشة بناءً على حالة الإشارة
    for idx, poly in polygons.items():
        color = (0, 255, 0) if idx == current_phase else (0, 0, 255)
        cv2.polylines(frame, [poly], isClosed=True, color=color, thickness=2)
        cv2.putText(frame, f"Lane {idx}", tuple(poly[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # عدّ السيارات والتأكد من وجودها داخل المضلعات المخصصة
    for box in results.boxes:
        bx1, by1, bx2, by2 = map(int, box.xyxy[0])
        cls = int(box.cls[0])
        cx, cy = (bx1 + bx2) // 2, (by1 + by2) // 2 

        for idx, poly in polygons.items():
            if cv2.pointPolygonTest(poly, (cx, cy), False) >= 0:
                obs[idx] += 1
                
                # افتراضياً: الشاحنة (7) تعامل كسيارة طوارئ للتجربة
                if cls == 7: 
                    emergency_detected = idx
                    cv2.putText(frame, "EMERGENCY!", (bx1, by1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                
                cv2.circle(frame, (cx, cy), 5, (255, 0, 0), -1)
                break 

    obs[4] = current_phase
    obs[5] = time_on_same_phase

    # اتخاذ القرار (النظام الهجين + PPO)
    if emergency_detected != -1:
        current_phase = emergency_detected
        time_on_same_phase = 0
        status_text = "EMERGENCY OVERRIDE!"
    else:
        action, _ = rl_brain.predict(obs, deterministic=True)
        queue_on_current_green = obs[current_phase]
        total_other_cars = sum(obs[:4]) - queue_on_current_green

        if action == 1 or time_on_same_phase >= 40: 
            if total_other_cars == 0 and queue_on_current_green > 0:
                time_on_same_phase = 0 
                status_text = "Smart Hold (Others empty)"
            else:
                current_phase = (current_phase + 1) % 4
                time_on_same_phase = 0
                status_text = "Switching Phase..."
        else:
            if queue_on_current_green == 0:
                current_phase = (current_phase + 1) % 4
                time_on_same_phase = 0
                status_text = "Phase Skipped (Empty)"
            else:
                time_on_same_phase += 1
                status_text = f"Holding Green... ({time_on_same_phase}/40)"

    # لوحة المعلومات
    cv2.rectangle(frame, (10, 10), (450, 150), (0, 0, 0), -1)
    cv2.putText(frame, f"AI Status: {status_text}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, f"Green Light: {phases_names[current_phase]}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, f"Queues: L0:{obs[0]} | L1:{obs[1]} | L2:{obs[2]} | L3:{obs[3]}", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    cv2.imshow("AI Smart Traffic Controller", frame)

    # اضغط q للخروج في أي وقت
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()