import os
import sys
import traci
from sumolib import checkBinary

script_dir = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.join(script_dir, "sim.sumocfg")

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("الرجاء التأكد من إضافة SUMO_HOME إلى متغيرات بيئة الويندوز")

# أضفنا الـ delay لترى المحاكاة بوضوح
sumoCmd = [checkBinary('sumo-gui'), "-c", config_path, "--start", "--delay", "150"]

print("جاري الاتصال...")
traci.start(sumoCmd)
print("السيطرة الآن لبايثون! جاري تطبيق نظام الشرق الأوسط...")

tls_id = traci.trafficlight.getIDList()[0]
num_signals = len(traci.trafficlight.getRedYellowGreenState(tls_id))
signals_per_dir = num_signals // 4 # تقسيم الإشارات على 4 اتجاهات

# =================================================================
# بناء المراحل الأربعة (نظام الشرق الأوسط: كل اتجاه يفتح لوحده)
# =================================================================
middle_east_phases = []
for i in range(4):
    state = ['r'] * num_signals # نبدأ بجعل كل شيء أحمر
    
    # نفتح الإشارة الخضراء (G) للاتجاه الحالي فقط
    start_idx = i * signals_per_dir
    for j in range(start_idx, start_idx + signals_per_dir):
        state[j] = 'G'
        
    middle_east_phases.append("".join(state))

# طباعة المراحل للرؤية
print("المراحل التي تم توليدها:")
for idx, phase in enumerate(middle_east_phases):
    print(f"Phase {idx}: {phase}")
# =================================================================

step = 0
current_phase_index = 0
frames_in_current_phase = 0
PHASE_DURATION = 100 # كل اتجاه سيأخذ 100 خطوة زمنية (تقريباً 10 ثوانٍ)

while step < 1000:
    traci.simulationStep() 
    
    # تبديل الإشارة للاتجاه التالي بعد انتهاء الوقت المخصص
    if frames_in_current_phase >= PHASE_DURATION:
        current_phase_index = (current_phase_index + 1) % 4 # الانتقال (0, 1, 2, 3 ثم العودة لـ 0)
        frames_in_current_phase = 0
        print(f"تم التبديل إلى الاتجاه رقم: {current_phase_index}")

    # إرسال أمر الإشارة (المرحلة الحالية) إلى التقاطع
    traci.trafficlight.setRedYellowGreenState(tls_id, middle_east_phases[current_phase_index])
    
    frames_in_current_phase += 1
    step += 1

traci.close()
print("تم انتهاء المحاكاة.")