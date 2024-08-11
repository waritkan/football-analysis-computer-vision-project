from ultralytics import YOLO

model = YOLO('models/best.pt')

result = model.predict('D:/boxing_analysis/input_videos/video_test.mp4',save=True)

print(result[0])
print('\t')

for box in result[0].boxes :
    print(box)