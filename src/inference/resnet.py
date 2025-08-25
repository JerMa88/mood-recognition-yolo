import cv2
import torch
from torchvision.models import resnet18
import numpy as np

import torch.nn as nn
import torchvision.transforms as transforms

class EmotionRecognizer:
    def __init__(self, model_path):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.classes = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise'] 
        self.model = self._load_model(model_path)
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])
        ])

    def _load_model(self, model_path):
        model = resnet18(weights=None)
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, len(self.classes))
        checkpoint = torch.load(model_path, map_location=self.device)
        if 'resnet' in checkpoint:
                state_dict = checkpoint['resnet']
        else:
                state_dict = checkpoint

        model.load_state_dict(state_dict)
        # model.load_state_dict(torch.load(model_path, map_location=self.device))
        model.to(self.device)
        model.eval()
        return model

    def predict(self, face_img):
        img_tensor = self.transform(face_img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            outputs = self.model(img_tensor)
            _, predicted = torch.max(outputs, 1)
            return self.classes[predicted.item()]

def main():
    # Initialize face detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # Initialize emotion recognizer
    emotion_recognizer = EmotionRecognizer('../../data/saved_models/facial_expression_resnet.pth')
    
    # Open webcam
    cap = cv2.VideoCapture(0)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Convert to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        
        for (x, y, w, h) in faces:
            # Draw rectangle around face
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            
            # Extract face ROI
            face_roi = frame[y:y+h, x:x+w]
            
            # Predict emotion
            try:
                emotion = emotion_recognizer.predict(face_roi)
                
                # Display emotion text
                cv2.putText(frame, emotion, (x, y-10), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.9, 
                          (36,255,12), 2)
            except:
                continue
        
        # Display the frame
        cv2.imshow('Facial Expression Recognition', frame)
        
        # Break loop on 'q' press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
