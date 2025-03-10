# src/emotion_detector.py

import cv2
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
from facenet_pytorch import MTCNN

class EmotionDetector:
    def __init__(self, model_path, device=None):
        # Set device (GPU if available; else CPU)
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Define emotion classes
        self.emotion_classes = ["Anger", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]

        # Load EfficientNet-B2 model without pretrained weights
        self.model = models.efficientnet_b2(pretrained=False)
        # (Optional) Modify first layer if needed
        self.model.features[0][0] = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False)
        # Modify classifier head for 7 classes
        in_features = self.model.classifier[1].in_features
        self.model.classifier = nn.Sequential(
            nn.Dropout(p=0.4),
            nn.Linear(in_features, len(self.emotion_classes))
        )
        # Load the fine-tuned weights
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

        # Define the image transformation pipeline (matches training preprocessing)
        self.transform = transforms.Compose([
            transforms.Resize((260, 260)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        # Initialize MTCNN for face detection
        self.mtcnn = MTCNN(keep_all=True, device=self.device)

    def preprocess_face(self, face_image):
        """
        Preprocess a face image (BGR format) for the model.
        Returns a tensor ready for prediction.
        """
        pil_image = Image.fromarray(cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB))
        processed_image = self.transform(pil_image)
        processed_image = processed_image.unsqueeze(0)
        return processed_image.to(self.device)

    def predict_emotion(self, face_image):
        """
        Predict emotion probabilities for a cropped face image.
        Returns a dictionary mapping emotion names to their probability.
        """
        processed_image = self.preprocess_face(face_image)
        with torch.no_grad():
            outputs = self.model(processed_image)
            probabilities = torch.softmax(outputs, dim=1)
            probabilities_np = probabilities.cpu().numpy()[0]
        emotion_dict = {emotion: float(prob) for emotion, prob in zip(self.emotion_classes, probabilities_np)}
        return emotion_dict

    def detect_and_predict(self, frame):
        """
        Detect faces in the frame using MTCNN and predict emotions.
        Returns a list of tuples: (bounding_box, emotion_dict)
        where bounding_box is (x1, y1, x2, y2).
        """
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        boxes, _ = self.mtcnn.detect(rgb_frame)
        results = []
        if boxes is not None:
            for box in boxes:
                x1, y1, x2, y2 = [int(coord) for coord in box]
                face_image = frame[y1:y2, x1:x2]
                if face_image.size == 0:
                    continue
                emotion_dict = self.predict_emotion(face_image)
                results.append(((x1, y1, x2, y2), emotion_dict))
        return results
