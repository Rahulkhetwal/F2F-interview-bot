import nltk
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
import speech_recognition as sr
import pyttsx3
import cv2
import threading
import random
import os

class VoiceAssistant:
    def __init__(self):
        self.tfidf_vectorizer = None
        self.label_encoder = None
        self.model = None
        self.engine = pyttsx3.init()
        self.voices = self.engine.getProperty('voices')
        self.engine.setProperty('voice', self.voices[1].id)  # Choosing a female voice
        self.engine.setProperty('rate', 150)  # Adjust the speaking rate if needed
        self.video_thread = threading.Thread(target=self.play_video)
        self.video_thread.daemon = True  # This thread will terminate when the main thread ends
        self.video_thread.start()
        self.train_intent_classifier()
        self.start_interview()

    def train_intent_classifier(self):
        # Sample dataset of user queries and intents
        self.data = [
            ("What is the full name of Mr Rahul?", "Rahul khetwal"),
            ("What is the college timing of Rahul sir?", "its 9:15 am"),
            ("What is the branch of Mr Rahul ?", "CSE"),
            ("what course is Mister Rahul pursuing?", "Bachelor of technology"),
            ("what's your relation with Rahul?", "I am his friend"),
            ("What types of payment do you accept?", "upi payments"),
            ("what is the full form of csv files?", "comma separated value files"),
            ("what is the full form of computer?", "common operating machine particularly used for technical education and research"),
            ("what is the full form of API?", "application programming interface"),
            ("what is the full fokrm of ssd?", "solid state drive"),
            ("what is the full form of AJAX?", "asynchronous javascript and XML"),
            ("what is the full form of IDE?", "Integrated development environment"),
            ("how many buttons are there in a standard shirt?", "eleven buttons"),
            ("What is failure?", "failure is learning"),
            ("What do you mean by teamwork?", "Teamwork means collaboration"),
            ("What does the projects indicate", "projects indicate Experience"),
            ("What full form do you know is of fail?", "first attempt in learning"),
            ("What principle does stack follows?", "LIFO that is last in first out"),
            ("What principle does Queue follows?", "FIFO That is first in first out"),
           
            # Add more examples...
        ]

        # Train the intent classifier model using the provided data
        preprocessed_data = [(word_tokenize(text.lower()), intent) for text, intent in self.data]
        X = [" ".join(tokens) for tokens, _ in preprocessed_data]
        y = [intent for _, intent in preprocessed_data]
        self.tfidf_vectorizer = TfidfVectorizer()
        X_tfidf = self.tfidf_vectorizer.fit_transform(X)
        self.label_encoder = LabelEncoder()
        y_encoded = self.label_encoder.fit_transform(y)
        self.model = SVC(kernel='linear')
        self.model.fit(X_tfidf, y_encoded)

    def predict_intent(self, text):
        preprocessed_text = " ".join(word_tokenize(text.lower()))
        X_tfidf = self.tfidf_vectorizer.transform([preprocessed_text])
        predicted_label = self.model.predict(X_tfidf)
        predicted_intent = self.label_encoder.inverse_transform(predicted_label)[0]
        return predicted_intent

    def start_interview(self):
        self.speak("Hello! I am Monika and I will take your interview today.")
        self.speak("So, shall we begin with the interview?")
        while True:
            response = self.get_user_input().lower()
            if response in ["roger","yes", "sure","fine","yes ma'am","sure mam","yes,i am ready"]:
                self.speak("Are you ready ?")
                response = self.get_user_input().lower()
                if "yes" or "absolutely" in response:
                    self.speak("That's great! The interview begins.")
                    self.conduct_interview()
                    break
                else:
                    self.speak("Okay, let me know when you are ready.")
            else:
                self.speak("Shall we begin with the interview, now?")

    def conduct_interview(self):
        correct_answers = 0
        random.shuffle(self.data)
        for question, answer in self.data[:5]:  # Asking 5 random questions
            self.speak(question)
            response = self.get_user_response().lower()
            if response == answer:
                self.speak("Correct answer!")
                correct_answers += 1
            else:
                self.speak("Wrong answer. The correct answer is: " + answer)
        if correct_answers >= 3:
            self.speak("Congratulations! You are selected for the job.")
        else:
            self.speak("Sorry, you are rejected. Better luck next time.")

    def get_user_input(self):
        print("Listening for user input...")
        r = sr.Recognizer()
        with sr.Microphone() as source:
            r.pause_threshold = 0.8
            audio = r.listen(source)

        try:
            user_input = r.recognize_google(audio)
            print(f"You: {user_input}")
            return user_input
        except sr.UnknownValueError:
            self.speak("Sorry, I didn't catch that. Could you please repeat?")
            return self.get_user_input()
        except sr.RequestError as e:
            print(f"Could not request results from Google Speech Recognition service; {e}")
            return None

    def get_user_response(self):
        print("Listening for user response...")
        r = sr.Recognizer()
        with sr.Microphone() as source:
            r.pause_threshold = 0.8
            audio = r.listen(source)

        try:
            response = r.recognize_google(audio)
            print(f"You: {response}")
            return response
        except sr.UnknownValueError:
            self.speak("Sorry, I didn't catch that. Could you please repeat?")
            return self.get_user_response()
        except sr.RequestError as e:
            print(f"Could not request results from Google Speech Recognition service; {e}")
            return None

    def speak(self, text):
        print("Speaking:", text)
        self.engine.say(text)
        self.engine.runAndWait()

    def play_video(self):
        cap = cv2.VideoCapture("anim_vid.mp4")
        fps = cap.get(cv2.CAP_PROP_FPS)  # Get the frame rate of the video

        while True:
            ret, frame = cap.read()
            if not ret:
                # Reset video to beginning when it reaches the end
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue

            cv2.imshow("Video", frame)
            if cv2.waitKey(int(1000 / fps)) & 0xFF == ord('q'):  # Adjust delay based on FPS
                break

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    assistant =VoiceAssistant()  # Create an instance of VoiceAssistant
