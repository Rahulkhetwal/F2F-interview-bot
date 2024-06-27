import speech_recognition as sr

def command(language='en-us', pause_threshold=0.8):
    """
    This function uses the speech_recognition library to recognize speech from the default microphone.
    It takes two optional arguments:
    - language: the language code for speech recognition (default: 'en-us')
    - pause_threshold: the minimum duration of silence (in seconds) before the recognizer stops listening (default: 0.8)
    """
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening...")
        r.pause_threshold = pause_threshold
        audio = r.listen(source)

    try:
        ask = r.recognize_google(audio, language=language)
        print("you said : {ask}")
        return ask
    except sr.UnknownValueError:
        print("Could not understand audio")
        return None
    except sr.RequestError as e:
        print(f"Could not request results from Google Speech Recognition service; {e}")
        return None

# Call the function and print the output
print(command())