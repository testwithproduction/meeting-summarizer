import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav
from dotenv import load_dotenv
import whisper
from openai import OpenAI

import os
import threading
import atexit
import sys
import torch
from PyQt5 import QtWidgets, QtGui, QtCore

# Load environment variables from .env file
load_dotenv()

# Global flag to control the recording
is_recording = False
audio_filename = "recorded_audio.wav"

# Prompts for different summarization styles
SUMMARIZATION_PROMPTS = {
    "Meeting": """Summarize the following meeting transcript.
The meeting summary should be in the standard business writing style. The meeting summary needs to write in following structure. 

The date, time, and location
The meeting participants, and their roles. If You do not know the name, You will write their name as Person A, Person B, and etc.
Meeting purpose: Summarize the main purpose of the meeting, and provide an overview of the agenda items
Main discussion points: Summarize the main discussion points within each topic area
Key decisions: Summarize the key decisions made during the meeting, and their significance
Action items: List action items, and assign clear responsibilities and deadlines
Future directions: Capture the future directions or strategies discussed in the meeting
    {}""",
    "Interview": """Summarize the following interview transcript:

    {}""",
    "Open Discussion": """Summarize the following open discussion:
The summary should be in the standard business writing style. The summary needs to write in following structure. 

Main discussion points: Summarize the main discussion points within each topic area
Key decisions: Summarize the key decisions made during the meeting, and their significance
    {}"""
}

# Function to record audio from the microphone
def record_audio(filename, fs=44100):
    global is_recording
    is_recording = True
    print("Recording...")
    recording = []

    # Detect the number of input channels
    input_device_info = sd.query_devices(kind='input')
    channels = input_device_info['max_input_channels']

    def callback(indata, frames, time, status):
        if status:
            print(status)
        if is_recording:
            recording.append(indata.copy())

    with sd.InputStream(samplerate=fs, channels=channels, callback=callback):
        while is_recording:
            sd.sleep(100)

    print("Recording complete.")
    wav.write(filename, fs, np.concatenate(recording))

# Function to transcribe audio using the Whisper model with GPU support
def transcribe_audio(filename):
    model = whisper.load_model("base")
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        model = model.to(device)
        print("Using GPU")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        model = model.to(device)
        print("Using MPS")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    """
    result = model.transcribe(filename)
    return result["text"]

# Function to summarize text using OpenAI GPT-4 model
def summarize_text(api_key, text, style):
    prompt = SUMMARIZATION_PROMPTS[style].format(text)
    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(model="gpt-4o",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ],
    max_tokens=150,
    temperature=0.5)
    return response.choices[0].message.content.strip()

# Function to start recording in a separate thread
def start_recording_thread(filename):
    threading.Thread(target=record_audio, args=(filename,), daemon=True).start()

# Cleanup function to remove the audio file on exit
def cleanup():
    if os.path.exists(audio_filename):
        os.remove(audio_filename)
        print(f"Removed file: {audio_filename}")

# Register the cleanup function to be called on exit
atexit.register(cleanup)

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.transcription = ""

    def initUI(self):
        self.setWindowTitle("Audio Recorder and Summarizer")
        self.setGeometry(100, 100, 1080, 960)  # Increased size by 20%
        self.center()
        self.setWindowIcon(QtGui.QIcon('icon.png'))

        self.status_label = QtWidgets.QLabel("", self)
        self.status_label.setGeometry(10, 10, 1060, 20)

        self.start_button = QtWidgets.QPushButton("Start Recording", self)
        self.start_button.setGeometry(10, 40, 200, 40)
        self.start_button.clicked.connect(self.start_recording)

        self.stop_button = QtWidgets.QPushButton("Stop Recording", self)
        self.stop_button.setGeometry(220, 40, 200, 40)
        self.stop_button.clicked.connect(self.stop_recording)
        self.stop_button.setDisabled(True)

        self.progress_bar = QtWidgets.QProgressBar(self)
        self.progress_bar.setGeometry(10, 90, 1060, 30)
        self.progress_bar.setRange(0, 0)
        self.progress_bar.setVisible(False)

        self.transcription_label = QtWidgets.QLabel("Transcription:", self)
        self.transcription_label.setGeometry(10, 130, 1060, 20)

        self.transcription_text = QtWidgets.QTextEdit(self)
        self.transcription_text.setGeometry(10, 160, 1060, 250)
        self.transcription_text.setReadOnly(True)

        self.copy_transcription_button = QtWidgets.QPushButton("Copy Transcription", self)
        self.copy_transcription_button.setGeometry(10, 420, 200, 40)
        self.copy_transcription_button.clicked.connect(self.copy_transcription)

        self.style_label = QtWidgets.QLabel("Select Summarization Style:", self)
        self.style_label.setGeometry(10, 470, 200, 20)

        self.style_combo = QtWidgets.QComboBox(self)
        self.style_combo.setGeometry(220, 470, 150, 30)
        self.style_combo.addItems(["Meeting", "Interview", "Open Discussion"])

        self.summarize_button = QtWidgets.QPushButton("Summarize", self)
        self.summarize_button.setGeometry(380, 470, 200, 40)
        self.summarize_button.clicked.connect(self.summarize_text)

        self.summary_label = QtWidgets.QLabel("Summary:", self)
        self.summary_label.setGeometry(10, 520, 1060, 20)

        self.summary_text = QtWidgets.QTextEdit(self)
        self.summary_text.setGeometry(10, 550, 1060, 300)
        self.summary_text.setReadOnly(True)

        self.copy_summary_button = QtWidgets.QPushButton("Copy Summary", self)
        self.copy_summary_button.setGeometry(10, 860, 200, 40)
        self.copy_summary_button.clicked.connect(self.copy_summary)

        self.exit_button = QtWidgets.QPushButton("Exit", self)
        self.exit_button.setGeometry(220, 860, 200, 40)
        self.exit_button.clicked.connect(self.close)

    def center(self):
        frame_geometry = self.frameGeometry()
        screen = QtWidgets.QApplication.desktop().screenNumber(QtWidgets.QApplication.desktop().cursor().pos())
        center_point = QtWidgets.QApplication.desktop().screenGeometry(screen).center()
        frame_geometry.moveCenter(center_point)
        self.move(frame_geometry.topLeft())

    def start_recording(self):
        global is_recording
        if not is_recording:
            is_recording = True
            self.start_button.setDisabled(True)
            self.stop_button.setDisabled(False)
            self.status_label.setText("Recording... Press 'Stop Recording' to stop.")
            self.progress_bar.setVisible(True)
            self.progress_bar.setValue(0)
            start_recording_thread(audio_filename)

    def stop_recording(self):
        global is_recording
        if is_recording:
            is_recording = False
            self.start_button.setDisabled(False)
            self.stop_button.setDisabled(True)
            self.progress_bar.setVisible(False)
            self.status_label.setText("Recording stopped. Transcribing...")
            self.animate_transcription()
            self.transcription = transcribe_audio(audio_filename)
            self.stop_transcription_animation()
            self.transcription_text.setText(self.transcription)
            self.status_label.setText("Transcription complete. Select style and click 'Summarize'.")

    def summarize_text(self):
        style = self.style_combo.currentText()
        self.status_label.setText("Summarizing transcription...")
        self.animate_summary()
        self.summary_text.setText("Summarizing transcription...")
        QtCore.QCoreApplication.processEvents()
        summary = summarize_text(self.openai_api_key, self.transcription, style)
        self.stop_summary_animation()
        self.summary_text.setText(summary)
        self.status_label.setText("Summarization complete.")

    def animate_transcription(self):
        self.transcription_animation = QtCore.QTimer()
        self.transcription_animation.timeout.connect(self.update_transcription_animation)
        self.transcription_animation.start(300)  # Slower interval

    def update_transcription_animation(self):
        current_text = self.transcription_text.toPlainText()
        if current_text.endswith("..."):
            self.transcription_text.setText("Transcribing audio")
        else:
            self.transcription_text.setText(current_text + ".")

    def stop_transcription_animation(self):
        self.transcription_animation.stop()

    def animate_summary(self):
        self.summary_animation = QtCore.QTimer()
        self.summary_animation.timeout.connect(self.update_summary_animation)
        self.summary_animation.start(300)  # Slower interval

    def update_summary_animation(self):
        current_text = self.summary_text.toPlainText()
        if current_text.endswith("..."):
            self.summary_text.setText("Summarizing transcription")
        else:
            self.summary_text.setText(current_text + ".")

    def stop_summary_animation(self):
        self.summary_animation.stop()

    def copy_transcription(self):
        clipboard = QtWidgets.QApplication.clipboard()
        clipboard.setText(self.transcription_text.toPlainText())
        self.status_label.setText("Transcription copied to clipboard!")

    def copy_summary(self):
        clipboard = QtWidgets.QApplication.clipboard()
        clipboard.setText(self.summary_text.toPlainText())
        self.status_label.setText("Summary copied to clipboard!")

def main():
    app = QtWidgets.QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
