from flask import Flask, render_template, request, jsonify, send_file, send_from_directory
import os
import wave
import time
import logging
import pyaudio
import threading
from pydub import AudioSegment
from transformers import pipeline
from fpdf import FPDF
import tempfile

app = Flask(__name__)

# Initialize Whisper model for transcription
transcriber = pipeline("automatic-speech-recognition", model="openai/whisper-small")
# Initialize summarization pipeline
summarizer = pipeline("summarization")

# Setup logging
logging.basicConfig(level=logging.INFO)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/start_recording', methods=['POST'])
def start_recording():
    global is_recording, audio_data
    is_recording = True
    audio_data = []
    threading.Thread(target=record_audio).start()
    return jsonify({"status": "Recording started"})

@app.route('/stop_recording', methods=['POST'])
def stop_recording():
    global is_recording
    is_recording = False
    
    audio_path = "static/recorded_audio.wav"
    with wave.open(audio_path, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(pyaudio.get_sample_size(pyaudio.paInt16))
        wf.setframerate(16000)
        wf.writeframes(b''.join(audio_data))
    
    return jsonify({"status": "Recording stopped and audio saved", "audio_url": audio_path})

@app.route('/process_audio', methods=['POST'])
def process_audio():
    audio_path = "static/recorded_audio.wav"
    
    try:
        # Transcribe the audio file using Whisper model
        transcription = transcriber(audio_path)["text"]
        
        # Summarize the transcription
        summary = summarizer(transcription, max_length=150, min_length=30, do_sample=False)
        summary_text = summary[0]['summary_text']
        
        # Save the results
        with open('transcription.txt', 'w') as file:
            file.write(transcription)
        
        with open('summary.txt', 'w') as file:
            file.write(summary_text)
        
        return jsonify({"status": "success", "transcription": transcription, "summary": summary_text, "message": "Processing completed"})
    
    except Exception as e:
        logging.error(f"Error processing file: {str(e)}")
        return jsonify({"status": "error", "message": "Error processing file"})

def record_audio():
    global is_recording, audio_data

    cap = pyaudio.PyAudio()
    stream = cap.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=8192)
    stream.start_stream()

    while is_recording:
        data = stream.read(4096)
        audio_data.append(data)

    stream.stop_stream()
    stream.close()
    cap.terminate()

@app.route('/upload_audio', methods=['POST'])
def upload_audio():
    if 'file' not in request.files:
        logging.error("No file part in the request")
        return jsonify({"status": "error", "message": "No file part"})
    
    file = request.files['file']
    if file.filename == '':
        logging.error("No file selected for uploading")
        return jsonify({"status": "error", "message": "No selected file"})
    
    if file:
        try:
            filename = os.path.join("uploads", file.filename)
            file.save(filename)
            logging.info("File uploaded successfully")

            # Start time tracking
            start_time = time.time()

            # Convert audio file to required format if necessary
            audio = AudioSegment.from_file(filename)
            audio = audio.set_frame_rate(16000).set_channels(1)
            audio.export(filename, format="wav")
            logging.info("Audio converted to WAV format")

            # Transcribe the audio file using Whisper model
            transcription = transcriber(filename)["text"]
            
            # Summarize the transcription
            summary = summarizer(transcription, max_length=150, min_length=30, do_sample=False)
            summary_text = summary[0]['summary_text']
            
            # End time tracking
            end_time = time.time()
            processing_time = end_time - start_time
            logging.info(f"Processing completed in {processing_time:.2f} seconds")
            
            return jsonify({"status": "success", "transcription": transcription, "summary": summary_text, "message": "Processing completed"})
        
        except Exception as e:
            logging.error(f"Error processing file: {str(e)}")
            return jsonify({"status": "error", "message": "Error processing file"})
    
    logging.error("File upload failed")
    return jsonify({"status": "error", "message": "File upload failed"})

@app.route('/get_results', methods=['GET'])
def get_results():
    if os.path.exists('transcription.txt') and os.path.exists('summary.txt'):
        with open('transcription.txt', 'r') as file:
            transcription_text = file.read()
        with open('summary.txt', 'r') as file:
            summary_text = file.read()
        return jsonify({"transcription": transcription_text, "summary": summary_text})
    else:
        return jsonify({"transcription": "", "summary": ""})

@app.route('/download_pdf', methods=['POST'])
def download_pdf():
    transcription = request.form['transcription']
    summary = request.form['summary']

    pdf = FPDF()
    pdf.add_page()

    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 10, "Transcription:\n\n" + transcription)
    pdf.add_page()
    pdf.multi_cell(0, 10, "Summary:\n\n" + summary)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_pdf:
        pdf.output(tmp_pdf.name)
        tmp_pdf.seek(0)
        return send_file(tmp_pdf.name, as_attachment=True, attachment_filename="transcription_summary.pdf")

if __name__ == '__main__':
    if not os.path.exists('static'):
        os.makedirs('static')
    app.run(debug=True)
