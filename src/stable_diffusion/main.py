import sys
import argparse
import random
import json

import numpy as np
import cv2
import pyaudio
import torch
from vosk import Model, KaldiRecognizer
from transformers import MarianTokenizer, MarianMTModel
from PyQt5.QtWidgets import QApplication, QWidget, QMessageBox
from PyQt5.QtCore import QThread, pyqtSignal
from diffusers import LMSDiscreteScheduler

from stable_diffusion.front import Ui_Form
from stable_diffusion.stable_diffusion_engine import StableDiffusionEngine

class SpeechToTextThread(QThread):
    # stringを投げるシグナルを定義
    text_received = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.vosk = parent.vosk
        self.device_id  = 1                 # uv run device_list.pyで取得したID
        self.format     = pyaudio.paInt16   # 16bit
        self.channels   = 1                 # Voskがモノラルに最適化されてるのでモノラルにする
        self.rate       = 16000             # サンプルレートもVoskに合わせる
        self.chunk      = 4096              # データのチャンクサイズ
        self.running    = False
    
    def run(self):
        voice = self.recording_voice()
        # シグナルはemitメソッドを叩くと送信される
        self.text_received.emit(voice)

    def recording_voice(self):
        audio = pyaudio.PyAudio()
        stream = audio.open(
            input_device_index=self.device_id,
            format=self.format,
            channels=self.channels,
            frames_per_buffer=self.chunk,
            rate=self.rate,
            input=True
        )
        stream.start_stream()
        rec = KaldiRecognizer(self.vosk, self.rate)
        recording = True
        while recording:
            data = stream.read(self.chunk // 2, exception_on_overflow=False)
            if rec.AcceptWaveform(data):
                recording = False
                stream.close()
                audio.terminate()
                self.quit()
                return json.loads(rec.Result()).get("text")



class MyApp(QWidget):
    def __init__(self):
        super().__init__()
        self.ui = Ui_Form()
        self.ui.setupUi(self)
        self.sd_model = None
        self.scheduler = None
        try:
            self.load_sd()
            self.vosk = Model("assets/vosk-model-ja-0.22")
            self.device = torch.device("cpu")
            self.translate_model = MarianMTModel.from_pretrained("assets/fugumt-ja-en").eval().to(self.device)
            self.tokenizer = MarianTokenizer.from_pretrained("assets/fugumt-ja-en")
            # スレッドのtext_receivedシグナルにgenerate_imageメソッドを接続
            self.transcription_thread = SpeechToTextThread(self)
            self.transcription_thread.text_received.connect(self.generate_image)
        except Exception as e:
            QMessageBox.critical(self, "エラー", f"モデルの読み込みに失敗しました: {str(e)}")
            sys.exit(1)

        self.ui.pushButton.clicked.connect(self.start_recording_thread)

    def load_sd(self):
        args = argparse.Namespace(
            model="assets/stable-diffusion-v1-4-openvino",
            device="GPU", # GPU環境がない人はここにCPUを設定
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            tokenizer="assets/clip-vit-large-patch14"
        )

        self.scheduler = LMSDiscreteScheduler(
            beta_start=args.beta_start,
            beta_end=args.beta_end,
            beta_schedule=args.beta_schedule,
            tensor_format="np"
        )

        self.sd_model = StableDiffusionEngine(
            model=args.model,
            scheduler=self.scheduler,
            tokenizer=args.tokenizer,
            device=args.device
        )

    def start_recording_thread(self):
        self.ui.label.setText("recording...")
        self.ui.label_2.setText("録音中...")
        self.ui.textBrowser.setText("")
        self.ui.textBrowser_2.setText("")
        QApplication.processEvents()
        self.ui.pushButton.setEnabled(False)
        # startした時にスレッドのrunメソッドが走る
        self.transcription_thread.start()

    def translate_voice(self, voice):
        # 辞書内包でtoken生成とdeviceへの転送を同時に行うと負荷が増す気がして分割（諸説あり）
        input_ids_ram = self.tokenizer(voice, return_tensors="pt", padding=True, truncation=True)
        input_ids_device = {key: value.to(self.device) for key, value in input_ids_ram.items()}
        with torch.no_grad():
            return self.translate_model.generate(**input_ids_device)

    def generate_image(self,voice):
        prompt = self.tokenizer.decode(self.translate_voice(voice)[0], skip_special_tokens=True)
        self.ui.label.setText("generating...")
        self.ui.label_2.setText("検出しました 生成中...")
        self.ui.textBrowser.setText(prompt)
        self.ui.textBrowser_2.setText(voice)
        QApplication.processEvents()
        seed = None
        
        try:
            args = argparse.Namespace(
                seed=seed,
                num_inference_steps=8, # 8～32くらいの間でお好みで調整
                guidance_scale=7.5,
                eta=0.0,
                prompt=prompt,
                init_image=None,
                strength=0.5,
                mask=None
            )
            image = self.run_image_generation(args)
            self.show_image(image)

        except Exception as e:
            QMessageBox.critical(self, "エラー", str(e))

    def run_image_generation(self, args):
        if args.seed is None:
            args.seed = random.randint(0, 2**30)
        np.random.seed(args.seed)

        image = self.sd_model(
            prompt=args.prompt,
            init_image=None,
            mask=None,
            strength=args.strength,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            eta=args.eta
        )
        self.ui.label.setText("generated")
        self.ui.label_2.setText("生成完了")
        self.ui.pushButton.setEnabled(True)

        return image

    def show_image(self, image):
        cv2.imshow("Generated Image", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def main():
    app = QApplication([])
    window = MyApp()
    window.show()
    app.exec_()
