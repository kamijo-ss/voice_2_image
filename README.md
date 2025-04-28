## 前置き
https://github.com/bes-dev/stable_diffusion.openvino にUIと音声入力をつけたものです  
音声の翻訳に https://huggingface.co/staka/fugumt-ja-en を使わせて頂いてます  
プロジェクト管理はuvでやります  
GPUを使うのでIntel Core Ultraシリーズ搭載のPCを想定しています


## このレポジトリをクローンしたらやること
1. **uvをインストール**  
powershell -ExecutionPolicy Bypass -Command "irm https://astral.sh/uv/install.ps1 | iex"

2. **モデル配置**  
https://huggingface.co/bes-dev/stable-diffusion-v1-4-openvino/blob/main/unet.bin  
https://huggingface.co/bes-dev/stable-diffusion-v1-4-openvino/blob/main/text_encoder.bin  
をダウンロードし、名前を変えずにassets/stable-diffusion-v1-4-openvinoに保存してください  
https://huggingface.co/staka/fugumt-ja-en/blob/main/pytorch_model.bin  
をダウンロードし、名前を変えずにassets/fugumt-ja-enに保存してください  
https://alphacephei.com/vosk/models/vosk-model-ja-0.22.zip  
からVoskを落として解凍し、「vosk-model-ja-0.22」って名前でassets直下に保存してください  
（別途モデルのDL用スクリプト組んでもよかったが億劫だった）

3. **仮想環境作成**  
uv sync  
uv run stable_diffusionって叩いて1分くらい待つとアプリが起動します

**（おまけ1）UI編集のについて**  
stable_diffusion_uv\\.venv\Lib\site-packages\qt5_applications\Qt\bin\designer を使ってfront.uiを編集  
pyuic5 -x front.ui -o front.py　でfront.uiからfront.pyが出来上がります 


**（おまけ2）ビルド**  
ビルドもできるようになっています  
uv build  
uv tool install dist\stable_diffusion-0.1.0-py3-none-any.whl  
（.exeファイルから起動する時はassets以下も同一階層に置いてください）


## 音声が認識されない時は
uv run device_list.py  
↑音声デバイスの一覧が表示されます  
お使いの入力デバイスのIDを控えてsrc/stable_diffusion/main.pyの26行目(device_id)に記述してください
