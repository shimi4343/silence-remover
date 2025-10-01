import os
import time
import tempfile
from pathlib import Path

import numpy as np
import librosa
import soundfile as sf
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

def detect_and_remove_silence(audio_path, output_path, silence_threshold=-40, min_silence_duration=0.1, target_silence_duration=0.1):
    """
    音声ファイルから無音部分を検出し、0.1秒以上の無音を0.1秒に短縮する

    Args:
        audio_path: 入力音声ファイルのパス
        output_path: 出力音声ファイルのパス
        silence_threshold: 無音と判定するレベル（dB）
        min_silence_duration: 短縮対象の最小無音時間（秒）デフォルト0.1秒
        target_silence_duration: 短縮後の無音時間（秒）デフォルト0.1秒
    """
    try:
        # 音声ファイルを読み込み
        audio, sr = librosa.load(audio_path, sr=None)
        
        print(f"元ファイル: {len(audio)/sr:.2f}秒, サンプルレート: {sr}Hz")
        
        # 処理
        processed_audio = process_audio_simple(audio, sr, silence_threshold, min_silence_duration, target_silence_duration)
        
        print(f"処理後: {len(processed_audio)/sr:.2f}秒")
        
        # 結果を保存
        sf.write(output_path, processed_audio, sr)
        print(f"処理完了: {audio_path} -> {output_path}")
        
    except Exception as e:
        print(f"エラーが発生しました ({audio_path}): {e}")

def process_audio_simple(audio, sr, silence_threshold=-40, min_silence_duration=0.1, target_silence_duration=0.1):
    """
    シンプルな無音検出と削除
    """
    # RMS エネルギーを計算（フレームサイズを大きくして安定化）
    frame_length = int(0.025 * sr)  # 25ms フレーム
    hop_length = int(0.010 * sr)    # 10ms ホップ
    
    rms = librosa.feature.rms(y=audio, frame_length=frame_length, hop_length=hop_length)[0]
    
    # dBに変換
    rms_db = librosa.amplitude_to_db(rms, ref=np.max)
    
    # 無音フレームを検出
    silent_frames = rms_db < silence_threshold
    
    segments = []
    i = 0
    
    while i < len(silent_frames):
        if not silent_frames[i]:
            # 音声部分の開始
            audio_start = i
            while i < len(silent_frames) and not silent_frames[i]:
                i += 1
            audio_end = i
            
            # 音声部分をサンプル単位に変換して追加
            start_sample = audio_start * hop_length
            end_sample = min(audio_end * hop_length, len(audio))
            segments.append(audio[start_sample:end_sample])
            
        else:
            # 無音部分の開始
            silence_start = i
            while i < len(silent_frames) and silent_frames[i]:
                i += 1
            silence_end = i
            
            # 無音部分の時間を計算
            silence_duration = (silence_end - silence_start) * hop_length / sr
            
            # 0.1秒以上の無音は0.1秒に短縮
            if silence_duration >= min_silence_duration:
                silence_samples = int(target_silence_duration * sr)
                segments.append(np.zeros(silence_samples))
            else:
                # 短い無音はそのまま保持
                start_sample = silence_start * hop_length
                end_sample = min(silence_end * hop_length, len(audio))
                if end_sample > start_sample:
                    segments.append(audio[start_sample:end_sample])
    
    # セグメントを結合
    if segments:
        result = np.concatenate(segments)
        print(f"セグメント数: {len(segments)}")
        return result
    else:
        return audio

def process_folder(input_folder, output_folder, silence_threshold=-40, min_silence_duration=0.1, target_silence_duration=0.1):
    """
    フォルダ内の全ての音声ファイルを処理
    
    Args:
        input_folder: 入力フォルダのパス
        output_folder: 出力フォルダのパス
        silence_threshold: 無音と判定するレベル（dB）
        min_silence_duration: 短縮対象の最小無音時間（秒）デフォルト0.1秒
        target_silence_duration: 短縮後の無音時間（秒）デフォルト0.1秒
    """
    input_path = Path(input_folder)
    output_path = Path(output_folder)
    
    # 出力フォルダを作成
    output_path.mkdir(exist_ok=True)
    
    # サポートする音声ファイル形式
    supported_extensions = {'.wav', '.mp3', '.flac', '.aac', '.m4a', '.ogg'}
    
    # voice.wavのみを処理対象とする
    audio_files = []
    voice_file = input_path / "voice.wav"
    if voice_file.exists():
        audio_files = [voice_file]
    
    if not audio_files:
        print(f"音声ファイルが見つかりませんでした: {input_folder}")
        return
    
    print(f"{len(audio_files)}個の音声ファイルを処理します...")
    
    for audio_file in audio_files:
        # 出力ファイル名を生成（元ファイルと同じ場所に保存）
        output_file = audio_file.parent / f"{audio_file.stem}_processed.wav"
        
        detect_and_remove_silence(
            str(audio_file),
            str(output_file),
            silence_threshold,
            min_silence_duration,
            target_silence_duration
        )

class VoiceFileHandler(FileSystemEventHandler):
    """voice.wavの変更を監視するハンドラー"""
    
    def __init__(self, watch_dir):
        self.watch_dir = Path(watch_dir)
        self.processing = False
        
    def on_modified(self, event):
        if event.is_directory:
            return
            
        file_path = Path(event.src_path)
        
        # voice.wavが更新された場合のみ処理
        if file_path.name == "voice.wav" and not self.processing:
            print(f"\n[{time.strftime('%H:%M:%S')}] voice.wavの更新を検出しました")
            
            # 処理中フラグを立てる
            self.processing = True
            
            try:
                # 少し待機してファイルの書き込みが完了するのを待つ
                time.sleep(1)
                
                # 無音削除処理を実行
                output_path = file_path.parent / f"{file_path.stem}_processed.wav"
                detect_and_remove_silence(
                    str(file_path),
                    str(output_path),
                    silence_threshold=-40,
                    min_silence_duration=0.1,
                    target_silence_duration=0.1
                )
                print(f"[{time.strftime('%H:%M:%S')}] 自動処理が完了しました\n")
                
            except Exception as e:
                print(f"[{time.strftime('%H:%M:%S')}] 処理中にエラーが発生しました: {e}\n")
            
            finally:
                # 処理中フラグを解除
                self.processing = False

def watch_voice_file(watch_dir="."):
    """voice.wavファイルの変更を監視"""
    watch_path = Path(watch_dir).resolve()
    
    print("=" * 50)
    print("🎵 Voice File Watcher 開始")
    print("=" * 50)
    print(f"監視フォルダ: {watch_path}")
    print(f"監視対象: voice.wav")
    print(f"処理条件: 0.1秒以上の無音を0.1秒に短縮")
    print(f"出力先: voice_processed.wav")
    print()
    print("voice.wavが更新されるたびに自動で無音削除処理が実行されます")
    print("終了するには Ctrl+C を押してください")
    print("=" * 50)
    
    # 初回処理（既存のvoice.wavがある場合）
    voice_file = watch_path / "voice.wav"
    if voice_file.exists():
        print(f"\n[{time.strftime('%H:%M:%S')}] 既存のvoice.wavを処理します...")
        try:
            output_path = voice_file.parent / f"{voice_file.stem}_processed.wav"
            detect_and_remove_silence(
                str(voice_file),
                str(output_path),
                silence_threshold=-40,
                min_silence_duration=0.1,
                target_silence_duration=0.1
            )
            print(f"[{time.strftime('%H:%M:%S')}] 初回処理が完了しました")
        except Exception as e:
            print(f"[{time.strftime('%H:%M:%S')}] 初回処理でエラーが発生しました: {e}")
    
    print(f"\n[{time.strftime('%H:%M:%S')}] 監視を開始しました...")
    
    # ファイル監視を開始
    event_handler = VoiceFileHandler(watch_dir)
    observer = Observer()
    observer.schedule(event_handler, str(watch_path), recursive=False)
    observer.start()
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
        print(f"\n[{time.strftime('%H:%M:%S')}] 監視を終了しました")
        print("=" * 50)

    observer.join()


def create_web_app():
    """シンプルなFlaskアプリを生成"""
    from flask import Flask, request, send_file, render_template_string
    from werkzeug.utils import secure_filename

    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024  # 50MB 上限

    page_template = """<!doctype html>
    <html lang=\"ja\">
    <head>
        <meta charset=\"utf-8\">
        <title>Silence Remover</title>
        <style>
            body { font-family: sans-serif; background: #f2f2f2; margin: 0; padding: 2rem; }
            main { max-width: 640px; margin: 0 auto; background: #fff; padding: 2rem; border-radius: 0.5rem; box-shadow: 0 0 16px rgba(0,0,0,0.1); }
            h1 { margin-top: 0; }
            form { display: grid; gap: 1rem; }
            label { display: flex; flex-direction: column; font-weight: bold; }
            input[type=\"number\"] { padding: 0.5rem; }
            input[type=\"file\"] { padding: 0.5rem 0; }
            button { padding: 0.75rem 1rem; font-size: 1rem; cursor: pointer; }
            .error { color: #d32f2f; font-weight: bold; }
            .hint { color: #666; font-size: 0.9rem; font-weight: normal; }
        </style>
    </head>
    <body>
        <main>
            <h1>Silence Remover</h1>
            <p>音声ファイルをアップロードすると、0.1秒以上の無音が短縮されたファイルをダウンロードできます。</p>
            {% if error %}
                <p class=\"error\">{{ error }}</p>
            {% endif %}
            <form method=\"post\" enctype=\"multipart/form-data\">
                <label>音声ファイル
                    <input type=\"file\" name=\"audio\" accept=\"audio/*\" required>
                </label>
                <label>無音判定閾値 (dB)<span class=\"hint\">既定値: -40</span>
                    <input type=\"number\" name=\"silence_threshold\" value=\"{{ silence_threshold }}\" step=\"1\" min=\"-120\" max=\"0\">
                </label>
                <label>短縮対象の最小無音時間 (秒)<span class=\"hint\">既定値: 0.1</span>
                    <input type=\"number\" name=\"min_silence_duration\" value=\"{{ min_silence_duration }}\" step=\"0.05\" min=\"0\">
                </label>
                <label>短縮後の無音時間 (秒)<span class=\"hint\">既定値: 0.1</span>
                    <input type=\"number\" name=\"target_silence_duration\" value=\"{{ target_silence_duration }}\" step=\"0.05\" min=\"0\">
                </label>
                <button type=\"submit\">処理してダウンロード</button>
            </form>
            <p class=\"hint\">CLI や監視機能は <code>python silence_remover.py</code> / <code>python silence_remover.py watch</code> で利用できます。</p>
        </main>
    </body>
    </html>"""

    @app.route("/", methods=["GET", "POST"])
    def index():
        default_params = {
            "silence_threshold": -40,
            "min_silence_duration": 0.1,
            "target_silence_duration": 0.1,
        }

        if request.method == "GET":
            return render_template_string(page_template, error=None, **default_params)

        params = {
            "silence_threshold": request.form.get("silence_threshold", default_params["silence_threshold"]),
            "min_silence_duration": request.form.get("min_silence_duration", default_params["min_silence_duration"]),
            "target_silence_duration": request.form.get("target_silence_duration", default_params["target_silence_duration"]),
        }

        error = None
        uploaded_file = request.files.get("audio")
        if uploaded_file is None or uploaded_file.filename == "":
            error = "音声ファイルを選択してください"
            return render_template_string(page_template, error=error, **params)

        try:
            silence_threshold = float(params["silence_threshold"] or default_params["silence_threshold"])
            min_silence_duration = float(params["min_silence_duration"] or default_params["min_silence_duration"])
            target_silence_duration = float(params["target_silence_duration"] or default_params["target_silence_duration"])
        except ValueError:
            error = "数値パラメータが正しくありません"
            return render_template_string(page_template, error=error, **params)

        params.update(
            {
                "silence_threshold": silence_threshold,
                "min_silence_duration": min_silence_duration,
                "target_silence_duration": target_silence_duration,
            }
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_dir_path = Path(tmp_dir)
            original_name = secure_filename(uploaded_file.filename or "input.wav")
            if not original_name:
                original_name = "input.wav"

            input_path = tmp_dir_path / original_name
            uploaded_file.save(input_path)

            output_path = input_path.with_name(f"{input_path.stem}_processed.wav")

            try:
                detect_and_remove_silence(
                    str(input_path),
                    str(output_path),
                    silence_threshold=silence_threshold,
                    min_silence_duration=min_silence_duration,
                    target_silence_duration=target_silence_duration,
                )
            except Exception as exc:
                error = f"処理中にエラーが発生しました: {exc}"
                return render_template_string(page_template, error=error, **params)

            download_name = f"{Path(uploaded_file.filename).stem or 'audio'}_processed.wav"
            return send_file(output_path, as_attachment=True, download_name=download_name)

    return app


def run_web_app(host="127.0.0.1", port=5000):
    """Flask開発サーバーを起動"""
    app = create_web_app()
    app.run(host=host, port=port, debug=False)

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "watch":
        # 監視モード
        watch_voice_file(".")
    elif len(sys.argv) > 1 and sys.argv[1] == "web":
        # Webアプリモード
        print("Webインターフェースを起動します。ブラウザで http://127.0.0.1:5000 を開いてください。")

        host = "127.0.0.1"
        port = 5000

        if len(sys.argv) > 2:
            host = sys.argv[2]

        if len(sys.argv) > 3:
            try:
                port = int(sys.argv[3])
            except ValueError:
                print("ポート番号が正しくありません。デフォルトの5000番を使用します。")
                port = 5000

        run_web_app(host=host, port=port)
    else:
        # 通常の一回だけの処理
        current_dir = "."

        print("音声ファイルの無音削除処理を開始します...")
        print(f"入力フォルダ: {current_dir}")
        print(f"出力先: 元ファイルと同じフォルダ（_processed.wavとして保存）")
        print(f"処理条件: 0.1秒以上の無音を0.1秒に短縮")
        print()
        print("監視モードで起動する場合は: python3 silence_remover.py watch")
        print()
        
        process_folder(current_dir, current_dir)
        print("処理が完了しました！")
