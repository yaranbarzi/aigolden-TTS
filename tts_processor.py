# Created by aigolden
# Source: https://github.com/yaranbarzi/aigolden-TTS
# License: MIT

import base64
import mimetypes
import os
import re
import struct
import time
import zipfile
from google import genai
from google.genai import types
from IPython.display import Audio
from google.colab import userdata
from google.colab import files
try:
    from pydub import AudioSegment
    PYDUB_AVAILABLE = True
except ImportError:
    PYDUB_AVAILABLE = False
    print("⚠️ pydub در دسترس نیست. فایل‌ها جداگانه ذخیره می‌شوند.")

def save_binary_file(file_name, data):
    """Save binary data to a file."""
    with open(file_name, "wb") as f:
        f.write(data)
    print(f"✅ فایل در مسیر زیر ذخیره شد: {file_name}")
    return file_name

def convert_to_wav(audio_data: bytes, mime_type: str) -> bytes:
    """Convert audio data to WAV format."""
    parameters = parse_audio_mime_type(mime_type)
    bits_per_sample = parameters["bits_per_sample"]
    sample_rate = parameters["rate"]
    num_channels = 1
    data_size = len(audio_data)
    bytes_per_sample = bits_per_sample // 8
    block_align = num_channels * bytes_per_sample
    byte_rate = sample_rate * block_align
    chunk_size = 36 + data_size  # اندازه کل فایل WAV (هدر + داده)

    header = struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF",
        chunk_size,
        b"WAVE",
        b"fmt ",
        16,  # اندازه زیرقسمت fmt
        1,   # فرمت PCM
        num_channels,
        sample_rate,
        byte_rate,
        block_align,
        bits_per_sample,
        b"data",
        data_size
    )
    return header + audio_data

def parse_audio_mime_type(mime_type: str) -> dict[str, int | None]:
    """Parse audio MIME type to extract parameters."""
    bits_per_sample = 16
    rate = 24000  # نرخ پیش‌فرض

    parts = mime_type.split(";")
    for param in parts:
        param = param.strip()
        if param.lower().startswith("rate="):
            try:
                rate_str = param.split("=", 1)[1]
                rate = int(rate_str)
            except (ValueError, IndexError):
                pass
        elif param.startswith("audio/L"):
            try:
                bits_per_sample = int(param.split("L", 1)[1])
            except (ValueError, IndexError):
                pass
    return {"bits_per_sample": bits_per_sample, "rate": rate}

def load_text_file():
    """Load text file containing only the main text (no prompt)."""
    print("📁 لطفاً فایل متنی خود را آپلود کنید...")
    print("💡 فایل فقط باید شامل متن اصلی باشد (پرامپت از فیلد بالا خوانده می‌شود)")
    
    uploaded = files.upload()
    
    if not uploaded:
        print("❌ هیچ فایلی آپلود نشد.")
        return ""
    
    file_name = list(uploaded.keys())[0]
    print(f"✅ فایل '{file_name}' با موفقیت آپلود شد.")
    
    try:
        with open(file_name, 'r', encoding='utf-8') as f:
            content = f.read().strip()
        
        print(f"📖 متن بارگذاری شده: {len(content)} کاراکتر")
        print(f"📝 نمونه متن: '{content[:100]}{'...' if len(content) > 100 else ''}'")
        
        return content
    
    except Exception as e:
        print(f"❌ خطا در خواندن فایل: {e}")
        return ""

def smart_text_split(text, max_size=3800):
    """Split text into chunks without breaking sentences."""
    if len(text) <= max_size:
        return [text]
    
    chunks = []
    current_chunk = ""
    
    sentences = re.split(r'(?<=[.!?])\s+', text)
    
    for sentence in sentences:
        if len(current_chunk) + len(sentence) + 1 > max_size:
            if current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = sentence
            else:
                words = sentence.split()
                temp_chunk = ""
                for word in words:
                    if len(temp_chunk) + len(word) + 1 > max_size:
                        if temp_chunk:
                            chunks.append(temp_chunk.strip())
                            temp_chunk = word
                        else:
                            chunks.append(word[:max_size])
                            word = word[max_size:]
                            while len(word) > max_size:
                                chunks.append(word[:max_size])
                                word = word[max_size:]
                            if word:
                                temp_chunk = word
                    else:
                        temp_chunk += (" " if temp_chunk else "") + word
                current_chunk = temp_chunk
        else:
            current_chunk += (" " if current_chunk else "") + sentence
    
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return chunks

def merge_audio_files_func(file_paths, output_path):
    """Merge multiple audio files into one."""
    if not PYDUB_AVAILABLE:
        print("❌ pydub در دسترس نیست. نمی‌توان فایل‌ها را ادغام کرد.")
        return False
    
    try:
        print(f"🔗 در حال ادغام {len(file_paths)} فایل صوتی...")
        
        combined = AudioSegment.empty()
        
        for i, file_path in enumerate(file_paths):
            if os.path.exists(file_path):
                print(f"📎 اضافه کردن فایل {i+1}: {file_path}")
                audio = AudioSegment.from_file(file_path)
                combined += audio
                if i < len(file_paths) - 1:
                    combined += AudioSegment.silent(duration=500)
            else:
                print(f"⚠️ فایل پیدا نشد: {file_path}")
        
        combined.export(output_path, format="wav")
        print(f"✅ فایل ادغام شده ذخیره شد: {output_path}")
        return True
        
    except Exception as e:
        print(f"❌ خطا در ادغام فایل‌ها: {e}")
        return False

def create_zip_file(file_paths, zip_name):
    """Create a zip file containing all audio files."""
    try:
        with zipfile.ZipFile(zip_name, 'w') as zipf:
            for file_path in file_paths:
                if os.path.exists(file_path):
                    zipf.write(file_path, os.path.basename(file_path))
        print(f"📦 فایل ZIP ایجاد شد: {zip_name}")
        return True
    except Exception as e:
        print(f"❌ خطا در ایجاد فایل ZIP: {e}")
        return False

def generate_audio(text_input, prompt_input, selected_voice, output_base_name, 
                   api_key_input_field, model, temperature, use_file=False, 
                   max_chunk_size=3800, sleep_time=2, merge_files=True, 
                   delete_partials=True):
    """
    Generate audio from text input with specified parameters.
    
    Created by aigolden
    Source: https://github.com/yaranbarzi/aigolden-TTS
    License: MIT
    """
    print("🚀 شروع فرآیند تبدیل متن به گفتار...")
    
    global final_audio_file, generated_files
    final_audio_file = None
    generated_files = []

    # Handle file input if enabled
    if use_file:
        print("📁 حالت فایل فعال است. در حال آپلود فایل...")
        file_text = load_text_file()
        if not file_text:
            print("❌ خطا: متن استخراج شده از فایل خالی است.")
            return None, None
        text_input = file_text
        print("✅ متن از فایل با موفقیت بارگذاری شد.")
    else:
        print("⌨️ حالت ورودی دستی فعال است.")

    # API Key Retrieval and Validation
    api_key = None
    if api_key_input_field:
        api_key = api_key_input_field
        print("🔑 کلید API از فیلد ورودی بارگذاری شد.")
    else:
        api_key = userdata.get("GEMINI_API_KEY")
        if api_key:
            print("🔑 کلید API از Colab Secrets بارگذاری شد.")
        else:
            print("❌ خطا: کلید API جمینای (GEMINI_API_KEY) پیدا نشد.")
            print("لطفاً کلید API خود را در فیلد بالا وارد کنید یا مطمئن شوید که آن را در Colab Secrets به درستی ذخیره کرده‌اید.")
            print("💡 راهنما: روی آیکون قفل 🔑 در نوار کناری سمت چپ Colab کلیک کنید، یک 'راز' (Secret) جدید با نام `GEMINI_API_KEY` ایجاد کنید و کلید خود را به عنوان مقدار آن قرار دهید. سپس 'Notebook access' را فعال کنید.")
            return None, None

    os.environ["GEMINI_API_KEY"] = api_key
    print("🔧 متغیر محیطی GEMINI_API_KEY تنظیم شد.")

    # Initialize GenAI Client
    try:
        print("🛠️ در حال ایجاد کلاینت جمینای...")
        client = genai.Client(
            api_key=os.environ.get("GEMINI_API_KEY"),
        )
        print("✅ کلاینت جمینای با موفقیت ایجاد شد.")
    except Exception as e:
        print(f"❌ خطا در ایجاد کلاینت جمینای: {e}")
        print("لطفاً از صحت کلید API خود اطمینان حاصل کنید. ممکن است کلید منقضی شده یا نامعتبر باشد.")
        return None, None

    # Validate Text Input
    if not text_input or text_input.strip() == "":
        print("❌ خطا: متن ورودی برای تبدیل به گفتار خالی است. لطفاً متنی را وارد کنید.")
        return None, None

    # Split text into chunks
    text_chunks = smart_text_split(text_input, max_chunk_size)
    print(f"📊 متن به {len(text_chunks)} قطعه تقسیم شد.")
    
    for i, chunk in enumerate(text_chunks):
        print(f"📝 قطعه {i+1}: {len(chunk)} کاراکتر")

    # Generate audio for each chunk
    for i, chunk in enumerate(text_chunks):
        print(f"\n🔊 تولید صدا برای قطعه {i+1}/{len(text_chunks)}...")
        
        final_text = f'"{prompt_input}"\n{chunk}' if prompt_input.strip() else chunk

        contents = [
            types.Content(
                role="user",
                parts=[
                    types.Part.from_text(text=final_text),
                ],
            ),
        ]
        generate_content_config = types.GenerateContentConfig(
            temperature=temperature,
            response_modalities=[
                "audio",
            ],
            speech_config=types.SpeechConfig(
                voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(
                        voice_name=selected_voice
                    )
                )
            ),
        )

        try:
            chunk_filename = f"{output_base_name}_part_{i+1:03d}"
            
            for chunk_data in client.models.generate_content_stream(
                model=model,
                contents=contents,
                config=generate_content_config,
            ):
                if (
                    chunk_data.candidates
                    and chunk_data.candidates[0].content
                    and chunk_data.candidates[0].content.parts
                    and chunk_data.candidates[0].content.parts[0].inline_data
                ):
                    inline_data = chunk_data.candidates[0].content.parts[0].inline_data
                    data_buffer = inline_data.data
                    file_extension = mimetypes.guess_extension(inline_data.mime_type)
                    print(f"ℹ️ MIME Type: {inline_data.mime_type}, Data Size: {len(data_buffer)} bytes")
                    if inline_data.mime_type == "audio/wav":
                        file_extension = ".wav"
                    else:
                        file_extension = ".wav"
                        data_buffer = convert_to_wav(inline_data.data, inline_data.mime_type)

                    generated_file_path = save_binary_file(f"{chunk_filename}{file_extension}", data_buffer)
                    generated_files.append(generated_file_path)
                    print(f"✅ قطعه {i+1} تولید شد: {generated_file_path}")
                    break
                else:
                    if chunk_data.text:
                        print(f"ℹ️ پیام متنی از API: {chunk_data.text}")

        except Exception as e:
            print(f"❌ خطا در تولید قطعه {i+1}: {e}")
            continue
        
        if i < len(text_chunks) - 1:
            print(f"⏱️ انتظار {sleep_time} ثانیه...")
            time.sleep(sleep_time)

    if not generated_files:
        print("❌ هیچ فایل صوتی تولید نشد!")
        return None, None

    print(f"\n🎉 {len(generated_files)} فایل صوتی با موفقیت تولید شد!")

    # Merge audio files if enabled
    if merge_files and len(generated_files) > 1:
        merged_filename = f"{output_base_name}_merged.wav"
        if merge_audio_files_func(generated_files, merged_filename):
            final_audio_file = merged_filename
            print(f"🎵 فایل نهایی ادغام شده: {merged_filename}")
            
            if delete_partials:
                for file_path in generated_files:
                    try:
                        os.remove(file_path)
                        print(f"🗑️ فایل جزئی حذف شد: {file_path}")
                    except:
                        pass
        else:
            print("⚠️ ادغام ممکن نبود. فایل‌های جداگانه حفظ شدند.")
    
    # Create zip file if no merge
    if not final_audio_file and len(generated_files) > 1:
        zip_filename = f"{output_base_name}_all_parts.zip"
        create_zip_file(generated_files, zip_filename)

    # Play the audio
    if final_audio_file and os.path.exists(final_audio_file):
        print(f"▶️ پخش فایل ادغام شده: {final_audio_file}")
        display(Audio(final_audio_file, autoplay=True))
    elif generated_files and os.path.exists(generated_files[0]):
        print(f"▶️ پخش اولین قطعه: {generated_files[0]}")
        display(Audio(generated_files[0], autoplay=True))
    else:
        print("🛑 پخش صدا امکان‌پذیر نیست زیرا فایل صوتی تولید نشده است.")

    return generated_files, final_audio_file
