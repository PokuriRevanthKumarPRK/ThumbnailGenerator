
import subprocess
import imageio_ffmpeg
import whisper
from groq import Groq
import streamlit as st
import tempfile
import os
from gradio_client import Client

@st.cache_resource
def load_whisper():
    return whisper.load_model("tiny")

@st.cache_resource
def load_summariser():
    return pipeline("summarization", model="facebook/bart-large-cnn")

st.title("Thumbnail Generator")
st.write("By Pokuri Revanth Kumar")
st.text("This thumbnail generator takes a video as input and automatically creates a visually engaging image suitable for a YouTube Shorts thumbnail. For faster results, a maximum video length of 2 minutes is recommended.")

ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()

#This funtion is to convert the video file into audio so that I can use whisper model .
def vid_to_aud(uploaded_video):
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_vid:
        tmp_vid.write(uploaded_video.read())
        video_path = tmp_vid.name

    audio_path = video_path.replace(".mp4", ".mp3")

    try:
        
        subprocess.run([
            "ffmpeg", "-y", "-i", video_path,
            "-vn", "-acodec", "libmp3lame", audio_path
        ], check=True, capture_output=True)
        return audio_path, video_path
    except Exception as e:
        st.error(f"FFmpeg conversion failed: {e}")
        return None, video_path    

user_video = st.file_uploader("Upload Your Video",type=["mp4"])
if user_video:
    audio_file, video_temp_path = vid_to_aud(user_video)
    if audio_file:
        
        try:
            
            model =load_whisper()
            result =model.transcribe(str(audio_file))
            st.success("Audio has been transcribed properly")
            transcript = result["text"]
        except Exception as e:
            st.error(f"Whisper transcription failed: {e}")
            transcript = ""
        
        st.subheader("Transcript")
        st.write(transcript)
        if transcript:
            try:

                client = Groq(api_key=st.secrets["GROQ_API_KEY"])
                chat_completion1 = client.chat.completions.create(
                    messages=[
                        {
                            "role":"user",
                            "content":f"You are an AI that summarizes transcripts accurately without adding new information. DO NOT introduce unrelated details. Focus ONLY on the key ideas discussed in the transcript. Generate a clear, coherent summary between 100 and 250 words, prioritizing the most important points, and ensure the output does not exceed 3000 characters. Transcript: {transcript} "

                        }
                    ],
                    model = "llama-3.3-70b-versatile",
                )
                summary = chat_completion1.choices[0].message.content
            except Exception as e:
                summary = transcript # if summariser fails, it will take the transcript as the summary
            st.subheader("Summary")
            st.write(summary)
            os.remove(audio_file)
            try:
                
                chat_completion = client.chat.completions.create(
                    messages=[
                        {
                            "role":"user",
                            "content":f"You are an AI that generates highly relevant YouTube thumbnail prompts.DO NOT make up unrelated topics. Focus ONLY on what the transcript discusses.Analyze the transcript and generate a concise and engaging thumbnail description The image should not contain any text. Now, generate a YouTube thumbnail prompt based on the following summary:{summary} "

                        }
                    ],
                    model = "llama-3.3-70b-versatile",
                )
                generated_prompt= chat_completion.choices[0].message.content
            except Exception as e:
                st.error(f"Groq API failed: {e}")
                generated_prompt = ""
            st.subheader("Generated Thumbnail Prompt")
            st.write(generated_prompt)

            if generated_prompt:
            # Converting from text to image

                try:
                    grad_client = Client("black-forest-labs/FLUX.1-schnell")
                    result = grad_client.predict(
                        prompt=generated_prompt,
                        seed= 0,
                        width=512,
                        height=912,
                        num_inference_steps=4,
                        api_name="/infer"
                    )
                
                    if isinstance(result, tuple) or isinstance(result, list):
                        image_path = result[0]
                    else:
                        image_path = result

                    st.image(image_path, caption="Generated Thumbnail")

                    with open(image_path,"rb") as download_image_file:
                        st.download_button(
                            label = "Download Thumbnail",
                            data= download_image_file,
                            file_name="thumbnail.webp",
                            mime="image/webp"
                        )
                except Exception as e :
                    st.error(f"Image generation failed: {e}")

                

    else:
        st.text("Audio Transcription Error")
