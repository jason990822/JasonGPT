import json
import time
from fastapi import FastAPI, UploadFile, BackgroundTasks, Header, Form, File, Query
from fastapi.responses import FileResponse
import openai
import shutil
import uuid
from gtts import gTTS
import ffmpeg
import base64
from fastapi.staticfiles import StaticFiles
import os
from pydub import AudioSegment
import uvicorn
import jasonassist


openai.api_key = ""
AI_COMPLETION_MODEL = os.getenv("AI_COMPLETION_MODEL", "gpt-3.5-turbo")
LANGUAGE = os.getenv("LANGUAGE", "zh")
AUDIO_SPEED = os.getenv("AUDIO_SPEED", None)
app = FastAPI()
user_mode_flag = {}
user_content_flag = {}
user_flag = {}
jasonsql = {
    '1209601741': 100
}  # 模拟数据库
USER_DIRTION_PREFIX = 'C:\\Users\\12096\\Desktop\\JasonGPT\\app'


@app.post("/inference")
def infer(audio: UploadFile, background_tasks: BackgroundTasks,
          conversation: str = Header(default=None), link: str = Header(default=None), userid: str = Header(default=None), docfile: str = Header(default=None)) -> FileResponse:

    # print(json.loads(base64.b64decode(link)))
    # conversation == mode 的意思
    # 1、看模式，再根据模式判断
    mode = json.loads(base64.b64decode(conversation))
    mylink = json.loads(base64.b64decode(link))
    myid = json.loads(base64.b64decode(userid))
    mydoc = json.loads(base64.b64decode(docfile))

    if myid not in jasonsql or jasonsql.get(myid) <= 0:
        return

    if mode == 'Bilibili' or mode == 'Youtube' or mode == 'URL':
        if myid in user_mode_flag:
            pre_mode = user_mode_flag.get(myid)
            if pre_mode == 'URL':
                if mylink == user_content_flag.get(myid):
                    user_flag.update({myid: False})
                else:
                    user_flag.update({myid: True})
                    user_content_flag.update({myid: mylink})
            else:
                user_flag.update({myid: True})
                user_content_flag.update({myid: mylink})
        else:
            user_flag.update({myid: True})
            user_content_flag.update({myid: mylink})
            user_mode_flag.update({myid: 'URL'})

    elif mode == 'Document':
        if myid in user_mode_flag:
            pre_mode = user_mode_flag.get(myid)
            if pre_mode == 'Document':
                print(user_content_flag.get(myid))
                if mydoc == user_content_flag.get(myid):
                    user_flag.update({myid: False})
                else:
                    user_flag.update({myid: True})
                    user_content_flag.update({myid: mydoc})
            else:
                user_flag.update({myid: True})
                user_content_flag.update({myid: mydoc})
        else:
            user_flag.update({myid: True})
            user_content_flag.update({myid: mydoc})
            user_mode_flag.update({myid: 'Document'})

    # return
    user_prompt = transcribe(audio)

    ai_response = get_completion(user_prompt, mode, mylink, mydoc, myid)

    output_audio_filepath = to_audio(ai_response)
    background_tasks.add_task(delete_file, output_audio_filepath)

    return FileResponse(path=output_audio_filepath, media_type="audio/mpeg",
                        headers={"text": construct_response_header(user_prompt, ai_response)})


@app.post("/upload")
async def documentupload(userid: str = Form(...), files: UploadFile = File()):
    if userid in jasonsql:
        try:
            print(userid+"===")
            # 判断文件夹是否存在,如果不存在就创建
            folder_path = os.path.join(
                USER_DIRTION_PREFIX, 'text_content', userid)
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)

            file_location = os.path.join(
                USER_DIRTION_PREFIX, 'text_content', userid, files.filename)
            with open(file_location, "wb") as f:
                contents = await files.read()
                await f.write(contents)

        except:
            return "worry"


@app.post("/test")
def test(audio: UploadFile):
    print("received request")

    initial_filepath = f"C:/Users/12096/Desktop/VoxGPT-main/app/audio/{uuid.uuid4()}{audio.filename}"

    with open(initial_filepath, "wb+") as file_object:
        shutil.copyfileobj(audio.file, file_object)

    print("done")


# 查询剩余次数
@app.get("/remainder")
def remainder(userid: str = Query(None)):
    if userid in jasonsql:
        return jasonsql[userid]
    else:
        return 'The key does not exist'


@app.get("/list")
def listDocument(userid: str = Query(None)):
    if userid in jasonsql:
        path = os.path.join(USER_DIRTION_PREFIX, 'text_content', userid)
        allowed_extensions = ('.pdf', '.docx', '.doc', '.txt', '.pptx', '.mp4')
        txt_files = [os.path.basename(file) for file in os.listdir(
            path) if os.path.splitext(file)[-1] in allowed_extensions]
        return txt_files


def transcribe(audio):
    start_time = time.time()
    initial_filepath = f"C:/Users/12096/Desktop/JasonGPT/app/audio/{uuid.uuid4()}{audio.filename}"

    with open(initial_filepath, "wb+") as file_object:
        shutil.copyfileobj(audio.file, file_object)

    converted_filepath = f"C:/Users/12096/Desktop/JasonGPT/app/audio/ffmpeg-{uuid.uuid4()}{audio.filename}"

    print("running through ffmpeg")
    (
        ffmpeg
        .input(initial_filepath)
        .output(converted_filepath, loglevel="error")
        .run()
    )
    print("ffmpeg done")

    delete_file(initial_filepath)

    with open(converted_filepath, "rb") as read_file:
        transcription = (openai.Audio.transcribe(
            "whisper-1", read_file))["text"]

    delete_file(converted_filepath)

    return transcription


def get_completion(user_prompt, mode, ulink, udoc, uid):

    if mode == 'ChatBox':
        jasonsql[uid] -= 1
        if jasonsql[uid] < 0:
            return
        completion = jasonassist.creat_chat_rob(user_prompt, uid)
        return completion

    elif mode == 'SerperGoogle':
        jasonsql[uid] -= 1
        if jasonsql[uid] < 0:
            return
        completion = jasonassist.google_serper(user_prompt, uid)
        return completion

    elif mode == 'Youtube':
        jasonsql[uid] -= 2
        if jasonsql[uid] < 0:
            return
        completion = jasonassist.talk_2_youtube(
            uid, ulink, user_prompt, user_flag.get(uid))
        return completion

    elif mode == 'Bilibili':
        jasonsql[uid] -= 2
        if jasonsql[uid] < 0:
            return
        completion = jasonassist.talk_2_bilibili(
            uid, [ulink], user_prompt, user_flag.get(uid))
        return completion

    elif mode == 'URL':
        jasonsql[uid] -= 2
        if jasonsql[uid] < 0:
            return
        completion = jasonassist.talk_2_url(
            uid, [ulink], user_prompt, user_flag.get(uid))
        return completion

    elif mode == 'Document':
        # 遍历udoc文件列表
        jasonsql[uid] -= 3
        if jasonsql[uid] < 0:
            return
        if user_flag.get(uid):
            dir_path = os.path.join(USER_DIRTION_PREFIX, 'data', uid)
            jasonassist.delete_files(dir_path)
            for filenode in udoc:
                jasonassist.files_2_store(filenode, uid)
        completion = jasonassist.talk_2_letter_without_db(
            uid, user_prompt, user_flag.get(uid))
        return completion


def to_audio(text):

    tts = gTTS(text, lang=LANGUAGE)
    filepath = f"C:/Users/12096/Desktop/JasonGPT/app/audio/{uuid.uuid4()}.mp3"
    tts.save(filepath)

    speed_adjusted_filepath = adjust_audio_speed(filepath)

    return speed_adjusted_filepath


def adjust_audio_speed(audio_filepath):
    if AUDIO_SPEED is None:
        return audio_filepath

    audio = AudioSegment.from_mp3(audio_filepath)
    faster_audio = audio.speedup(playback_speed=float(AUDIO_SPEED))

    speed_adjusted_filepath = f"C:/Users/12096/Desktop/JasonGPT/app/audio/{uuid.uuid4()}.mp3"
    faster_audio.export(speed_adjusted_filepath, format="mp3")

    delete_file(audio_filepath)

    return speed_adjusted_filepath


def delete_file(filepath: str):
    os.remove(filepath)


def construct_response_header(user_prompt, ai_response):
    return base64.b64encode(
        json.dumps(
            [{"role": "user", "content": user_prompt}, {"role": "assistant", "content": ai_response}]).encode(
            'utf-8')).decode("utf-8")


app.mount("/", StaticFiles(directory="app/static", html=True), name="static")

if __name__ == '__main__':
    uvicorn.run(app, host="127.0.0.1", port=9000)
