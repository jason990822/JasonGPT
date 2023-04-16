from langchain.llms import OpenAI
import os
from langchain.utilities import GoogleSerperAPIWrapper
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains.llm import LLMChain
from langchain.callbacks.base import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains.conversational_retrieval.prompts import CONDENSE_QUESTION_PROMPT, QA_PROMPT
from langchain.chains.question_answering import load_qa_chain
from langchain.memory import ConversationBufferMemory
from langchain import OpenAI, LLMChain, PromptTemplate
from pydub import AudioSegment
from moviepy.editor import *
from langchain.document_loaders import UnstructuredWordDocumentLoader
from langchain.document_loaders import YoutubeLoader
import youtube_transcript_api
from langchain.document_loaders import SeleniumURLLoader
from langchain.document_loaders import UnstructuredPDFLoader
from langchain.document_loaders import UnstructuredPowerPointLoader
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferWindowMemory
import shutil
import openai
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter


openai.api_key = ""
USER_DIRTION_PREFIX = 'C:\\Users\\12096\\Desktop\\JasonGPT\\app'
os.environ["SERPER_API_KEY"] = ""
Memorys = {}  # 与机器人交流时的记忆
Chat_histories = {}  # 与用户对应的聊天记录
db = {}  # 用户对应的数据库


def add_memory(user_id):
    memory = ConversationBufferWindowMemory(memory_key='chat_history', k=3)
    Memorys.update({user_id: memory})
    return memory


def search_memory(user_id):
    if user_id in Memorys:
        return Memorys[user_id]
    else:
        return add_memory(user_id)


def creat_chat_rob(message, user_id):
    llm = OpenAI(streaming=True, callback_manager=CallbackManager(
        [StreamingStdOutCallbackHandler()]), verbose=True, temperature=0)
    template = """You are a chatbot having a conversation with a human.

    {chat_history}
    Human: {human_input}
    Chatbot:"""

    prompt = PromptTemplate(
        input_variables=["chat_history", "human_input"],
        template=template
    )
    memory = search_memory(user_id)

    llm_chain = LLMChain(
        llm=llm,
        prompt=prompt,
        verbose=True,
        memory=memory,
    )
    answer = llm_chain.predict(human_input=message)
    return answer

# start-当chat_rob效果不好时，选择GoogleSerper模式


def google_serper(message, user_id):
    search = GoogleSerperAPIWrapper()
    result = search.run(message)
    memory = search_memory(user_id)
    memory.save_context({"input": message}, {"ouput": result})
    return result
# end-当chat_rob效果不好时，选择GoogleSerper模式


def files_2_store(file_name, user_id):
    extension = os.path.splitext(file_name)[1]  # 文件后缀
    if extension == '.txt':
        txt_store_document(file_name, user_id)
    elif extension == '.doc' or extension == '.docx':
        word_store_document(file_name, user_id)
    elif extension == '.pdf':
        pdf_store_document(file_name, user_id)
    elif extension == '.pptx':
        ppt_store_document(file_name, user_id)
    elif extension == '.mp4':
        video_2_txt(file_name, user_id)
    # elif extension == '.mp3':
    #     audio_2_txt(os.path.splitext(filename)[0], user_id)


def video_2_txt(video_name, user_id):
    video_path = os.path.join(
        USER_DIRTION_PREFIX, 'text_content', user_id, video_name)
    my_audio_clip = AudioFileClip(video_path)
    prefix = video_name[:video_name.index(".")]
    my_audio_clip.write_audiofile(os.path.join(
        USER_DIRTION_PREFIX, 'text_content', user_id, prefix+'.mp3'))
    audio_2_txt(prefix, user_id)
    # 删除视频文件
    os.remove(video_path)
    txt_store_document(os.path.splitext(video_path)[0]+'.txt', user_id)


def audio_2_txt(prefix, user_id):
    AUDIOS_DIRTION = os.path.join(USER_DIRTION_PREFIX, 'text_content', user_id)
    song = AudioSegment.from_mp3(os.path.join(AUDIOS_DIRTION, prefix+".mp3"))
    duration = song.duration_seconds
    duration_minutes = duration / 60

    if duration_minutes > 20:
        start = 0
        end = 20 * 60 * 1000
        counter = 1
        while end < len(song):
            newAudio = song[start:end]
            newAudio.export(os.path.join(AUDIOS_DIRTION, prefix +
                            "_part"+str(counter)+".mp3"), format="mp3")
            speech_2_text(os.path.join(AUDIOS_DIRTION, prefix +
                          "_part"+str(counter)+".mp3"), prefix, user_id)
            os.remove(os.path.join(AUDIOS_DIRTION,
                      prefix+"_part"+str(counter)+".mp3"))
            start = end
            end = start + 20 * 60 * 1000
            counter += 1
        newAudio = song[start:len(song)]
        newAudio.export(os.path.join(AUDIOS_DIRTION, prefix +
                        "_part"+str(counter)+".mp3"), format="mp3")
        speech_2_text(os.path.join(AUDIOS_DIRTION, prefix +
                      "_part"+str(counter)+".mp3"), prefix, user_id)
        os.remove(os.path.join(AUDIOS_DIRTION,
                  prefix+"_part"+str(counter)+".mp3"))
    else:
        speech_2_text(os.path.join(AUDIOS_DIRTION,
                      prefix+".mp3"), prefix, user_id)
        os.remove(os.path.join(AUDIOS_DIRTION, prefix+".mp3"))


def speech_2_text(path, text_retain_name, user_id):
    audio_file = open(path, "rb")
    transcript = openai.Audio.transcribe("whisper-1", audio_file)
    txt_file = open(os.path.join(USER_DIRTION_PREFIX, 'text_content', user_id, text_retain_name+".txt"),
                    mode='a', encoding='gbk')
    txt_file.write(transcript["text"])
    txt_file.close()


# 当文档不存在数据库中
def txt_store_document(file_name, user_id):
    loader = TextLoader(os.path.join(USER_DIRTION_PREFIX,
                        'text_content', user_id, file_name), encoding="utf-8")
    persist_letter_db(loader, user_id)


def word_store_document(file_name, user_id):
    loader = UnstructuredWordDocumentLoader(os.path.join(
        USER_DIRTION_PREFIX, 'text_content', user_id, file_name))
    persist_letter_db(loader, user_id)


def pdf_store_document(file_name, user_id):
    loader = UnstructuredPDFLoader(file_path=os.path.join(
        USER_DIRTION_PREFIX, 'text_content', user_id, file_name))
    persist_letter_db(loader, user_id)

# 输入ppt,将对网页内容进行向量化，存储在数据库中，返回数据库


def ppt_store_document(file_name, user_id):
    loader = UnstructuredPowerPointLoader(os.path.join(
        USER_DIRTION_PREFIX, 'text_content', user_id, file_name))
    persist_letter_db(loader, user_id)


def persist_letter_db(loader, user_id):
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=600, chunk_overlap=20)
    docs = text_splitter.split_documents(documents)
    embeddings = OpenAIEmbeddings()
    documentdata = os.path.join(USER_DIRTION_PREFIX, 'data', user_id)
    os.makedirs(documentdata, exist_ok=True)
    db = Chroma.from_documents(
        docs, embeddings, persist_directory=documentdata)
    db.persist()
    db = None
    # db.delete_collection()


def talk_2_letter(user_id, user_db, message, flag):
    # dir_path = os.path.join(USER_DIRTION_PREFIX, 'data', user_id)
    if flag:
        dele_chat_history(user_id)  # 之前的聊天记录也要删除
        user_chat_history = get_document_chat_history(user_id)
        ans = communication_document(user_db, user_chat_history, message)
        return ans
    else:
        user_chat_history = get_document_chat_history(user_id)
        ans = communication_document(user_db, user_chat_history, message)
        return ans


def talk_2_letter_without_db(user_id, message, flag):
    dirpath = os.path.join(USER_DIRTION_PREFIX, 'data', user_id)
    vdb = get_vectorstore(dirpath)
    ans = talk_2_letter(user_id, vdb, message, flag)
    # vdb.delete_collection()
    vdb = None
    return ans


def get_vectorstore(persist_directory):
    # Now we can load the persisted database from disk, and use it as normal.
    embedding = OpenAIEmbeddings()
    vectordb = Chroma(persist_directory=persist_directory,
                      embedding_function=embedding)
    return vectordb


def add_document_history(user_id):
    history = []
    Chat_histories.update({user_id: history})
    return history


def dele_chat_history(user_id):
    if user_id in Chat_histories:
        del Chat_histories[user_id]


def get_document_chat_history(user_id):
    if user_id in Chat_histories:
        return Chat_histories[user_id]
    else:
        return add_document_history(user_id)

# 根据数据库,让用户以及文本内容交流，不同的用户应该对应的是不同的chat_history，可以把chat_history
# 可以把chat_history存储在前端或者内存中，目前是存储在内存中


def communication_document(vectorstore, chat_history, query):
    llm = OpenAI(temperature=0)
    streaming_llm = OpenAI(streaming=True, callback_manager=CallbackManager(
        [StreamingStdOutCallbackHandler()]), verbose=True, temperature=0)

    question_generator = LLMChain(llm=llm, prompt=CONDENSE_QUESTION_PROMPT)
    doc_chain = load_qa_chain(
        streaming_llm, chain_type="stuff", prompt=QA_PROMPT)

    qa = ConversationalRetrievalChain(
        retriever=vectorstore.as_retriever(), combine_docs_chain=doc_chain, question_generator=question_generator)

    result = qa({"question": query, "chat_history": chat_history})
    # 对于文档只保留最近3个对话，防止tokens溢出
    if len(chat_history) > 3:
        chat_history.pop(0)
    chat_history.append((query, result["answer"]))

    return result["answer"]


def youtube_store_document(user_id, youtube_url):
    video_id = youtube_url.split("youtube.com/watch?v=")[-1]
    transcript_list = youtube_transcript_api.YouTubeTranscriptApi.list_transcripts(
        video_id)
    for transcript in transcript_list:
        language = transcript.language_code
        loader = YoutubeLoader(video_id=video_id, language=language)
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=200, chunk_overlap=20)
        docs = text_splitter.split_documents(documents)
        embeddings = OpenAIEmbeddings()
        db = Chroma.from_documents(docs, embeddings)
        return db


def talk_2_youtube(user_id, youtube_url, message, flag):
    if flag:
        user_db = youtube_store_document(user_id, youtube_url)
        if user_id in db:  # 更新之前的db，必须要把之前的db给释放了
            db.get(user_id).delete_collection()
        db.update({user_id: user_db})
        dele_chat_history(user_id)  # 之前的聊天记录也要删除
        user_chat_history = get_document_chat_history(user_id)
        ans = communication_document(user_db, user_chat_history, message)
        return ans
    else:
        user_db = db.get(user_id)
        user_chat_history = get_document_chat_history(user_id)
        ans = communication_document(user_db, user_chat_history, message)
        return ans


# 输入url,将对网页内容进行向量化，存储在数据库中，返回数据库
def url_store_document(user_id, urls):
    loader = SeleniumURLLoader(urls=urls)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=100, chunk_overlap=10)
    docs = text_splitter.split_documents(documents)
    embeddings = OpenAIEmbeddings()
    db = Chroma.from_documents(docs, embeddings)
    return db


def talk_2_url(user_id, link_url, message, flag):
    if flag:
        user_db = url_store_document(user_id, link_url)
        if user_id in db:  # 更新之前的db，必须要把之前的db给释放了
            db.get(user_id).delete_collection()
        db.update({user_id: user_db})
        dele_chat_history(user_id)  # 之前的聊天记录也要删除
        user_chat_history = get_document_chat_history(user_id)
        ans = communication_document(user_db, user_chat_history, message)
        return ans
    else:
        user_db = db.get(user_id)
        user_chat_history = get_document_chat_history(user_id)
        ans = communication_document(user_db, user_chat_history, message)
        return ans


def delete_files(directory):
    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        if os.path.isfile(filepath):
            os.remove(filepath)
        elif os.path.isdir(filepath):
            shutil.rmtree(filepath)
