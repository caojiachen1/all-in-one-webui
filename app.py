import gradio as gr
from TTService import TTService
import file, os, numpy, requests, json, contextlib
from time import time
from torch import float16

os.environ["TRANSFORMERS_CACHE"] = r"E:\models"
os.environ["HUGGINGFACE_HUB_CACHE"] = r"E:\models"

tokenizer, model_dir, model = None, None, None
sdxlturbo_pipe = None
sdturbo_pipe = None
whisper_pipe = None
distil_whisper_v3_pipe, distill_whisper_v3_processor, distil_whisper_v3_model = None, None, None

MODEL_NAME = "openai/whisper-large-v3"
BATCH_SIZE = 8
FILE_LIMIT_MB = 1000

def openai_respond(openai_base_url, openai_api_key, gpt_modelname, history, temperature=None, top_p=None, ):
    if gpt_modelname not in ["gpt-3.5-turbo", "gpt-4"]:
        return 
    messages = []
    for idx, (user, bot) in enumerate(history):
        messages.append({"role": "user", "content": user})
        if idx != len(history) - 1:
            messages.append({"role": "assistant", "content": bot})

    # openai_api_key = os.environ["OPENAI_SB_API_KEY"]
    # openai_base_url = r"https://api.openai-sb.com/v1/chat/completions"
    data = {
        "model" : gpt_modelname,
        "messages" : messages,
        "stream" : True,
        "top_p" : top_p,
        "temperature" : temperature,
    }
    header = {
        "Authorization" : f"Bearer {openai_api_key}",
        "Content-Type" : "application/json"
    }
    result = requests.post(openai_base_url , headers = header , json = data , stream = True)
    for line in result.iter_lines():
        if line != b"":
            with contextlib.suppress(Exception):
                a = line.decode("utf-8").split(":" , 1)[1].strip()
                a = json.loads(a)
                respond = a["choices"][0]["delta"].get("content", "")
                yield respond

def clear_gpu_cache():
    from torch.cuda import empty_cache
    global tokenizer, model_dir, model, sdxlturbo_pipe, whisper_pipe, sdturbo_pipe
    global distil_whisper_v3_pipe, distill_whisper_v3_processor, distil_whisper_v3_model
    del distil_whisper_v3_pipe, distill_whisper_v3_processor, distil_whisper_v3_model
    del tokenizer, model_dir, model, sdxlturbo_pipe, whisper_pipe, sdturbo_pipe
    tokenizer, model_dir, model = None, None, None
    sdxlturbo_pipe = None, None
    whisper_pipe = None
    sdturbo_pipe = None
    distil_whisper_v3_pipe, distill_whisper_v3_processor, distil_whisper_v3_model = None, None, None
    empty_cache()
    gr.Info("Success")

def scan_model():
    models_scan_folder = r".\models"
    configs_scan_folder = r".\configs"
    models_ = file.folder(models_scan_folder)
    configs_ = file.folder(configs_scan_folder)
    scan_models = [os.path.basename(i) for i in models_.files]
    scan_configs = [os.path.basename(i) for i in configs_.files]
    for model in scan_models:
        config = file.splitname(model)[0].split("_")[0] + ".json"
        if config not in scan_configs:
            print(f"Model {model} has no config! Removing it from the model list...")
            scan_models.remove(model)
    return scan_models

def gpu_memory_status():
    import pynvml
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    used_memory = mem_info.used / 1024**3
    free_memory = mem_info.free / 1024**3
    pynvml.nvmlShutdown()
    return free_memory, used_memory 

def chat_llm(llm_name, input, chatbot_history, temperature, top_p, openai_base_url=r"https://api.openai-sb.com/v1/chat/completions", openai_api_key=os.environ["OPENAI_SB_API_KEY"]):
    chatbot_history = chatbot_history or []
    load_localhost_models(llm_name)
    if llm_name == "chatglm2-6b":
        try:
            chatbot_history = list(map(tuple, chatbot_history))
            if gpu_memory_status()[0] < 0.3:
                chatbot_history = []
            for respond , history in model.stream_chat(tokenizer , input , history = chatbot_history, temperature=temperature, top_p=top_p):
                yield history
        except:
            return [(input, "Error!")]
    
    elif llm_name == "gpt-3.5-turbo":
        chatbot_history = chatbot_history or []
        chatbot_history.append([input, ""])

        prediction = openai_respond(
            openai_base_url=openai_base_url,
            openai_api_key=openai_api_key,
            gpt_modelname="gpt-3.5-turbo",
            history=chatbot_history,
            temperature=temperature,
            top_p=top_p
        )

        for delta_text in prediction:
            chatbot_history[-1][1] += delta_text
            yield chatbot_history
    
    elif llm_name == "gpt-4":
        chatbot_history = chatbot_history or []
        chatbot_history.append([input, ""])

        prediction = openai_respond(
            openai_base_url=openai_base_url,
            openai_api_key=openai_api_key,
            gpt_modelname="gpt-4",
            history=chatbot_history,
            temperature=temperature,
            top_p=top_p
        )

        for delta_text in prediction:
            chatbot_history[-1][1] += delta_text
            yield chatbot_history

def load_localhost_models(modelname):
    global model, model_dir, tokenizer, sdxlturbo_pipe, whisper_pipe, sdturbo_pipe
    global distil_whisper_v3_pipe, distil_whisper_v3_model, distill_whisper_v3_processor
    
    if modelname == "chatglm2-6b":
        if tokenizer is None or model is None or model_dir is None:
            if gpu_memory_status()[0] < 5:
                clear_gpu_cache()
            gr.Info("Loading chatglm2-6b...")
            from modelscope import AutoTokenizer, AutoModel, snapshot_download
            os.environ["MODELSCOPE_CACHE"] = r"E:\models"
            model_dir = snapshot_download("ZhipuAI/chatglm2-6b-int4", revision="v1.0.1")
            tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
            model = AutoModel.from_pretrained(model_dir, trust_remote_code=True).half().cuda().eval()
            gr.Info("Success!")
    
    elif modelname == "sdxlturbo":
        if sdxlturbo_pipe is None:
            clear_gpu_cache()
            gr.Info("Loading sdxl-turbo...")
            from diffusers import AutoPipelineForText2Image
            sdxlturbo_pipe = AutoPipelineForText2Image.from_pretrained("stabilityai/sdxl-turbo", torch_dtype=float16, use_safetensors=True, variant="fp16")
            sdxlturbo_pipe = sdxlturbo_pipe.to("cuda")
            gr.Info("Success")
    
    elif modelname == "sdturbo":
        if sdturbo_pipe is None:
            if gpu_memory_status()[0] < 5:
                clear_gpu_cache()
            gr.Info("Loading sd-turbo...")
            from diffusers import AutoPipelineForText2Image
            sdturbo_pipe = AutoPipelineForText2Image.from_pretrained("stabilityai/sd-turbo", torch_dtype=float16, use_safetensors=True, variant="fp16")
            sdturbo_pipe = sdturbo_pipe.to("cuda")
            gr.Info("Success")

    elif modelname == "whisper-large-v3":
        if whisper_pipe is None:
            if gpu_memory_status()[0] < 7:
                clear_gpu_cache()
            gr.Info("Loading whisper-large-v3...")
            from transformers import pipeline
            clear_gpu_cache()
            whisper_pipe = pipeline(
                task="automatic-speech-recognition",
                model=MODEL_NAME,
                chunk_length_s=30,
                device=0,
            )
            gr.Info("Success")
    
    elif modelname == "distilled-whisper-large-v3":
        if distil_whisper_v3_pipe is None or distil_whisper_v3_model is None or distill_whisper_v3_processor is None:
            clear_gpu_cache()
            gr.Info("Loading distilled whisper-large-v3...")
            from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
            distil_whisper_v3_model = AutoModelForSpeechSeq2Seq.from_pretrained(
                "Aspik101/distil-whisper-large-v3-pl", torch_dtype=float16, low_cpu_mem_usage=True, use_safetensors=False
            ).to("cuda")
            distill_whisper_v3_processor = AutoProcessor.from_pretrained("Aspik101/distil-whisper-large-v3-pl")
            distil_whisper_v3_pipe = pipeline(
                "automatic-speech-recognition",
                model=distil_whisper_v3_model,
                tokenizer=distill_whisper_v3_processor.tokenizer,
                feature_extractor=distill_whisper_v3_processor.feature_extractor,
                max_new_tokens=128,
                torch_dtype=float16,
                chunk_length_s=30,
                device="cuda",
            )
            gr.Info("Success")

    else:
        return

def generate_tts(modelpath, text, speed):
    cfg_path = file.splitname(modelpath)[0].split("_")[0] + ".json"
    a = TTService(cfg = os.path.join(os.getcwd(), "configs", cfg_path), 
                    model = os.path.join(os.getcwd(), "models", modelpath), 
                    speed = speed)
    length = len(text)
    if length > 300:
        audio = a.read("")
        for i in range(0, length, 300):
            slice_string = text[i : min(i + 300, length)]
            audio = numpy.concatenate((audio, a.read(slice_string)))
    else:
         audio = a.read(text)
    return "Success", (a.hps.data.sampling_rate, audio)

def generate_sdxlturbo(prompt, steps):
    load_localhost_models("sdxlturbo")
    return sdxlturbo_pipe(prompt=prompt, num_inference_steps=steps, guidance_scale=0.0).images[0]

def generate_sdturbo(prompt, steps):
    load_localhost_models("sdturbo")
    return sdturbo_pipe(prompt=prompt, num_inference_steps=steps, guidance_scale=0.0).images[0]

def generate_sd(modelname, prompt, steps):
    if modelname == "sd-turbo":
        return generate_sdturbo(prompt, steps)
    elif modelname == "sdxl-turbo":
        return generate_sdxlturbo(prompt, steps)
    else:
        return None

def transcribe_whisper(inputs, task, modelname):
    if inputs is None:
        gr.Warning("No audio file submitted! Please upload or record an audio file before submitting your request.")
        return None, None

    if modelname == "whisper-large-v3":
        load_localhost_models("whisper-large-v3")
        a = time()
        output = whisper_pipe(inputs, batch_size=BATCH_SIZE, generate_kwargs={"task": task}, return_timestamps=True)["text"]
        return output, time() - a
    
    elif modelname == "distilled-whisper-large-v3":
        load_localhost_models("distilled-whisper-large-v3")
        a = time()
        output = distil_whisper_v3_pipe(inputs, batch_size=BATCH_SIZE, generate_kwargs={"task": task}, return_timestamps=True)["text"]
        return output, time() - a

if __name__ == "__main__":
    scan_models = scan_model()
    with gr.Blocks() as app:
        with gr.Tabs():
            with gr.TabItem("TTS"):
                with gr.Tabs():
                    with gr.Row():
                        with gr.Column():
                            model_select = gr.Dropdown(choices=scan_models, value=scan_models[0], label="Model")
                            textbox = gr.TextArea(label="Text",
                                                    placeholder="Type your sentence here",
                                                    value="你好啊", elem_id="tts-input")
                            duration_slider = gr.Slider(minimum=0.1, maximum=5, value=1, step=0.1,
                                                        label="速度 Speed")
                        with gr.Column():
                            text_output = gr.TextArea(label="Output",lines=3)
                            audio_output = gr.Audio(label="Output Audio", elem_id="tts-audio")
                            tts_generate = gr.Button("Generate!")
                            tts_generate.click(generate_tts,
                                        inputs=[model_select, textbox, duration_slider],
                                        outputs=[text_output, audio_output])
                            clear_gpucache = gr.Button("卸载Model")

            with gr.TabItem("LLM"):
                with gr.Tabs():
                    with gr.Row():
                        chatbot = gr.Chatbot(height=512)
                        with gr.Column():
                            llms = ["gpt-3.5-turbo", "gpt-4", "chatglm2-6b"]
                            llm_select = gr.Dropdown(choices=llms, label="Model")
                            
                            msg = gr.Textbox(label="Input", placeholder="Type here")
                            with gr.Accordion(label="Settings", open=False):
                                openai_base_url_input = gr.Textbox(label="Openai base url", 
                                                                   visible=True,
                                                                   value=r"https://api.openai-sb.com/v1/chat/completions",)
                                openai_api_key_input = gr.Textbox(label="Openai api key",
                                                                  type="password",
                                                                  visible=True,
                                                                  value=os.environ["OPENAI_SB_API_KEY"])
                                llm_temperature = gr.Slider(0.0, 1.0, label="Temperature", step=0.1, value=0.5)
                                llm_top_p = gr.Slider(0.0, 1.0, label="Top P", step=0.05, value=1.0)
                            with gr.Row():
                                llm_generate = gr.Button("Generate")
                                clear_gpucache2 = gr.Button("卸载本地LLM")
                            with gr.Row():
                                # stop_llm = gr.Button("Stop")
                                clear_llm = gr.Button("Clear")

            with gr.TabItem("SD"):
                with gr.Row():
                    sd_generated_pic = gr.Image(width=512, height=512)
                    with gr.Column():
                        sd_models = ["sdxl-turbo", "sd-turbo"]
                        sd_model_select = gr.Dropdown(choices=sd_models, label="Model")
                        sd_prompt = gr.Textbox(placeholder="Input you prompt.", label="Prompt")
                        sd_step = gr.Slider(minimum=1, maximum=50, step=1, value=1, label="Step")
                        with gr.Row():
                            sd_generate = gr.Button("Generate")
                            clear_image_sd = gr.Button("Clear")
                        with gr.Row():
                            clear_gpucache3 = gr.Button("卸载SD")

                clear_gpucache3.click(clear_gpu_cache)
                clear_image_sd.click(lambda : None, None, sd_generated_pic, queue=False)
                sd_prompt.submit(generate_sd, inputs=[sd_model_select, sd_prompt, sd_step], outputs=[sd_generated_pic])
                sd_generate.click(generate_sd, inputs=[sd_model_select, sd_prompt, sd_step], outputs=[sd_generated_pic])
            
            with gr.TabItem("ASR"):
                asr_models = ["whisper-large-v3", "distilled-whisper-large-v3"]
                with gr.Row():
                    with gr.Column():
                        asr_audio_input = gr.Audio(type="filepath", label="Audio file")
                        asr_task = gr.Radio(choices=["transcribe", "translate"], label="Task", value="transcribe")
                        with gr.Row():
                            asr_clear = gr.Button(value="Clear")
                            asr_submit = gr.Button(value="Submit")
                    with gr.Column():
                        asr_model_select = gr.Dropdown(choices=asr_models, value=asr_models[0], label="Model")
                        asr_time = gr.Textbox(label="Time")
                        asr_output = gr.Textbox(label="Output")

            asr_clear.click(lambda : (None, None, None), None, [asr_audio_input, asr_time, asr_output], queue=False)

            asr_submit.click(transcribe_whisper, inputs=[asr_audio_input, asr_task, asr_model_select], outputs=[asr_output, asr_time])

            llm_select.input(load_localhost_models, inputs=[llm_select], outputs=[])

            clear_gpucache.click(clear_gpu_cache)

            clear_gpucache2.click(clear_gpu_cache)

            clear_llm.click(lambda: None, None, chatbot, queue=False)

            msg.submit(chat_llm, 
                       inputs=[llm_select, msg, chatbot, llm_temperature, llm_top_p, openai_base_url_input, openai_api_key_input], 
                       outputs=[chatbot], queue=True)

            llm_generate.click(chat_llm, 
                               inputs=[llm_select, msg, chatbot, llm_temperature, llm_top_p, openai_base_url_input, openai_api_key_input], 
                               outputs=[chatbot], queue=True)

            llm_generate.click(lambda : gr.update(value=""), [], [chatbot])

    app.queue().launch(show_api=True, share=False, inbrowser=True, show_error=True)
