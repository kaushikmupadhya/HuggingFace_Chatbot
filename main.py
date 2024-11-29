import wget 

def bar_custom(current, total, width=80):
    print("Downloading %d%% [%d / %d] bytes" % (current / total * 100, current, total))

model_url = "https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF/resolve/main/llama-2-7b-chat.Q2_K.gguf"
wget.download(model_url, bar=bar_custom)

# !pip -q install streamlit


# !pip install llama-index-embeddings-huggingface
# !pip install llama-index-llms-llama-cpp
 
# !streamlit run app.py & npx localtunnel --port 8501
