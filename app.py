# import streamlit as st 
# from llama_index.core import SimpleDirectoryReader
# from llama_index.core import VectorStoreIndex
# from llama_index.core import ServiceContext

# from llama_index.llms.llama_cpp import LlamaCPP
# from llama_index.llms.llama_cpp.llama_utils import (
#     messages_to_prompt,
#     completion_to_prompt,
# )

# from langchain.schema import(SystemMessage, HumanMessage, AIMessage)

# def init_page() -> None:
#   st.set_page_config(
#     page_title="Personal Chatbot"
#   )
#   st.header("Persoanl Chatbot")
#   st.sidebar.title("Options")

# def select_llm() -> LlamaCPP:
#   return LlamaCPP(
#     model_path="/content/llama-2-7b-chat.Q2_K.gguf",
#     temperature=0.1,
#     max_new_tokens=500,
#     context_window=3900,
#     generate_kwargs={},
#     model_kwargs={"n_gpu_layers":1},
#     messages_to_prompt=messages_to_prompt,
#     completion_to_prompt=completion_to_prompt,
#     verbose=True,
#   )

# def init_messages() -> None:
#   clear_button = st.sidebar.button("Clear Conversation", key="clear")
#   if clear_button or "messages" not in st.session_state:
#     st.session_state.messages = [
#       SystemMessage(
#         content="you are a helpful AI assistant. Reply your answer in markdown format."
#       )
#     ]

# def get_answer(llm, messages) -> str:
#   response = llm.complete(messages)
#   return response.text

# def main() -> None:
#   init_page()
#   llm = select_llm()
#   init_messages()

#   if user_input := st.chat_input("Input your question!"):
#     st.session_state.messages.append(HumanMessage(content=user_input))
#     with st.spinner("Bot is typing ..."):
#       answer = get_answer(llm, user_input)
#       print(answer)
#     st.session_state.messages.append(AIMessage(content=answer))
    

#   messages = st.session_state.get("messages", [])
#   for message in messages:
#     if isinstance(message, AIMessage):
#       with st.chat_message("assistant"):
#         st.markdown(message.content)
#     elif isinstance(message, HumanMessage):
#       with st.chat_message("user"):
#         st.markdown(message.content)

# if __name__ == "__main__":
#   main()


import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

class ChatBot:
    def __init__(self, model_name="tiiuae/falcon-7b-instruct"):
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        
        # Move model to GPU if available
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)

    def generate_response(self, input_text, chat_history_ids=None):
        # Tokenize the input
        new_user_input_ids = self.tokenizer.encode(
            input_text + self.tokenizer.eos_token, 
            return_tensors='pt'
        ).to(self.device)
        
        # Append to chat history if exists
        if chat_history_ids is not None:
            bot_input_ids = torch.cat([chat_history_ids, new_user_input_ids], dim=-1)
        else:
            bot_input_ids = new_user_input_ids

        # Generate a response 
        chat_history_ids = self.model.generate(
            bot_input_ids, 
            max_length=2000,
            pad_token_id=self.tokenizer.eos_token_id,
            temperature=0.4,
            do_sample=True
        )
        
        # Decode the response
        response = self.tokenizer.decode(
            chat_history_ids[:, bot_input_ids.shape[-1]:][0], 
            skip_special_tokens=True
        )
        
        return response, chat_history_ids

def init_page():
    st.set_page_config(page_title="EduPoints - Practitioner & Patients Chathelpbot")
    st.header("EduPoint - Practitioner & Patients Chathelp bot")
    st.sidebar.title("Chathelpbot Settings")

def main():
    init_page()
    
    # Initialize chatbot
    chatbot = ChatBot()
    
    # Initialize chat history in session state
    if 'chat_history_ids' not in st.session_state:
        st.session_state['chat_history_ids'] = None
    
    if 'messages' not in st.session_state:
        st.session_state.messages = []

    # Clear conversation button
    if st.sidebar.button("Clear Conversation"):
        st.session_state.messages = []
        st.session_state.chat_history_ids = None

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # User input
    if prompt := st.chat_input("Enter your message"):
        # Add user message to chat history
        st.session_state.messages.append(
            {"role": "user", "content": prompt}
        )
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate bot response
        with st.chat_message("assistant"):
            with st.spinner("Generating response..."):
                response, chat_history_ids = chatbot.generate_response(
                    prompt, 
                    st.session_state.chat_history_ids
                )
                st.markdown(response)
        
        # Update chat history
        st.session_state.messages.append(
            {"role": "assistant", "content": response}
        )
        st.session_state.chat_history_ids = chat_history_ids

if __name__ == "__main__":
    main()