import streamlit as st
# from streamlit.components.v1 import html
from bot import chat_with_alpaca

#
# my_js = """
# function resize() {
# try {
# var height1 = document.querySelector(".stMainBlockContainer")?.clientHeight; var height2 = document.querySelector('div[data-testid="stBottomBlockContainer"]').clientHeight; console.log(height1 + height2); window.parent.postMessage(["setHeight", height1 + height2], "*");
# } catch(e) {}
# }
# document.addEventListener('DOMContentLoaded', function() {
#       resize();
#       setInterval(resize, 200);
#       alert('1');
#     })
# """
# my_html = f"<script>{my_js}</script>"

# html(f'<script src="https://cdn.jsdelivr.net/npm/@iframe-resizer/child@5.4.6"></script>')
st.set_page_config(page_title="Alpaca Chatbot")
with open("./style.css") as css:
    st.markdown(f'<style>{css.read()}</style>', unsafe_allow_html=True)
# st.title("Alpaca AI")
# st.subheader("Ask about the system, the site — or how Alpaca can boost your clinic")

# Initialize session state
if "history" not in st.session_state:
    st.session_state.history = []

# User input
user_input = st.chat_input("Ask your question:")

# Handle user input
if user_input:
    with st.spinner("Thinking..."):
        reply = chat_with_alpaca(user_input)
        st.session_state.history.append(("user", user_input))
        st.session_state.history.append(("bot", reply))

# Display chat history
for speaker, message in st.session_state.history:
    if speaker == "user":
        with st.chat_message("user", avatar='./user.png'):
            st.markdown(message)
    else:
        with st.chat_message("assistant", avatar='./alpaca.png'):
            st.markdown(message)
# st.markdown(my_html, unsafe_allow_html=True)
# html(my_html)
