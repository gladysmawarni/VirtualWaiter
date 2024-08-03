import warnings
import streamlit as st
import base64
import requests
import os

from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI


# Suppress warnings related to date parsing
warnings.filterwarnings("ignore")

# openai API key
os.environ['OPENAI_API_KEY'] = st.secrets['OPENAI_API_KEY']

# webapp title
st.title('Virtual Waiter')

# initialize columns for the buttons
col1, col2 = st.columns(2)

## SESSION STATES
if 'ready' not in st.session_state:
    st.session_state.ready = False
if 'read_image' not in st.session_state:
    st.session_state.read_image = True
if 'menu' not in st.session_state:
    st.session_state.menu = None
if 'db' not in st.session_state:
    st.session_state.db = None
if 'memories' not in st.session_state:
    st.session_state.memories = []
if 'counter' not in st.session_state:
    st.session_state.counter = 0
if 'pictures' not in st.session_state:
    st.session_state.pictures = []
if 'init' not in st.session_state:
    st.session_state.init = False
if 'submitted' not in st.session_state:
    st.session_state.submitted = False
    
if 'camera' not in st.session_state:
    st.session_state.camera = False
if 'file_input' not in st.session_state:
    st.session_state.file_input = False


### -------------------- CAMERA ------------------------
# Toggle the state of 'camera'
def toggle_camera():
    st.session_state.camera = not st.session_state.camera

if st.session_state.read_image:
    image_button = st.empty()
    with image_button.container():
        # Create a button that toggles the 'camera' state
        col1.button('Toggle Camera', on_click=toggle_camera)

    # If 'camera' is True, show the camera input
    if st.session_state.camera:
        picture = st.camera_input("Take a picture")

        if picture:
            st.session_state.pictures.append(picture)
            st.session_state.init = True

    images_placeholder = st.empty()
    submit_button = st.empty()

    if st.session_state.init:
        images_placeholder.image(st.session_state.pictures, width=300)

        submitted =  submit_button.button(label="SUBMIT", on_click=toggle_camera)

        if submitted:
            st.session_state.submitted = True


    if st.session_state.submitted:
        image_button.empty()
        images_placeholder.empty()
        submit_button.empty()
        

        with st.spinner('Processing image...'):
            # Prepare the content list with the text instruction
            content_list = [
                {
                    "type": "text",
                    "text": """Extract the menu in this picture, just give me the menu without any introduction. Keep the restaurant information.
                            Make sure to get the categories, the name of the item, the item description, and the price."""
                }
            ]

            # Loop through the list of image paths and add each image to the content list
            for image_path in st.session_state.pictures:
                base64_picture = base64.b64encode(image_path.getvalue()).decode('utf-8')
                content_list.append(
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_picture}"
                        }
                    }
                )
            
            # Create the payload with the content list
            payload = {
                "model": "gpt-4o-mini",
                "messages": [
                    {
                        "role": "user",
                        "content": content_list
                    }
                ]
            }

            # Send the request to the OpenAI API
            response = requests.post(
                "https://api.openai.com/v1/chat/completions",
                
                headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}"
                },

                json=payload
            )

            raw_data = response.json()
            menu = raw_data['choices'][0]['message']['content']
            print(menu)

            st.session_state.menu = raw_data['choices'][0]['message']['content']
            st.session_state.ready = True

            st.session_state.read_image = False
            st.session_state.menu = menu

### -------------------- FILE UPLOAD ------------------------
def toggle_file_input():
    st.session_state.file_input = not st.session_state.file_input

if st.session_state.read_image:
    file_button = st.empty()
    with file_button.container():
        col2.button('Input Image', on_click=toggle_file_input)

    # If 'file_input' is True, show the file uploader
    if st.session_state.file_input:
        with st.form("my-form", clear_on_submit=True):
            uploaded_files = st.file_uploader("FILE UPLOADER", type=['png', 'jpeg', 'webp', 'jpg'], accept_multiple_files=True)
            submitted = st.form_submit_button("SUBMIT")

        
        if uploaded_files:
            file_button.empty()

            with st.spinner('Processing image...'):
                 # Prepare the content list with the text instruction
                content_list = [
                    {
                        "type": "text",
                        "text": """Extract the menu in this picture, just give me the menu without any introduction. Keep the restaurant information.
                                Make sure to get the categories, the name of the item, the item description, and the price."""
                    }
                ]
                # Loop through the list of image paths and add each image to the content list
                for file_path in uploaded_files:
                    base64_picture = base64.b64encode(file_path.getvalue()).decode('utf-8')
                    content_list.append(
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_picture}"
                            }
                        }
                    )

                # Create the payload with the content list
                payload = {
                    "model": "gpt-4o-mini",
                    "messages": [
                        {
                            "role": "user",
                            "content": content_list
                        }
                    ]
                }

                # Send the request to the OpenAI API
                response = requests.post(
                    "https://api.openai.com/v1/chat/completions",
                    
                    headers = {
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}"
                    },

                    json=payload
                )

                raw_data = response.json()
                menu = raw_data['choices'][0]['message']['content']
                print(menu)

                st.session_state.menu = raw_data['choices'][0]['message']['content']
                st.session_state.ready = True

                st.session_state.read_image = False
                st.session_state.menu = menu

### ---------------------- FUNCTIONS ----------------------------

# a function to check whether the question asked is on topic (relevant to the restaurant's menu)
def check_question(question):

    # system message - grader role
    system = """As a context-aware evaluator of restaurant customer interactions, your task is to:

        1. Assess if the input is appropriate for restaurant staff, considering the ongoing conversation.
        2. Determine if the input indicates the customer's intent to continue the conversation.
        3. Evaluate if the input is related to dining, menu items, restaurant services, or general polite conversation.

        Respond ONLY with 'True' if ANY of the following apply:
        - The input is a question about food, drinks, menu items, pricing, or restaurant services.
        - The input is a statement providing information relevant to their dining experience or order.
        - The input is a polite conversational remark typical in a restaurant setting.
        - The input seeks clarification about anything related to the restaurant or dining experience.
        - The input is a direct response to a previous question or statement from the waiter (e.g., 'Yes, please', 'No, thank you', 'That sounds good').
        - The input continues the flow of conversation, even if it's brief (e.g., 'Okay', 'Great', 'I see').

        Respond ONLY with 'False' if ALL of the following apply:
        - The input is completely unrelated to dining, the restaurant, or normal restaurant conversation.
        - The input is inappropriate, rude, or out of context for a restaurant setting.
        - The input clearly indicates the customer wants to end the conversation or leave, without any ambiguity.

        Your response must be ONLY 'True' or 'False' without any additional text or explanation.

        Remember: 
        1. Always consider the context of the ongoing conversation.
        2. Brief affirmative or negative responses should usually be considered 'True' as they often indicate engagement.
        3. When in doubt, lean towards 'True' to maintain the conversation flow."""

    # prompt template, format the system message and user question
    TEMPLATE = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", "User question: {question}"),
        ]
      )

    prompt = TEMPLATE.format(question=question)

    # call LLM model to generate the answer based on the given context and query
    model = ChatOpenAI(temperature=0)
    response_text = model.invoke(prompt)

    return response_text.content

# function in the case the question is not on topic
def off_topic_response(conversation):
  if conversation <= 1:
    answer = "\nI apologize, I can't answer that question. I can only respond to questions about the menu in this restaurant."
    st.session_state.memories.append({"role": "assistant", "content": answer})

    with st.chat_message("assistant"):
        st.write(answer)
  else:
    answer = "\nHappy to help!"
    st.session_state.memories.append({"role": "assistant", "content": answer})

    with st.chat_message("assistant"):
        st.write(answer)


# function to ask llm model question and generate answer
def generate(question, menu):
  model = ChatOpenAI()
  memory = st.session_state.memories

  # system message - waiter role
  system = """As a restaurant waiter, your role is to:
        1. Answer customer questions STRICTLY based on the provided menu.
        2. Never invent, assume, or add information not explicitly stated in the menu.
        3. If a question can't be answered using only menu information, politely inform the customer of this limitation.
        4. Always refer directly to menu content in your responses.
        5. Respond naturally, as if in conversation, without any introductions or self-references.
        6. Absolutely NO dialogue tags or speaker label like "Assistant:" or "Waiter:"
        7. Maintain a polite, professional tone throughout the interaction. Suitable to a high-end restaurant.
        8. If asked about recommendations or specialties, only suggest items that are clearly listed on the menu.
        9. For questions about allergies or dietary restrictions, only confirm information explicitly stated in the menu.
        10. If uncertain about any detail, err on the side of caution and inform the customer that it is not mentioned in the menu.
        11. Don't suggest something that you have no knowledge of.

        Remember: Your knowledge is limited to the menu. Stick to it rigorously. """
  # prompt template, format the system message and user question
  TEMPLATE = ChatPromptTemplate.from_messages([
    ("system", system),
    ("human", "Menu: {menu}"),
    ("human", "Conversation history: {memory}"),
    ("human", "Customer question: {question}")
    ])

  prompt = TEMPLATE.format(menu=menu, memory=memory, question=question)

  # call LLM model to generate the answer based on the given context and query
  model = ChatOpenAI(temperature=0)
  response_text = model.invoke(prompt)
  print(response_text.content)
  
  st.session_state.memories.append({"role": "assistant", "content": response_text.content})
  
  with st.chat_message("assistant"):
    st.write(response_text.content)

# function to repeatedly ask for the user's input
def further_question(state):
  prompt = st.chat_input("Say something")
  st.session_state.memories.append({"role": "user", "content": prompt})

  state['question'] = prompt
  state['conversation'] += 1

  return state


# -----------------
if st.session_state.ready:
    # Display chat memories from history on app rerun
    for memory in st.session_state.memories:
        with st.chat_message(memory["role"]):
            st.write(memory["content"])
    
    if st.session_state.counter == 0:
        with st.chat_message("assistant"):
            st.write('Hello! Welcome to our restaurant, I will be your waiter for today. How can I help you?\n')

    # Accept user input
    if user_input := st.chat_input("Say Something"):
        st.session_state.counter += 1
        # Add user message to chat history
        st.session_state.memories.append({"role": "user", "content": user_input})
        # Display user message in chat message container
        with st.chat_message("user"):
            st.write(user_input)
        
        check_bool = check_question(user_input)

        if check_bool == "False":
            off_topic_response(st.session_state.counter)
        
        else:
            answer = generate(user_input, st.session_state.menu)

    

       
    