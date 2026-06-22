import os
import streamlit as st
from groq import Groq
from dotenv import load_dotenv

# 1. Load the Groq API Key from your .env file
load_dotenv()

# 2. Configure the web page layout
st.set_page_config(page_title="FelineFriend AI", page_icon="🐈", layout="wide")

# 3. Sidebar: Gather Information about the Cat
with st.sidebar:
    st.header("🐈 Cat Profile")
    st.write("Help the AI give more relevant answers by detailing your cat's profile:")
    cat_name = st.text_input("Cat's Name", value="Luna")
    cat_age = st.text_input("Age (e.g., 3 years, 6 months)", value="2 years")
    is_spayed = st.selectbox("Spayed/Neutered?", ["Yes", "No", "Unknown"])
    lifestyle = st.selectbox("Lifestyle", ["Strictly Indoor", "Outdoor", "Indoor/Outdoor mix"])
    diet = st.text_input("Dietary Type (e.g., Dry kibble, Wet food)", value="Dry Kibble")
    health_issues = st.text_area("Any known medical conditions?", placeholder="None")

# 4. Initialize the Groq Client
api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    st.warning("⚠️ Groq API Key not detected. Please verify your .env file contains GROQ_API_KEY=gsk_...")
    st.stop()

client = Groq(api_key=api_key)

# 5. Define the "System Prompt" (The Instructions/Rules for the AI)
SYSTEM_PROMPT = f"""
You are "Kitty ," an expert AI virtual assistant specializing in cat care, behavior, and feline nutrition. 
Your goal is to help cat owners understand their pets better and provide practical, compassionate advice.

You are currently consulting about a specific cat with the following profile:
- Name: {cat_name}
- Age: {cat_age}
- Spayed/Neutered: {is_spayed}
- Lifestyle: {lifestyle}
- Diet: {diet}
- Existing Health Issues: {health_issues if health_issues else 'None reported'}

Keep this profile in mind for all your answers to make your advice customized to {cat_name}.

CRITICAL SAFETY & MEDICAL RULES:
1. You are an AI, NOT a veterinarian. Always include a brief, gentle disclaimer when answering medical or health-related queries, reminding the user that you cannot diagnose illnesses.
2. If the user mentions any of the following emergency "RED FLAG" symptoms, immediately stop giving DIY home remedies and strongly urge them to seek immediate emergency veterinary care:
   - Straining to urinate, crying in the litterbox, or inability to urinate (especially critical for male cats).
   - Difficulty breathing or open-mouth breathing/panting.
   - Extreme lethargy, unresponsiveness, or sudden inability to walk.
   - Refusing food or water for more than 24 hours.
   - Ingestion of toxic items (like lilies, onions, garlic, chocolate, human medication).
   - Uncontrolled bleeding or severe physical trauma.

TONE:
- Warm, empathetic, and encouraging.
- Break down complex feline behavior or medical jargon into clear, digestible explanations.
- Use the cat's name ({cat_name}) where natural to make the interaction personal.
"""

# 6. Main Page Layout
st.title("Kitty: Your Cat Care AI Assistant")
st.write(f"Ask any questions about **{cat_name}**'s behavior, health, nutrition, or general wellness.")

# 7. Initialize Chat Memory (Session State)
# Streamlit apps rerun from top to bottom on every click. 
# "Session State" acts as a persistent memory so the chat history doesn't disappear.
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "system", "content": SYSTEM_PROMPT}
    ]

# 8. Display Existing Chat Messages
for message in st.session_state.messages:
    if message["role"] != "system":
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

# 9. Handle New User Inputs
if user_query := st.chat_input(f"Ask about {cat_name}..."):
    
    # Show user message in the browser UI
    with st.chat_message("user"):
        st.markdown(user_query)
    
    # Save user message to memory
    st.session_state.messages.append({"role": "user", "content": user_query})
    
    # Query Groq and display the streamed response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        try:
            # We use the open-source Llama-3.3-70B model hosted on Groq
            stream = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": m["role"], "content": m["content"]}
                    for m in st.session_state.messages
                ],
                stream=True,
            )
            
            # Stream the text onto the screen word-by-word
            for chunk in stream:
                if chunk.choices[0].delta.content:
                    full_response += chunk.choices[0].delta.content
                    message_placeholder.markdown(full_response + "▌")
            
            # Finalize response formatting
            message_placeholder.markdown(full_response)
            
        except Exception as e:
            st.error(f"Error communicating with Groq: {e}")
            full_response = "I encountered an issue processing that query. Please check your setup and try again."
            message_placeholder.markdown(full_response)
            
    # Save the AI's response to memory
    st.session_state.messages.append({"role": "assistant", "content": full_response})
