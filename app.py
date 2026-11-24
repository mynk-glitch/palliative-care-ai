import os
import time
import re
import streamlit as st
from ollama import Client
from matcher import PalliativeCaseMatcher
import requests


os.environ["NO_PROXY"] = "localhost,127.0.0.1"
local_client = Client(host='http://127.0.0.1:11434')

st.set_page_config(
    page_title="Care & Study AI Hub",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="collapsed"
)

@st.cache_resource
def load_matcher():
    return PalliativeCaseMatcher()

matcher = load_matcher()


if "active_tab" not in st.session_state:
    st.session_state.active_tab = "illness"
if "illness_history" not in st.session_state:
    st.session_state.illness_history = []
if "study_notes" not in st.session_state:
    st.session_state.study_notes = []
if "selected_term" not in st.session_state:
    st.session_state.selected_term = ""
if "selected_symptom" not in st.session_state:
    st.session_state.selected_symptom = ""




def ask_local_ai(system_prompt: str, user_text: str) -> str:
    """Uses Groq API when deployed online, or local Ollama when running on laptop."""
    # 1. Cloud Mode (for your teacher's live Streamlit link)
    groq_key = st.secrets.get("GROQ_API_KEY", None) if hasattr(st, "secrets") else None
    if groq_key:
        try:
            res = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={"Authorization": f"Bearer {groq_key}", "Content-Type": "application/json"},
                json={
                    "model": "openai/gpt-oss-20b",
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_text}
                    ]
                },
                timeout=30
            )
            res.raise_for_status()
            return res.json()["choices"][0]["message"]["content"]
        except Exception:
            pass

    # 2. Local Mode (Ollama on your laptop)
    for model_name in ["llama3.2", "qwen2.5-coder:7b"]:
        try:
            res = local_client.chat(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_text}
                ]
            )
            return res["message"]["content"]
        except Exception:
            continue
    raise RuntimeError("Could not connect to Cloud AI or local Ollama.")


def detect_perspective(text: str, manual_choice: str) -> str:
    if manual_choice == "For Myself":
        return "SELF"
    if manual_choice == "For Someone Else":
        return "OTHER"
    
    words = re.findall(r"\b\w+\b", text.lower())
    other_words = {"he", "she", "him", "her", "his", "dad", "mom", "father", "mother", "grandpa", "grandma", "brother", "sister", "friend", "patient", "they", "them"}
    self_words = {"i", "me", "my", "myself", "im", "ive", "am", "feeling"}
    
    other_count = sum(1 for w in words if w in other_words)
    self_count = sum(1 for w in words if w in self_words)
    
    # If they say "my dad" or "my mom", 'my' + 'dad' appears, so weight other_words higher
    if other_count > 0:
        return "OTHER"
    if self_count > 0:
        return "SELF"
    return "SELF"


is_illness = st.session_state.active_tab == "illness"

theme_css = f"""
<style>
@keyframes fadeSlideUp {{
    0% {{ opacity: 0; transform: translateY(18px); }}
    100% {{ opacity: 1; transform: translateY(0); }}
}}

@keyframes gentlePulse {{
    0%, 100% {{ transform: scale(1); opacity: 0.85; }}
    50% {{ transform: scale(1.06); opacity: 1; }}
}}

@keyframes breatheCircle {{
    0%, 100% {{ transform: scale(0.75); background: rgba(129, 199, 132, 0.25); box-shadow: 0 0 15px rgba(129, 199, 132, 0.2); }}
    50% {{ transform: scale(1.15); background: rgba(77, 182, 172, 0.45); box-shadow: 0 0 30px rgba(77, 182, 172, 0.5); }}
}}

@keyframes shimmerBorder {{
    0% {{ background-position: 0% 50%; }}
    50% {{ background-position: 100% 50%; }}
    100% {{ background-position: 0% 50%; }}
}}

/* Dynamic Screen Background */
.stApp {{
    background: {"radial-gradient(circle at top left, #132a26 0%, #0f1c24 50%, #0b1319 100%)" if is_illness else "radial-gradient(circle at top right, #261e14 0%, #191724 55%, #111019 100%)"};
    transition: background 0.6s ease-in-out;
}}

/* Animated Container Wrapper */
.animated-screen {{
    animation: fadeSlideUp 0.55s cubic-bezier(0.16, 1, 0.3, 1) forwards;
}}

/* Custom Hero Cards */
.chill-hero {{
    background: linear-gradient(135deg, rgba(77, 182, 172, 0.16) 0%, rgba(129, 199, 132, 0.12) 100%);
    border: 1px solid rgba(129, 199, 132, 0.3);
    border-radius: 20px;
    padding: 26px 30px;
    margin-bottom: 22px;
    box-shadow: 0 12px 32px rgba(0, 0, 0, 0.22);
}}

.study-hero {{
    background: linear-gradient(135deg, rgba(245, 176, 65, 0.16) 0%, rgba(155, 89, 182, 0.14) 100%);
    border: 1px solid rgba(245, 176, 65, 0.35);
    border-radius: 20px;
    padding: 26px 30px;
    margin-bottom: 22px;
    box-shadow: 0 12px 32px rgba(0, 0, 0, 0.25);
}}

/* Output Cards */
.care-card {{
    background: rgba(255, 255, 255, 0.04);
    border-left: 5px solid #4db6ac;
    border-radius: 16px;
    padding: 24px;
    margin-top: 16px;
    animation: fadeSlideUp 0.45s ease-out;
    box-shadow: 0 8px 24px rgba(0,0,0,0.2);
}}

.study-card {{
    background: rgba(255, 248, 231, 0.05);
    border-left: 5px solid #f5b041;
    border-radius: 16px;
    padding: 24px;
    margin-top: 16px;
    animation: fadeSlideUp 0.45s ease-out;
    box-shadow: 0 8px 24px rgba(0,0,0,0.25);
}}

.side-panel-box {{
    background: rgba(255, 255, 255, 0.035);
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 18px;
    padding: 20px;
    margin-bottom: 16px;
}}

.breathing-orb {{
    width: 90px;
    height: 90px;
    border-radius: 50%;
    margin: 18px auto;
    animation: breatheCircle 8s infinite ease-in-out;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 13px;
    color: #e0f2f1;
    font-weight: 600;
}}

/* Smooth Inputs & Buttons */
.stTextArea textarea, .stTextInput input {{
    border-radius: 14px !important;
    background: rgba(255, 255, 255, 0.05) !important;
    border: 1px solid {"rgba(77, 182, 172, 0.4)" if is_illness else "rgba(245, 176, 65, 0.4)"} !important;
    color: #f5f5f5 !important;
    padding: 14px !important;
}}

.stButton button {{
    border-radius: 12px !important;
    font-weight: 600 !important;
    transition: all 0.25s ease !important;
}}
.stButton button:hover {{
    transform: translateY(-2px);
    box-shadow: 0 6px 16px rgba(0,0,0,0.3);
}}
</style>
"""
st.markdown(theme_css, unsafe_allow_html=True)


head_left, head_right = st.columns([1.3, 1], vertical_alignment="center")

with head_left:
    if is_illness:
        st.markdown("## 🌿 Serenity Health & Symptom Space")
    else:
        st.markdown("## 📚 Medical Lexicon & Pharmacology Desk")

with head_right:
    nav_col1, nav_col2 = st.columns(2)
    with nav_col1:
        if st.button(
            "🌿 Illness & Comfort Care",
            type="primary" if is_illness else "secondary",
            use_container_width=True
        ):
            st.session_state.active_tab = "illness"
            st.rerun()
    with nav_col2:
        if st.button(
            "📚 Medical Dictionary",
            type="primary" if not is_illness else "secondary",
            use_container_width=True
        ):
            st.session_state.active_tab = "dictionary"
            st.rerun()

st.markdown("<hr style='margin: 8px 0 24px 0; opacity: 0.15;'>", unsafe_allow_html=True)


if is_illness:
    st.markdown('<div class="animated-screen">', unsafe_allow_html=True)
    
    main_col, side_col = st.columns([1.85, 1], gap="large")
    
    with main_col:
        st.markdown("""
        <div class="chill-hero">
            <h3 style="margin:0; color:#b2dfdb;">✨ Take a deep breath. You are in a calm, supportive space.</h3>
            <p style="margin:8px 0 0 0; opacity:0.88; font-size:15px;">
                Share how you are feeling—or what someone close to you is experiencing—in plain, everyday words. 
                You will receive gentle, clear, step-by-step relief guidance and medication insights.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Perspective selector + Quick Symptom Chips
        p_col1, p_col2 = st.columns([1, 1.6], vertical_alignment="center")
        with p_col1:
            who_mode = st.radio(
                "Who is this for?",
                ["Auto-Detect", "For Myself", "For Someone Else"],
                horizontal=True,
                help="Helps the assistant speak directly to you or guide you in helping another person."
            )
        
        with p_col2:
            st.caption("Quick-fill common situations:")
            chip1, chip2, chip3 = st.columns(3)
            if chip1.button("😮‍💨 Short of Breath", use_container_width=True):
                st.session_state.selected_symptom = "I am feeling short of breath and tightness in my chest."
            if chip2.button("🛌 Severe Pain / Ache", use_container_width=True):
                st.session_state.selected_symptom = "I have constant body pain and trouble resting comfortably."
            if chip3.button("💧 Dry Mouth / Nausea", use_container_width=True):
                st.session_state.selected_symptom = "My mouth feels extremely dry and I feel nauseous when trying to eat."

        symptom_input = st.text_area(
            "How are you feeling today?",
            value=st.session_state.selected_symptom,
            placeholder="Example: 'I have had a dull ache and nausea since yesterday...' OR 'My father is having noisy breathing...'",
            height=130
        )

        if st.button("🌱 Get Gentle Relief & Care Guide", type="primary", use_container_width=True):
            if not symptom_input.strip():
                st.warning("Please share a few words about the symptoms first.")
            else:
                perspective = detect_perspective(symptom_input, who_mode)
                perspective_label = "Addressing You Directly" if perspective == "SELF" else "Helping Someone Else"
                
                with st.status(f"🌿 Preparing your personalized comfort guide ({perspective_label})...", expanded=True) as status:
                    st.write("🔍 Reviewing clinical relief protocols...")
                    results = matcher.find_matching_cases(symptom_input, top_k=1)
                    
                    has_match = results and results[0]["similarity_score"] > 50
                    if has_match:
                        match = results[0]
                        case = match["case_details"]
                        st.write(f"✨ Found matching relief guidance for **{case['condition']}**")
                        interventions = "\n".join([f"- {i}" for i in case["recommended_interventions"]])
                        warnings = "\n".join([f"- {w}" for w in case["contraindications_warnings"]])
                        context_block = f"Condition: {case['condition']}\nHelpful Actions:\n{interventions}\nThings to Avoid:\n{warnings}"
                        citation_text = f"Verified Clinical Reference: {case['case_id']} ({match['similarity_score']}% Match) — {case['evidence_reference']}"
                    else:
                        st.write("💊 Preparing general symptom & medication overview...")
                        context_block = "Use safe, standard palliative and supportive symptom-relief principles."
                        citation_text = "General Supportive Medical Guidance (Consult a physician for prescriptions)"

                    st.write("🌸 Formatting calm, easy-to-read steps...")
                    
                    if perspective == "SELF":
                        tone_instruction = (
                            "The user is the PATIENT asking for THEMSELVES. Speak directly to them using 'you' and 'your'. "
                            "NEVER refer to them as a caregiver or talk about 'the patient' in the third person. "
                            "Keep the tone warm, grounded, calm, and reassuring—like a supportive doctor who genuinely cares."
                        )
                    else:
                        tone_instruction = (
                            "The user is asking about SOMEONE ELSE (a family member or friend). "
                            "Address the user clearly and practically without being overly dramatic or robotic."
                        )

                    illness_prompt = f"""
                    {tone_instruction}
                    
                    Use the medical context below to provide a clear, soothing, well-structured guide:
                    {context_block}
                    
                    Structure your response with these exact headings:
                    ### 🌿 Gentle Steps You Can Take Right Now
                    (Clear, calming, practical actions for immediate comfort)
                    
                    ### 💊 Common Medications & Why They Help
                    (Explain what medicines doctors commonly use for this and *why* they work in simple terms)
                    
                    ### 🛡️ Things to Avoid & When to Call a Doctor
                    (Simple safety precautions without causing panic)
                    """

                    try:
                        ai_reply = ask_local_ai(illness_prompt, symptom_input)
                        status.update(label="✨ Your Comfort & Relief Guide is Ready", state="complete", expanded=False)
                        
                        st.session_state.illness_history.insert(0, {
                            "query": symptom_input,
                            "perspective": perspective_label,
                            "reply": ai_reply,
                            "citation": citation_text
                        })
                    except Exception as e:
                        status.update(label="Connection Issue", state="error", expanded=False)
                        st.error(f"Please ensure Ollama is running in the background. Details: {e}")

        # Display latest and past care cards
        for idx, item in enumerate(st.session_state.illness_history):
            st.markdown(f"""
            <div class="care-card">
                <div style="font-size:12px; color:#80cbc4; text-transform:uppercase; letter-spacing:1px; margin-bottom:6px;">
                    🌿 {item['perspective']} • Query: "{item['query'][:60]}..."
                </div>
            </div>
            """, unsafe_allow_html=True)
            st.markdown(item["reply"])
            st.caption(f"📋 {item['citation']}")
            st.download_button(
                label="📥 Save This Comfort Guide (.md)",
                data=item["reply"] + f"\n\n---\n{item['citation']}",
                file_name=f"Comfort_Guide_{idx+1}.md",
                mime="text/markdown",
                key=f"dl_care_{idx}"
            )
            st.divider()

    with side_col:
        st.markdown("""
        <div class="side-panel-box" style="text-align:center;">
            <h4 style="margin:0; color:#a7ffeb;">🌬️ Calm Breathing Pacer</h4>
            <p style="font-size:13px; opacity:0.8; margin-top:6px;">
                Follow the glowing circle below to slow your heart rate and ease physical tension.
            </p>
            <div class="breathing-orb">Inhale / Exhale</div>
            <p style="font-size:12px; opacity:0.7; margin:0;">
                Breathe in gently as it expands... breathe out slowly as it softens.
            </p>
        </div>
        
        <div class="side-panel-box">
            <h4 style="margin:0 0 10px 0; color:#b2dfdb;">🍵 Quick Comfort Checklist</h4>
            <ul style="font-size:13.5px; line-height:1.7; opacity:0.88; padding-left:18px; margin:0;">
                <li><b>Posture:</b> Prop pillows slightly under your head and knees to ease chest and back strain.</li>
                <li><b>Airflow:</b> A gentle cool breeze from a fan or window often eases breathlessness fast.</li>
                <li><b>Hydration:</b> Take tiny sips of room-temperature water or ice chips rather## than gulping.</li>
                <li><b>Lighting:</b> Dim harsh overhead lights to help your nervous system settle.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)


else:
    st.markdown('<div class="animated-screen">', unsafe_allow_html=True)
    
    study_main, study_side = st.columns([1.85, 1], gap="large")
    
    with study_main:
        st.markdown("""
        <div class="study-hero">
            <h3 style="margin:0; color:#f8c471;">📖 Interactive Medical Lexicon & Study Desk</h3>
            <p style="margin:8px 0 0 0; opacity:0.9; font-size:15px;">
                Explore medical terminology, drug classes, and how medicines work inside the human body. 
                Designed like a personal study companion—clear, insightful, and free of confusing jargon.
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.caption("🔥 Popular Study Topics (Click to auto-fill):")
        t_col1, t_col2, t_col3, t_col4 = st.columns(4)
        if t_col1.button("💊 Opioids & Pain", use_container_width=True):
            st.session_state.selected_term = "What are Opioids and how do they work in the body?"
        if t_col2.button("🫁 Dyspnea", use_container_width=True):
            st.session_state.selected_term = "What does the medical term Dyspnea mean?"
        if t_col3.button("🧪 Antiemetics", use_container_width=True):
            st.session_state.selected_term = "What are Antiemetic medicines and why are they prescribed?"
        if t_col4.button("🩺 Edema", use_container_width=True):
            st.session_state.selected_term = "What is Edema and what causes it?"

        term_input = st.text_input(
            "Search any medicine, medical term, or biological concept:",
            value=st.session_state.selected_term,
            placeholder="e.g., 'What is Morphine?', 'How do beta blockers work?', 'Meaning of Prognosis'..."
        )

        if st.button("🔎 Study & Explain This Topic", type="primary", use_container_width=True):
            if not term_input.strip():
                st.warning("Type a medical term or medicine name above to start studying.")
            else:
                # Notice: NO database search, NO clinical report generation! Pure study mode.
                with st.spinner("📖 Opening medical encyclopedia & pharmacology notes..."):
                    study_prompt = """
                    You are an engaging, brilliant medical professor and pharmacology study guide.
                    The user is studying or looking up a medical term, medicine, or health concept.
                    Do NOT act like a hospice caregiver and do NOT refuse to explain regulated medicines like opioids or morphine—provide objective, educational science.
                    
                    Format your explanation as an engaging Study Flashcard with these exact sections:
                    ### 💡 Plain-English Definition
                    (A crystal-clear, 2-sentence summary anyone can understand)
                    
                    ### ⚙️ How It Works in the Body (Mechanism)
                    (Explain the biology or pharmacology step-by-step in an interesting way)
                    
                    ### 🏥 Real-World Medical Use
                    (When and why doctors use this term or prescribe this medication)
                    
                    ### 🧠 Quick Memory Hook / Key Takeaway
                    (One memorable bullet point or analogy so the reader never forgets it)
                    """
                    try:
                        study_reply = ask_local_ai(study_prompt, term_input)
                        st.session_state.study_notes.insert(0, {
                            "term": term_input,
                            "explanation": study_reply
                        })
                    except Exception as e:
                        st.error(f"Please ensure Ollama is running in the background. Details: {e}")

        
        for item in st.session_state.study_notes:
            st.markdown(f"""
            <div class="study-card">
                <div style="font-size:12px; color:#f5b041; text-transform:uppercase; letter-spacing:1px; margin-bottom:4px;">
                    📌 STUDY FLASHCARD • {item['term']}
                </div>
            </div>
            """, unsafe_allow_html=True)
            st.markdown(item["explanation"])
            st.divider()

    with study_side:
        st.markdown("""
        <div class="side-panel-box">
            <h4 style="margin:0 0 8px 0; color:#f8c471;">🗂️ Your Session Study Notebook</h4>
            <p style="font-size:13px; opacity:0.8; margin-bottom:12px;">
                Every term or medication you explore is automatically bookmarked here for quick revision:
            </p>
        """, unsafe_allow_html=True)
        
        if not st.session_state.study_notes:
            st.caption("✨ No terms looked up yet. Search a term on the left to build your study deck!")
        else:
            for idx, note in enumerate(st.session_state.study_notes):
                st.markdown(f"**{idx+1}.** `{note['term']}`")
            
            if st.button("🗑️ Clear Study Notes", use_container_width=True):
                st.session_state.study_notes = []
                st.rerun()

        st.markdown("""
        </div>
        <div class="side-panel-box">
            <h4 style="margin:0 0 8px 0; color:#d7bde2;">🧬 Quick Pharmacology Roots</h4>
            <p style="font-size:13px; opacity:0.85; line-height:1.6; margin:0;">
                • <b>-pril / -sartan:</b> Blood pressure & heart support<br>
                • <b>-prazole:</b> Stomach acid & ulcer protection<br>
                • <b>-setron:</b> Anti-nausea (Antiemetics)<br>
                • <b>-statin:</b> Cholesterol regulation<br>
                • <b>Anti-pyretic:</b> Fever reducer<br>
                • <b>Analgesic:</b> Pain reliever
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)
