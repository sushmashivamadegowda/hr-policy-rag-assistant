import streamlit as st
import os
import time
import importlib
import rag_engine as rag_engine_module
importlib.reload(rag_engine_module)
from rag_engine import HRPolicyRAG
from document_parsers import load_documents_from_directory

# Directory Path for Admin Uploaded Documents
DOCS_DIR = os.path.join(os.path.dirname(__file__), "data", "documents")

# Configure Streamlit Page
st.set_page_config(
    page_title="HR Policy Assistant",
    page_icon="✦",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# -----------------------------------------------------------------------------
# Curated Extraordinary Theme Palettes
# -----------------------------------------------------------------------------
THEMES = {
    "🌌 Obsidian Violet": {
        "name": "Obsidian Violet",
        "bg_app": "#090b10",
        "bg_gradient": "radial-gradient(at 50% 0%, rgba(99, 102, 241, 0.16) 0px, transparent 55%), radial-gradient(at 100% 100%, rgba(139, 92, 246, 0.06) 0px, transparent 50%)",
        "bg_sidebar": "#06070a",
        "text_main": "#f1f5f9",
        "text_muted": "#94a3b8",
        "accent": "#818cf8",
        "user_bubble": "linear-gradient(135deg, #4338ca 0%, #6366f1 100%)",
        "user_text": "#ffffff",
        "user_shadow": "0 4px 20px -2px rgba(99, 102, 241, 0.4)",
        "assistant_bubble": "rgba(18, 22, 34, 0.85)",
        "assistant_text": "#e2e8f0",
        "chip_bg": "rgba(255, 255, 255, 0.05)",
        "chip_hover": "rgba(99, 102, 241, 0.2)",
        "chip_border": "rgba(255, 255, 255, 0.08)",
        "chip_text": "#cbd5e1",
        "input_bg": "rgba(17, 21, 34, 0.95)",
        "input_border": "rgba(255, 255, 255, 0.12)",
        "input_focus": "#6366f1",
        "card_bg": "rgba(17, 21, 34, 0.8)",
        "card_border": "rgba(255, 255, 255, 0.08)",
        "status_color": "#10b981",
    },
    "🌊 Cyber Emerald": {
        "name": "Cyber Emerald",
        "bg_app": "#050d0e",
        "bg_gradient": "radial-gradient(at 50% 0%, rgba(16, 185, 129, 0.18) 0px, transparent 55%), radial-gradient(at 100% 100%, rgba(6, 182, 212, 0.08) 0px, transparent 50%)",
        "bg_sidebar": "#030809",
        "text_main": "#f0fdf4",
        "text_muted": "#86efac",
        "accent": "#34d399",
        "user_bubble": "linear-gradient(135deg, #059669 0%, #0d9488 100%)",
        "user_text": "#ffffff",
        "user_shadow": "0 4px 20px -2px rgba(16, 185, 129, 0.4)",
        "assistant_bubble": "rgba(10, 25, 26, 0.85)",
        "assistant_text": "#e6fffa",
        "chip_bg": "rgba(16, 185, 129, 0.07)",
        "chip_hover": "rgba(16, 185, 129, 0.22)",
        "chip_border": "rgba(16, 185, 129, 0.15)",
        "chip_text": "#a7f3d0",
        "input_bg": "rgba(9, 23, 24, 0.95)",
        "input_border": "rgba(16, 185, 129, 0.25)",
        "input_focus": "#10b981",
        "card_bg": "rgba(9, 23, 24, 0.8)",
        "card_border": "rgba(16, 185, 129, 0.15)",
        "status_color": "#34d399",
    },
    "🌅 Sunset Amber": {
        "name": "Sunset Amber",
        "bg_app": "#120e0c",
        "bg_gradient": "radial-gradient(at 50% 0%, rgba(245, 158, 11, 0.16) 0px, transparent 55%), radial-gradient(at 100% 100%, rgba(239, 68, 68, 0.08) 0px, transparent 50%)",
        "bg_sidebar": "#0b0907",
        "text_main": "#fffbeb",
        "text_muted": "#fbbf24",
        "accent": "#f59e0b",
        "user_bubble": "linear-gradient(135deg, #d97706 0%, #ea580c 100%)",
        "user_text": "#ffffff",
        "user_shadow": "0 4px 20px -2px rgba(217, 119, 6, 0.4)",
        "assistant_bubble": "rgba(28, 22, 18, 0.85)",
        "assistant_text": "#fef3c7",
        "chip_bg": "rgba(245, 158, 11, 0.08)",
        "chip_hover": "rgba(245, 158, 11, 0.22)",
        "chip_border": "rgba(245, 158, 11, 0.15)",
        "chip_text": "#fde68a",
        "input_bg": "rgba(25, 19, 16, 0.95)",
        "input_border": "rgba(245, 158, 11, 0.25)",
        "input_focus": "#f59e0b",
        "card_bg": "rgba(25, 19, 16, 0.8)",
        "card_border": "rgba(245, 158, 11, 0.15)",
        "status_color": "#f59e0b",
    },
    "⚡ Tokyo Neon": {
        "name": "Tokyo Neon",
        "bg_app": "#0a0a14",
        "bg_gradient": "radial-gradient(at 50% 0%, rgba(236, 72, 153, 0.18) 0px, transparent 55%), radial-gradient(at 100% 100%, rgba(139, 92, 246, 0.12) 0px, transparent 50%)",
        "bg_sidebar": "#06060c",
        "text_main": "#fdf4ff",
        "text_muted": "#f472b6",
        "accent": "#ec4899",
        "user_bubble": "linear-gradient(135deg, #db2777 0%, #9333ea 100%)",
        "user_text": "#ffffff",
        "user_shadow": "0 4px 20px -2px rgba(236, 72, 153, 0.4)",
        "assistant_bubble": "rgba(24, 18, 38, 0.85)",
        "assistant_text": "#f5d0fe",
        "chip_bg": "rgba(236, 72, 153, 0.08)",
        "chip_hover": "rgba(236, 72, 153, 0.22)",
        "chip_border": "rgba(236, 72, 153, 0.18)",
        "chip_text": "#f5d0fe",
        "input_bg": "rgba(20, 15, 32, 0.95)",
        "input_border": "rgba(236, 72, 153, 0.25)",
        "input_focus": "#ec4899",
        "card_bg": "rgba(20, 15, 32, 0.8)",
        "card_border": "rgba(236, 72, 153, 0.18)",
        "status_color": "#ec4899",
    },
    "❄️ Arctic Studio": {
        "name": "Arctic Studio",
        "bg_app": "#f8fafc",
        "bg_gradient": "radial-gradient(at 50% 0%, rgba(224, 231, 255, 0.5) 0px, transparent 60%), radial-gradient(at 100% 100%, rgba(241, 245, 249, 0.8) 0px, transparent 50%)",
        "bg_sidebar": "#f1f5f9",
        "text_main": "#0f172a",
        "text_muted": "#64748b",
        "accent": "#4f46e5",
        "user_bubble": "#0f172a",
        "user_text": "#ffffff",
        "user_shadow": "0 4px 14px rgba(15, 23, 42, 0.12)",
        "assistant_bubble": "#ffffff",
        "assistant_text": "#1e293b",
        "chip_bg": "#ffffff",
        "chip_hover": "#f1f5f9",
        "chip_border": "#e2e8f0",
        "chip_text": "#334155",
        "input_bg": "#ffffff",
        "input_border": "#cbd5e1",
        "input_focus": "#0f172a",
        "card_bg": "#ffffff",
        "card_border": "#e2e8f0",
        "status_color": "#10b981",
    }
}

# Initialize Theme in Session State
if "current_theme" not in st.session_state:
    st.session_state.current_theme = "🌌 Obsidian Violet"

theme = THEMES[st.session_state.current_theme]

# Dynamic CSS Injection using Curated Theme Variables
st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

    /* Global Typography */
    html, body, .stApp, p, h1, h2, h3, h4, input, textarea, label {{
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important;
        -webkit-font-smoothing: antialiased;
    }}

    /* Preserve Streamlit Material Icons font ligatures */
    span[data-testid="stIconMaterial"],
    [data-testid*="stIcon"],
    [class*="material-symbols"],
    [class*="material-icons"],
    button[data-testid="stSidebarCollapseButton"] span,
    [data-testid="stSidebarHeader"] span,
    header[data-testid="stHeader"] span {{
        font-family: 'Material Symbols Rounded', 'Material Symbols Outlined', 'Material Icons' !important;
        font-weight: normal !important;
        font-style: normal !important;
        letter-spacing: normal !important;
        text-transform: none !important;
        display: inline-block !important;
        white-space: nowrap !important;
        word-wrap: normal !important;
        direction: ltr !important;
    }}

    /* Prevent horizontal scrolling app-wide and in sidebar */
    html, body, .stApp, 
    section[data-testid="stSidebar"], 
    div[data-testid="stSidebarContent"], 
    div[data-testid="stSidebarUserContent"],
    div[data-testid="stSidebarHeader"],
    .block-container {{
        overflow-x: hidden !important;
        max-width: 100vw;
        box-sizing: border-box !important;
    }}

    /* Dynamic Canvas */
    .stApp {{
        background-color: {theme['bg_app']} !important;
        background-image: {theme['bg_gradient']} !important;
        color: {theme['text_main']} !important;
        transition: background-color 0.3s ease, color 0.3s ease;
    }}

    /* Transparent Streamlit Header */
    header[data-testid="stHeader"] {{
        background: transparent !important;
    }}

    /* Centered Content Container */
    .block-container {{
        max-width: 760px !important;
        padding-top: 3.8rem !important;
        padding-bottom: 6.5rem !important;
        margin: 0 auto !important;
    }}

    /* Header Section */
    .hero-container {{
        padding: 0.5rem 0 1.8rem 0;
        margin-bottom: 1.8rem;
        border-bottom: 1px solid rgba(255, 255, 255, 0.06);
    }}

    .hero-badge {{
        display: inline-flex;
        align-items: center;
        gap: 7px;
        background: {theme['chip_bg']};
        border: 1px solid {theme['chip_border']};
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.74rem;
        font-weight: 600;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        color: {theme['accent']};
        margin-bottom: 0.8rem;
    }}

    .status-dot {{
        width: 7px;
        height: 7px;
        background-color: {theme['status_color']};
        border-radius: 50%;
        display: inline-block;
        box-shadow: 0 0 10px {theme['status_color']};
        animation: pulseGlow 2s infinite ease-in-out;
    }}

    @keyframes pulseGlow {{
        0%, 100% {{ opacity: 1; transform: scale(1); }}
        50% {{ opacity: 0.5; transform: scale(0.85); }}
    }}

    .hero-heading {{
        margin: 0;
        font-size: 1.85rem;
        font-weight: 800;
        color: {theme['text_main']} !important;
        letter-spacing: -0.03em;
        line-height: 1.25;
    }}

    .hero-subtext {{
        margin-top: 0.4rem;
        margin-bottom: 0;
        font-size: 0.92rem;
        color: {theme['text_muted']};
        line-height: 1.55;
    }}

    /* Enterprise Metric Badges */
    .hero-metrics {{
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
        margin-top: 0.9rem;
    }}

    .metric-pill {{
        display: inline-flex;
        align-items: center;
        gap: 5px;
        font-size: 0.76rem;
        font-weight: 500;
        color: {theme['text_muted']};
        background: {theme['chip_bg']};
        padding: 3px 9px;
        border-radius: 8px;
    }}

    /* ========================================================
       CHAT LAYOUT: BORDERLESS, SHRINK-WRAPPED, CLEAN SEPARATION
       ======================================================== */
    [data-testid="stChatMessage"] {{
        margin-top: 1.5rem !important;
        margin-bottom: 1.5rem !important;
        padding: 0 !important;
        background: transparent !important;
        border: none !important;
        box-shadow: none !important;
        display: flex !important;
        align-items: flex-start !important;
    }}

    /* Hide avatars to keep clean conversational flow */
    [data-testid="stChatMessageAvatarUser"],
    [data-testid="stChatMessageAvatarAssistant"] {{
        display: none !important;
    }}

    /* USER MESSAGE: Shrink-wrapped to exact text length, right-aligned, NO BORDER */
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {{
        flex-direction: row-reverse !important;
        margin-left: auto !important;
        margin-right: 0 !important;
        width: fit-content !important;
        max-width: 75% !important;
        justify-content: flex-start !important;
    }}

    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) [data-testid="stChatMessageContent"] {{
        background: {theme['user_bubble']} !important;
        border: none !important;
        border-radius: 18px 18px 4px 18px !important;
        padding: 0.7rem 1.2rem !important;
        box-shadow: {theme['user_shadow']} !important;
        width: fit-content !important;
        max-width: 100% !important;
        flex: 0 1 auto !important;
        flex-grow: 0 !important;
        margin: 0 !important;
        display: inline-block !important;
    }}

    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) .stMarkdown,
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) [data-testid="stMarkdownContainer"] {{
        width: fit-content !important;
        display: inline-block !important;
    }}

    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) [data-testid="stChatMessageContent"] p,
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) [data-testid="stChatMessageContent"] span {{
        color: {theme['user_text']} !important;
        font-weight: 500 !important;
        font-size: 0.95rem !important;
        line-height: 1.5 !important;
        margin: 0 !important;
        padding: 0 !important;
        display: inline !important;
    }}

    /* ASSISTANT MESSAGE: Shrink-wrapped to exact text length, left-aligned, NO BORDER */
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) {{
        flex-direction: row !important;
        margin-right: auto !important;
        margin-left: 0 !important;
        width: fit-content !important;
        max-width: 85% !important;
        justify-content: flex-start !important;
    }}

    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) [data-testid="stChatMessageContent"] {{
        background: {theme['assistant_bubble']} !important;
        border: none !important;
        border-radius: 18px 18px 18px 4px !important;
        padding: 0.85rem 1.35rem !important;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.15) !important;
        width: fit-content !important;
        max-width: 100% !important;
        flex: 0 1 auto !important;
        flex-grow: 0 !important;
        margin: 0 !important;
        display: inline-block !important;
    }}

    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) .stMarkdown,
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) [data-testid="stMarkdownContainer"] {{
        width: fit-content !important;
        display: inline-block !important;
    }}

    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) [data-testid="stChatMessageContent"] p,
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) [data-testid="stChatMessageContent"] span {{
        color: {theme['assistant_text']} !important;
        font-weight: 400 !important;
        font-size: 0.95rem !important;
        line-height: 1.65 !important;
        margin: 0 !important;
        padding: 0 !important;
    }}

    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) [data-testid="stChatMessageContent"] strong {{
        color: {theme['text_main']} !important;
        font-weight: 600 !important;
    }}

    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) [data-testid="stChatMessageContent"] li {{
        color: {theme['assistant_text']} !important;
        font-weight: 400 !important;
        font-size: 0.95rem !important;
        line-height: 1.6 !important;
        margin-bottom: 0.35rem !important;
    }}

    /* QUICK SUGGESTION CHIPS */
    .stButton>button {{
        background: {theme['chip_bg']} !important;
        border: 1px solid {theme['chip_border']} !important;
        border-radius: 12px !important;
        color: {theme['chip_text']} !important;
        font-size: 0.86rem !important;
        font-weight: 500 !important;
        padding: 0.6rem 0.95rem !important;
        box-shadow: 0 2px 6px rgba(0, 0, 0, 0.15) !important;
        transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1) !important;
        text-align: left !important;
    }}

    .stButton>button:hover {{
        background: {theme['chip_hover']} !important;
        color: #ffffff !important;
        border-color: {theme['accent']} !important;
        transform: translateY(-2px) !important;
    }}

    /* MINIMALIST LOADER */
    .minimal-loader {{
        display: inline-flex;
        align-items: center;
        gap: 8px;
        padding: 7px 15px;
        background: {theme['card_bg']};
        border-radius: 14px;
        color: {theme['text_muted']};
        font-size: 0.84rem;
        font-weight: 500;
        margin: 6px 0;
    }}

    .minimal-dots span {{
        width: 5px;
        height: 5px;
        background-color: {theme['accent']};
        border-radius: 50%;
        display: inline-block;
        animation: subtlePulse 1.2s infinite ease-in-out both;
        box-shadow: 0 0 6px {theme['accent']};
    }}

    .minimal-dots span:nth-child(1) {{ animation-delay: -0.30s; }}
    .minimal-dots span:nth-child(2) {{ animation-delay: -0.15s; }}
    .minimal-dots span:nth-child(3) {{ animation-delay: 0s; }}

    @keyframes subtlePulse {{
        0%, 80%, 100% {{ opacity: 0.25; transform: scale(0.8); }}
        40% {{ opacity: 1; transform: scale(1.15); }}
    }}

    /* FLOATING CHAT INPUT */
    [data-testid="stChatInput"] {{
        border-radius: 20px !important;
        border: 1px solid {theme['input_border']} !important;
        background: {theme['input_bg']} !important;
        box-shadow: 0 8px 30px rgba(0, 0, 0, 0.4) !important;
        transition: border-color 0.2s, box-shadow 0.2s;
    }}

    [data-testid="stChatInput"]:focus-within {{
        border-color: {theme['input_focus']} !important;
        box-shadow: 0 8px 30px rgba(0, 0, 0, 0.5), 0 0 0 1px {theme['input_focus']} !important;
    }}

    [data-testid="stChatInput"] textarea {{
        color: {theme['text_main']} !important;
    }}

    [data-testid="stChatInput"] textarea::placeholder {{
        color: {theme['text_muted']} !important;
        opacity: 0.8 !important;
    }}

    /* Fixed bottom bar transparency */
    div[data-testid="stBottom"],
    div[data-testid="stBottom"] > div,
    .stChatFloatingInputContainer {{
        background-color: transparent !important;
        background: transparent !important;
    }}

    footer {{
        display: none !important;
        visibility: hidden !important;
    }}

    /* Selectbox styling in sidebar */
    [data-testid="stSidebar"] [data-baseweb="select"] > div {{
        background-color: {theme['card_bg']} !important;
        border-color: {theme['card_border']} !important;
        color: {theme['text_main']} !important;
        border-radius: 10px !important;
    }}

    [data-testid="stSidebar"] [data-baseweb="select"] * {{
        color: {theme['text_main']} !important;
    }}

    div[data-baseweb="popover"],
    ul[data-testid="stSelectboxVirtualDropdown"] {{
        background-color: {theme['bg_sidebar']} !important;
        border: 1px solid {theme['card_border']} !important;
        border-radius: 10px !important;
    }}

    li[role="option"] {{
        background-color: {theme['bg_sidebar']} !important;
        color: {theme['text_main']} !important;
    }}

    li[role="option"]:hover,
    li[aria-selected="true"] {{
        background-color: {theme['chip_hover']} !important;
        color: {theme['accent']} !important;
    }}

    /* SIDEBAR STYLING */
    section[data-testid="stSidebar"] {{
        background-color: {theme['bg_sidebar']} !important;
        border-right: 1px solid rgba(255, 255, 255, 0.06) !important;
        overflow-x: hidden !important;
    }}

    section[data-testid="stSidebar"] div[data-testid="stSidebarContent"],
    section[data-testid="stSidebar"] div[data-testid="stSidebarUserContent"] {{
        overflow-x: hidden !important;
        padding-left: 1rem !important;
        padding-right: 1rem !important;
    }}

    .sidebar-card {{
        background: {theme['card_bg']} !important;
        border: 1px solid {theme['card_border']} !important;
        border-radius: 10px !important;
        padding: 12px 14px !important;
        margin-bottom: 12px !important;
        max-width: 100% !important;
        box-sizing: border-box !important;
        overflow-x: hidden !important;
        word-wrap: break-word !important;
    }}

    .sidebar-card-title {{
        font-size: 0.72rem !important;
        font-weight: 600 !important;
        letter-spacing: 0.05em !important;
        text-transform: uppercase !important;
        color: {theme['accent']} !important;
        margin-bottom: 4px !important;
    }}

    .sidebar-card-value {{
        font-size: 0.92rem !important;
        font-weight: 600 !important;
        color: {theme['text_main']} !important;
        word-break: break-word !important;
    }}

    .sidebar-doc-badge {{
        font-size: 0.78rem !important;
        color: {theme['chip_text']} !important;
        background: {theme['chip_bg']} !important;
        padding: 6px 10px !important;
        border-radius: 6px !important;
        margin-top: 6px !important;
        font-family: monospace !important;
        word-break: break-all !important;
        white-space: normal !important;
        max-width: 100% !important;
        box-sizing: border-box !important;
        display: block !important;
    }}
</style>
""", unsafe_allow_html=True)

# Function to refresh vector index from documents folder
def reload_knowledge_base():
    docs = load_documents_from_directory(DOCS_DIR)
    st.session_state.raw_docs = docs
    st.session_state.rag_engine.build_index(
        docs, 
        chunk_size=st.session_state.get("chunk_size", 400), 
        chunk_overlap=st.session_state.get("chunk_overlap", 50)
    )

# Initialize Session State
if "rag_engine" not in st.session_state or "RESPONSE_TIME" not in getattr(st.session_state.rag_engine, "intent_exemplars", {}):
    st.session_state.rag_engine = HRPolicyRAG(model_name="all-MiniLM-L6-v2")
    st.session_state.raw_docs = []
    reload_knowledge_base()

org_title = getattr(st.session_state.rag_engine, "organization_name", "Company")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        {
            "role": "assistant",
            "content": f"Hello! 👋 I am your **{org_title} HR Assistant**. Feel free to ask me anything about leave rules, office hours, attendance, notice periods, or workplace guidelines."
        }
    ]

# Minimalist Top Hero Header
st.markdown(f"""
<div class="hero-container">
    <div class="hero-badge">
        <span class="status-dot"></span>
        <span>{org_title} • HR Intelligence</span>
    </div>
    <h1 class="hero-heading">HR Policy Assistant</h1>
    <p class="hero-subtext">Instant, verified answers based on official company HR policy documentation.</p>
    <div class="hero-metrics">
        <span class="metric-pill">⚡ Sub-Second Retrieval</span>
        <span class="metric-pill">📄 100% Grounded</span>
        <span class="metric-pill">🔒 Private & Auditable</span>
    </div>
</div>
""", unsafe_allow_html=True)

# Sidebar Navigation, Theme Customizer & Settings
with st.sidebar:
    st.markdown("### ⚙️ System & Controls")
    
    # Visual Theme Customizer
    st.markdown("**🎨 Visual Theme Preference**")
    selected_theme = st.selectbox(
        "Select Theme",
        options=list(THEMES.keys()),
        index=list(THEMES.keys()).index(st.session_state.current_theme),
        label_visibility="collapsed",
        key="theme_picker"
    )
    if selected_theme != st.session_state.current_theme:
        st.session_state.current_theme = selected_theme
        st.rerun()

    st.markdown("---")

    # Active Organization Card
    st.markdown(f"""
    <div class="sidebar-card">
        <div class="sidebar-card-title">Active Organization</div>
        <div class="sidebar-card-value">{org_title}</div>
    </div>
    """, unsafe_allow_html=True)

    # Active Documents Status
    active_docs = [f for f in os.listdir(DOCS_DIR) if not f.startswith(".")] if os.path.exists(DOCS_DIR) else []
    if active_docs:
        docs_html = "".join([f'<div class="sidebar-doc-badge">📄 {d}</div>' for d in active_docs])
        st.markdown(f"""
        <div class="sidebar-card">
            <div class="sidebar-card-title">Active Knowledge Base</div>
            {docs_html}
            <div style="font-size: 0.75rem; color: {theme['text_muted']}; margin-top: 6px;">{len(st.session_state.rag_engine.chunks)} indexed semantic chunks</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.warning("⚠️ No documents indexed. Upload a document below.")

    st.markdown("---")
    
    # Response Engine
    synthesis_mode = st.radio(
        "Response Engine",
        options=["⚡ Semantic Precision (Fast & Factual)", "🤖 LLM Generative"],
        index=0,
        help="Semantic Precision extracts exact factual sentences without hallucination. LLM Generative synthesizes phrasing."
    )
    use_llm_mode = (synthesis_mode == "🤖 LLM Generative")
    
    api_key_input = None
    if use_llm_mode:
        api_key_input = st.text_input(
            "Groq / OpenAI API Key (Optional)",
            type="password",
            placeholder="gsk_... or sk-...",
            help="Free Groq API key delivers ultra-fast generation. Leave blank to use local generator."
        )

    st.markdown("---")

    if st.button("🗑️ Clear Conversation", use_container_width=True):
        st.session_state.chat_history = [
            {
                "role": "assistant",
                "content": f"Hello! 👋 I am your **{org_title} HR Assistant**. Feel free to ask me anything about leave rules, office hours, attendance, notice periods, or workplace guidelines."
            }
        ]
        st.rerun()

    st.markdown("---")
    
    # Document Ingestion (Admin)
    with st.expander("📁 Document Management", expanded=False):
        admin_file = st.file_uploader(
            "Upload Policy Document", 
            type=["pdf", "docx", "doc", "html", "htm", "txt", "md"],
            key="admin_uploader"
        )
        if admin_file:
            if st.button("💾 Save & Reindex", type="primary", use_container_width=True):
                file_path = os.path.join(DOCS_DIR, admin_file.name)
                with open(file_path, "wb") as f:
                    f.write(admin_file.read())
                reload_knowledge_base()
                st.success(f"'{admin_file.name}' indexed successfully!")
                st.rerun()

        if st.button("🔄 Reload Documents", use_container_width=True):
            reload_knowledge_base()
            st.success("Knowledge base reloaded!")
            st.rerun()

# Render Chat History with Distinct Spacing
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Quick Suggested Question Chips (Displayed when conversation just started)
chip_selected = None
if len(st.session_state.chat_history) <= 1:
    st.markdown(f"<div style='font-size: 0.76rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; color: {theme['text_muted']}; margin-top: 1.2rem; margin-bottom: 0.6rem;'>Suggested Inquiries</div>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        if st.button("⏰ What are the working hours?", use_container_width=True):
            chip_selected = "What are the working hours and office timings?"
        if st.button("💼 What is the notice period policy?", use_container_width=True):
            chip_selected = "What is the policy for notice period?"
    with c2:
        if st.button("📅 How many casual leaves are allowed?", use_container_width=True):
            chip_selected = "How many casual leaves are allowed?"
        if st.button("⚡ How fast do you answer?", use_container_width=True):
            chip_selected = "how many min will u take to answer a specific question"

# Input Box
chat_input_val = st.chat_input("Ask a question about HR policies...")
user_query = chip_selected if chip_selected else chat_input_val

if user_query:
    # Append & display user message
    st.session_state.chat_history.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    # Assistant Response
    with st.chat_message("assistant"):
        loader_placeholder = st.empty()
        loader_placeholder.markdown("""
        <div class="minimal-loader">
            <div class="minimal-dots">
                <span></span><span></span><span></span>
            </div>
            <span>Searching policy...</span>
        </div>
        """, unsafe_allow_html=True)
        
        try:
            response = st.session_state.rag_engine.generate_answer(
                query=user_query,
                chat_history=st.session_state.chat_history,
                top_k=4,
                score_threshold=0.38,
                use_llm=use_llm_mode,
                api_key=api_key_input
            )
            answer_text = response.get("answer", "")
        except Exception as e:
            print(f"Error handling query '{user_query}': {e}")
            answer_text = "I apologize, but I had a momentary issue retrieving that policy detail. Please try asking again."
            
        loader_placeholder.empty()
        st.markdown(answer_text)

    # Store conversation history
    st.session_state.chat_history.append({
        "role": "assistant",
        "content": answer_text
    })
    
    # Rerun to cleanly update the view if a chip was clicked
    if chip_selected:
        st.rerun()
