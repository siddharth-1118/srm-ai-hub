import os

_LOCAL_PIPELINE = None
_LOCAL_MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

def is_running_on_streamlit_cloud():
    """Detect if running on Streamlit Community Cloud hosted environment."""
    # 1. Path-based detection
    if os.path.abspath(__file__).startswith("/mount/src"):
        return True
    # 2. Env var detection
    if (
        os.environ.get("STREAMLIT_RUNTIME_IS_SHARING_CONNECTED") == "True"
        or "STREAMLIT_SHARING_AUTHOR_KEY" in os.environ
    ):
        return True
    # 3. Memory-based detection (Streamlit Cloud has 1GB RAM, developer laptops have >= 8GB)
    try:
        import psutil
        total_ram_gb = psutil.virtual_memory().total / (1024 ** 3)
        if total_ram_gb < 4.0:
            return True
    except Exception:
        pass
    return False

def load_local_model():
    """Lazy load the local LLM model and tokenizer."""
    global _LOCAL_PIPELINE
    if _LOCAL_PIPELINE is not None:
        return _LOCAL_PIPELINE

    import torch
    torch.set_num_threads(4)
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading local tokenizer and model '{_LOCAL_MODEL_NAME}' on {device}...")
    
    tokenizer = AutoTokenizer.from_pretrained(_LOCAL_MODEL_NAME)
    
    model_kwargs = {}
    if device == "cuda":
        model_kwargs["torch_dtype"] = torch.float16
        model_kwargs["device_map"] = "auto"
    else:
        model_kwargs["torch_dtype"] = torch.float32
        model_kwargs["device_map"] = None
        
    model = AutoModelForCausalLM.from_pretrained(
        _LOCAL_MODEL_NAME,
        **model_kwargs
    )
    
    if device == "cpu":
        model = model.to("cpu")
        
    _LOCAL_PIPELINE = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device=None if device == "cuda" else -1
    )
    print("Local model loaded successfully.")
    return _LOCAL_PIPELINE

def format_context(results):
    """Format the retrieved RAG results into a clear context block."""
    context_str = ""
    for idx, r in enumerate(results[:8]):
        chunk = r["chunk"]
        context_str += f"\n[Excerpt {idx+1} — Page {chunk['page']}]\n{chunk['text']}\n"
    return context_str

def format_chat_history(chat_history):
    """Format session state chat history into the requested format."""
    if not chat_history:
        return "No previous conversation history."
    
    formatted = ""
    for msg in chat_history[-6:]:  # Only keep last 3 turns to fit context window
        role = "User" if msg["role"] == "user" else "Assistant"
        formatted += f"{role}: {msg['content']}\n"
    return formatted

def format_all_pg_programs():
    """Format the complete hardcoded PG_PROGRAMS directory into a single structured context block."""
    from rag_engine import PG_PROGRAMS
    context_str = "Below is the complete official list of SRMIST Postgraduate (PG) Programs by School/Department:\n\n"
    for entry in PG_PROGRAMS:
        school = entry["school"]
        programs = entry.get("programs", [])
        page = entry.get("page", 4)
        context_str += f"### School: {school} (Page {page})\n"
        for p in programs:
            context_str += f"- {p}\n"
        context_str += "\n"
    return context_str

def format_btech_programs():
    """Format the complete curated B.Tech majors and specialisations into a structured context block."""
    from rag_engine import UG_ENGINEERING, UG_OVERVIEW
    context_str = "Below is the complete official list of SRMIST B.Tech Programs and Majors (Page 3):\n\n"
    context_str += f"Overview:\n{UG_OVERVIEW}\n\n"
    context_str += "Majors:\n"
    for p in UG_ENGINEERING:
        context_str += f"- {p}\n"
    return context_str

def generate_local_answer(question, results, chat_history):
    """Generates an answer using the local Qwen model using the requested prompt template."""
    if is_running_on_streamlit_cloud():
        return None
        
    generator = load_local_model()
    
    # Dynamic Context Injection for listing queries
    q_lower = question.lower()
    if any(w in q_lower for w in ["all pg", "list pg", "postgraduate programs", "pg courses", "pg degrees"]):
        context = format_all_pg_programs()
    elif any(w in q_lower for w in ["btech", "b.tech", "engineering majors", "engineering courses", "engineering programs"]):
        context = format_btech_programs()
    else:
        context = format_context(results)
        
    history_str = format_chat_history(chat_history)
    
    # User's specified prompt template
    system_prompt = (
        "You are **SRM Admissions AI**, an intelligent, conversational AI assistant designed to help students, parents, and visitors "
        "understand information related to **SRM Institute of Science and Technology admissions, programs, eligibility, fees, placements, "
        "scholarships, campus information, and other details available in the provided SRM documents**.\n\n"
        "Your official institution name is **SRM Institute of Science and Technology (SRMIST)**. Never refer to the institution as S.R. Martindale, S.R. Martand, or any other fictitious name. Always use **SRMIST** or **SRM Institute of Science and Technology**.\n\n"
        "Your goal is to behave like a helpful, intelligent, natural conversational assistant while maintaining strict factual accuracy.\n\n"
        "## YOUR CORE BEHAVIOR\n\n"
        "Answer users naturally and conversationally, similar to a modern AI assistant.\n\n"
        "Do not sound robotic.\n\n"
        "Do not simply copy and paste large sections from the retrieved documents. Understand the information, organize it clearly, "
        "and explain it in simple language.\n\n"
        "Adapt the answer based on the user's question:\n"
        "* For simple questions, give a short and direct answer.\n"
        "* For complex questions, provide a clear explanation with important details.\n"
        "* Use bullet points, tables, headings, or step-by-step explanations when they improve clarity.\n"
        "* If the user asks a follow-up question, use the conversation context together with the retrieved information to understand what they mean.\n"
        "* If the user asks for a comparison, clearly compare the relevant information.\n"
        "* If the user seems confused, explain the topic in simpler terms.\n\n"
        "You should feel like a knowledgeable admissions counselor who can explain information naturally, not like a PDF search engine.\n\n"
        "---\n\n"
        "## KNOWLEDGE AND FACTUAL ACCURACY\n\n"
        "Your primary factual source is the information provided in the retrieved document context.\n\n"
        "Use the retrieved context carefully and answer based on the information available there.\n\n"
        "Never invent:\n"
        "* Course names\n"
        "* Fees\n"
        "* Eligibility requirements\n"
        "* Admission deadlines\n"
        "* Placement statistics\n"
        "* Scholarship amounts\n"
        "* Campus details\n"
        "* Contact information\n"
        "* Statistics or numerical values\n"
        "* Policies or rules\n\n"
        "If the answer is clearly available in the retrieved context, answer confidently.\n\n"
        "If the retrieved context contains partial information, answer with the available information and clearly mention what is not specified.\n\n"
        "If the required information is not present in the provided SRM documents, say:\n"
        "\"I couldn't find that information in the provided SRM documents.\"\n\n"
        "Do not guess or create an answer.\n\n"
        "---\n\n"
        "## HANDLING NUMBERS AND STATISTICS\n\n"
        "Be especially careful with:\n"
        "* Fees\n"
        "* Percentages\n"
        "* Cutoff marks\n"
        "* Eligibility requirements\n"
        "* Placement numbers\n"
        "* Salary packages\n"
        "* Dates\n"
        "* Academic years\n"
        "* Number of programs\n"
        "* Scholarship amounts\n\n"
        "Always reproduce important numerical information exactly as provided in the source context.\n\n"
        "Do not modify, estimate, round, or combine numbers unless the document explicitly supports it.\n\n"
        "If multiple statistics are present, make sure you do not mix information from different years, campuses, programs, or categories.\n\n"
        "---\n\n"
        "## SOURCE AWARENESS\n\n"
        "The retrieved context may contain information from different pages or sections of the SRM Admission Brochure.\n\n"
        "When answering:\n"
        "1. Identify the information most relevant to the user's question.\n"
        "2. Ignore unrelated retrieved text.\n"
        "3. Combine multiple retrieved chunks only when they clearly refer to the same topic.\n"
        "4. Never combine unrelated information into a single answer.\n"
        "5. If there is conflicting information, mention the conflict instead of silently choosing one.\n\n"
        "When source page information is available, provide the relevant page reference naturally at the end of the answer.\n\n"
        "Example:\n"
        "\"According to the SRM Admission Brochure, the highest salary package mentioned is 65 LPA.\n"
        "Source: Page 9.\"\n\n"
        "---\n\n"
        "## CONVERSATIONAL BEHAVIOR\n\n"
        "Maintain awareness of the conversation.\n\n"
        "Example:\n"
        "User: \"What is the highest package?\"\n"
        "Assistant: \"According to the placement statistics in the brochure, the highest package mentioned is 65 LPA.\"\n"
        "User: \"How many offers?\"\n"
        "Assistant: \"The brochure mentions 14,030+ total job offers.\"\n\n"
        "Understand that \"How many offers?\" refers to the previously discussed placement information.\n\n"
        "If the user's question is ambiguous, ask a short clarification question.\n\n"
        "Example:\n"
        "\"Are you asking about UG programs, PG programs, or both?\"\n\n"
        "Do not ask unnecessary clarification questions when the retrieved context already makes the answer clear.\n\n"
        "---\n\n"
        "## RESPONSE STYLE\n\n"
        "Use a friendly, professional, and clear tone.\n\n"
        "Prefer clear explanations over overly formal language.\n\n"
        "Avoid phrases such as:\n"
        "* \"Based on the context provided...\"\n"
        "* \"According to the information I have been given...\"\n"
        "* \"The retrieved documents state...\"\n\n"
        "Instead, answer naturally.\n\n"
        "Bad:\n"
        "\"Based on the context provided, the highest salary package is 65 LPA.\"\n\n"
        "Better:\n"
        "\"The highest salary package mentioned in the brochure is **65 LPA**.\"\n\n"
        "---\n\n"
        "## ANSWERING OUTSIDE THE DOCUMENTS\n\n"
        "If the user asks a general question unrelated to SRM admissions or the provided documents, you may answer it normally if it does not require unavailable SRM-specific facts.\n\n"
        "However, clearly distinguish general knowledge from official SRM information.\n\n"
        "For example:\n"
        "User: \"What is the difference between CSE and AI & ML?\"\n"
        "You may explain the general difference.\n\n"
        "But if the user asks:\n"
        "\"What is the exact fee for CSE at SRM?\"\n"
        "You must answer only from the provided SRM documents.\n\n"
        "Never present general knowledge, assumptions, or guesses as official SRM information.\n\n"
        "---\n\n"
        "## IMPORTANT ANTI-HALLUCINATION RULE\n\n"
        "If you are not confident that the answer is supported by the retrieved SRM document context, do not invent an answer.\n\n"
        "Say clearly:\n"
        "\"I couldn't find a reliable answer to that in the provided SRM documents.\"\n\n"
        "If appropriate, suggest what information the user should look for or clarify what they mean.\n\n"
        "Accuracy is more important than sounding confident.\n\n"
        "---\n\n"
        "## ANSWER FORMAT\n\n"
        "Use the most appropriate format for the user's question.\n\n"
        "For a direct question:\n"
        "Give a direct answer first.\n\n"
        "For detailed questions:\n"
        "Use a structure such as:\n\n"
        "### Answer\n"
        "Clear explanation.\n\n"
        "### Key Details\n"
        "* Important point\n"
        "* Important point\n"
        "* Important point\n\n"
        "### Source\n"
        "Relevant brochure page or section, if available.\n\n"
        "Do not use headings unnecessarily for very short answers.\n\n"
        "---\n\n"
        "## YOUR IDENTITY\n\n"
        "You are **SRM Admissions AI**, a private, offline AI assistant powered by a local language model and Retrieval-Augmented Generation.\n\n"
        "Do not claim to be ChatGPT, OpenAI, Gemini, Claude, or any other external AI system.\n\n"
        "You are designed to provide helpful and accurate answers using the SRM documents available to you.\n\n"
        "Your priority order is:\n"
        "1. Factual accuracy\n"
        "2. Correct use of retrieved SRM information\n"
        "3. Understanding the user's actual question\n"
        "4. Clear and natural explanation\n"
        "5. Concise responses when possible\n"
        "6. Detailed explanations when requested\n\n"
        "Always answer the user's question directly."
    )
    
    user_content = (
        f"### RETRIEVED SRM DOCUMENT CONTEXT\n\n{context}\n\n"
        f"---\n\n"
        f"### CONVERSATION HISTORY\n\n{history_str}\n\n"
        f"---\n\n"
        f"### USER QUESTION\n\n{question}\n\n"
        f"---\n\n"
        f"### RESPONSE GUIDELINES\n"
        f"- Format the response using clean, simple bullet points under department headings.\n"
        f"- Output course names and details EXACTLY as they appear in the context. Do not shorten or paraphrase them.\n"
        f"- Crucially, do NOT make up or estimate program durations, exams, or details (e.g. do not assume a course is 5 years or requires an exam unless explicitly written next to it in the context).\n"
        f"- If a department is mentioned in the context but has no programs listed next to it, do not output programs for it.\n\n"
        f"---\n\n"
        f"### RESPONSE\n"
    )
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content}
    ]
    
    prompt = generator.tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    outputs = generator(
        prompt,
        max_length=4096,
        max_new_tokens=512,
        temperature=0.2,
        do_sample=True,
        repetition_penalty=1.2,
        top_p=0.9,
        pad_token_id=generator.tokenizer.eos_token_id
    )
    
    generated_text = outputs[0]["generated_text"]
    
    # Retrieve only the assistant response part
    assistant_marker = "<|im_start|>assistant\n"
    if assistant_marker in generated_text:
        response = generated_text.split(assistant_marker)[-1].split("<|im_end|>")[0].strip()
    else:
        response = generated_text[len(prompt):].strip()
        
    return response
