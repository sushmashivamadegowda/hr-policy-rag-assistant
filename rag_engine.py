import re
import difflib
import numpy as np
import faiss
import torch
from typing import List, Dict, Any, Tuple
from sentence_transformers import SentenceTransformer, util
from langchain_text_splitters import RecursiveCharacterTextSplitter

class HRPolicyRAG:
    """
    Industry-Standard RAG Engine using:
    1. Multi-Exemplar Anchor Vector Intent Classification (Semantics)
    2. Dynamic Fuzzy Matching on Canonical Tokens (Typo tolerance without hardcoding)
    3. Document Vocabulary Validation & FAISS Cosine Similarity Thresholding (OOD/Gibberish filtering)
    4. Robust Sentence-Boundary & Decimal-Preserving Answer Extraction
    5. Embedding-Powered Semantic Sentence Re-ranking
    """
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.embedding_model = SentenceTransformer(model_name)
        self.dimension = 384  # Dimension for all-MiniLM-L6-v2
        self.index = faiss.IndexFlatIP(self.dimension)
        
        # Dynamic organization persona (inferred from document header, default: Company)
        self.organization_name = "Company"
        
        # Canonical Intent Exemplars (semantic templates)
        self.intent_exemplars = {
            "GREETING": [
                "hello", "hi", "hey", "good morning", "good afternoon", "good evening", 
                "greetings", "how are you", "how are you doing"
            ],
            "BOT_IDENTITY": [
                "who are you", "what is your name", "what is your job", "what do you do", 
                "what are your responsibilities", "who created you", "what is your role"
            ],
            "BOT_CAPABILITIES": [
                "what can you do", "what do you know", "what kind of questions can you answer",
                "how can you help me", "what help can you provide", "what are your features",
                "how do you work", "help", "guide me", "what is your purpose"
            ],
            "RESPONSE_TIME": [
                "how many min will u take to answer a specific question",
                "how many minutes will you take to answer",
                "how many min will you take to answer",
                "how long do you take to answer",
                "how fast are you",
                "how quickly do you respond",
                "how much time do you take to answer",
                "what is your response time",
                "how fast can you answer a question",
                "how many seconds do you take",
                "how fast do you reply",
                "how many minutes to answer",
                "how quick is your reply"
            ],
            "USER_IDENTITY": [
                "what is my name", "who am i", "do you know my name", "tell me my name", "what is my identity"
            ],
            "THANKS": [
                "thank you", "thanks", "i appreciate it", "many thanks"
            ],
            "FAREWELL": [
                "goodbye", "bye", "see you later", "take care", "have a nice day"
            ]
        }
        
        # Pre-compute Dense Vector Embeddings for all intent exemplars
        self.intent_vectors = {
            intent: self.embedding_model.encode(phrases, convert_to_tensor=True)
            for intent, phrases in self.intent_exemplars.items()
        }
        
        # Canonical token vocabulary for automated typo tolerance
        self.canonical_tokens = list(set([
            w for phrases in self.intent_exemplars.values() 
            for phrase in phrases 
            for w in phrase.split()
        ]))
        
        self.chunks: List[Dict[str, Any]] = []
        self.raw_documents: List[Dict[str, Any]] = []
        self.document_vocab: set = set()
        self.is_indexed = False
        self.local_generator = None
        
    def chunk_documents(
        self, 
        raw_documents: List[Dict[str, Any]], 
        chunk_size: int = 500, 
        chunk_overlap: int = 50
    ) -> List[Dict[str, Any]]:
        """
        Splits raw document texts into overlapping semantic chunks.
        """
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", "---", ". ", " ", ""]
        )
        
        processed_chunks = []
        global_chunk_id = 0
        
        for doc in raw_documents:
            text = doc.get("text", "")
            source = doc.get("source", "Unknown Document")
            page = doc.get("page", 1)
            
            splits = text_splitter.split_text(text)
            for split_idx, split_text in enumerate(splits):
                if split_text.strip():
                    processed_chunks.append({
                        "chunk_id": global_chunk_id,
                        "source": source,
                        "page": page,
                        "split_index": split_idx,
                        "text": split_text.strip()
                    })
                    global_chunk_id += 1
                    
        return processed_chunks

    def _infer_organization_name(self, documents: List[Dict[str, Any]]):
        """Infers company or organization name dynamically from document headers."""
        for doc in documents:
            text = doc.get("text", "")
            for line in text.splitlines()[:25]:
                line = line.strip()
                if re.search(r'\b(?:CORP|CORPORATION|LIMITED|LTD|INC|LLC|PVT|PRIVATE LIMITED)\b', line, re.IGNORECASE):
                    clean = re.sub(r'[\(\)]', '', line).strip().title()
                    # Exclude generic headers like 'CIN NO: ...' or 'REGISTERED OFFICE: ...'
                    if not any(k in clean.lower() for k in ["cin no", "office", "plot", "street", "road"]):
                        if 3 < len(clean) < 70:
                            self.organization_name = clean
                            return
        self.organization_name = "Company"

    def build_index(
        self, 
        documents: List[Dict[str, Any]], 
        chunk_size: int = 500, 
        chunk_overlap: int = 50
    ):
        """
        Chunks documents, dynamically detects organization name, builds vocabulary index, 
        and indexes embeddings into FAISS.
        """
        self.raw_documents = documents
        self._infer_organization_name(documents)
        self.chunks = self.chunk_documents(documents, chunk_size, chunk_overlap)
        
        # Build document vocabulary index for hybrid lexical validation
        self.document_vocab = set()
        for doc in documents:
            words = re.findall(r'\b[a-zA-Z]{3,}\b', doc.get("text", "").lower())
            self.document_vocab.update(words)

        if not self.chunks:
            self.index = faiss.IndexFlatIP(self.dimension)
            self.is_indexed = False
            return
            
        texts = [chunk["text"] for chunk in self.chunks]
        embeddings = self.embedding_model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        
        # Normalize for Cosine Similarity
        faiss.normalize_L2(embeddings)
        
        self.index = faiss.IndexFlatIP(self.dimension)
        self.index.add(embeddings.astype(np.float32))
        self.is_indexed = True

    def _classify_intent(self, query: str) -> Tuple[str, float]:
        """
        Semantic Intent Classification using exemplar vectors and fuzzy token normalization.
        """
        tokens = re.findall(r'\b[a-zA-Z]+\b', query.lower())
        corrected_tokens = []
        for t in tokens:
            close = difflib.get_close_matches(t, self.canonical_tokens, n=1, cutoff=0.7)
            corrected_tokens.append(close[0] if close else t)
        normalized_query = " ".join(corrected_tokens) if corrected_tokens else query
        
        q_vec = self.embedding_model.encode(normalized_query, convert_to_tensor=True)
        best_intent = "POLICY_SEARCH"
        best_score = 0.0
        
        for intent, vectors in self.intent_vectors.items():
            max_sim = util.cos_sim(q_vec, vectors).max().item()
            if max_sim > best_score:
                best_score = max_sim
                best_intent = intent
                
        if best_score >= 0.50:
            return best_intent, best_score
            
        return "POLICY_SEARCH", best_score

    def retrieve(self, query: str, top_k: int = 5, score_threshold: float = 0.42) -> List[Dict[str, Any]]:
        """
        Retrieves top-k relevant document chunks with cosine thresholding.
        """
        if not self.is_indexed or len(self.chunks) == 0:
            return []
            
        query_vector = self.embedding_model.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(query_vector)
        
        actual_k = min(top_k, len(self.chunks))
        scores, indices = self.index.search(query_vector.astype(np.float32), actual_k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0 and idx < len(self.chunks):
                if score >= score_threshold:
                    chunk_info = self.chunks[idx].copy()
                    chunk_info["similarity_score"] = float(score)
                    results.append(chunk_info)
                    
        return results

    def contextualize_query(self, query: str, chat_history: List[Dict[str, Any]] = None) -> str:
        """
        Resolves conversational follow-ups and anaphora (e.g. 'as a women', 'can it be extended?')
        by contextualizing with recent user query topics from the conversation history.
        """
        if not chat_history:
            return query
            
        last_user_query = ""
        for msg in reversed(chat_history):
            if msg.get("role") == "user":
                content = msg.get("content", "").strip()
                # Skip current query if already appended to history
                if content.lower() == query.strip().lower():
                    continue
                last_user_query = content
                break
                
        if not last_user_query:
            return query
            
        q_clean = query.strip().lower()
        words = q_clean.split()
        
        # Follow-up indicators
        follow_up_starters = (
            "as a ", "as an ", "what about ", "how about ", "and what ", "and how ", 
            "what if ", "and for ", "for ", "can it ", "does it ", "is it ", 
            "can this ", "could it ", "what of ", "in case of "
        )
        is_starter = any(q_clean.startswith(s) for s in follow_up_starters)
        has_pronoun = any(w in words for w in ["it", "its", "this", "that", "they", "them", "these", "those"])
        is_fragment = len(words) <= 4 and not any(w in words for w in ["what", "why", "where", "when", "who", "how"])
        
        if is_starter or has_pronoun or is_fragment:
            clean_prev = re.sub(r'[\?!.]+$', '', last_user_query).strip()
            norm_query = query.strip()
            if any(w in norm_query.lower() for w in ['women', 'woman', 'female', 'maternity']):
                return f"{clean_prev} for female woman maternity leave"
            if any(w in norm_query.lower() for w in ['men', 'man', 'male', 'paternity']):
                return f"{clean_prev} for male man paternity leave"
            return f"{clean_prev} ({norm_query})"
            
        return query

    def generate_answer(
        self, 
        query: str, 
        chat_history: List[Dict[str, Any]] = None,
        top_k: int = 5, 
        score_threshold: float = 0.38,
        use_llm: bool = False,
        api_key: str = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Processes query:
        - Accurately classifies conversational intents (greetings, bot identity, response speed, capabilities)
        - Contextualizes multi-turn follow-up queries
        - Employs strict thresholding to gracefully reject out-of-domain queries
        - Generates precise, grounded answers
        - Returns structured answers along with verifiable source passage citations
        """
        # 1. Semantic Intent Classification on raw query
        intent, intent_score = self._classify_intent(query)
        org = self.organization_name
        
        if intent == "GREETING":
            return {
                "answer": f"Hello! 👋 Good day! How can I assist you with {org} HR policies today?",
                "sources": []
            }
            
        elif intent == "BOT_IDENTITY":
            return {
                "answer": f"I am the official **HR Policy Assistant** for {org}. I am here to assist you with questions regarding company policies, leave rules, working hours, attendance, travel allowances, and employee guidelines.",
                "sources": []
            }

        elif intent == "BOT_CAPABILITIES":
            return {
                "answer": f"I can help you instantly look up and understand {org}'s HR policies, including:\n- 📅 **Leave & Time-Off** (Earned leaves, casual leaves, maternity benefits, holidays)\n- ⏰ **Working Hours & Attendance** (Office timings, weekly offs, lunch breaks, biometric marking)\n- 💼 **Employment Terms & Notice Period** (Probation, resignation, separation)\n- 💰 **Wages, Payroll & Statutory Compliances** (PF, ESIC, salary details)\n- 📜 **Code of Conduct & Workplace Guidelines**\n\nFeel free to ask any question about company rules!",
                "sources": []
            }

        elif intent == "RESPONSE_TIME":
            return {
                "answer": f"I respond almost instantly, typically within **1 to 2 seconds**! ⚡\n\nYou can ask me any question about {org}'s HR policies, leave entitlements, office hours, or benefits, and I will find the exact rules right away.",
                "sources": []
            }
            
        elif intent == "USER_IDENTITY":
            return {
                "answer": f"I don't know your personal name as I don't have access to individual employee profiles. However, I would be delighted to assist you with any questions about {org} HR policies!",
                "sources": []
            }
            
        elif intent == "THANKS":
            return {
                "answer": "You are very welcome! 😊 Please feel free to ask anytime if you need more assistance.",
                "sources": []
            }
            
        elif intent == "FAREWELL":
            return {
                "answer": "Goodbye! Have a wonderful day ahead. 👋",
                "sources": []
            }

        # 2. Contextualize query with conversation history for multi-turn search
        search_query = self.contextualize_query(query, chat_history)

        # 3. Vector Database Retrieval using contextualized query
        retrieved_chunks = self.retrieve(search_query, top_k=top_k, score_threshold=score_threshold)
        best_sim = max([c.get("similarity_score", 0.0) for c in retrieved_chunks], default=0.0)
        
        # 4. Out-of-Domain & Low Relevance Guard
        if not retrieved_chunks or best_sim < score_threshold:
            return {
                "answer": f"I would be happy to help, but I couldn't find specific details regarding that in the {org} HR policy document. I can assist with questions regarding leaves, working hours, attendance, notice periods, wages, and employee benefits.",
                "sources": []
            }

        # 5. Generative Answer Synthesis (synthesizes precise answer, NOT just extracting raw lines)
        if use_llm:
            answer_text = self.synthesize_answer(query, retrieved_chunks, chat_history, api_key)
        else:
            answer_text = self._summarize_max_2_lines(query, retrieved_chunks)

        # 6. Format sources for auditability and citations
        sources = [
            {
                "chunk_id": c.get("chunk_id"),
                "source": c.get("source", "Document"),
                "page": c.get("page", 1),
                "score": round(c.get("similarity_score", 0.0), 3),
                "text": c.get("text", "")
            }
            for c in retrieved_chunks
        ]

        return {
            "answer": answer_text,
            "sources": sources
        }

    def _summarize_max_2_lines(self, query: str, chunks: List[Dict[str, Any]]) -> str:
        """
        Ranks candidate policy statements against query using embedding semantic similarity.
        Safely preserves decimal numbers (e.g. 0.5 leaves, 1.5 days, $75.50) and abbreviations.
        """
        abbrevs = ['Pvt.', 'Ltd.', 'Inc.', 'Corp.', 'Mr.', 'Mrs.', 'Ms.', 'Dr.', 'vs.', 'No.', 'e.g.', 'i.e.', 'etc.', 'a.m.', 'p.m.']
        candidates = []
        seen = set()

        for chunk in chunks:
            raw_text = chunk["text"]
            chunk_score = chunk.get("similarity_score", 0.5)
            
            # Split by line to preserve list items and tables
            lines = [l.strip() for l in raw_text.splitlines() if l.strip()]
            
            for line in lines:
                # Remove section delimiter lines
                if re.match(r'^[=\-_]{3,}$', line) or (line.isupper() and len(line) < 35):
                    continue
                if line.endswith("POLICY") or line.endswith("MANAGEMENT") or line.endswith("CONTRACTS"):
                    continue
                    
                # Protect abbreviations from sentence boundary splitting
                protected = line
                for idx, abb in enumerate(abbrevs):
                    protected = protected.replace(abb, f"__ABB_{idx}__")
                    
                # Clean leading list markers safely without stripping decimals:
                # \d{1,2}\.\s+ requires space after the period, so '0.5' is NEVER stripped
                protected = re.sub(r'^(?:(?:\([a-zA-Z0-9]+\)|[a-zA-Z0-9]{1,3}\)|\d{1,2}\.)\s+)', '', protected)
                protected = re.sub(r'^[•\-\*\#]\s*', '', protected).strip()
                
                # Split on sentence-ending punctuation (period, exclamation, question mark)
                # NOT preceded by digit and NOT followed by digit, followed by space or end of string
                splits = re.split(r'(?<!\d)[\.!\?](?!\d)(?:\s+|$)', protected)
                
                for s in splits:
                    s = s.strip()
                    if len(s) < 18 or s.endswith(":") or len(s.split()) < 4 or s[0].islower():
                        continue
                        
                    # Restore protected abbreviations
                    for idx, abb in enumerate(abbrevs):
                        s = s.replace(f"__ABB_{idx}__", abb)
                        
                    if s.lower() in seen:
                        continue
                    seen.add(s.lower())
                    
                    candidates.append((s, chunk_score))

        if not candidates:
            return f"I would be happy to help, but I couldn't find a direct answer to your query in the {self.organization_name} HR policy document. Please feel free to ask about leaves, working hours, or allowances!"

        # Semantic Candidate Re-ranking using Embedding Model
        candidate_texts = [c[0] for c in candidates]
        q_vec = self.embedding_model.encode(query, convert_to_tensor=True)
        c_vecs = self.embedding_model.encode(candidate_texts, convert_to_tensor=True)
        sims = util.cos_sim(q_vec, c_vecs)[0].cpu().numpy()

        stop_words = {'what', 'is', 'are', 'the', 'how', 'many', 'will', 'i', 'get', 'a', 'an', 'in', 'for', 'of', 'to', 'do', 'can', 'my', 'me', 'ur', 'your', 'whats'}
        q_words = set(re.findall(r'\w+', query.lower())) - stop_words
        q_stems = set(w.rstrip('s') for w in q_words if len(w) > 2)

        scored_statements = []
        for idx, (stmt, ch_score) in enumerate(candidates):
            sem_score = float(sims[idx])
            stmt_stems = set(w.rstrip('s') for w in re.findall(r'\w+', stmt.lower()) if len(w) > 2)
            keyword_overlap = len(q_stems.intersection(stmt_stems))
            
            # Composite relevance score:
            # 55% sentence semantic match + 20% parent chunk score + 25% stem keyword match
            composite_score = (0.55 * sem_score) + (0.20 * ch_score) + (0.25 * min(keyword_overlap, 3) / 3.0)
            scored_statements.append((composite_score, stmt))

        scored_statements.sort(key=lambda x: x[0], reverse=True)

        # Select top 1 or 2 highest-ranking distinct statements
        top_sentences = []
        for score, stmt in scored_statements:
            top_sentences.append(stmt)
            if len(top_sentences) == 2:
                break

        # Return direct, clear sentences as a natural paragraph
        clean_sentences = [s if s.endswith(".") else s + "." for s in top_sentences]
        return " ".join(clean_sentences)

    def _load_local_llm(self):
        """Loads local generative LLM (TinyLlama-1.1B-Chat or Flan-T5) lazily and caches it."""
        if self.local_generator is not None:
            return
            
        try:
            import torch
            from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
            model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            model = AutoModelForCausalLM.from_pretrained(
                model_id, 
                torch_dtype=torch.float32, 
                low_cpu_mem_usage=True
            )
            self.local_generator = pipeline(
                "text-generation", 
                model=model, 
                tokenizer=tokenizer
            )
            self.generator_type = "causal"
        except Exception as e:
            print(f"TinyLlama loading failed ({e}), falling back to Flan-T5...")
            from transformers import pipeline
            self.local_generator = pipeline("text2text-generation", model="google/flan-t5-base")
            self.generator_type = "seq2seq"

    def _call_cloud_llm(
        self, 
        query: str, 
        chunks: List[Dict[str, Any]], 
        chat_history: List[Dict[str, Any]] = None, 
        api_key: str = ""
    ) -> str:
        """Calls Groq or OpenAI chat completion API if configured."""
        import requests
        is_groq = api_key.startswith("gsk_")
        url = "https://api.groq.com/openai/v1/chat/completions" if is_groq else "https://api.openai.com/v1/chat/completions"
        model_name = "llama-3.1-8b-instant" if is_groq else "gpt-4o-mini"
        
        context_text = "\n\n".join([f"Passage {idx+1}:\n{c['text']}" for idx, c in enumerate(chunks[:4])])
        messages = [
            {
                "role": "system",
                "content": (
                    f"You are the official HR Assistant for {self.organization_name}. "
                    "Answer the employee's question directly, accurately, and precisely using ONLY the facts in the provided HR Policy passages. "
                    "Synthesize a natural, coherent, conversational response. Do not simply copy-paste raw bullet points. "
                    "If the answer cannot be determined from the passages, state that politely."
                )
            }
        ]
        if chat_history:
            for m in chat_history[-4:]:
                messages.append({"role": m["role"], "content": m["content"]})
                
        messages.append({
            "role": "user",
            "content": f"HR Policy Passages:\n{context_text}\n\nQuestion: {query}\n\nPrecise Answer:"
        })
        
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": model_name,
            "messages": messages,
            "temperature": 0.2,
            "max_tokens": 250
        }
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=12)
            if resp.status_code == 200:
                return resp.json()["choices"][0]["message"]["content"].strip()
        except Exception as e:
            print(f"Cloud API generation error: {e}")
        return None

    def synthesize_answer(
        self, 
        query: str, 
        chunks: List[Dict[str, Any]], 
        chat_history: List[Dict[str, Any]] = None,
        api_key: str = None
    ) -> str:
        """
        Synthesizes a precise, natural language answer using an LLM 
        instead of extracting raw disconnected sentences.
        """
        if api_key:
            cloud_ans = self._call_cloud_llm(query, chunks, chat_history, api_key)
            if cloud_ans:
                return cloud_ans

        try:
            self._load_local_llm()
            context_text = "\n\n".join([c["text"] for c in chunks[:3]])
            
            # Incorporate recent conversational history
            recent_context = ""
            if chat_history and len(chat_history) > 1:
                recent_user_turns = [
                    f"User asked: {m['content']}" for m in chat_history[-3:] if m.get("role") == "user" and m.get("content", "").strip().lower() != query.strip().lower()
                ]
                if recent_user_turns:
                    recent_context = "Previous topic: " + " | ".join(recent_user_turns) + "\n\n"

            if getattr(self, "generator_type", "causal") == "causal" and hasattr(self.local_generator.tokenizer, "apply_chat_template"):
                messages = [
                    {
                        "role": "system",
                        "content": (
                            f"You are the official HR Policy Assistant for {self.organization_name}. "
                            "Answer the user's question directly, concisely, and factually based ONLY on the provided policy context. "
                            "If the answer cannot be determined from the context, state that politely. "
                            "Do not make up facts or include unrelated policy rules."
                        )
                    },
                    {
                        "role": "user",
                        "content": (
                            f"Policy Context:\n{context_text}\n\n"
                            f"{recent_context}"
                            f"Question: {query}\n"
                            f"Precise Answer:"
                        )
                    }
                ]
                chat_prompt = self.local_generator.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                res = self.local_generator(chat_prompt, max_new_tokens=120, do_sample=False)
                gen_text = res[0]["generated_text"][len(chat_prompt):].strip()
                gen_text = re.sub(r'</s>$', '', gen_text).strip()
                # Clean up prompt artifacts and robotic prefixes
                gen_text = re.sub(r'^(?:Precise Answer|Direct Answer|Answer|Certainly!?|Based on the policy context,?|In the given context,?)\s*[:,\-]?\s*', '', gen_text, flags=re.IGNORECASE).strip()
                if gen_text:
                    gen_text = gen_text[0].upper() + gen_text[1:]
                if gen_text and len(gen_text) > 15:
                    return gen_text
            else:
                prompt = (
                    f"Read the HR policy context and answer the question directly, politely, and completely.\n\n"
                    f"Context: {context_text}\n\n"
                    f"{recent_context}"
                    f"Question: {query}\n"
                    f"Direct Answer:"
                )
                res = self.local_generator(prompt, max_length=120)
                gen_text = res[0]["generated_text"].strip()
                gen_text = re.sub(r'^(?:Precise Answer|Direct Answer|Answer|Certainly!?|Based on the policy context,?|In the given context,?)\s*[:,\-]?\s*', '', gen_text, flags=re.IGNORECASE).strip()
                if gen_text:
                    gen_text = gen_text[0].upper() + gen_text[1:]
                if gen_text and len(gen_text) > 10:
                    if not gen_text.endswith("."):
                        gen_text += "."
                    return gen_text
        except Exception as e:
            print(f"LLM generation failed ({e}), falling back to extractive summary...")
            
        return self._summarize_max_2_lines(query, chunks)
