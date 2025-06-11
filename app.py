import os
import re
import json
import uuid
import pickle
from pathlib import Path

import fitz 
import faiss
import numpy as np
from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from agno.agent import Agent
from agno.media import File
from agno.models.google import Gemini
from agno.exceptions import ModelProviderError

# Load environment variables
load_dotenv()

# Configuration
UPLOAD_FOLDER = 'uploads'
INDEX_FOLDER = 'index'
ALLOWED_EXTENSIONS = {'pdf'}

# Flask app setup
app = Flask(__name__)
app.secret_key = os.getenv('FLASK_SECRET_KEY', 'supersecretkey')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['INDEX_FOLDER'] = INDEX_FOLDER

# Create directories
Path(UPLOAD_FOLDER).mkdir(exist_ok=True)
Path(INDEX_FOLDER).mkdir(exist_ok=True)

# Initialize models
embedder = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

# Gemini setup
gemini_api_key = os.getenv('GEMINI_API_KEY')
gemini_model = Gemini(api_key=gemini_api_key)
agent = Agent(model=gemini_model)

# Insurance domain synonyms dictionary
INSURANCE_SYNONYMS = {
    'wypadek': ['zdarzenie', 'incydent', 'przypadek', 'szkoda', 'uszkodzenie', 'sytuacja'],
    'świadczenie': ['wypłata', 'odszkodowanie', 'kompensata', 'rekompensata', 'zwrot', 'ubezpieczenie'],
    'składka': ['premia', 'opłata', 'koszt', 'wkład'],
    'ubezpieczony': ['klient', 'posiadacz polisy', 'właściciel'],
    'ubezpieczyciel': ['towarzystwo', 'firma ubezpieczeniowa', 'zakład'],
    'polisa': ['umowa', 'kontrakt', 'dokument'],
    'szkoda': ['strata', 'uszkodzenie', 'zniszczenie', 'utrata'],
    'franszyza': ['udział własny', 'odliczenie'],
    'okres karencji': ['czas oczekiwania', 'okres wyczekiwania'],
    'likwidacja': ['rozpatrzenie', 'obsługa szkody', 'proces szkodowy'],
    'roszczenie': ['żądanie', 'wniosek o wypłatę'],
    'wypowiedzenie': ['rozwiązanie', 'anulowanie', 'zakończenie'],
    'choroba': ['schorzenie', 'dolegliwość', 'stan chorobowy'],
    'leczenie': ['terapia', 'kuracja', 'rehabilitacja'],
    'hospitalizacja': ['pobyt w szpitalu', 'leczenie szpitalne'],
}


def allowed_file(filename):
    """Check if file has allowed extension."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def normalize_text(text):
    """Normalize text for better matching."""
    text = text.lower()
    # Preserve Polish characters
    text = re.sub(r'[^\w\sąćęłńóśźż]', ' ', text, flags=re.UNICODE)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def extract_fragments_from_pdf(pdf_path):
    """Extract text fragments from PDF, splitting by paragraphs."""
    doc = fitz.open(pdf_path)
    fragments = []
    
    for page_num, page in enumerate(doc):
        text = page.get_text().strip()
        if text:
            # Split into paragraphs (double newline)
            paragraphs = re.split(r'\n\s*\n', text)
            for para_idx, paragraph in enumerate(paragraphs):
                paragraph = paragraph.strip()
                if len(paragraph) > 50:  # Ignore very short fragments
                    fragments.append({
                        'page': page_num + 1,
                        'paragraph': para_idx + 1,
                        'text': paragraph,
                        'text_normalized': normalize_text(paragraph)
                    })
    
    doc.close()
    return fragments


def extract_definitions_advanced(fragments):
    """Extract definitions using various patterns."""
    definitions = {}
    definition_patterns = [
        r'(.+?)\s*-\s*oznacza\s*(.+?)(?=\n|$)',
        r'(.+?)\s*to\s*(.+?)(?=\n|$)',
        r'przez\s+(.+?)\s+rozumie\s+się\s*(.+?)(?=\n|$)',
        r'definicja\s*:\s*(.+?)\s*-\s*(.+?)(?=\n|$)',
        r'(.+?)\s*definiuje\s+się\s+jako\s*(.+?)(?=\n|$)',
    ]
    
    for frag in fragments:
        text = frag['text']
        for pattern in definition_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE | re.MULTILINE)
            for match in matches:
                term = match.group(1).strip()
                definition = match.group(2).strip()
                if len(term) < 100 and len(definition) > 10:  # Basic validation
                    definitions[normalize_text(term)] = {
                        'term': term,
                        'definition': definition,
                        'source': f"strona {frag['page']}"
                    }
    
    return definitions


def expand_query_with_synonyms(query):
    """Expand query with domain-specific synonyms."""
    expanded_terms = set()
    query_words = normalize_text(query).split()
    
    for word in query_words:
        expanded_terms.add(word)
        # Search for synonyms
        for key, synonyms in INSURANCE_SYNONYMS.items():
            if word in synonyms or word == key:
                expanded_terms.add(key)
                expanded_terms.update(synonyms)
    
    return list(expanded_terms)


def create_tfidf_index(fragments):
    """Create TF-IDF index for fragments."""
    texts = [frag['text_normalized'] for frag in fragments]
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 3),  # Unigrams, bigrams, and trigrams
        max_features=10000,
        stop_words=None  # Don't remove stop words as they might be important in legal context
    )
    tfidf_matrix = vectorizer.fit_transform(texts)
    return vectorizer, tfidf_matrix


def index_pdf(pdf_path, index_path, meta_path):
    """Index PDF with multiple search methods."""
    fragments = extract_fragments_from_pdf(pdf_path)
    
    # Embedding index
    texts = [frag['text'] for frag in fragments]
    embeddings = embedder.encode(texts, convert_to_numpy=True)
    dim = embeddings.shape[1]
    embedding_index = faiss.IndexFlatL2(dim)
    embedding_index.add(embeddings)
    
    # TF-IDF index
    tfidf_vectorizer, tfidf_matrix = create_tfidf_index(fragments)
    
    # Extract definitions
    definitions = extract_definitions_advanced(fragments)
    
    # Save all indices
    faiss.write_index(embedding_index, index_path)
    
    metadata = {
        'fragments': fragments,
        'definitions': definitions,
        'tfidf_vectorizer': tfidf_vectorizer,
        'tfidf_matrix': tfidf_matrix
    }
    
    with open(meta_path, 'wb') as f:
        pickle.dump(metadata, f)


def load_index(index_path, meta_path):
    """Load all indices and metadata."""
    embedding_index = faiss.read_index(index_path)
    with open(meta_path, 'rb') as f:
        metadata = pickle.load(f)
    return embedding_index, metadata


def extract_keywords(text):
    """Extract keywords from text."""
    # Extended stopwords list
    stopwords = {
        'czy', 'jest', 'to', 'na', 'w', 'z', 'za', 'do', 'od', 'po', 'i', 'albo', 'lub',
        'a', 'o', 'u', 'dla', 'przez', 'się', 'nie', 'tak', 'jak', 'która', 'który',
        'które', 'którą', 'którzy', 'którego', 'której', 'ktory', 'ktora', 'ktorych',
        'ktorym', 'ktorymi', 'może', 'można', 'będzie', 'była', 'było', 'były', 'są',
        'co', 'gdy', 'gdzie', 'kiedy', 'dlaczego', 'ile', 'jakie', 'jaki', 'jaka'
    }
    
    # Add synonyms to keywords
    normalized = normalize_text(text)
    words = [w for w in normalized.split() if w not in stopwords and len(w) > 2]
    
    # Expand with synonyms
    expanded_words = set(words)
    for word in words:
        synonyms = expand_query_with_synonyms(word)
        expanded_words.update(synonyms)
    
    return expanded_words


def hybrid_search(question, embedding_index, metadata, top_k=10):
    """Hybrid search combining embeddings with TF-IDF."""
    fragments = metadata['fragments']
    tfidf_vectorizer = metadata['tfidf_vectorizer']
    tfidf_matrix = metadata['tfidf_matrix']
    definitions = metadata['definitions']
    
    # 1. Embedding search
    q_emb = embedder.encode([question], convert_to_numpy=True)
    D_emb, I_emb = embedding_index.search(q_emb, top_k * 2)  # Get more candidates
    
    # 2. TF-IDF search
    question_expanded = ' '.join(expand_query_with_synonyms(question))
    question_tfidf = tfidf_vectorizer.transform([normalize_text(question_expanded)])
    tfidf_scores = cosine_similarity(question_tfidf, tfidf_matrix).flatten()
    
    # 3. Combine results
    candidates = {}
    
    # Add embedding results (convert distance to similarity)
    for i, idx in enumerate(I_emb[0]):
        if idx < len(fragments):
            similarity_emb = 1 / (1 + D_emb[0][i])  # Convert distance to similarity
            candidates[idx] = candidates.get(idx, 0) + similarity_emb * 0.6  # 60% weight
    
    # Add TF-IDF results
    for idx, score in enumerate(tfidf_scores):
        if score > 0:
            candidates[idx] = candidates.get(idx, 0) + score * 0.4  # 40% weight
    
    # 4. Check definitions
    keywords = extract_keywords(question)
    definition_boost = {}
    for keyword in keywords:
        for def_key, def_data in definitions.items():
            if keyword in def_key or any(kw in def_data['definition'].lower() for kw in keywords):
                # Find fragments containing this definition
                for idx, frag in enumerate(fragments):
                    if def_data['term'].lower() in frag['text'].lower():
                        definition_boost[idx] = definition_boost.get(idx, 0) + 0.3
    
    # Add definition boost
    for idx, boost in definition_boost.items():
        candidates[idx] = candidates.get(idx, 0) + boost
    
    # Sort and return top results
    sorted_candidates = sorted(candidates.items(), key=lambda x: x[1], reverse=True)
    
    result_fragments = []
    for idx, score in sorted_candidates[:top_k]:
        fragment = fragments[idx].copy()
        fragment['relevance_score'] = score
        result_fragments.append(fragment)
    
    return result_fragments


def simple_markdown_to_html(text):
    """
    Prosta konwersja markdown na HTML (listy, nagłówki, pogrubienia, nowe linie).
    Poprawia formatowanie wypunktowań i numeracji, obsługuje <ul> i <ol> oraz nie generuje zbędnych <br>.
    Unika lookbehind, aby nie powodować błędów re.
    """
    import html
    text = html.escape(text)

    # Zamień numerowane listy na <ol><li>...</li></ol>
    def ol_wrap(match):
        items = match.group(0)
        return f"<ol>{items}</ol>"
    text = re.sub(r'((<li>\d+\..*?</li>\s*){2,})', ol_wrap, text, flags=re.DOTALL)

    # Zamień linie zaczynające się od liczby i kropki na <li>...</li>
    text = re.sub(r'^[ \t]*\d+\.[ \t]+(.+)', r'<li>\1</li>', text, flags=re.MULTILINE)

    # Zamień wypunktowania na <ul><li>...</li></ul>
    def ul_wrap(match):
        items = match.group(0)
        return f"<ul>{items}</ul>"
    text = re.sub(r'((<li>.*?</li>\s*){2,})', ul_wrap, text, flags=re.DOTALL)

    # Zamień linie zaczynające się od *, -, •, → na <li>...</li>
    text = re.sub(r'^[ \t]*[\*\-•→][ \t]+(.+)', r'<li>\1</li>', text, flags=re.MULTILINE)

    # Pogrubienie **text** i __text__
    text = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', text)
    text = re.sub(r'__(.+?)__', r'<strong>\1</strong>', text)
    # Kursywa _text_
    text = re.sub(r'_(.+?)_', r'<em>\1</em>', text)

    # Nagłówki (##, #)
    text = re.sub(r'^\s*#{2,}\s*(.+)$', r'<h3>\1</h3>', text, flags=re.MULTILINE)
    text = re.sub(r'^\s*#\s*(.+)$', r'<h2>\1</h2>', text, flags=re.MULTILINE)

    # Zamień podwójne nowe linie na <br><br>
    text = re.sub(r'\n{2,}', '<br><br>', text)
    # Zamień pojedyncze nowe linie na <br>
    text = text.replace('\n', '<br>')

    # Usuń <br> bezpośrednio przed <ul>, <ol> oraz po </ul>, </ol>
    text = re.sub(r'(<br>\s*)+(<ul>|<ol>)', r'\2', text)
    text = re.sub(r'(</ul>|</ol>)(\s*<br>)+', r'\1', text)
    text = re.sub(r'^(<br>\s*)+', '', text)
    text = re.sub(r'(<br>\s*)+$', '', text)

    return text


def generate_contextual_answer(question, relevant_fragments, definitions, agent):
    """Generate contextual answer based on fragments and definitions."""
    # Prepare definitions
    definitions_text = ""
    if definitions:
        definitions_text = "DEFINICJE:\n"
        for def_data in list(definitions.values())[:10]:  # Max 10 definitions
            definitions_text += f"- {def_data['term']}: {def_data['definition']}\n"
        definitions_text += "\n"

    # Prepare fragments
    context_parts = []
    for i, frag in enumerate(relevant_fragments[:7]):  # Max 7 fragments
        context_parts.append(f"Fragment {i+1} (strona {frag['page']}): {frag['text']}")

    context_text = "\n\n".join(context_parts)

    prompt = f"""Jesteś ekspertem od ubezpieczeń analizującym dokument OWU. 

ZASADY ODPOWIEDZI:
1. Odpowiadaj WYŁĄCZNIE na podstawie podanych fragmentów i definicji
2. Jeśli informacji nie ma w dokumentach, napisz: "Nie mogę znaleźć odpowiedzi w przesłanym dokumencie OWU"
3. Używaj konkretnych odniesień do fragmentów (np. "Według fragmentu 1...")
4. Jeśli odpowiedź wymaga interpretacji kilku fragmentów, wyjaśnij to krok po kroku
5. Fragmenty są posortowane wg trafności

{definitions_text}

FRAGMENTY DOKUMENTU (posortowane wg trafności):
{context_text}

PYTANIE: {question}

ODPOWIEDŹ:"""

    try:
        response = agent.run(prompt)
        if hasattr(response, "content"):
            raw_answer = response.content
        elif hasattr(response, "text"):
            raw_answer = response.text
        else:
            raw_answer = str(response)
        return simple_markdown_to_html(raw_answer)
    except Exception as e:
        return f"Błąd podczas generowania odpowiedzi: {str(e)}"


def query_index_improved(question, embedding_index, metadata, top_k=7):
    """Main search function with improved algorithms."""
    # Hybrid search
    relevant_fragments = hybrid_search(question, embedding_index, metadata, top_k)
    
    # Check if we found anything relevant
    if not relevant_fragments or all(frag.get('relevance_score', 0) < 0.1 for frag in relevant_fragments):
        return None, "Nie znaleziono odpowiedzi na podstawie przesłanego dokumentu. Spróbuj przeformułować pytanie."
    
    # Generate answer
    answer = generate_contextual_answer(question, relevant_fragments, metadata['definitions'], agent)
    
    return relevant_fragments, answer


def generate_paraphrases(question):
    """Generate question paraphrases for better search."""
    prompt = f"""Wygeneruj 4 różne wersje tego pytania dotyczącego ubezpieczeń:

Oryginalne pytanie: {question}

Wygeneruj:
1. Wersja formalna (język prawniczy/ubezpieczeniowy)
2. Wersja uproszczona (potoczny język)
3. Wersja szczegółowa (z dodatkowymi warunkami)
4. Wersja negatywna (kiedy NIE przysługuje/NIE obowiązuje)

Każdą wersję rozpocznij od numeru (1., 2., 3., 4.)"""

    try:
        response = agent.run(prompt)
        if hasattr(response, "content"):
            text = response.content
        elif hasattr(response, "text"):
            text = response.text
        else:
            text = str(response)
            
        lines = [l.strip() for l in text.split('\n') if l.strip()]
        paraphrases = []
        
        for line in lines:
            if len(line) > 2 and line[0] in "1234" and line[1] == ".":
                paraphrases.append(line[2:].strip())
            elif line.startswith("- ") and len(paraphrases) < 4:
                paraphrases.append(line[2:].strip())
                
        return paraphrases[:4] if paraphrases else [question]
    except Exception:
        return [question]


def get_pdf_id(pdf_path):
    """Get unique identifier based on PDF filename."""
    return os.path.splitext(os.path.basename(pdf_path))[0]


# Routes
@app.route('/', methods=['GET'])
def index():
    """Main page with chat history."""
    if 'history' not in session:
        session['history'] = []
    return render_template('index.html', history=session.get('history', []))


@app.route('/upload', methods=['POST'])
def upload():
    """Handle PDF upload and indexing."""
    if 'pdf' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['pdf']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
        
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Index PDF
        pdf_id = get_pdf_id(filepath)
        index_path = os.path.join(app.config['INDEX_FOLDER'], f"{pdf_id}.index")
        meta_path = os.path.join(app.config['INDEX_FOLDER'], f"{pdf_id}.meta")
        
        try:
            index_pdf(filepath, index_path, meta_path)
            session['pdf_id'] = pdf_id
            session['history'] = []
            return jsonify({
                'success': True, 
                'filename': filename, 
                'message': 'Plik został przesłany i zindeksowany. System jest gotowy do odpowiadania na pytania.'
            })
        except Exception as e:
            return jsonify({'error': f'Błąd podczas indeksowania: {str(e)}'}), 500
    else:
        return jsonify({'error': 'Invalid file type. Only PDF allowed.'}), 400


@app.route('/ask', methods=['POST'])
def ask():
    """Handle question asking."""
    question = request.form.get('question')
    pdf_id = session.get('pdf_id')
    
    if not pdf_id:
        msg = 'Nie przesłano pliku OWU. Najpierw wgraj plik PDF.'
        if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
            return jsonify({'error': msg}), 400
        flash(msg)
        return redirect(url_for('index'))
        
    if not question:
        msg = 'Nie podano pytania.'
        if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
            return jsonify({'error': msg}), 400
        flash(msg)
        return redirect(url_for('index'))
    
    index_path = os.path.join(app.config['INDEX_FOLDER'], f"{pdf_id}.index")
    meta_path = os.path.join(app.config['INDEX_FOLDER'], f"{pdf_id}.meta")
    
    if not os.path.exists(index_path) or not os.path.exists(meta_path):
        msg = 'Indeks nie został znaleziony. Spróbuj ponownie przesłać plik.'
        if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
            return jsonify({'error': msg}), 400
        flash(msg)
        return redirect(url_for('index'))

    try:
        # Load indices
        embedding_index, metadata = load_index(index_path, meta_path)
        
        # Try with original question
        fragments, answer = query_index_improved(question, embedding_index, metadata)
        
        if fragments is None:
            # Try with paraphrases
            paraphrases = generate_paraphrases(question)
            for paraphrase in paraphrases:
                fragments, answer = query_index_improved(paraphrase, embedding_index, metadata)
                if fragments is not None:
                    break
        
        if fragments is None:
            answer = "Nie mogę znaleźć odpowiedzi w przesłanym dokumencie OWU. Spróbuj przeformułować pytanie, używając bardziej konkretnych terminów ubezpieczeniowych."
        
        # Save to history
        history = session.get('history', [])
        history.append({
            'user': question,
            'bot': answer,
            'fragments_count': len(fragments) if fragments else 0
        })
        session['history'] = history
        
        if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
            return jsonify({
                'user': question,
                'bot': answer,
                'fragments_count': len(fragments) if fragments else 0
            })
            
        return redirect(url_for('index') + '?gotochat=1')
        
    except Exception as e:
        error_msg = f"Błąd podczas przetwarzania pytania: {str(e)}"
        if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
            return jsonify({'error': error_msg}), 500
        flash(error_msg)
        return redirect(url_for('index'))


if __name__ == '__main__':
    app.run(debug=True)