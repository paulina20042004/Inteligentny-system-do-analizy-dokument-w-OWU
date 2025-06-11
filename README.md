# 🤖 Inteligentny asystent do analizy dokumentów OWU

Asystent wykorzystujący technologie **RAG (Retrieval-Augmented Generation)** oraz **zaawansowane wyszukiwanie hybrydowe** do analizy **Ogólnych Warunków Ubezpieczenia (OWU)**.

Aplikacja stworzona tylko na podstawie działania asystentów AI.

---

## 🎯 Opis

Aplikacja umożliwia przesyłanie dokumentów **PDF** z warunkami ubezpieczenia oraz zadawanie pytań w **języku naturalnym**.

---

## 🛠️ Technologie

**Backend**:  
- Flask  
- Python

**AI / ML**:  
- Google Gemini *(generowanie odpowiedzi)*  
- SentenceTransformers *(embeddings)*  
- FAISS *(wyszukiwanie wektorowe)*  
- scikit-learn *(TF-IDF)*

**Przetwarzanie PDF**:  
- PyMuPDF (`fitz`)

**Frontend**:  
- HTML  
- CSS  
- JavaScript

**Inne**:  
- Agno framework

---

## 📋 Wymagania

**Python**: 3.8+

**Główne zależności**:
- `flask`
- `pymupdf`
- `faiss-cpu`
- `numpy`
- `sentence-transformers`
- `scikit-learn`
- `python-dotenv`
- `werkzeug`

---

## ▶️ Uruchom aplikację

```bash
python app.py
```

Aplikacja będzie dostępna pod adresem: http://localhost:5000

## 🔧 Konfiguracja

### Zmienne środowiskowe

- `GEMINI_API_KEY` – Klucz API Google Gemini (**wymagany**)  
- `FLASK_SECRET_KEY` – Klucz sesji Flask (**opcjonalny**, domyślnie ustawiona wartość)

---

### Obsługiwane formaty

- **Pliki wejściowe**: PDF  
- **Języki**: Polski (z obsługą polskich znaków diakrytycznych)  
- **Domeny**: Ubezpieczenia (ze słownikiem synonimów branżowych)

---

## 📖 Użytkowanie

1. Prześlij dokument PDF z warunkami ubezpieczenia  
2. Zadaj pytanie w języku naturalnym, np.:
   - *"Kiedy przysługuje odszkodowanie za wypadek?"*
   - *"Jaka jest wysokość franszyzy redukcyjnej?"*
   - *"Czy choroba zawodowa jest objęta ochroną?"*
3. Otrzymaj odpowiedź z odniesieniami do konkretnych fragmentów dokumentu

---

## 🎯 Biznes Case

Firmy ubezpieczeniowe i klienci borykają się z następującymi wyzwaniami:

- **Złożoność dokumentów OWU** – Warunki ubezpieczenia są często obszerne i napisane skomplikowanym językiem prawniczym  
- **Czas obsługi klienta** – Konsultanci spędzają 60–80% czasu na wyszukiwaniu informacji w dokumentach zamiast pomagać klientom  
- **Błędy interpretacyjne** – Nieprecyzyjne odpowiedzi prowadzą do sporów, reklamacji i strat finansowych  
- **Koszty szkolenia** – Nowi pracownicy potrzebują miesięcy, aby opanować wszystkie produkty ubezpieczeniowe  
- **Dostępność 24/7** – Klienci oczekują natychmiastowych odpowiedzi poza godzinami pracy

  ## Wygląd:
  ![image](https://github.com/user-attachments/assets/28d97e30-f4d1-47df-ab5c-c9e2d09e0ab3)
  ![image](https://github.com/user-attachments/assets/f4434127-47f2-4c47-ab26-5968255c0891)



