from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline
from deep_translator import GoogleTranslator
import requests
import os

app = Flask(__name__)

similarity_model = SentenceTransformer("all-MiniLM-L6-v2") 
ai_detector = pipeline("text-classification", model="roberta-base-openai-detector") 

UPLOAD_FOLDER = "uploads"
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

def extract_text_from_pdf(pdf_path):
    text = ""
    with open(pdf_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            text += page.extract_text() + "\n"
    return text

def extract_text_from_docx(docx_path):
    doc = docx.Document(docx_path)
    text = "\n".join([para.text for para in doc.paragraphs])
    return text

@app.route("/upload", methods=["POST"])
def upload_file():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files["file"]

    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    file_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
    file.save(file_path)

   
    if file.filename.endswith(".pdf"):
        extracted_text = extract_text_from_pdf(file_path)
    elif file.filename.endswith(".docx"):
        extracted_text = extract_text_from_docx(file_path)
    else:
        return jsonify({"error": "Unsupported file format"}), 400

    return jsonify({"extracted_text": extracted_text})

@app.route("/detect_plagiarism", methods=["POST"])
def detect_plagiarism():
    data = request.get_json()
    original_text = data.get("original_text")
    student_text = data.get("student_text")

    if not original_text or not student_text:
        return jsonify({"error": "Missing input texts"}), 400

    embedding1 = similarity_model.encode(original_text, convert_to_tensor=True)
    embedding2 = similarity_model.encode(student_text, convert_to_tensor=True)

    similarity_score = util.pytorch_cos_sim(embedding1, embedding2).item()

    return jsonify({
        "similarity_score": similarity_score,
        "message": "Plagiarism detected" if similarity_score > 0.75 else "Text is original"
    })


@app.route("/detect_ai_generated", methods=["POST"])
def detect_ai_generated():
    data = request.get_json()
    text = data.get("text")

    if not text:
        return jsonify({"error": "Missing text"}), 400

    result = ai_detector(text)
    return jsonify({
        "ai_score": result[0]["score"],
        "message": "Likely AI-generated" if result[0]["label"] == "LABEL_1" else "Likely Human-written"
    })


@app.route("/cross_language_check", methods=["POST"])
def cross_language_check():
    data = request.get_json()
    original_text = data.get("original_text")
    student_text = data.get("student_text")

    if not original_text or not student_text:
        return jsonify({"error": "Missing input texts"}), 400

    translated_student_text = GoogleTranslator(source="auto", target="en").translate(student_text)

    embedding1 = similarity_model.encode(original_text, convert_to_tensor=True)
    embedding2 = similarity_model.encode(translated_student_text, convert_to_tensor=True)

    similarity_score = util.pytorch_cos_sim(embedding1, embedding2).item()

    return jsonify({
        "similarity_score": similarity_score,
        "message": "Cross-language plagiarism detected" if similarity_score > 0.75 else "Text is original"
    })


@app.route("/web_check", methods=["POST"])
def web_check():
    data = request.get_json()
    query = data.get("query")

    if not query:
        return jsonify({"error": "Missing query"}), 400

  
    serp_api_key = "YOUR_SERPAPI_KEY"  
    search_url = f"https://serpapi.com/search.json?q={query}&api_key={serp_api_key}"
    response = requests.get(search_url)

    if response.status_code != 200:
        return jsonify({"error": "Failed to fetch search results"}), 500

    search_results = response.json().get("organic_results", [])

    return jsonify({"search_results": search_results})

if __name__ == "__main__":
    app.run(debug=True)
