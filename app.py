from flask import Flask, request, jsonify, render_template, send_file
from flask_cors import CORS
from transformers import T5ForConditionalGeneration, T5Tokenizer
import PyPDF2, torch, os, translate
import torch
from googletrans import Translator
from deep_translator import GoogleTranslator
from fpdf import FPDF
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase import pdfmetrics


app = Flask(__name__)
CORS(app)

# Load the model and tokenizer before defining routes
model = T5ForConditionalGeneration.from_pretrained('./model')
tokenizer = T5Tokenizer.from_pretrained('./model')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

@app.route('/')
def index():
    return render_template('index.html')  # Ensure your HTML file is named index.html

# @app.route('/summary', methods=['POST'])
# def summary():
#     data = request.get_json()
#     text = data.get('text', '')
#     inputs = tokenizer.encode("summarize: " + text, return_tensors="pt", max_length=512, truncation=True).to(device)
#     outputs = model.generate(inputs, max_length=512, num_beams=4, early_stopping=True)
#     translated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
#     # print(translated_text)

    # return jsonify({'text': translated_text})

@app.route('/extract_text', methods=['POST'])
def extract_text():
    file = request.files['file']
    reader = PyPDF2.PdfReader(file)
    text = ''.join([page.extract_text().replace('\n', ' ') for page in reader.pages])
    inputs = tokenizer.encode("summarize: " + text, return_tensors="pt", max_length=1000, truncation=True).to(device)
    outputs = model.generate(inputs, max_length=1000, min_length=100, length_penalty=2.0, num_beams=4, early_stopping=True)
    
    sumtext = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # print(text)
    # print(sumtext)
    return jsonify({
        'textorigin': text,
        'textsum': sumtext
        })


@app.route('/translate', methods=['POST'])
def translate():
    data = request.get_json()
    text = data.get('text', '')
    #print(request.data)
    target_language = data.get('target_language', 'fr')  # Default to French if not specified
    print(f"Translating to: {target_language}")  # Add this line for debugging
    language_map = {
        'fr': "translate English to French: ",
        'de': "translate English to German: ",
        'ro': "translate English to Romanian: ",
        'th': "translate English to Thai: "
    }
    if(target_language=='th'):
        translator = Translator()
        translated = translator.translate(text, dest='th')  
        translated_text = translated.text 
    else:
        translation_input = language_map.get(target_language, "translate English to French: ") + text  # Default to French
        inputs = tokenizer.encode(translation_input, return_tensors="pt", max_length=1000, truncation=True).to(device)
        outputs = model.generate(inputs, max_length=1000, num_beams=4, early_stopping=True)
        translated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    

    return jsonify({'translated_text': translated_text})

@app.route('/download_pdf', methods=['POST'])
def download_pdf():
    data = request.get_json()
    text = data.get('text', '')

    pdf = FPDF()
    pdf.add_page()

    # Ensure to use the correct path for the NotoSansThai font file
    font_path = "C:\\Users\\User\\Downloads\\Noto_Sans_Thai\\NotoSansThai-VariableFont_wdth,wght.ttf"  # Change this to your font path
    pdf.add_font("NotoSansThai", "", font_path, uni=True)  
    pdf.set_font("NotoSansThai", size=12)

    # Set line height for better readability
    line_height = 10  

    # Use multi_cell to handle text wrapping
    pdf.multi_cell(0, line_height, text)  # 0 means auto width

    # Save the PDF to a temporary file
    pdf_file_path = "summary.pdf"
    pdf.output(pdf_file_path)

    # Return the PDF file
    return send_file(pdf_file_path, as_attachment=True)
    
if __name__ == '__main__':
    app.run(debug=True)
