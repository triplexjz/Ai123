from flask import Flask, request, jsonify, render_template, send_file
from flask_cors import CORS
from transformers import T5ForConditionalGeneration, T5Tokenizer
from fpdf import FPDF
import PyPDF2, torch, os


app = Flask(__name__)
CORS(app)

# model = MBartForConditionalGeneration.from_pretrained('facebook/mbart-large-50')
# tokenizer = MBart50TokenizerFast.from_pretrained('facebook/mbart-large-50')
# Load the model and tokenizer before defining routes
model = T5ForConditionalGeneration.from_pretrained("./model")
tokenizer = T5Tokenizer.from_pretrained("./model")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

@app.route('/')
def index():
    return render_template('index.html')  # Ensure your HTML file is named index.html

@app.route('/extract_text', methods=['POST'])
def extract_text():
    file = request.files['file']
    reader = PyPDF2.PdfReader(file)
    text = ''.join([page.extract_text() for page in reader.pages])

    inputs = tokenizer.encode("summarize: " + text, return_tensors="pt", max_length=1000, truncation=True).to(device)
    outputs = model.generate(inputs, max_length=1000, min_length=100, length_penalty=2.0, num_beams=4, early_stopping=True)
    summary = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return jsonify({'text': summary})


@app.route('/translate', methods=['POST'])
def translate():
    data = request.get_json()
    text = data.get('text', '')
    target_language = data.get('target_language', 'fr')  # Default to French if not specified
    print(f"Translating to: {target_language}")  # Add this line for debugging
    language_map = {
        'fr': "translate English to French: ",
        'de': "translate English to German: ",
        'ro': "translate English to Romanian: ",
        'th': "translate English to Thai: "
    }

    translation_input = language_map.get(target_language, "translate English to French: ") + text  # Default to French

    # Prepare input for translation
    inputs = tokenizer.encode(translation_input, return_tensors="pt", max_length=512, truncation=True).to(device)
    outputs = model.generate(inputs, max_length=512, num_beams=4, early_stopping=True)
    translated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return jsonify({'translated_text': translated_text})
@app.route('/download_pdf', methods=['POST'])
def download_pdf():
    data = request.get_json()
    text = data.get('text', '')

    # Generate a PDF
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 10, text)

    # Save the PDF to a temporary file
    pdf_file_path = "summary.pdf"
    pdf.output(pdf_file_path)

    # Return the PDF file
    return send_file(pdf_file_path, as_attachment=True)

    # Optionally, you may want to delete the PDF after sending it
    os.remove(pdf_file_path)


if __name__ == '__main__':
    app.run(debug=True)
