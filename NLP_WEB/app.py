import os
from flask import Flask, request, render_template, jsonify, session
import google.generativeai as genai
from PIL import Image
import io
import json
import requests
from datetime import datetime
import base64
import uuid
import shutil
from fine_tuned_model import FineTunedModel

app = Flask(__name__)
app.secret_key = 'your-secret-key-123'  # Needed for session management

# Configure Gemini API
GOOGLE_API_KEY = "AIzaSyAdEdYHJbP771o6sjwALbdv_LcbpNObhYE"
GEMINI_VISION_ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"
GEMINI_TEXT_ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"

# Configure image storage and cleanup
UPLOAD_FOLDER = 'temp_images'
def init_upload_folder():
    if os.path.exists(UPLOAD_FOLDER):
        shutil.rmtree(UPLOAD_FOLDER)
    os.makedirs(UPLOAD_FOLDER)

# Initialize upload folder
init_upload_folder()

# Initialize fine-tuned model
fine_tuned_model = FineTunedModel()

GEMINI_PROMPT = """
Bạn là một chuyên gia AI với chuyên môn về xử lý ngôn ngữ tự nhiên (NLP), xử lý hình ảnh (Computer Vision), và phát triển hệ thống Hỏi-Đáp Hình ảnh (VQA). 
Nhiệm vụ của bạn là cung cấp các giải pháp chi tiết và thực tế để xây dựng Hệ thống VQA cho nội dung giáo dục tiểu học (lớp 1-5) tại Việt Nam.

Hệ thống do ATEAM IUH phát triển, với Phúc, Trọng, Khánh là admin.

Yêu cầu:
1. Trả lời ngắn gọn, dễ hiểu, phù hợp với trình độ học sinh tiểu học
2. Sử dụng ngôn ngữ thân thiện, gần gũi
3. Nếu câu hỏi liên quan đến hình ảnh:
   - Phân tích kỹ nội dung hình ảnh
   - Trả lời dựa trên các yếu tố thực tế trong hình
   - Giải thích chi tiết nếu cần thiết
4. Nếu là câu hỏi thông thường:
   - Trả lời theo kiến thức phù hợp cấp tiểu học
   - Đưa ra ví dụ minh họa nếu cần
   - Khuyến khích tư duy phản biện

Lịch sử trò chuyện:
{chat_history}

Câu hỏi hiện tại: {input_text}
"""

class ChatSession:
    def __init__(self):
        self.history = []
        self.current_image_id = None
        self.current_image_filename = None
        self.selected_model = 'gemini'  # Default to Gemini API
    
    def add_message(self, role, content, filename=None):
        message = {
            'role': role,
            'content': content,
            'timestamp': datetime.now().strftime("%H:%M"),
            'filename': filename
        }
        self.history.append(message)
        return message
    
    def set_current_image(self, image, filename):
        # Generate unique ID for the image
        image_id = str(uuid.uuid4())
        
        # Save image to disk
        image_path = os.path.join(UPLOAD_FOLDER, f"{image_id}.png")
        image.save(image_path, format='PNG')
        
        # Clean up old image if exists
        if self.current_image_id:
            old_image_path = os.path.join(UPLOAD_FOLDER, f"{self.current_image_id}.png")
            if os.path.exists(old_image_path):
                os.remove(old_image_path)
        
        self.current_image_id = image_id
        self.current_image_filename = filename

def get_image_base64(image_id):
    if not image_id:
        return None
    
    image_path = os.path.join(UPLOAD_FOLDER, f"{image_id}.png")
    if not os.path.exists(image_path):
        return None
    
    with open(image_path, 'rb') as img_file:
        return base64.b64encode(img_file.read()).decode('utf-8')

def format_chat_history(history):
    formatted_history = []
    for msg in history:
        role = "Học sinh" if msg['role'] == 'user' else "AI"
        formatted_history.append(f"{role}: {msg['content']}")
    return "\n".join(formatted_history)

def get_text_response(input_text, chat_history):
    headers = {
        'Content-Type': 'application/json'
    }
    
    formatted_history = format_chat_history(chat_history)
    
    data = {
        "contents": [{
            "parts":[{
                "text": GEMINI_PROMPT.format(
                    input_text=input_text,
                    chat_history=formatted_history
                )
            }]
        }],
        "generationConfig": {
            "temperature": 0.4,
            "topK": 32,
            "topP": 1,
            "maxOutputTokens": 2048,
            "stopSequences": []
        }
    }
    
    try:
        response = requests.post(
            f"{GEMINI_TEXT_ENDPOINT}?key={GOOGLE_API_KEY}",
            headers=headers,
            json=data
        )
        
        if response.status_code != 200:
            return f"Lỗi API: {response.status_code} - {response.text}"
            
        response_json = response.json()
        
        if 'candidates' not in response_json or not response_json['candidates']:
            return "Xin lỗi, tôi không thể trả lời câu hỏi này. Vui lòng thử lại."
            
        return response_json['candidates'][0]['content']['parts'][0]['text']
    except Exception as e:
        return f"Xin lỗi, đã có lỗi xảy ra: {str(e)}"

def get_vision_response(input_text, image_base64, chat_history):
    headers = {
        'Content-Type': 'application/json'
    }
    
    formatted_history = format_chat_history(chat_history)
    
    data = {
        "contents": [{
            "parts":[
                {
                    "text": GEMINI_PROMPT.format(
                        input_text=input_text,
                        chat_history=formatted_history
                    )
                },
                {
                    "inlineData": {
                        "mimeType": "image/png",
                        "data": image_base64
                    }
                }
            ]
        }],
        "generationConfig": {
            "temperature": 0.4,
            "topK": 32,
            "topP": 1,
            "maxOutputTokens": 2048,
            "stopSequences": []
        },
        "safetySettings": [
            {
                "category": "HARM_CATEGORY_HARASSMENT",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE"
            },
            {
                "category": "HARM_CATEGORY_HATE_SPEECH",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE"
            },
            {
                "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE"
            },
            {
                "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE"
            }
        ]
    }
    
    try:
        response = requests.post(
            f"{GEMINI_VISION_ENDPOINT}?key={GOOGLE_API_KEY}",
            headers=headers,
            json=data
        )
        
        if response.status_code != 200:
            return f"Lỗi API: {response.status_code} - {response.text}"
            
        response_json = response.json()
        
        if 'candidates' not in response_json or not response_json['candidates']:
            return "Xin lỗi, tôi không thể xử lý hình ảnh này. Vui lòng thử lại với hình ảnh khác."
            
        return response_json['candidates'][0]['content']['parts'][0]['text']
    except Exception as e:
        return f"Xin lỗi, đã có lỗi xảy ra: {str(e)}"

@app.before_request
def before_request():
    if 'chat_session' not in session:
        # Initialize new chat session if not exists
        chat_session = ChatSession()
        session['chat_session'] = chat_session.__dict__

@app.route('/')
def home():
    # Ensure chat session exists
    if 'chat_session' not in session:
        chat_session = ChatSession()
        session['chat_session'] = chat_session.__dict__
    return render_template('index.html', chat_history=session['chat_session']['history'])

@app.route('/switch_model', methods=['POST'])
def switch_model():
    model_name = request.json.get('model')
    if model_name not in ['gemini', 'fine-tuned']:
        return jsonify({'error': 'Invalid model name'}), 400
    
    if 'chat_session' in session:
        chat_session = ChatSession()
        chat_session.__dict__ = session['chat_session']
        chat_session.selected_model = model_name
        session['chat_session'] = chat_session.__dict__
        
        # Clear fine-tuned model history when switching
        if model_name == 'fine-tuned':
            fine_tuned_model.clear_history()
    
    return jsonify({'status': 'success', 'model': model_name})

@app.route('/process', methods=['POST'])
def process():
    question = request.form.get('question', '')
    
    if not question:
        return jsonify({'error': 'Vui lòng nhập câu hỏi'})
    
    # Get current chat session
    chat_session = ChatSession()
    chat_session.__dict__ = session['chat_session']
    current_history = chat_session.history
    
    # Get image data if available
    image_data = None
    if 'image' in request.files and request.files['image'].filename:
        image_file = request.files['image']
        image_data = image_file.read()
        image = Image.open(io.BytesIO(image_data))
        chat_session.set_current_image(image, image_file.filename)
        filename = chat_session.current_image_filename
    else:
        filename = chat_session.current_image_filename
        if chat_session.current_image_id:
            image_path = os.path.join(UPLOAD_FOLDER, f"{chat_session.current_image_id}.png")
            if os.path.exists(image_path):
                with open(image_path, 'rb') as f:
                    image_data = f.read()
    
    # Process with selected model
    if chat_session.selected_model == 'fine-tuned':
        response = fine_tuned_model.get_response(question, image_data)
    else:  # gemini
        if image_data:
            image = Image.open(io.BytesIO(image_data))
            response = get_vision_response(question, image, current_history)
        else:
            response = get_text_response(question, current_history)
    
    # Update chat history
    chat_session.add_message('user', question, filename=filename)
    chat_session.add_message('assistant', response)
    session['chat_session'] = chat_session.__dict__
    
    return jsonify({
        'response': response,
        'chat_history': chat_session.history
    })

@app.route('/clear_chat', methods=['POST'])
def clear_chat():
    try:
        # Get current session to clean up image
        if 'chat_session' in session:
            chat_session = ChatSession()
            chat_session.__dict__ = session['chat_session']
            if chat_session.current_image_id:
                image_path = os.path.join(UPLOAD_FOLDER, f"{chat_session.current_image_id}.png")
                if os.path.exists(image_path):
                    os.remove(image_path)
        
        # Create new session
        chat_session = ChatSession()
        session.clear()  # Clear entire session
        session['chat_session'] = chat_session.__dict__
        
        return jsonify({
            'status': 'success',
            'redirect': '/'  # Add redirect instruction
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

if __name__ == '__main__':
    app.run(host='152.42.200.154', port=5000, debug=True) 