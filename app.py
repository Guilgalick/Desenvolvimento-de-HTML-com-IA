from flask import Flask, render_template, request, jsonify
from classifier import email_classifier
import json

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/classify', methods=['POST'])
def classify_email():
    try:
        data = request.get_json()
        email_text = data.get('email_text', '')
        
        if not email_text:
            return jsonify({"error": "Texto do email não fornecido"}), 400
        
        # Classificar o email
        classification_result = email_classifier.classify_email(email_text)
        
        if "error" in classification_result:
            return jsonify(classification_result), 500
        
        # Gerar resposta automática
        response = email_classifier.generate_response(
            email_text, 
            classification_result["category"]
        )
        
        classification_result["suggested_response"] = response
        
        return jsonify(classification_result)
    
    except Exception as e:
        return jsonify({"error": f"Erro no servidor: {str(e)}"}), 500

@app.route('/examples')
def get_examples():
    examples = {
        "productive": [
            "Prezados, estou com um problema no sistema de relatórios. Não consigo gerar o relatório de vendas do último trimestre. O sistema exibe a mensagem 'Erro 504: Timeout'. Gostaria de solicitar urgência na resolução deste problema, pois preciso apresentar esses dados em uma reunião amanhã.",
            "Olá, gostaria de saber se há previsão para a implementação da funcionalidade de exportação em PDF no módulo financeiro. Essa feature seria muito útil para nossa equipe, que precisa compartilhar relatórios com stakeholders externos."
        ],
        "unproductive": [
            "Bom dia! Gostaria de agradecer pelo excelente suporte prestado durante a implantação do novo sistema. A equipe foi extremamente prestativa e profissional, o que tornou toda a transição muito mais suave do que esperávamos.",
            "Caros colegas, quero parabenizar a todos pelo sucesso do último projeto. O lançamento foi um grande sucesso e recebemos muitos feedbacks positivos dos clientes. Isso só foi possível graças ao empenho e dedicação de cada um de vocês."
        ]
    }
    return jsonify(examples)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)