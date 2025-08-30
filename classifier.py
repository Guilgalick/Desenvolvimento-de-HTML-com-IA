import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import joblib
import os

# Baixar recursos do NLTK (apenas na primeira execução)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

class EmailClassifier:
    def __init__(self):
        self.model = None
        self.vectorizer = TfidfVectorizer()
        self.initialize_model()
    
    def initialize_model(self):
        """Inicializa o modelo com dados de treinamento"""
        # Dados de treinamento - emails exemplo para cada categoria
        training_emails = [
            # Emails produtivos
            "Preciso de suporte técnico com o sistema de relatórios.",
            "Não consigo acessar minha conta, preciso de ajuda urgente.",
            "Como faço para resetar minha senha?",
            "O sistema está apresentando erro 404 ao tentar gerar relatório.",
            "Preciso de uma atualização sobre meu caso aberto #12345.",
            "Gostaria de solicitar uma nova funcionalidade para o módulo financeiro.",
            "Há um problema de performance no servidor principal.",
            "Como posso exportar os dados em formato CSV?",
            "Preciso de assistência para configurar as permissões de usuário.",
            "O sistema não está salvando as alterações feitas no perfil.",
            
            # Emails improdutivos
            "Obrigado pela ajuda na reunião de hoje!",
            "Parabéns pelo excelente trabalho no projeto!",
            "Agradeço pelo suporte prestado durante a implantação.",
            "Feliz natal para toda a equipe!",
            "Ótimo fim de semana para todos!",
            "Só queria agradecer pelo atendimento de qualidade.",
            "Muito obrigado pela paciência e explicações.",
            "Que bom ver que tudo foi resolvido, obrigado!",
            "Aprecio muito o trabalho de vocês!",
            "Estou muito satisfeito com os resultados, parabéns!"
        ]
        
        # Rótulos correspondentes (0 = Improdutivo, 1 = Produtivo)
        training_labels = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        
        # Criar e treinar o modelo
        self.model = Pipeline([
            ('tfidf', TfidfVectorizer(
                stop_words=stopwords.words('portuguese'),
                max_features=1000,
                ngram_range=(1, 2)
            )),
            ('clf', MultinomialNB())
        ])
        
        self.model.fit(training_emails, training_labels)
    
    def preprocess_text(self, text):
        """Pré-processa o texto para análise"""
        # Converter para minúsculas
        text = text.lower()
        
        # Remover caracteres especiais e números
        text = re.sub(r'[^a-záàâãéèêíïóôõöúçñ\s]', '', text)
        
        # Tokenizar
        tokens = word_tokenize(text)
        
        # Remover stopwords
        stop_words = set(stopwords.words('portuguese'))
        tokens = [token for token in tokens if token not in stop_words]
        
        # Stemming
        stemmer = PorterStemmer()
        tokens = [stemmer.stem(token) for token in tokens]
        
        return ' '.join(tokens)
    
    def classify_email(self, email_text):
        """Classifica o email em produtivo ou improdutivo"""
        if not email_text or not email_text.strip():
            return {"error": "Texto do email vazio"}
        
        try:
            # Pré-processar o texto
            processed_text = self.preprocess_text(email_text)
            
            # Fazer a previsão
            prediction = self.model.predict([email_text])[0]
            prediction_proba = self.model.predict_proba([email_text])[0]
            
            # Determinar a categoria
            category = "Produtivo" if prediction == 1 else "Improdutivo"
            confidence = prediction_proba[1] if prediction == 1 else prediction_proba[0]
            
            return {
                "category": category,
                "confidence": round(confidence * 100, 2),
                "original_text": email_text,
                "processed_text": processed_text
            }
        except Exception as e:
            return {"error": f"Erro na classificação: {str(e)}"}
    
    def generate_response(self, email_text, category):
        """Gera uma resposta automática baseada na categoria do email"""
        if category == "Produtivo":
            responses = [
                "Agradecemos pelo seu contato. Nossa equipe técnica já foi notificada e está trabalhando na resolução do problema. Retornaremos em breve com uma atualização.",
                "Obrigado pela sua solicitação. Um de nossos especialistas entrará em contato para fornecer o suporte necessário dentro das próximas 24 horas.",
                "Recebemos sua dúvida e estamos analisando a melhor forma de ajudá-lo. Retornaremos em breve com mais informações.",
                "Agradecemos pelo feedback sobre a funcionalidade. Nossa equipe de desenvolvimento já foi notificada e avaliará sua sugestão."
            ]
            
            # Selecionar resposta com base no conteúdo
            if any(word in email_text.lower() for word in ['erro', 'problema', 'bug', 'não funciona']):
                return responses[0]
            elif any(word in email_text.lower() for word in ['dúvida', 'como', 'ajuda', 'orientação']):
                return responses[2]
            elif any(word in email_text.lower() for word in ['funcionalidade', 'implementação', 'sugestão', 'melhoria']):
                return responses[3]
            else:
                return responses[1]
        else:
            responses = [
                "Agradecemos muito pelo seu feedback positivo! Ficamos felizes em saber que nosso trabalho está sendo reconhecido.",
                "Muito obrigado pelas felicitações! É um prazer poder contribuir para o sucesso de sua empresa.",
                "Agradecemos suas palavras de incentivo. Nosso time se sente motivado com esse reconhecimento.",
                "Ficamos muito contentes com seu agradecimento. Estamos sempre à disposição para oferecer o melhor suporte."
            ]
            return responses[np.random.randint(0, len(responses))]

# Instância global do classificador
email_classifier = EmailClassifier()

if __name__ == "__main__":
    # Teste do classificador
    test_emails = [
        "Preciso de ajuda com o sistema, não consigo fazer login.",
        "Obrigado pelo suporte excelente na implantação do projeto!",
        "Como faço para exportar os relatórios em PDF?",
        "Parabéns pelo trabalho impecável!"
    ]
    
    for email in test_emails:
        result = email_classifier.classify_email(email)
        if "error" not in result:
            response = email_classifier.generate_response(email, result["category"])
            print(f"Email: {email}")
            print(f"Categoria: {result['category']} (Confiança: {result['confidence']}%)")
            print(f"Resposta: {response}")
            print("-" * 50)