import json
import re
import os
from fastapi import HTTPException
import joblib
import pandas as pd
from typing import List
from openai import OpenAI
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

from src.schemas import OpenAIConfig, ClassificationMethod, CATEGORIES, PRIORITIES


class TicketClassifierService:
    def __init__(self):
        self.category_model = None
        self.priority_model = None
        self.openai_client = None
        self.openai_config = None
        self.load_models()
        self.setup_openai()

    def setup_openai(self):
        """
        Configuration OpenAI Client
        """
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key:
            self.openai_client = OpenAI(api_key=api_key)
            self.openai_config = OpenAIConfig(api_key=api_key)
        else:
            print("OpenAI API Key not set")

    def update_openai_config(self, config: OpenAIConfig):
        self.openai_config = config
        self.openai_client = OpenAI(api_key=self.openai_config.api_key)

    def preprocess_text(self, text: str) -> str:
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def extract_features(self, title: str, description: str) -> str:
        return f"{title} {description}"

    def train_models(self, training_data: List[dict]):
        try:
            df = pd.DataFrame(training_data)
            print(f"DataFrame created with columns: {df.columns.tolist()}")
            print(f"DataFrame shape: {df.shape}")
            print(f"First row: {df.iloc[0].to_dict() if len(df) > 0 else 'Empty'}")
            
            df['complete_text'] = df.apply(
                lambda x: self.extract_features(x['title'], x['description']),
                axis=1
            )
            df['processed_text'] = df['complete_text'].apply(self.preprocess_text)
        except Exception as e:
            print(f"Error in train_models: {e}")
            print(f"Training data: {training_data}")
            raise

        self.category_model = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=1000, stop_words='english')),
            ('classifier', MultinomialNB())
        ])
        
        self.priority_model = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=1000, stop_words='english')),
            ('classifier', MultinomialNB())
        ])

        X = df['processed_text']
        y_category = df['category']
        y_priority = df['priority']

        self.category_model.fit(X, y_category)
        self.priority_model.fit(X, y_priority)

        
        models_dir = '/app/models'
        os.makedirs(models_dir, exist_ok=True)
        joblib.dump(self.category_model, f'{models_dir}/category_model.pkl')
        joblib.dump(self.priority_model, f'{models_dir}/priority_model.pkl')

        return True

    def load_models(self):
        try:
            models_dir = '/app/models'
            if os.path.exists(f'{models_dir}/category_model.pkl'):
                self.category_model = joblib.load(f'{models_dir}/category_model.pkl')
            if os.path.exists(f'{models_dir}/priority_model.pkl'):
                self.priority_model = joblib.load(f'{models_dir}/priority_model.pkl')
        except Exception as e:
            print(f"Failed to load models: {e}")
            self._create_default_models()

    def _create_default_models(self):
        sample_data = [
            {"title": "Cannot log in", "description": "My password is not working", "category": "Login", "priority": "Medium"},
            {"title": "Connection error", "description": "I don't have internet access", "category": "Network", "priority": "High"},
            {"title": "Software installation", "description": "I need to install Office", "category": "Software", "priority": "Low"},
            {"title": "Computer running slow", "description": "My PC is very slow", "category": "Hardware", "priority": "Medium"},
            {"title": "Problem with invoice", "description": "My bill is incorrect", "category": "Billing", "priority": "Medium"},
            {"title": "System down", "description": "Main server is not responding", "category": "Hardware", "priority": "High"},
            {"title": "Email not working", "description": "Cannot send emails", "category": "Email", "priority": "Medium"},
            {"title": "Virus detected", "description": "My antivirus detected malware", "category": "Security", "priority": "High"}
        ]

        os.makedirs('/app/models', exist_ok=True)
        self.train_models(sample_data)

    def classify_with_openai(self, title: str, description: str) -> dict:
        if not self.openai_client:
            raise HTTPException(status_code=400, detail="OpenAI not configured")
        prompt = f"""
            You are an expert in technical support ticket classification. Analyze the following ticket and provide detailed classification.
            
            TICKET:
            Title: {title}
            Description: {description}
            
            AVAILABLE CATEGORIES: {', '.join(CATEGORIES)}
            AVAILABLE PRIORITIES: {', '.join(PRIORITIES)}
            
            Provide your response in JSON format with the following structure:
            {{
                "category": "chosen_category",
                "priority": "chosen_priority",
                "category_confidence": 0.95,
                "priority_confidence": 0.90,
                "sentiment_detected": "frustrated/neutral/satisfied",
                "urgency_level": "critical/high/medium/low",
                "entities_extracted": ["entity1", "entity2"],
                "reasoning": "Brief explanation of why you chose this classification"
            }}
            
            IMPORTANT RULES:
            - Confidence must be a number between 0.0 and 1.0
            - Only use the categories and priorities listed above
            - Extract entities like products, versions, systems mentioned
            - Reasoning should be concise but informative
            """
        try:
            response = self.openai_client.chat.completions.create(
                model = self.openai_config.model,
                messages = [{"role": "user", "content": prompt}],
                # max_tokens = self.openai_config.max_tokens,
                temperature = self.openai_config.temperature,
            )
            result_text = response.choices[0].message.content.strip()

            if result_text.startswith('```json'):
                result_text = result_text.replace('```json', '').replace('```', '').strip()
            result = json.loads(result_text)

            if result['category'] not in CATEGORIES:
                result["category"] = "General"
            if result['priority'] not in PRIORITIES:
                result["priority"] = "Medium"
            
            # Convert entities_extracted list to string if it exists
            if 'entities_extracted' in result and isinstance(result['entities_extracted'], list):
                result['entities_extracted'] = ', '.join(result['entities_extracted'])

            return result

        except json.JSONDecodeError:
            return self._openai_fallback_classification(title, description)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error with OpenAI: {str(e)}")

    def _openai_fallback_classification(self, title: str, description: str) -> dict:
        """Fallback classification if OpenAI fails"""
        # Use local model as fallback
        local_result = self.classify_ticket_local(title, description)
        return {
            "category": local_result["category"],
            "priority": local_result["priority"],
            "category_confidence": local_result["category_confidence"] * 0.7,
            "priority_confidence": local_result["priority_confidence"] * 0.7,
            "sentiment_detected": "neutral",
            "urgency_level": "medium",
            "entities_extracted": [],
            "reasoning": "Fallback classification (local model)"
        }

    def classify_ticket_local(self, title: str, description: str) -> dict:
        """Classify ticket using local models"""
        if not self.category_model or not self.priority_model:
            raise HTTPException(status_code=500, detail="Local models not trained")

        full_text = self.extract_features(title, description)
        processed_text = self.preprocess_text(full_text)

        
        category_pred = self.category_model.predict([processed_text])[0]
        priority_pred = self.priority_model.predict([processed_text])[0]

        category_proba = max(self.category_model.predict_proba([processed_text])[0])
        priority_proba = max(self.priority_model.predict_proba([processed_text])[0])

        
        entities = []
        text_lower = f"{title} {description}".lower()
        
        tech_keywords = ['error', 'code', 'installation', 'software', 'application', 'system']
        network_keywords = ['wifi', 'internet', 'connection', 'network', 'vpn']
        email_keywords = ['email', 'outlook', 'sync', 'mail']
        billing_keywords = ['billing', 'charge', 'subscription', 'invoice', 'payment']
        
        for keyword in tech_keywords + network_keywords + email_keywords + billing_keywords:
            if keyword in text_lower:
                entities.append(keyword)
        
        # Basic sentiment detection
        negative_words = ['error', 'failed', 'problem', 'issue', 'cannot', 'not working', 'broken']
        sentiment = "frustrated" if any(word in text_lower for word in negative_words) else "neutral"
        
        # Basic urgency mapping
        urgency_map = {"High": "high", "Medium": "medium", "Low": "low"}
        urgency = urgency_map.get(priority_pred, "medium")

        return {
            "category": category_pred,
            "priority": priority_pred,
            "category_confidence": round(category_proba, 3),
            "priority_confidence": round(priority_proba, 3),
            "sentiment_detected": sentiment,
            "urgency_level": urgency,
            "extracted_entities": ", ".join(entities[:3]) if entities else None,  # Max 3 entities
            "reasoning": f"Local classification based on text analysis: {category_pred.lower()} issue with {priority_pred.lower()} priority"
        }

    def classify_ticket_hybrid(self, title: str, description: str) -> dict:
        """Hybrid classification: use OpenAI if local confidence is low"""

        local_result = self.classify_ticket_local(title, description)

        average_confidence = (local_result["category_confidence"] + local_result["priority_confidence"]) / 2

        if average_confidence < 0.7 and self.openai_client:
            try:
                openai_result = self.classify_with_openai(title, description)
                openai_result["method_used"] = "openai_hybrid"
                return openai_result
            except:
                # If OpenAI fails, use local result
                local_result["method_used"] = "local_fallback"
                return local_result
        else:
            local_result["method_used"] = "local_confident"
            return local_result

    def classify_ticket(self, title: str, description: str,
                        method: ClassificationMethod = ClassificationMethod.HYBRID) -> dict:
        """Classify ticket according to specified method"""
        if method == ClassificationMethod.LOCAL:
            result = self.classify_ticket_local(title, description)
            result["method_used"] = "local"

        elif method == ClassificationMethod.OPENAI:
            if not self.openai_client:
                # Fallback to local if OpenAI is not available
                result = self.classify_ticket_local(title, description)
                result["method_used"] = "local_no_openai"
            else:
                result = self.classify_with_openai(title, description)
                result["method_used"] = "openai"

        else:  # HYBRID
            result = self.classify_ticket_hybrid(title, description)

        return result

classifier = TicketClassifierService()







