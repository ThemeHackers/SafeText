import argparse
from flask import Flask, render_template, request
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, get_scheduler
from datasets import Dataset
import pandas as pd
import torch
import re
import art
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch.optim import AdamW
from torch.utils.data import DataLoader

app = Flask(__name__)

def fine_tune_model(dataset_path="/home/kali/SafeText/threats.csv"):

    df = pd.read_csv(dataset_path)
    df.to_csv("threats.csv", index=False)
    print("CSV file saved!")
    datasets = Dataset.from_pandas(df).train_test_split(test_size=0.2)
    train_datasets = datasets['train']
    test_datasets = datasets['test']

 
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")

    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, padding="max_length")

    tokenized_train_datasets = train_datasets.map(tokenize_function, batched=True)
    tokenized_test_datasets = test_datasets.map(tokenize_function, batched=True)

 
    model = AutoModelForSequenceClassification.from_pretrained(
        "distilbert-base-uncased-finetuned-sst-2-english", num_labels=2
    )


    training_args = TrainingArguments(
        output_dir="./results",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=10,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        fp16=True, 
        gradient_accumulation_steps=2,  
    )


    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = logits.argmax(axis=-1)
        precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='binary')
        acc = accuracy_score(labels, predictions)
        return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}


    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=tokenized_train_datasets,
        eval_dataset=tokenized_test_datasets,
        compute_metrics=compute_metrics,
    )


    trainer.train()


    results = trainer.evaluate(tokenized_test_datasets)

    
    print(f"Test Accuracy: {results['eval_accuracy']}")


    model.save_pretrained("./fine_tuning_model")
    tokenizer.save_pretrained("./fine_tuning_model")
    print("Model fine-tuned and saved successfully!")


class ThreatAnalyzer:
    def __init__(self):
        model_path = './fine_tuning_model'
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)

    def analyze_threat(self, input_text):
        with torch.no_grad():
            inputs = self.tokenizer(input_text, return_tensors="pt", truncation=True, padding=True)
            outputs = self.model(**inputs)
            probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)  
            threat_score = probabilities[0][1].item()  

        threat_patterns = {
            "phishing": re.compile(r"\b(verify|urgent|suspend|account|click here)\b", re.IGNORECASE),
            "social_engineering": re.compile(r"\b(tech support|password|credentials|urgent)\b", re.IGNORECASE),
            "malware": re.compile(r"\b(download|patch|update|critical)\b", re.IGNORECASE),
            "impersonation": re.compile(r"\b(admin|support|verify identity)\b", re.IGNORECASE)
        }

        detected_patterns = [threat_type for threat_type, pattern in threat_patterns.items() if pattern.search(input_text)]

        recommendations = self._generate_recommendations(threat_score, detected_patterns)

        return {
            "threat_score": threat_score,
            "threat_level": self._classify_threat_level(threat_score),
            "detected_patterns": detected_patterns,
            "recommendations": recommendations,
            "model_probabilities": probabilities[0].tolist()  
        }

    def _generate_recommendations(self, threat_score, detected_patterns):
        recommendations = []
        if threat_score > 0.7:
            recommendations.extend([ 
                "🚨 High Risk: Do NOT interact with this message",
                "Immediately report to IT security",
                "Do not click any links or download attachments"
            ])
        elif threat_score > 0.4:
            recommendations.extend([ 
                "⚠️ Potential Threat Detected",
                "Verify the source independently",
                "Contact sender through official channels"
            ])

        if "phishing" in detected_patterns:
            recommendations.append("Phishing Indicator: Check sender's email address carefully")
        if "social_engineering" in detected_patterns:
            recommendations.append("Social Engineering Alert: Never share personal credentials")
        if "malware" in detected_patterns:
            recommendations.append("Malware Risk: Scan with updated antivirus before opening")

        return recommendations

    def _classify_threat_level(self, score):
        if score > 0.8: return "CRITICAL THREAT"
        if score > 0.6: return "HIGH THREAT"
        if score > 0.4: return "MODERATE THREAT"
        if score > 0.2: return "LOW THREAT"
        return "NO SIGNIFICANT THREAT"
    
    def calculate_dataset_statistics(self, dataset):
        
        df = pd.DataFrame(dataset)
        summary_stats = df.describe()  
        return summary_stats

    def calculate_mean_std(self, dataset):
        
        df = pd.DataFrame(dataset)
        mean = df.mean()
        std_dev = df.std()
        return mean, std_dev

    def calculate_expected_value(self, dataset):
     
        df = pd.DataFrame(dataset)
        expected_value = df.mean(axis=0)  
        return expected_value

@app.route("/")
def home():

    return render_template("home.html", result=None)

@app.route("/about")

def about():
    return render_template("about.html")

@app.route("/statistics")

def statistics():
    return render_template("statistics.html")

@app.route("/start-model", methods=["GET", "POST"])
def start_model():
    if request.method == "POST":
        input_text = request.form["inputText"]
        analyzer = ThreatAnalyzer()
        result = analyzer.analyze_threat(input_text)
        ascii_art = art.text2art(result["threat_level"], "block")

       
        result_metrics = {
            "accuracy": result.get("accuracy"),
            "f1": result.get("f1"),
            "precision": result.get("precision"),
            "recall": result.get("recall")
        }

        return render_template(
            "start-model.html",
            result=result,
            input_text=input_text,
            detected_patterns=", ".join(result["detected_patterns"]) or "No specific patterns",
            ascii_art=ascii_art,
            result_metrics=result_metrics
        )
    return render_template("start-model.html", result=None)

def run_cli(args):
    if args.input_text:
        analyzer = ThreatAnalyzer()
        result = analyzer.analyze_threat(args.input_text)
        print(f"Threat Level: {result['threat_level']}")
        print(f"Detected Patterns: {', '.join(result['detected_patterns'])}")
        print(f"Recommendations:\n{chr(10).join(result['recommendations'])}")
    else:
        print("Error: Please provide text for analysis.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze potential threats in messages or fine-tune the model.")
    parser.add_argument("--cli", action="store_true", help="Run the CLI version.")
    parser.add_argument("--input_text", type=str, help="Suspicious text or message to analyze.")
    parser.add_argument("--fine_tuning", action="store_true", help="Fine-tune the model with a dataset.")
    parser.add_argument("--datasets", type=str, help="Path to the dataset for fine-tuning (CSV format).")

    args = parser.parse_args()

    if args.fine_tuning:
        dataset_path = args.datasets or "threats.csv"
        fine_tune_model(dataset_path)
    elif args.cli:
        run_cli(args)
    else:
        app.run(host="0.0.0.0", debug=True)
