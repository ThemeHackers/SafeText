# Threat Detection System with Fine-Tuning and Analysis

This project provides a web application and CLI tool for analyzing threats in text messages using a fine-tuned transformer model. It supports fine-tuning on custom datasets, threat-level analysis, pattern detection, and generating actionable recommendations.

---

## Features

- **Fine-Tuning**: Fine-tune a pre-trained transformer model (`distilbert-base-uncased-finetuned-sst-2-english`) on a custom dataset for threat detection.
- **Threat Analysis**: Analyze text messages for threats, classify threat levels, and detect patterns such as phishing, social engineering, malware, and impersonation.
- **Recommendations**: Generate actionable recommendations based on the threat score and detected patterns.
- **Web Interface**: Simple web interface for threat analysis and visualization.
- **CLI Support**: Command-line support for fine-tuning and threat analysis.
- **About threats.csv file**: You can customize this file yourself.
---

## Installation and Run

   ```bash
   git clone https://github.com/ThemeHackers/SafeText.git
   cd SafeText
   python3 -m venv .venv
   source .venv/bin/activate
   pip3 install -r requirements.txt
   pip install transformers[torch]
   python3 safetext.py --fine_tuning --datasets threats.csv
```
## The data format you want to know can be found by running this command
   ```bash
   python3 info.py
```
## File Structure
   ```bash
SafeText/
│
├── app.py                 # Main application file
├── templates/             # HTML templates for the web app
│   ├── home.html
│   ├── about.html
│   ├── statistics.html
|   └── start-model.html
├── requirements.txt       # Python dependencies
├── threats.csv            # Sample dataset for fine-tuning
├── README.md              # Documentation
├── .gitignore             # .gitignore file
└── fine_tuning_model/     # Folder for saving the fine-tuned model

```
