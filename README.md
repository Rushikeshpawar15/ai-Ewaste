# AI _Powered E-Waste Recycling Guide
AI-Powered E-Waste Recycling Guide
An intelligent system that leverages Natural Language Processing (NLP) and Image Classification (CNN) to identify electronic waste items, predict their category, and provide guidance on safe disposal, recoverable materials, and hazardous contents.
This project was developed as part of an AI-Sustainability Internship to promote responsible e-waste management through AI.
Features
Text Query (NLP Model)
Accepts user input like "iPhone", "mi charger", "HP laptop"
Predicts e-waste category (e.g., mobile phone, laptop, charger)
Returns disposal instructions, recoverable metals, and hazardous substances
Image Upload (CNN Model)
Accepts an uploaded image of an e-waste item
Classifies into one of 10 categories
Returns disposal guidance and hazard information
Interactive Web App (Flask + HTML/CSS/JS)
User-friendly interface with text input & image upload
Real-time predictions
Beautiful UI with clear results section
Project Overview
Introduction
E-waste is one of the fastest-growing waste streams in the world, containing both valuable recoverable materials and hazardous chemicals. Proper classification and disposal of e-waste are critical for sustainability and environmental safety.
This project integrates AI to make e-waste classification accessible, scalable, and interactive for the public.
Project Description
NLP Model (Text-based):
Trained using a dataset of 100+ variations for each e-waste category
Uses TF-IDF + Logistic Regression for classification
Outputs category, disposal method, materials, and hazards
CNN Model (Image-based):
Trained on an E-Waste Image Dataset (~2400 images, 10 classes)
Uses MobileNetV2 Transfer Learning
Achieves >95% test accuracy
Solution Offered
The system provides:
Automated Classification → via text queries or image uploads
Safe Disposal Guidance → how to dispose responsibly
Material Recovery Info → gold, copper, lithium, etc.
Hazard Awareness → lead, mercury, CFCs, etc.
End Users
Consumers: Dispose old devices responsibly
Recyclers: Quickly categorize items
Policymakers & NGOs: Promote sustainable e-waste handling
Students/Researchers: Educational and experimental tool
 Technology Used
Programming Language: Python
Libraries/Frameworks:
NLP: Scikit-learn, Pandas, NumPy
Deep Learning: TensorFlow, Keras (MobileNetV2)
Web App: Flask, HTML, CSS, JavaScript
Dataset:
Custom NLP dataset (CSV, text variations)
E-Waste Image Dataset (10 categories, 2400+ images)
Model Storage:
NLP model: ewaste_model.pkl
CNN image model: ewaste_image_model_final.h5/.keras
Metadata: ewaste_data.json
