# ISY5004_ITSS_GC_Project_Team_20_Oil_Spill_Detection_System

## Project - Oil Spill Detection System with Synthetic Data Generation Pipeline and Dynamic Perceiver Architecture model 


# Oil Spill Detection System 🌊🛢️

## 📌 Project Overview  
This project aims to develop an **AI-based Oil Spill Detection System** using **Synthetic Aperture Radar (SAR) images**. The system enhances oil spill detection accuracy by leveraging **synthetic data generation** and the **Dynamic Perceiver architecture** for efficient image analysis.

## 🚀 Key Features  
✅ **Detection of oil spills from SAR images**  
✅ **Synthetic data generation** using **cDDPM (Conditional Denoising Diffusion Probabilistic Model)**  
✅ **High-accuracy model (>90%)** using **Dynamic Perceiver**  
✅ **Web-based interface**   
✅ **Optimized for Edge AI deployment**  

## 📊 Datasets  
- **Original SAR Dataset**: [CSIRO Sentinel-1 SAR Image Dataset](https://doi.org/10.25919/4v55-dn16)  
- **Synthetic Data**: Generated using **cDDPM**  
- **Combined Dataset**: Real + Synthetic images  

## 🏗️ Project Architecture  
1. **Data Collection & Preprocessing**
   - Acquire real SAR images  
   - Generate synthetic SAR images  
2. **Model Training**
   - Train **YOLOv4** as a baseline  
   - Implement **Dynamic Perceiver** for improved accuracy  
   - Evaluate performance using **IoU, Precision, Recall, FID**  
3. **Web-Based System**
   - Deploy an interface for uploading SAR images  
   - Display oil spill detection results  
4. **Edge AI Optimization**
   - Improve inference speed and reduce power consumption  

## 📜 Proposal Document  
The detailed project proposal is included in the repository:  
[📄 ISY5004_ITSS_GC_project_proposal.pptx](ISY5004_ITSS_GC_project_proposal_team_20_Sweta_Pattnaik.pptx)  

## 🔧 Installation & Usage  
### Clone the Repository  
```bash
git clone https://github.com/your-username/oil-spill-detection.git
cd oil-spill-detection
```

## 📚 References  

1. **CSIRO Sentinel-1 SAR Image Dataset** -  
   Blondeau-Patissier, D., Schroeder, T., Diakogiannis, F., & Li, Z. (2022).  
   CSIRO Sentinel-1 SAR image dataset of oil- and non-oil features for machine learning (Deep Learning).  
   [DOI: 10.25919/4v55-dn16](https://doi.org/10.25919/4v55-dn16)  

2. **SAR Image Synthesis with Diffusion Models** -  
   Research paper evaluating **DDPM vs. GANs** for SAR image generation.  
   [Arxiv Preprint](https://arxiv.org/pdf/2405.07776v1)  

3. **Dynamic Perceiver for Efficient Visual Recognition** -  
   A new adaptable **Perceiver-based** model for image recognition.  
   [GitHub Code Repository](https://github.com/LeapLabTHU/Dynamic_Perceiver)  

4. **YOLOv4 for Oil Spill Detection** -  
   Using **You Only Look Once (YOLOv4)** trained on Sentinel-1 SAR images.  
   - Copernicus Sentinel-1 Data for Oil Spill Detection  
   - [Sentinels Copernicus Case Study](https://sentinels.copernicus.eu/web/success-stories/-/copernicus-sentinel-1-data-enable-oil-spill-detection-in-south-eastern-mediterranean-sea)  

5. **Singapore Oil Spill Incident, June 2024** -  
   - [CNA News](https://www.channelnewsasia.com/singapore/oil-spill-slick-sentosa-beaches-east-coast-park-labrador-park-4412481)
   - ![image](https://github.com/user-attachments/assets/6c6b4f1e-bf11-4b2b-9f49-74f40736a729)

   - [Reuters News](https://www.reuters.com/pictures/pictures-singapore-intensifies-clean-up-oil-spill-spreads-along-coast-2024-06-17/)
   - ![image](https://github.com/user-attachments/assets/d895c827-04fe-4e03-8b25-02d2c6ce7351)




