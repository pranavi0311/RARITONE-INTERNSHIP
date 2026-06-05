# RARITONE-INTERNSHIP

**AI/ML Intern Final Submission - Virtual Try-On & Fashion Recommendation System**

##  Project Overview
AI-powered virtual try-on system to reduce fashion e-commerce returns. The project combines 2D garment warping, body landmark detection, and personalized recommendations to help users visualize clothes before purchase.

##  Key Contributions

### 1. Research Work
- **2D Virtual Try-On:** Comparative study of TPS, CP-VTON, VITON-HD, GMM for garment warping
- **AI Recommendation Systems:** Content-Based, Collaborative & Hybrid filtering techniques
- **AR Technologies:** Feasibility study for AR integration in fashion apps
- **Dataset Analysis:** VITON-HD, DeepFashion, Dress Code - pros, cons & use cases

### 2. Implementation Work
- **Body Landmark Detection:** Used MediaPipe to identify 33 pose keypoints for shoulder, hip, waist positions
- **Pose Detection & Skeleton Tracing:** Tested MediaPipe Pose on multiple images with varying poses/lighting
- **Body Measurement Estimation:** Extracted shoulder width, height, waist position from landmarks
- **Accuracy Analysis:** Evaluated detection performance across different body types and poses
- **2D Virtual Try-On:** Prototyped 2D try-on by warping cloth image to person image using TPS
- **3D Avatar Generation:** Explored 3D avatar generation using user provided image

## Tech Stack
**Languages:** Python  
**Libraries:** MediaPipe, OpenCV, NumPy, TensorFlow  
**Models Studied:** CP-VTON, VITON-HD, SMPL, GMM  
**Tools:** GitHub, VSCode, Google Colab

## ⚠️ Challenges Faced & Solutions

### 1. **Lighting & Image Quality Issues**
**Challenge:** MediaPipe Pose landmarks were inaccurate or completely missed in low-light photos and blurry images. Shoulder/hip points would jitter or shift by 20+ pixels.  
**Solution:** Added preprocessing step using OpenCV. Applied CLAHE for contrast adjustment and Gaussian blur for noise reduction. Also documented best practice: "Take photos in daylight, full body visible, plain background" for users.

### 2. **Occlusion in Complex Poses**
**Challenge:** When user’s arms crossed the torso or hands were on waist, MediaPipe mislabeled landmarks. Waist keypoint would snap to elbow, breaking body measurement calculation.  
**Solution:** Filtered extreme poses during input validation. Used confidence score threshold >0.7 from MediaPipe. For failed detections, prompted user: "Please stand straight with arms at sides".

### 3. **Inconsistent Body Measurements**
**Challenge:** Pixel distances from landmarks varied based on user’s distance from camera. Shoulder width in pixels was meaningless without scale reference.  
**Solution:** Implemented ratio-based measurements instead of absolute values. Used "shoulder_width / image_height" ratio to normalize. For future: Plan to add reference object like A4 sheet for cm conversion.

### 4. **2D Garment Warping Artifacts**
**Challenge:** TPS warping with OpenCV caused severe stretching for loose clothes like kurtas or dresses. Texture and logos got distorted unnaturally.  
**Solution:** Identified this as limitation of geometric warping. Researched and documented VITON-HD as solution - it uses deep learning + ALIAS generator to preserve texture. Marked TPS for real-time preview only.

##  Future Improvements
1. **3D Try-On:** Integrate SMPL or PIFuHD for 360° avatar view
2. **Real-time Mobile:** Replace MediaPipe with MoveNet for faster inference on phones
3. **Size Prediction:** Auto-recommend S/M/L/XL based on extracted measurements + brand size charts
4. **Indian Dataset:** Collect dataset with Indian body types and ethnic wear for fine-tuning
5. **AR Integration:** "View in Room" feature using ARCore/ARKit for outfit placement

##  Deliverables Checklist
1. **Research Documentation** 
2. **Code Implementation** 
3. **Results & Screenshots** 
4. **Architecture Diagrams** 
5. **Final Presentation** 


---
**Note:** This repository contains research findings and prototype code developed during the AI/ML Internship at Raritone. For production deployment, VITON-HD fine-tuning and mobile optimization are recommended.
