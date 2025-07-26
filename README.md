# Laptop Price Prediction Using Machine Learning
A complete machine learning pipeline for predicting laptop prices using regression algorithms with hyperparameter tuning (Optuna), model evaluation, and an interactive Streamlit app for deployment.

## Hosted Web Application
This project is hosted online at the following link : 

[Laptop Price Predictor](https://laptop-price-prediction-using-machine-lcqf.onrender.com/)
## Project Overview 
This project focuses on building an end-to-end machine learning model to predict laptop prices based on hardware specifications like RAM, SSD, processor, GPU brand, Os etc. The model is trained, tuned, evaluated, and deployed using Streamlit for interactive usage.

<img width="1919" height="1019" alt="image" src="https://github.com/user-attachments/assets/9db6ea13-2dda-43c3-86ce-7360f46b9a51" />
<img width="1919" height="998" alt="image" src="https://github.com/user-attachments/assets/1d914258-d0ab-4baf-949c-112df5f4591c" />




## Objective
- Predict the price of laptops using various regression algorithms.
- Optimize and compare model performance using Bayesian Optimization via Optuna.
- Deploy the best-performing model using Streamlit for real-time predictions.

## Workflow
1. #### Data Cleaning & Preprocessing
   - Loaded dataset 
   - Handled missing values and irrelevant columns
   - Converted string specifications into usable numeric features
2. #### Feature Engineering
   - Extracted processor types, SSD sizes, GPU brands etc.
   - Created new features like "Total Storage", "Brand", "PPI" etc.
3. #### EDA
   - Visualized distributions and correlations using `matplotlib` and `seaborn`
   - Identified skewness, outliers, and feature importance
4. #### Model Building
     - Implemented and compared multiple regression algorithms:
     - Linear Regression
     - Ridge & Lasso Regression
     - K-Nearest Neighbors
     - Decision Tree
     - Random Forest
     - Extra Trees
     - Support Vector Regressor (SVR)
     - XGBoost Regressor
     - Voting Regressor
     - Stacking Regressor
5. #### Hyperparameter Tuning
      - Used `Optuna` for Bayesian Optimization
      - Searched for best model and optimal hyperparameters
6. #### Model Selection and Evaluation
      - Selected best model based on cross-validation R² scores
      - Achieved **R² Score: 0.88** using **Random Forest Regressor**
      - Used `cross_val_score` for validating stability
7. #### Deployment with Streamlit
      - Built a clean and interactive UI for predictions
      - Allows user to input laptop specs and get predicted price instantly

## Tech Stack
- **Languages**: Python
- **Libraries**: pandas, numpy, scikit-learn, seaborn, matplotlib, Optuna, xgboost
- **Model Deployment**: Streamlit

## Future Scope
- Improve model by incorporating more data sources or live pricing APIs
- Add categorical encoding techniques for better performance
- Include deep learning models (e.g., ANN) for comparison
- Containerize the application using Docker for robust deployment
