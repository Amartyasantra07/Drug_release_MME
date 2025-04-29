import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
import joblib
import streamlit as st
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import plotly.express as px
from io import StringIO

# Load the dataset
df = pd.read_csv('taguchi1.csv')

# Prepare the data
X = df.drop(columns=['Run ', 'perc of Drug Release'])
y = df['perc of Drug Release']

# Data preprocessing
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Data augmentation using polynomial features
poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X_scaled)

# Hyperparameter tuning using GridSearchCV
param_grid = {'fit_intercept': [True, False]}
grid_search = GridSearchCV(LinearRegression(), param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(X_poly, y)

# Best model from GridSearchCV
best_model = grid_search.best_estimator_

# Cross-validation scores
cv_scores = cross_val_score(best_model, X_poly, y, cv=5, scoring='neg_mean_squared_error')

# Save the best model and preprocessing objects
joblib.dump(best_model, "mlr_model.pkl")
joblib.dump(poly, "poly_features.pkl")
joblib.dump(scaler, "scaler.pkl")

# Load trained model and preprocessing objects
loaded_model = joblib.load("mlr_model.pkl")
poly = joblib.load("poly_features.pkl")
scaler = joblib.load("scaler.pkl")

# Streamlit UI
def main():
    st.title("📊 Drug Release Prediction and Optimization System")
    st.write("""
    This application predicts drug release percentage based on formulation parameters 
    and recommends optimal conditions for maximum/minimum release.
    """)
    
    # Add specifications section
    st.sidebar.title("⚙️ System Specifications")
    st.sidebar.write("""
    - **Input Parameters**: Time(min), Drug Concentration(Mg), RPM, pH, Temperature
    - **Output**: Drug Release Percentage (%)
    - **Models Available**: Enhanced MLR, Decision Tree, Random Forest
    - **Features**: Prediction, Optimization, Visualization, Export
    - **Data Points**: 27 experimental runs
    - **Accuracy**: R² > 0.85 (MLR model)
    - **Validation**: 5-fold cross-validation
    """)
    
    # Initialize model with the loaded model
    model = loaded_model
    
    # Model selection
    model_option = st.selectbox("Select Prediction Model", 
                               ["Enhanced MLR", "Decision Tree", "Random Forest"])
    
    # If model changes, retrain
    if model_option != "Enhanced MLR":
        if model_option == "Decision Tree":
            model = DecisionTreeRegressor(max_depth=5)
        else:
            model = RandomForestRegressor(n_estimators=100, max_depth=5)
        model.fit(X_poly, y)
    
    tab1, tab2, tab3, tab4 = st.tabs(["🔮 Prediction", "📈 Visualization", "⚡ Optimization", "📤 Export"])
    
    with tab1:
        st.header("Drug Release Prediction")
        col1, col2 = st.columns(2)
        
        with col1:
            # User inputs with default values set to medians
            input_values = []
            for col in X.columns:
                median_val = float(X[col].median())
                value = st.number_input(
                    f"{col}", 
                    min_value=float(X[col].min()), 
                    max_value=float(X[col].max()),
                    value=median_val,
                    step=1.0 if col in ['Rpm', 'Temperature'] else 0.1
                )
                input_values.append(value)
        
        if st.button("Predict Drug Release"):
            input_values = np.array(input_values).reshape(1, -1)
            input_values_scaled = scaler.transform(input_values)
            input_values_poly = poly.transform(input_values_scaled)
            prediction = model.predict(input_values_poly)[0]
            
            # Display prediction with confidence interval
            y_pred = model.predict(X_poly)
            residuals = y - y_pred
            std_dev = np.std(residuals)
            
            st.success(f"""
            **Predicted Drug Release:** {prediction:.2f}%  
            **Confidence Interval (95%):** {prediction-1.96*std_dev:.2f}% to {prediction+1.96*std_dev:.2f}%
            """)
            
            # Model evaluation
            mse = mean_squared_error(y, y_pred)
            r2 = r2_score(y, y_pred)
            
            st.subheader("Model Performance")
            col_metric1, col_metric2, col_metric3 = st.columns(3)
            with col_metric1:
                st.metric("R² Score", f"{r2:.3f}")
            with col_metric2:
                st.metric("MSE", f"{mse:.3f}")
            with col_metric3:
                st.metric("CV MSE", f"{-np.mean(cv_scores):.3f}")
            
            # Feature importance (for tree-based models)
            if model_option != "Enhanced MLR":
                st.subheader("Feature Importance")
                if hasattr(model, 'feature_importances_'):
                    importance = model.feature_importances_
                    features = poly.get_feature_names_out(X.columns)
                    fig_imp = px.bar(x=features, y=importance, 
                                    labels={'x':'Features', 'y':'Importance'},
                                    title='Feature Importance Scores')
                    st.plotly_chart(fig_imp)
    
    with tab2:
        st.header("Data Visualization")
        
        # 3D Scatter plot
        st.subheader("3D Parameter Space Exploration")
        col_x, col_y, col_z = st.columns(3)
        with col_x:
            x_axis = st.selectbox("X-axis", X.columns, index=0)
        with col_y:
            y_axis = st.selectbox("Y-axis", X.columns, index=1)
        with col_z:
            z_axis = st.selectbox("Z-axis (color)", ['perc of Drug Release'] + list(X.columns), index=0)
        
        fig_3d = px.scatter_3d(df, x=x_axis, y=y_axis, z='perc of Drug Release',
                              color=z_axis, hover_name='Run ',
                              title=f"3D View: {x_axis} vs {y_axis} vs Drug Release")
        st.plotly_chart(fig_3d)
        
        # Actual vs Predicted plot
        st.subheader("Model Performance Visualization")
        y_pred = model.predict(X_poly)
        fig_perf = px.scatter(x=y, y=y_pred, 
                             labels={'x':'Actual Drug Release', 'y':'Predicted Drug Release'},
                             
                             title="Actual vs Predicted Drug Release")
        fig_perf.add_shape(type="line", x0=y.min(), y0=y.min(), x1=y.max(), y1=y.max(),
                          line=dict(color="Red", width=2, dash="dash"))
        st.plotly_chart(fig_perf)
    
    with tab3:
        st.header("Optimal Parameter Recommendation")
        st.write("Find parameters that maximize or minimize drug release based on the model.")
        
        target = st.radio("Optimization Target", ["Maximize Drug Release", "Minimize Drug Release"])
        
        if st.button("Find Optimal Parameters"):
            # Simple grid search for optimization (in real app, use more sophisticated method)
            n_samples = 1000
            random_samples = pd.DataFrame({
                'Time(min)': np.random.uniform(X['Time(min)'].min(), X['Time(min)'].max(), n_samples),
                'Drug_con(Mg)': np.random.uniform(X['Drug_con(Mg)'].min(), X['Drug_con(Mg)'].max(), n_samples),
                'Rpm': np.random.uniform(X['Rpm'].min(), X['Rpm'].max(), n_samples),
                'pH': np.random.uniform(X['pH'].min(), X['pH'].max(), n_samples),
                'Temperature': np.random.uniform(X['Temperature'].min(), X['Temperature'].max(), n_samples)
            })
            
            random_samples_scaled = scaler.transform(random_samples)
            random_samples_poly = poly.transform(random_samples_scaled)
            predictions = model.predict(random_samples_poly)
            
            if "Maximize" in target:
                optimal_idx = np.argmax(predictions)
            else:
                optimal_idx = np.argmin(predictions)
                
            optimal_params = random_samples.iloc[optimal_idx]
            optimal_value = predictions[optimal_idx]
            
            st.success(f"**Optimal Parameters for {target}:**")
            st.write(optimal_params)
            st.metric("Predicted Drug Release", f"{optimal_value:.2f}%")
    
    with tab4:
        st.header("Export Results")
        st.write("Download predictions or model details.")
        
        # Export predictions
        y_pred = model.predict(X_poly)
        results_df = df.copy()
        results_df['Predicted Release'] = y_pred
        results_df['Residual'] = y - y_pred
        
        csv = results_df.to_csv(index=False)
        st.download_button(
            label="Download Predictions (CSV)",
            data=csv,
            file_name='drug_release_predictions.csv',
            mime='text/csv'
        )
        
        # Export model details
        if st.button("Export Model Summary"):
            buffer = StringIO()
            buffer.write(f"Drug Release Prediction Model Summary\n")
            buffer.write(f"Model Type: {model_option}\n")
            buffer.write(f"R² Score: {r2_score(y, y_pred):.4f}\n")
            buffer.write(f"MSE: {mean_squared_error(y, y_pred):.4f}\n\n")
            buffer.write("Feature Importance:\n")
            
            if hasattr(model, 'feature_importances_'):
                features = poly.get_feature_names_out(X.columns)
                for feat, imp in zip(features, model.feature_importances_):
                    buffer.write(f"{feat}: {imp:.4f}\n")
            
            st.download_button(
                label="Download Model Summary",
                data=buffer.getvalue(),
                file_name='model_summary.txt',
                mime='text/plain'
            )

if __name__ == "__main__":
    main()