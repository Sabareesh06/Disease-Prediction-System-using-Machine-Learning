import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import accuracy_score, classification_report


@st.cache
def load_data():
    df = pd.read_csv('heart.csv')
    return df


def preprocess_data(df):
    
    imputer = SimpleImputer(strategy='median')
    df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
    
    
    Q1 = df_imputed.quantile(0.25)
    Q3 = df_imputed.quantile(0.75)
    IQR = Q3 - Q1
    df_cleaned = df_imputed[~((df_imputed < (Q1 - 1.5 * IQR)) | (df_imputed > (Q3 + 1.5 * IQR))).any(axis=1)]
    
    
    scaler = StandardScaler()
    df_cleaned[df_cleaned.columns[:-1]] = scaler.fit_transform(df_cleaned[df_cleaned.columns[:-1]])
    
    return df_cleaned

def select_features(X, y):
    k_best = SelectKBest(score_func=f_classif, k=8)  
    X_selected = k_best.fit_transform(X, y)
    selected_features = k_best.get_support(indices=True)
    return X_selected, selected_features

def train_model(df):
    df_cleaned = preprocess_data(df)
    X = df_cleaned.drop(columns=['target'])
    y = df_cleaned['target']
    X_selected, selected_features = select_features(X, y)
    
    
    X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42)
    
    
    rf_model = RandomForestClassifier(random_state=42)
    rf_model.fit(X_train, y_train)
    
    return rf_model, selected_features

def main():
    st.title('Heart Disease Prediction')
    st.write("This app predicts the presence of heart disease based on various medical attributes.")
    
    
    df = load_data()
    
    
    rf_model, selected_features = train_model(df)

    
    st.subheader('Enter the features for prediction:')
    user_input = []
    
   
    for feature in selected_features:
        user_input.append(st.number_input(f"Feature {feature}:", value=0.0))

    
    if st.button('Predict'):
        input_data = pd.DataFrame([user_input], columns=[df.columns[i] for i in selected_features])
        input_data_scaled = StandardScaler().fit_transform(input_data)  
        prediction = rf_model.predict(input_data_scaled)
        st.write(f"Prediction: {'Heart Disease' if prediction[0] == 1 else 'No Heart Disease'}")

if __name__ == '__main__':
    main()
