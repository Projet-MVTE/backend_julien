################################################################################################################################
############################### IA Code ########################################################################################
################################################################################################################################

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split
import sklearn.preprocessing
from imblearn.over_sampling import SMOTENC
import xgboost as xgb
from sklearn.metrics import precision_recall_curve
import lime
import lime.lime_tabular

def preprocess_new_sample(new_sample: dict):
    # Data Understanding
    column_names = ['Colonne1', 'IDENT', 'SEXE', 'DATMVTE', 'POIDS', 'TAILLE', 'IMC', 'AGEDIAG', 'AGEDIAG_cl',
                'MVTE_INITIALE_cl', 'DUREE_TTT', 'DUREE_TTT_cl', 'POSTOP', 'PLATRE', 'GROSSESSE', 'POSTPART',
                'HOSPM3', 'VOYAGE', 'CO', 'THS', 'ATCDMVT', 'ATCDFAM', 'MALADIE_INFLAM', 'avcISCHEMIQUE',
                'avecHEMORRAGIQUE', 'Pneumopathie_interstitielle', 'bpco', 'ATCD_HYPOTHYR', 'ATCD_HYPERTHYR',
                'ATCD_RENAL', 'ATCD_CARDIOPATH_ISCHEMIQUE', 'ATCD_INSUFHEP_CHRQ', 'ATCD_CARDIOPATH_RYTHMIQUE',
                'CANCER', 'FVL', 'FII_G20210A', 'RISK_FACTOR', 'RECIDE_VTE', 'exposition_risque_annees']

    # Load the data into a DataFrame
    file_path = 'donnees_medecins.csv'
    df = pd.read_csv(file_path, names=column_names, skiprows=1)

    numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns

    df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric, errors='coerce')

    # Replace 'NA' values with NaN
    df.replace("NA", np.nan, inplace=True)

    # Handle missing numeric values
    df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].median())

    # Handle missing categorical values with forward fill
    cat_cols = df.select_dtypes(include=['object', 'category']).columns
    df[cat_cols] = df[cat_cols].ffill()

    # Apply conditionals for 'SEXE' based columns
    df['GROSSESSE'] = df['GROSSESSE'].where(df['SEXE'] != '1',0)
    df['POSTPART'] = df['POSTPART'].where(df['SEXE'] != '1', 0)
    df['THS'] = df['THS'].where(df['SEXE'] != '1',0)
    df['CO'] = df['CO'].where(df['SEXE'] != '1', 0)
    # for male in the ihm it should be put 0 in the new input
    # Drop columns with too many missing values or irrelevant for modeling
    df.drop(columns=['FVL', 'FII_G20210A', 'Colonne1', 'IDENT', 'DATMVTE'], inplace=True)

    # Separate features and target
    X = df.drop(columns='RECIDE_VTE')
    y = df['RECIDE_VTE']

    # Transform gender-specific features based on 'SEXE'
    gender_specific_cols = ['GROSSESSE', 'POSTPART', 'THS', 'CO']
    for col in gender_specific_cols:
        df[col] = df[col] * (df['SEXE'] == '2').astype(int)

    # Identify categorical and numeric columns
    categorical_cols = X.select_dtypes(include=['object']).columns
    numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns

    

    # Convert all categorical columns to string type to avoid mixed types
    for col in categorical_cols:
        X[col] = X[col].astype(str)

    # Remove classes with fewer than 2 samples
    class_distribution = y.value_counts()
    valid_classes = class_distribution[class_distribution > 1].index
    X = X[y.isin(valid_classes)]
    y = y[y.isin(valid_classes)]




    # Dictionary to store LabelEncoders for each categorical column
    label_encoders = {}

    for col in categorical_cols:
        le = sklearn.preprocessing.LabelEncoder()
        X[col] = X[col].astype(str)  # Ensure all categorical values are strings
        X[col] = le.fit_transform(X[col])  # Fit and transform training data
        label_encoders[col] = le  # Store the trained encoder


    


    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    categorical_feature_indices = [X_train.columns.get_loc(col) for col in categorical_cols]

    # Apply SMOTENC with the appropriate categorical features
    smote_nc = SMOTENC(sampling_strategy=0.4, categorical_features=categorical_feature_indices, random_state=42)
    X_train_resampled, y_train_resampled = smote_nc.fit_resample(X_train, y_train)
   

    # Apply RobustScaler instead of StandardScaler
    from sklearn.preprocessing import RobustScaler

    robust_scaler = RobustScaler()
    X_train_balanced = robust_scaler.fit_transform(X_train_resampled)
    x_test_untouched = X_test
    X_test = robust_scaler.transform(X_test)

    # Data Understanding
    feature_names = [ 'SEXE','POIDS', 'TAILLE','IMC', 'AGEDIAG','AGEDIAG_cl',
                    'MVTE_INITIALE_cl', 'DUREE_TTT', 'DUREE_TTT_cl', 'POSTOP', 'PLATRE', 'GROSSESSE', 'POSTPART',
                    'HOSPM3', 'VOYAGE', 'CO', 'THS', 'ATCDMVT', 'ATCDFAM', 'MALADIE_INFLAM', 'avcISCHEMIQUE',
                    'avecHEMORRAGIQUE', 'Pneumopathie_interstitielle', 'bpco', 'ATCD_HYPOTHYR', 'ATCD_HYPERTHYR',
                    'ATCD_RENAL', 'ATCD_CARDIOPATH_ISCHEMIQUE', 'ATCD_INSUFHEP_CHRQ', 'ATCD_CARDIOPATH_RYTHMIQUE',
                    'CANCER','RISK_FACTOR','exposition_risque_annees']

    # Convert the numpy array to DataFrame
    df_X_train_balanced = pd.DataFrame(X_train_balanced, columns=feature_names)
    df_y_train_balanced = pd.DataFrame(y_train_resampled, columns=['RECIDE_VTE'])
    df_X_test = pd.DataFrame(X_test, columns=feature_names)
    df_y_test = pd.DataFrame(y_test, columns=['RECIDE_VTE']) 
    df_X_test_untouched = pd.DataFrame(x_test_untouched, columns=feature_names)


    df_X_train_balanced.to_csv('data/X_train_balanced_server.csv', index=False)
    df_y_train_balanced.to_csv('data/y_train_balanced_server.csv', index=False)
    df_X_test.to_csv('data/X_test_server.csv', index=False)
    df_y_test.to_csv('data/y_test_server.csv', index=False)
    df_X_test_untouched.to_csv('data/X_test_untouched_server.csv', index=False)

    new_sample_df = pd.DataFrame([new_sample])
    df_original_sample=new_sample_df
    df_original_sample.to_csv('data/original_sample.csv',columns=feature_names)
    # Convert all categorical columns to string type to avoid mixed types
    for col in categorical_cols:
        new_sample_df[col] = new_sample_df[col].astype(str)
    
    for col in categorical_cols:
        le = label_encoders[col]  # Use the correct encoder for this column

        # Check for unseen values and replace with most frequent category
        known_classes = set(le.classes_)
        if new_sample_df[col].iloc[0] not in known_classes:
             print(f"⚠️ Warning: {new_sample_df[col].iloc[0]} not in trained LabelEncoder for {col}")
    
        # Apply encoding
        new_sample_df[col] = new_sample_df[col].apply(lambda x: x if x in known_classes else le.classes_[0])
        new_sample_df[col] = le.transform(new_sample_df[col])

    new_sample_df = robust_scaler.transform(new_sample_df)
    return new_sample_df



def predict_and_explain():
    # Load the balanced training features and targets
    X_train_balanced = pd.read_csv('data/X_train_balanced_server.csv')
    y_train_balanced = pd.read_csv('data/y_train_balanced_server.csv')
    y_train_balanced = y_train_balanced.values.ravel()

    # Load the testing features and targets
    X_test = pd.read_csv('data/X_test_server.csv')
    y_test = pd.read_csv('data/y_test_server.csv')
    x_test_untouched = pd.read_csv('data/X_test_untouched_server.csv')

    # Load the preprocessed new sample
    new_sample_df = pd.read_csv('data/new_sample_server.csv')

    # Load the original new sample before preprocessing
    original_sample_df = pd.read_csv('data/original_sample.csv')
    original_values = original_sample_df.iloc[0].to_dict()  # Convert first row to dictionary

    # Define optimized XGBoost model with manually tuned hyperparameters
    optimized_modelXGB = xgb.XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        gamma=1,
        reg_alpha=0.1,
        reg_lambda=1,
        random_state=42,
        use_label_encoder=False,
        eval_metric='logloss'
    )

    # Train the optimized model
    optimized_modelXGB.fit(X_train_balanced, y_train_balanced)

    # Predict probabilities
    y_pred_probs = optimized_modelXGB.predict_proba(X_test)[:, 1]

    # Compute precision-recall values for different thresholds
    precision, recall, thresholds = precision_recall_curve(y_test, y_pred_probs)

    # Find the threshold that gives the best F1-score (balance of precision & recall)
    f1_scores = 2 * (precision * recall) / (precision + recall)
    best_threshold = thresholds[np.argmax(f1_scores)]

    print(f"Best Threshold for F1-Score: {best_threshold:.2f}")

    # Apply the best threshold
    y_pred_optimized = (y_pred_probs >= best_threshold).astype(int)
    # Predict probability
    sample_prob = optimized_modelXGB.predict_proba(new_sample_df)[:, 1]
    
    # Apply threshold to get final prediction
    sample_prediction = int(sample_prob >= best_threshold)
    
    print(f"Predicted Probability: {sample_prob[0]:.4f}")
    print(f"Final Prediction (Threshold {best_threshold:.2f}): {sample_prediction}")
    
    # Generate LIME explanation
    explainer = lime.lime_tabular.LimeTabularExplainer(
        X_train_balanced.to_numpy(),
        feature_names=X_train_balanced.columns.tolist(),
        class_names=['No Recurrence (0)', 'Recurrence (1)'],
        mode='classification'
    )
    
    explanation = explainer.explain_instance(new_sample_df.iloc[0].to_numpy(), optimized_modelXGB.predict_proba, num_features=20)
    
    # Modify table values to show original feature values
    feature_map = explanation.as_map()[1]  # Get feature importance mapping
    updated_values = []
    
    for feature_idx, importance in feature_map:
        feature_name = X_train_balanced.columns[feature_idx]  # Get feature name
        original_value = original_values.get(feature_name, "N/A")  # Fetch original value
        updated_values.append((feature_name, original_value))
    
    # Override LIME's table data with original values
    explanation.domain_mapper.feature_values = [original_values.get(f, "N/A") for f in X_train_balanced.columns]
    
    # Save updated LIME explanation
    explanation.save_to_file("lime_explanation_with_original_values.html")
    
    return sample_prediction, "lime_explanation_with_original_values.html"

feature_names = [ 'SEXE','POIDS', 'TAILLE','IMC', 'AGEDIAG','AGEDIAG_cl',
                'MVTE_INITIALE_cl', 'DUREE_TTT', 'DUREE_TTT_cl', 'POSTOP', 'PLATRE', 'GROSSESSE', 'POSTPART',
                'HOSPM3', 'VOYAGE', 'CO', 'THS', 'ATCDMVT', 'ATCDFAM', 'MALADIE_INFLAM', 'avcISCHEMIQUE',
                'avecHEMORRAGIQUE', 'Pneumopathie_interstitielle', 'bpco', 'ATCD_HYPOTHYR', 'ATCD_HYPERTHYR',
                'ATCD_RENAL', 'ATCD_CARDIOPATH_ISCHEMIQUE', 'ATCD_INSUFHEP_CHRQ', 'ATCD_CARDIOPATH_RYTHMIQUE',
                'CANCER','RISK_FACTOR','exposition_risque_annees']

################################################################################################################################
############################### Server Code ####################################################################################
################################################################################################################################

from flask import Flask, jsonify, send_file, request, render_template
import os
import re
import json

app = Flask(__name__)

# Définir une route pour la racine "/"
@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "IA-MVTE Application, developped at ENSTA Bretagne in 2024 by Dina, Johan, Julien, Jawad and Naim"}), 200
    
# Route pour télécharger un fichier JSON
@app.route("/download_tree", methods=["GET"])
def download_json():
    json_file_path = "arbre.json"
    try:
        return send_file(json_file_path, as_attachment=True, mimetype='application/json')
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/set_tree', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return "No file part", 400

    file = request.files['file']

    if file.filename == '':
        return "No selected file", 400

    file.save("arbre.json")  # Enregistre directement dans la racine
    return "File uploaded successfully", 200

@app.route('/askIA', methods=['GET'])
def askIA():
    try :
        # Récupération des paramètres de la requête
        gender = request.args.get('gender', type=str)
        if gender=="M":
            SEXE = int(1)
        elif gender=="F":
            SEXE = int(2)
        else :
            return jsonify({"error":"Gender not valid, "+str(gender)})
        
        weight = request.args.get('weight', type=float)
        POIDS = float(min(max(39, weight),165))
    
        height = request.args.get('height', type=float)
        TAILLE = float(min(max(142,height),198))
        
        ageDiagnostic = request.args.get('ageDiagnostic', type=float)
        AGEDIAG = float(min(max(18.07,ageDiagnostic),96.9))
        if 0<AGEDIAG<=50:
            AGEDIAG_cl = '(0,50]'
        elif 50<AGEDIAG<=65:
            AGEDIAG_cl = '(50,65]'
        elif 65<AGEDIAG<=100:
            AGEDIAG_cl = '(65,100]'
        else:
            return jsonify({"error":"ageDiagnostic not valid, "+str(ageDiagnostic)})
        
        anticoagulantDuration = request.args.get('anticoagulantDuration', type=float)
        DUREE_TTT = int(min(max(90,anticoagulantDuration),301))
        if 90<DUREE_TTT<=180:
            DUREE_TTT_cl = "90-180"
        elif 180<DUREE_TTT<=360:
            DUREE_TTT_cl = "180-360"
        elif 360<DUREE_TTT:
            DUREE_TTT_cl = ">360"
        else :
            return jsonify({"error":"anticoagulantDuration not valid, "+str(anticoagulantDuration)})
        
        diagnosticAgeCategory = request.args.get('diagnosticAgeCategory', type=str)
        
        mvteType = request.args.get('mvteType', type=str)
        if mvteType=="EP+TVP":
            MVTE_INITIALE_cl = "EP+TVP"
        elif mvteType=="TVP seule":
            MVTE_INITIALE_cl = "TVPseule"
        elif mvteType=="EP seule":
            MVTE_INITIALE_cl = "EPseule"
        else :
            return jsonify({"error":"mvteType not valid, "+str(mvteType)})
        
        anticoagulantDurationCategory = request.args.get('anticoagulantDurationCategory', type=str)
        
        chronicInflammatoryDisease = request.args.get('chronicInflammatoryDisease', type=str)
        if chronicInflammatoryDisease=="1":
            MALADIE_INFLAM = 1
        elif chronicInflammatoryDisease=="0":
            MALADIE_INFLAM = 0
        else:
            return jsonify({"error":"chronicInflammatoryDisease not valid, "+str(chronicInflammatoryDisease)})

        chirurgie, platre, grossesse, post_partum, hospitalisation, voyage, contraception, menopause, antecedent_MVTE, familial_MVTE, AVC_isch, AVC_hem, pneumopathie, BPCO,hypothyroidie, hyperthyroidie, maladie_renale, cardiopathie_coronarienne, hepatique, rythmique, cancer =  "chirurgie", "platre", "grossesse", "post_partum", "hospitalisation", "voyage", "contraception", "menopause", "antecedent_MVTE", "familial_MVTE", "AVC_isch", "AVC_hem", "pneumopathie", "BPCO","hypothyroidie", "hyperthyroidie", "maladie_renale", "cardiopathie_coronarienne", "hepatique", "rythmique", "cancer"     
        riskFactorsList = request.args.get('riskFactorsList', type=str)
        riskFactorsList = re.sub(r'(\w+):', r'"\1":', riskFactorsList)
        riskFactorsList = re.sub(r'\b(true|false)\b', r'"\1"', riskFactorsList)
        dic = json.loads(riskFactorsList)
        
        """
        dic = ast.literal_eval(riskFactorsList)

        if dic["chirurgie"]=="true":
            POSTOP=1
        elif dic["chirurgie"]=="false":
            POSTOP=0
        else:
            return jsonify({"error":"chirurgie not valid, "+str(dic["chirurgie"])})

        if dic["platre"]=="true":
            PLATRE=1
        elif dic["platre"]=="false":
            PLATRE=0
        else:
            return jsonify({"error":"platre not valid, "+str(dic["platre"])})

        if dic["grossesse"]=="true":
            GROSSESSE=1
        elif dic["grossesse"]=="false":
            GROSSESSE=0
        else:
            return jsonify({"error":"grossesse not valid, "+str(dic["grossesse"])})

        if dic["post_partum"]=="true":
            POSTPART=1
        elif dic["post_partum"]=="false":
            POSTPART=0
        else:
            return jsonify({"error":"post_partum not valid, "+str(dic["post_partum"])})

        if dic["hospitalisation"]=="true":
            HOSPM3=1
        elif dic["hospitalisation"]=="false":
            HOSPM3=0
        else:
            return jsonify({"error":"hospitalisation not valid, "+str(dic["hospitalisation"])})

        if dic["voyage"]=="true":
            VOYAGE=1
        elif dic["voyage"]=="false":
            VOYAGE=0
        else:
            return jsonify({"error":"voyage not valid, "+str(dic["voyage"])})

        if dic["contraception"]=="true":
            CO=1
        elif dic["contraception"]=="false":
            CO=0
        else:
            return jsonify({"error":"contraception not valid, "+str(dic["contraception"])})

        if dic["menopause"]=="true":
            THS=1
        elif dic["menopause"]=="false":
            THS=0
        else:
            return jsonify({"error":"menopause not valid, "+str(dic["menopause"])})

        if dic["antecedent_MVTE"]=="true":
            ATCDMVT=1
        elif dic["antecedent_MVTE"]=="false":
            ATCDMVT=0
        else:
            return jsonify({"error":"antecedent_MVTE not valid, "+str(dic["antecedent_MVTE"])})

        if dic["familial_MVTE"]=="true":
            ATCDFAM=1
        elif dic["familial_MVTE"]=="false":
            ATCDFAM=0
        else:
            return jsonify({"error":"familial_MVTE not valid, "+str(dic["familial_MVTE"])})
        """

        
        riskFactor = request.args.get('riskFactor', type=str)
        if riskFactor=='MAJOR_TRANSIENT':
            RISK_FACTOR = "MAJOR_TRANSIENT"
        elif riskFactor=='NON_PROVOQUE':
            RISK_FACTOR = "NON_PROVOQUE"
        elif riskFactor=='CANCER':
            RISK_FACTOR = "CANCER"
        elif riskFactor=='MINOR_TRANSIENT':
            RISK_FACTOR = "MINOR_TRANSIENT"
        elif riskFactor=='MINOR_PERSISTENT':
            RISK_FACTOR = "MINOR_PERSISTANT"
        else:
            return jsonify({"error":"riskFactor not valid, "+str(riskFactor)})
        
        expositionRisqueAnnee = request.args.get('expositionRisqueAnnee', type=float)
        exposition_risque_annees = float(min(max(0.01916548,expositionRisqueAnnee),17.84580002))
    
        #Preparation des parametres pour l'IA
        sample_6 = {
            'SEXE': SEXE,#2
            'POIDS': POIDS,#48.0
            'TAILLE': TAILLE, #163.0
            'IMC': 18.066167337875,
            'AGEDIAG': AGEDIAG,#75.01
            'AGEDIAG_cl': AGEDIAG_cl,#'(50,65]'
            'MVTE_INITIALE_cl': MVTE_INITIALE_cl,#'EP+TVP'
            'DUREE_TTT': DUREE_TTT,#231
            'DUREE_TTT_cl': DUREE_TTT_cl,#'180-360'
            'POSTOP': 0,
            'PLATRE': 0,
            'GROSSESSE': 0,
            'POSTPART': 0.0,
            'HOSPM3': 1.0,
            'VOYAGE': 0.0,
            'CO': 0.0,
            'THS': 0.0,
            'ATCDMVT': 0.0,
            'ATCDFAM': 1.0,
            'MALADIE_INFLAM': MALADIE_INFLAM,#0
            'avcISCHEMIQUE': 0,
            'avecHEMORRAGIQUE': 0,
            'Pneumopathie_interstitielle': 0,
            'bpco': 0,
            'ATCD_HYPOTHYR': 0,
            'ATCD_HYPERTHYR': 0,
            'ATCD_RENAL': 0,
            'ATCD_CARDIOPATH_ISCHEMIQUE': 0,
            'ATCD_INSUFHEP_CHRQ': 0,
            'ATCD_CARDIOPATH_RYTHMIQUE': 0,
            'CANCER': 0,
            'RISK_FACTOR': RISK_FACTOR,#'NON_PROVOQUE'
            'exposition_risque_annees': exposition_risque_annees #0.777570912276859
        }
    
        #Appel de l'IA
        preprocessed_new_sample = preprocess_new_sample(sample_6)
        df_new_sample=pd.DataFrame(preprocessed_new_sample,columns=feature_names)
        df_new_sample.to_csv('data/new_sample_server.csv',index=False)
        sample_prediction, explanation_file = predict_and_explain()

        # On met le resultat de l'IA dans le nom du fichier. Cela permet d'envoyer dans une seule requete le fichier et le resultat de l'ia
        final_file_name = str(sample_prediction)+explanation_file
        os.rename(explanation_file, final_file_name) 
        # Vérification des paramètres
        if None in [gender, weight, height, ageDiagnostic, anticoagulantDuration, diagnosticAgeCategory, mvteType, anticoagulantDurationCategory, chronicInflammatoryDisease, riskFactorsList, riskFactor, expositionRisqueAnnee]:
            return jsonify({"error": "Missing parameters "+ str([gender, weight, height, ageDiagnostic, anticoagulantDuration, diagnosticAgeCategory, mvteType, anticoagulantDurationCategory, chronicInflammatoryDisease, riskFactorsList, riskFactor, expositionRisqueAnnee])}), 400
        return send_file(final_file_name, as_attachment=True, mimetype='text/html')
    except Exception as e:
        return jsonify({"Error":e})

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))  # Utilisation du port fourni par Railway
    app.run(host="0.0.0.0", port=port)
