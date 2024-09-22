import pandas as pd

def read_csv(csv_file, new_csv_file=None):

    df = pd.read_csv(csv_file)
    
    if new_csv_file:
        df.to_csv(new_csv_file, index=False)
        print(f"Archivo CSV guardado como: {new_csv_file}")
    else:
        print(df)

# Especifica el archivo de entrada y salida
csv_file = '/Users/carpasgis/PycharmProjects/personal/IDEAS TFM/dataset_TFM.xls'  
new_csv_file = 'dataset_TFM.csv'  

read_csv(csv_file, new_csv_file)

### CARGA DE LIBRERIAS Y FUNCIONES ACCESORIAS##
import os
import pandas as pd
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import numpy as np

from langchain.document_loaders import PyPDFLoader 
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from transformers import AutoTokenizer, AutoModel
import torch
from langchain.embeddings.base import Embeddings

from langchain_openai import OpenAI

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(file_name):
    with open(file_name, 'rb') as f:
        return pickle.load(f)

def save(file_name, model):
    with open(file_name, 'wb') as f:
        pickle.dump(model, f)

def tryload(file_name):
    if os.path.exists(file_name):
        return load_model(file_name)
    return None


plt.ioff()

model_filename = 'random_forest_model1.pkl'


model1 = tryload(model_filename)

if not model1:

    df = pd.read_csv(r'/Users/carpasgis/PycharmProjects/personal/IDEAS TFM/dataset_TFM.csv')


    mapping_dict = {
        'Sex': {'M': 1, 'F': 0},
        'ChestPainType': {'ATA': 1, 'NAP': 2, 'ASY': 3, 'TA': 4},
        'RestingECG': {'Normal': 1, 'ST': 2, 'LVH': 3},
        'ExerciseAngina': {'Y': 1, 'N': 0},
        'ST_Slope': {'Up': 1, 'Flat': 2, 'Down': 3}
    }
    df.replace(mapping_dict, inplace=True)


    plt.figure(figsize=(15, 8))
    sns.heatmap(df.describe().T, annot=True, fmt=".2f", cmap='coolwarm')
    plt.title('Descripción del conjunto de datos de ataques cardíacos')
    plt.show()

    df.hist(bins=20, figsize=(15, 10), color='blue', edgecolor='black')
    plt.suptitle('Histogramas de las características numéricas')
    plt.show()


    X = df.drop('HeartDisease', axis=1)
    y = df['HeartDisease']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


    print("Entrenando el modelo de Bosque Aleatorio (model1)...")
    model1 = RandomForestClassifier(random_state=42, class_weight='balanced', n_estimators=100)
    model1.fit(X_train, y_train)
    print("Entrenamiento del Bosque Aleatorio (model1) completado.")

   
    save(model_filename, model1)

else:

    df = pd.read_csv(r'/Users/carpasgis/PycharmProjects/personal/IDEAS TFM/dataset_TFM.csv')


    mapping_dict = {
        'Sex': {'M': 1, 'F': 0},
        'ChestPainType': {'ATA': 1, 'NAP': 2, 'ASY': 3, 'TA': 4},
        'RestingECG': {'Normal': 1, 'ST': 2, 'LVH': 3},
        'ExerciseAngina': {'Y': 1, 'N': 0},
        'ST_Slope': {'Up': 1, 'Flat': 2, 'Down': 3}
    }
    df.replace(mapping_dict, inplace=True)


    X = df.drop('HeartDisease', axis=1)
    y = df['HeartDisease']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("El modelo de Bosque Aleatorio (model1) ya está entrenado y cargado desde el archivo.")

def evaluate_model(model, X_test, y_test, model_name="Modelo"):
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n=== Evaluación de {model_name} ===")
    print("Matriz de Confusión:")
    print(confusion_matrix(y_test, y_pred))
    print("\nReporte de Clasificación:")
    print(classification_report(y_test, y_pred))
    print(f"Accuracy: {acc:.4f}")


if model1:
    evaluate_model(model1, X_test, y_test, model_name="Bosque Aleatorio (model1)")


def plot_feature_importances(model, feature_names):
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    sorted_features = [feature_names[i] for i in indices]
    sorted_importances = importances[indices]

    plt.figure(figsize=(12, 8))
    sns.barplot(x=sorted_importances, y=sorted_features, palette='viridis')
    plt.title('Importancia de Características - Bosque Aleatorio (model1)')
    plt.xlabel('Importancia')
    plt.ylabel('Características')
    plt.show()

if model1:
    plot_feature_importances(model1, X.columns)


    
#xgb.XGBClassifier()   # accuracy 0,856
# LogisticRegression(max_iter=1000, solver='liblinear', class_weight='balanced') accurracy 0.8261
#Ddecission tree Accuracy: 0.8696

with open(model_filename, 'wb') as f:
    pickle.dump(model1, f)

print(f"Modelo entrenado y guardado en {model_filename}")

##INTRODUCIR EL IMPUT PARA LA PREDICCIÓN DEL MODELO DE MACHINE LEARNING##

def get_patient_data():
    try:
        age = input("Edad (20-100): ") or None
        sex = input("Sexo (0=Femenino, 1=Masculino): ") or None
        cp = input("Tipo de dolor en el pecho (0-3): ") or None
        trtbps = input("Presión arterial en reposo (90-200 mmHg): ") or None
        chol = input("Colesterol sérico (100-500 mg/dl): ") or None
        fbs = input("Azúcar en sangre en ayunas (0-200 mg/dl): ") or None
        restecg = input("Resultados electrocardiográficos en reposo (0-2): ") or None
        thalachh = input("Frecuencia cardíaca máxima alcanzada (60-200 ppm): ") or None
        exng = input("Angina inducida por ejercicio (0=No, 1=Sí): ") or None
        oldpeak = input("Depresión ST (0-6): ") or None
        slp = input("Pendiente del segmento ST (0-2): ") or None
        caa = input("Número de vasos coloreados por fluoroscopia (0-3): ") or None
        thall = input("Resultado de la prueba de thallium (1-3): ") or None

        patient_data = {
            'age': age,
            'sex': sex,
            'cp': cp,
            'trtbps': trtbps,
            'chol': chol,
            'fbs': fbs,
            'restecg': restecg,
            'thalachh': thalachh,
            'exng': exng,
            'oldpeak': oldpeak,
            'slp': slp,
            'caa': caa,
            'thall': thall
        }

        patient_df = pd.DataFrame([patient_data])
        patient_df = patient_df.apply(pd.to_numeric, errors='coerce')
        return patient_df, f"Datos del paciente: {patient_data}"

    except Exception as e:
        print(f"Error al ingresar datos: {e}")
        return None, None


def predict_heart_disease(model1, patient_data):
    try:
        prediction = model1.predict(patient_data)[0]
        return 'Alto riesgo' if prediction == 1 else 'Bajo riesgo'
    except Exception as e:
        return f"No se pudo realizar la predicción debido a un error: {e}"


##CARGA DE DOCUMENTOS Y FRAGMENTACIÓN##
def load_document(pdf_path):
    documents = []
    
    try:
        pdf_loader = PyPDFLoader(pdf_path)
        pages = pdf_loader.load_and_split()

        for page_number, page in enumerate(pages):
            page_content = page.page_content
            if page_content:
                doc = Document(
                    page_content=page_content,
                    metadata={
                        "filename": pdf_path.split('\\')[-1],
                        "page": page_number + 1
                    }
                )
                documents.append(doc)
    except Exception as e:
        print(f"No se pudo cargar el archivo PDF: {e}")
    
    return documents



DATA_PATH = r"C:\Users\cpascual\PycharmProjects\pythonProject\cursos_actividades\Manual_AMIR_12da_ed_Cardiologia_y_Cirugi.pdf"

document = tryload("documentML.pkl")
if not document:
    document = load_document(DATA_PATH)
    save("documentML.pkl", document)



documents = load_document(r"C:\Users\cpascual\PycharmProjects\pythonProject\cursos_actividades\Manual_AMIR_12da_ed_Cardiologia_y_Cirugi.pdf")
for doc in documents:
    if doc.metadata['page'] == 30:
        print(f"Contenido de la página {doc.metadata['page']}:")
        print(doc.page_content)
        print("\nMetadatos:")
        print(doc.metadata)
        break

def split_documents(document: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n"],
        chunk_size=1000,   
        chunk_overlap=120,  
        length_function=len,
        is_separator_regex=False,
    )
    
    chunks = []
    for doc in document:
        split_texts = text_splitter.split_text(doc.page_content)
        for chunk in split_texts:
            new_doc = Document(
                page_content=chunk,
                metadata=doc.metadata  
            )
            chunks.append(new_doc)
    return chunks

chunks = tryload("chunksML.pkl")
if not chunks:
    chunks = split_documents(document)
    save("chunksML.pkl", chunks)

##CREAR REPRESENTACIONES VECTORIALES DEL TEXTO Y ALMACENARLO##

class LangChain_Embeddings(Embeddings):
    def __init__(self, embedder: AutoModel):
        self.embedder = embedder
        super().__init__()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=512)
        with torch.no_grad():
            embeddings = self.embedder(**inputs).last_hidden_state.mean(dim=1)
        return embeddings.cpu().numpy().tolist()

    def embed_query(self, text: str) -> list[float]:
        inputs = tokenizer([text], return_tensors="pt", padding=True, truncation=True, max_length=200)
        with torch.no_grad():
            embedding = self.embedder(**inputs).last_hidden_state.mean(dim=1)
        return embedding.cpu().numpy().tolist()[0]

model_name = "jinaai/jina-embeddings-v2-base-es"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
model.to(device)

faiss_index = tryload("faiss_indexML.pkl")
if not faiss_index:
    faiss_index = FAISS.from_documents(chunks, LangChain_Embeddings(model))
    save("faiss_indexML.pkl", faiss_index)



##RAG##
faiss_retriever = faiss_index.as_retriever(search_kwargs={'k': 10})

def get_documents(query, retriever):
    unique_docs = retriever.invoke(query)
    documents_with_pages = []
    context = ""
    for doc in unique_docs:
        page_info = doc.metadata.get('page', 'Página desconocida')
        document_filename = doc.metadata.get('filename', 'Documento no identificado') 
        content = f"Página {page_info}:\n{doc.page_content}\n\n"
        context += content
        documents_with_pages.append({
            "page": page_info,
            "content": doc.page_content,
            "filename": document_filename
        })
    return context, documents_with_pages


##LLAMAR AL LARGE LENGUAJE MODEL##

client = OpenAI(base_url="http://localhost:11434/v1", api_key="lm_studio")

class BiomedicalAssistant:
    def __init__(self, client, retriever, model_name="model-identifier"):
        self.client = client
        self.retriever = retriever
        self.sys_message = ''' 
            Eres un médico virtual especializado en enfermedades cardíacas, experto en interpretar resultados de analíticas,
            y tienes una especialización en infarto de miocardio. Tu rol es proporcionar asesoramiento profesional y bien informado
            sobre temas relacionados con la salud cardíaca. No generes mas preguntas adicionales
        '''
    
    def generate_response(self, context: str, question: str, metadata: dict, temperature=0, max_tokens=600) -> str:

        prompt = f"{self.sys_message}\n\nDatos del paciente: {context}\n\nPregunta: {question}\nRespuesta:"
        
        try:

            response = self.client.generate(prompts=[prompt], temperature=temperature, max_tokens=max_tokens)
            
            if hasattr(response, 'generations') and isinstance(response.generations, list):
                generations = response.generations[0]
                if len(generations) > 0 and hasattr(generations[0], 'text'):
                    generated_text = generations[0].text.strip()

                    filename = metadata.get('filename', 'Documento no identificado')
                    page = metadata.get('page', 'Página desconocida')
                    metadata_info = f"\n\nOrigen del documento: {filename}\nNúmero de página: {page}"

                    return f"{generated_text}\n{metadata_info}"
            
            return "No se pudo generar una respuesta válida."
        except Exception as e:
            return f"Error al generar la respuesta: {str(e)}"



model1 = tryload(model_filename)


def main():
    faiss_retriever = faiss_index.as_retriever(search_kwargs={'k': 20})
    assistant = BiomedicalAssistant(client, faiss_retriever)

    print("Bienvenido al Asistente Biomédico Virtual. Primero ingresa los datos del paciente para hacer una predicción.\n")

    patient_data, context = get_patient_data()

    if patient_data is not None:
        print("Datos del paciente ingresados correctamente.\n")
        prediccion = predict_heart_disease(model1, patient_data)
        print(f"Predicción: {prediccion} de enfermedad cardíaca.\n")
    else:
        print("No se ingresaron datos del paciente o hubo un error.")

    print("\nPuedes hacerle cualquier pregunta relacionada con la salud cardíaca o escribir 'salir' para finalizar.")

    while True:
        query = input("\n¿Qué deseas saber sobre la salud cardíaca?:\n")
        
        if query.lower() in ['salir', 'exit', 'quit']:
            print("Gracias por usar el Asistente Biomédico. ¡Cuídate!")
            break

        print("Buscando respuesta...\n")
        document_context, documents_with_pages = get_documents(query, faiss_retriever)

        if documents_with_pages:
            metadata = {
                "filename": documents_with_pages[0].get('filename', 'Documento no identificado'),
                "page": documents_with_pages[0].get('page', 'Página desconocida')
            }
        else:
            metadata = {
                "filename": 'Documento no identificado',
                "page": 'Página desconocida'
            }
        response = assistant.generate_response(context, query, metadata)

        if response:
            print("Respuesta:\n" + response + "\n")
        else:
            print("No se pudo generar una respuesta válida en este momento.\n")


if __name__ == "__main__":
    main()
