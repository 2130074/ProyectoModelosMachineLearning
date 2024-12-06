import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from wordcloud import WordCloud
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, confusion_matrix

def clean_numeric_value(value):
    try:
        # Extrae solo los números del string
        if isinstance(value, str):
            return float(''.join(filter(str.isdigit, value)))
        return float(value)
    except:
        return None

# Cargar datos
file_path = "Dataset.xlsx"
dataset = pd.read_excel(file_path)

# Limpiar columnas
columns_to_drop = ['Marca temporal', 'Dirección de correo electrónico']
dataset_cleaned = dataset.drop(columns=columns_to_drop)
dataset_cleaned.columns = dataset_cleaned.columns.str.strip()

# Definir columnas
numeric_columns = [
    '¿Cuántas horas duermes en promedio cada noche?',
    '¿Cuántas horas a la semana dedicas a actividades recreativas o de ocio?'
]

categorical_columns = [
    '¿Con qué frecuencia te sientes estresado/a durante el semestre?',
    '¿Qué tan satisfecho/a estás con la calidad de tu sueño?',
    '¿Qué tan efectiva consideras que es la universidad en brindar apoyo a los estudiantes en temas de salud mental?',
    '¿Con qué frecuencia haces ejercicio físico como parte de tu rutina para el bienestar mental?'
]

text_columns = [
    '¿Cuáles son las principales fuentes de estrés en tu vida universitaria?',
    '¿Qué métodos utilizas para manejar el estrés?',
    '¿Qué tan importante consideras la gestión de la salud mental durante la vida universitaria? Explica por qué.',
    '¿Has acudido alguna vez a un servicio de consejería o terapia psicológica? Explica por qué sí o por qué no.'
]

# Limpiar datos numéricos
for column in numeric_columns:
    dataset_cleaned[column] = dataset_cleaned[column].apply(clean_numeric_value)

data_numeric = dataset_cleaned[numeric_columns]
data_categorical = dataset_cleaned[categorical_columns]
data_text = dataset_cleaned[text_columns]

# Eliminar filas con valores nulos en columnas numéricas
data_numeric = data_numeric.dropna()

print("\nEstadísticas descriptivas después de limpieza:")
print(data_numeric.describe())

# Visualizaciones
for column in numeric_columns:
    plt.figure(figsize=(10, 6))
    sns.histplot(data=data_numeric[column].dropna(), kde=True)
    plt.title(f"Distribución de {column}")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# Nube de palabras
for column in text_columns:
    text_data = " ".join(dataset_cleaned[column].astype(str))
    wordcloud = WordCloud(width=800, height=400, background_color="white").generate(text_data)
    plt.figure(figsize=(10, 6))
    plt.imshow(wordcloud)
    plt.title(f"Nube de palabras: {column}")
    plt.axis("off")
    plt.show()

# Clustering
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data_numeric)

# Método del codo
wcss = []
K = range(2, 8)
for k in K:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(data_scaled)
    wcss.append(kmeans.inertia_)

plt.figure(figsize=(10, 6))
plt.plot(K, wcss, 'bo-')
plt.title("Método del Codo")
plt.xlabel("Número de Clústeres")
plt.ylabel("WCSS")
plt.show()

# Aplicar K-means
kmeans = KMeans(n_clusters=3, random_state=42)
data_numeric['Cluster'] = kmeans.fit_predict(data_scaled)

# Visualizar clusters
plt.figure(figsize=(10, 6))
plt.scatter(data_scaled[:, 0], data_scaled[:, 1], c=data_numeric['Cluster'])
plt.title("Clusters")
plt.xlabel(numeric_columns[0])
plt.ylabel(numeric_columns[1])
plt.show()

# Clasificación
satisfaction_mapping = {
    "Muy insatisfecho/a": 0,
    "Insatisfecho/a": 0,
    "Neutro": 1,
    "Satisfecho/a": 2,
    "Muy satisfecho/a": 2
}

dataset_cleaned['Etiqueta'] = dataset_cleaned['¿Qué tan satisfecho/a estás con la calidad de tu sueño?'].map(satisfaction_mapping)

# Usar solo filas que tienen datos numéricos completos
valid_indices = data_numeric.index
X = data_numeric.loc[valid_indices].drop(columns=['Cluster'])
y = dataset_cleaned.loc[valid_indices, 'Etiqueta']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = GaussianNB()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("\nReporte de Clasificación:")
print(classification_report(y_test, y_pred))

plt.figure(figsize=(8, 6))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d')
plt.title("Matriz de Confusión")
plt.show()