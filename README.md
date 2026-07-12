# clustering-pca-nmf-20newsgroups
## Descripción general

Este repositorio contiene la resolución de tres casos prácticos de Machine Learning no supervisado, desarrollados en un único notebook de Jupyter/Google Colab:

1. **Clustering Jerárquico** sobre el dataset *20 Newsgroups* (scikit-learn).
2. **Reconocimiento de rostros con Eigenfaces (PCA)** sobre el dataset *Olivetti Faces* (scikit-learn).
3. **Sistema de recomendación de artículos** usando *Factorización No Negativa de Matrices (NMF)* y similitud de coseno, también sobre *20 Newsgroups*.


## Contenido del repositorio

```
├── trabajo5.ipynb   # Notebook con los 3 casos resueltos
├── README.md                       # Este archivo                        
```

## Caso 1 — Clustering Jerárquico (20 Newsgroups)

Se aplicó `AgglomerativeClustering` sobre representaciones **TF-IDF + TruncatedSVD (LSA)** de 2 937 documentos de 5 categorías temáticas, probando tres criterios de linkage: **ward**, **complete** y **single**.

**Resultado principal:**

| Linkage | ARI | Homogeneity | V-measure |
|---|---|---|---|
| **ward** | **0.1637** | **0.3309** | **0.3935** |
| complete | 0.0112 | 0.0526 | 0.0889 |
| single | 0.0000 | 0.0014 | 0.0027 |

`ward` fue el criterio más exitoso; `complete` y `single` colapsaron casi toda la colección en un único cluster gigante (efecto de encadenamiento).

## Caso 2 — Reconocimiento de rostros con Eigenfaces (PCA)

Se redujo el dataset *Olivetti Faces* (400 imágenes, 40 personas, 4096 píxeles por imagen) a **100 componentes principales (eigenfaces)** con PCA (`whiten=True`), conservando el **94.37%** de la varianza total. Sobre esas proyecciones se entrenaron dos clasificadores clásicos (sin redes neuronales):

| Modelo | Accuracy (test) |
|---|---|
| **SVM (kernel RBF)** | **98.00%** |
| KNN (k=1) | 92.00% |

## Caso 3 — Recomendador de artículos con NMF + similitud de coseno

Sobre 6 251 artículos de 8 categorías, se extrajeron **8 temas latentes** con NMF a partir de una matriz TF-IDF, y se recomendaron artículos similares usando similitud de coseno sobre los perfiles temáticos (`W`).

**Resultado principal:** precisión@N entre **0.64 y 0.67**, frente a ~0.13 de una recomendación aleatoria (≈5x mejor que el azar).

## Tecnologías utilizadas

- Python 3
- scikit-learn (`fetch_20newsgroups`, `fetch_olivetti_faces`, `TfidfVectorizer`, `TruncatedSVD`, `PCA`, `NMF`, `AgglomerativeClustering`, `SVC`, `KNeighborsClassifier`)
- NumPy, Pandas
- Matplotlib
- SciPy (`scipy.cluster.hierarchy`)
- Google Colab / Jupyter Notebook

## Cómo ejecutar

1. Clonar el repositorio:
   ```bash
   git clone <URL_DEL_REPOSITORIO>
   ```
2. Abrir `trabajo5.ipynb` en Google Colab o Jupyter Notebook.
3. Ejecutar las celdas en orden (`Entorno de ejecución > Ejecutar todas` en Colab). No se requieren archivos externos: los tres datasets se descargan automáticamente desde scikit-learn.

