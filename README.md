Prueba de Hipotesis - Estadistica Inferencial
----------------------------------------------

Aplicacion interactiva desarrollada en Streamlit con estetica de videojuego pixel art.
Permite cargar datos, visualizar distribuciones, realizar pruebas de hipotesis Z
e interpretar resultados con apoyo de Inteligencia Artificial (Google Gemini).

Este proyecto fue desarrollado con asistencia de Claude (Anthropic),
IA utilizada para la construccion del codigo, estructura de modulos y logica estadistica.

----------------------------------------------
MODULOS
----------------------------------------------

| Modulo            | Descripcion                                              |
|-------------------|----------------------------------------------------------|
| Cargar Datos      | Sube un archivo CSV o genera datos sinteticos            |
| Distribuciones    | Histograma, KDE y boxplot con analisis automatico        |
| Hipotesis         | Prueba Z con parametros personalizables                  |
| Modulo de IA      | Interpreta los resultados con Google Gemini              |

----------------------------------------------
REQUISITOS
----------------------------------------------

- Python 3.8 o superior
- Las siguientes librerias de Python:

  | Libreria              | Uso                                      |
  |-----------------------|------------------------------------------|
  | streamlit             | Interfaz de la aplicacion                |
  | pandas                | Manejo de datos y archivos CSV           |
  | numpy                 | Calculos numericos                       |
  | matplotlib            | Graficas de distribuciones               |
  | scipy                 | Calculos estadisticos y prueba Z         |
  | python-dotenv         | Lectura segura de la API Key             |
  | google-generativeai   | Conexion con la API de Google Gemini     |

- Conexion a internet (para el modulo de IA)
- API Key gratuita de Google Gemini

----------------------------------------------
INSTALACION
----------------------------------------------

1 - Clona el repositorio:

    git clone https://github.com/HectorCruz2007/Prueba-de-Hipotesis.git
    cd Prueba-de-Hipotesis

2 - Instala las dependencias:

    pip install streamlit pandas numpy matplotlib scipy python-dotenv google-generativeai

3 - Configura tu API Key (opcional):

    Crea un archivo .env en la carpeta del proyecto con el siguiente contenido:

    GROQ_API_KEY=TU_API_KEY_DE_GEMINI_AQUI

    Puedes obtener una API Key gratuita en: https://aistudio.google.com/app/apikey

    Si no creas el archivo .env, la app te pedira la API Key
    directamente en la interfaz cada vez que uses el modulo de IA.

4 - Ejecuta la aplicacion:

    streamlit run app.py

    La app se abrira en tu navegador en: http://localhost:8501

----------------------------------------------
USO
----------------------------------------------

1 - Ve al modulo Cargar Datos y carga o genera tu dataset
2 - Ve al modulo Distribuciones para visualizar y analizar tu variable
3 - Ve al modulo Hipotesis y define:
    - Media hipotetica (H0)
    - Desviacion estandar poblacional
    - Nivel de significancia
    - Tipo de prueba (bilateral, cola derecha o cola izquierda)
4 - Haz clic en Ejecutar Prueba Z
5 - Ve al modulo IA para obtener una interpretacion automatica

----------------------------------------------
ESTRUCTURA DEL PROYECTO
----------------------------------------------

Prueba-de-Hipotesis/
|-- app.py        - Aplicacion principal
|-- .gitignore    - Archivos ignorados por Git
|-- README.md     - Este archivo

El archivo .env no se incluye en el repositorio por seguridad.
Cada usuario debe crear el suyo propio con su propia API Key.

----------------------------------------------
AUTOR
----------------------------------------------

Hector Cruz - 2026
