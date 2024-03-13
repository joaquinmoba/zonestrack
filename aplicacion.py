# Datos ficticios de usuarios y deportistas
import pandas as pd
from dash import dcc, html, Input, Output, State, Dash, dash_table, callback
import dash_bootstrap_components as dbc
import re
import plotly.express as px
import dash_core_components as dcc
import dash_html_components as html
import datetime
import xml.etree.ElementTree as ET
import base64
import tempfile
import dash.exceptions
import time
import numpy as np
import json
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import dash




class Deportista:
    # Definición de la clase Deportista
    def __init__(self, nombre, apellido, edad, v_pc_atletismo, vam_atletismo, pc_ciclismo, pam, vc_natacion, fc_max_natacion,fc_max_atletismo,fc_max_ciclismo):
        self.nombre = nombre
        self.apellido = apellido
        self.edad = edad
        self.v_pc_atletismo = v_pc_atletismo
        self.vam_atletismo = vam_atletismo
        self.pc_ciclismo = pc_ciclismo
        self.pam = pam
        self.vc_natacion = vc_natacion
        self.fc_max_natacion = fc_max_natacion
        self.fc_max_atletismo = fc_max_atletismo
        self.fc_max_ciclismo = fc_max_ciclismo

class Usuario:
    # Definición de la clase Usuario
    def __init__(self, nombre, apellido, usuario, email, tipo, contraseña, atletas):
        self.nombre = nombre
        self.apellido = apellido
        self.usuario = usuario
        self.email = email
        self.tipo = tipo
        self.contraseña = contraseña
        self.atletas = atletas if atletas is not None else []

# Crear datos ficticios de usuarios y deportistas
deportista1 = Deportista("Emma", "Wright", 25, 15, 20, 300, 300, 72, 195, 195, 195)
deportista2 = Deportista("Joaquín", "Mojica", 21, 20, 20.5, 320, 320, 90, 198, 198, 198)
deportista4 = Deportista("Gerard", "", 27, 18, 19.5, 280, 280, 90, 191, 191, 191)
deportista5 = Deportista("Álvaro", "Rancé", 49, 18, 35,  260, 260, 90, 185, 180, 175)
deportista6 = Deportista("Genis", "Grau", 29, 21, 21, 320, 310, 90, 188, 188, 188)

deportista3 = Deportista("Genis", "Grau", 22, 18, 35, 280, 320, 90, 185, 180, 175)

atletas_data = [deportista1.__dict__, deportista2.__dict__, deportista3.__dict__]

usuario1 = Usuario("Juan", "Pérez", "juanperez", "juanperez@example.com", "deportista", "password123", [deportista1.__dict__, deportista2.__dict__, deportista4.__dict__, deportista5.__dict__, deportista6.__dict__])
usuario2 = Usuario("María", "García", "mariagarcia", "mariagarcia@example.com", "entrenador", "pass456", [deportista3.__dict__])

usuarios_data = [usuario1.__dict__, usuario2.__dict__]


juanperez_atletas = None
for usuario in usuarios_data:
    if usuario['usuario'] == 'juanperez':
        juanperez_atletas = usuario['atletas']
        break

# Convertir los datos en un DataFrame para la tabla
juanperez_df = pd.DataFrame(juanperez_atletas)

# Obtener las columnas disponibles para agregar

column_tooltips = {
    'v_pc_atletismo': 'Velocidad de Pico en Atletismo',
    'vam_atletismo': 'Velocidad Aeróbica Máxima en Atletismo',
    'pc_ciclismo': 'Pico de Potencia en Ciclismo',
    'pam': 'Potencia Anaeróbica Máxima',
    'vc_natacion': 'Velocidad Crítica en Natación',
    'fc_max_natacion': 'Frecuencia Cardíaca Máxima natación',
    'fc_max_ciclismo': 'Frecuencia Cardíaca Máxima ciclismo',
    'fc_max_atletismo': 'Frecuencia Cardíaca Máxima atletismo'
}

# Función para procesar el archivo TCX


def importar_archivo(contents):
    if contents is None:
        raise dash.exceptions.PreventUpdate

    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)

    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(decoded)
        tmp.seek(0)

        # Lee el archivo desde la ruta temporal en un ElementTree
        tree = ET.parse(tmp.name)
        root = tree.getroot()

        # Inicializar listas para almacenar los datos

        times = []
        time_differences = []  # Lista para almacenar las diferencias de tiempo en segundos
        distances = []
        heart_rates = []
        cadences = []
        speeds = []
        watts = []

        prev_time = None  # Variable para almacenar el tiempo anterior
        prev_distance = None

        # Iterar a través de los trackpoints y extraer los datos
        for trackpoint in root.findall('.//{http://www.garmin.com/xmlschemas/TrainingCenterDatabase/v2}Trackpoint'):
            time = trackpoint.find('{http://www.garmin.com/xmlschemas/TrainingCenterDatabase/v2}Time').text
            current_time = datetime.datetime.strptime(time, "%Y-%m-%dT%H:%M:%S.%fZ")
            if prev_time is not None:
                time_difference = (current_time - prev_time).total_seconds()
                time_difference = time_difference if time_difference < 30 else 1
            else:
                time_difference = 1
            distance_element = trackpoint.find('.//{http://www.garmin.com/xmlschemas/TrainingCenterDatabase/v2}DistanceMeters')
            distance = float(distance_element.text) if distance_element is not None else 0

            if prev_distance is not None:
                delta = (distance - prev_distance)
            else:
                delta = distance
            heart_rate_element = trackpoint.find('.//{http://www.garmin.com/xmlschemas/TrainingCenterDatabase/v2}HeartRateBpm/{http://www.garmin.com/xmlschemas/TrainingCenterDatabase/v2}Value')
            heart_rate = int(heart_rate_element.text) if heart_rate_element is not None else None
            speed_element = trackpoint.find('.//{http://www.garmin.com/xmlschemas/ActivityExtension/v2}Speed')
            speed_m_s = float(speed_element.text) if speed_element is not None else (delta/time_difference if delta is not None else 0)
            speed_kmh = speed_m_s * 3.6
            watt_element = trackpoint.find('.//{http://www.garmin.com/xmlschemas/ActivityExtension/v2}Watts')
            watt = int(watt_element.text) if watt_element is not None else 0

            times.append(time)
            time_differences.append(int(time_difference))
            distances.append(distance)
            heart_rates.append(heart_rate)
            speeds.append(speed_kmh)
            watts.append(watt)

            prev_distance = distance
            prev_time = current_time

        # Crear un DataFrame de Pandas
        data = {
            'Time': time_differences,  # Usar las diferencias de tiempo en segundos
            'Distance': distances,
            'Heart Rate': heart_rates,
            'Speed': speeds,
            'Watts': watts
        }

        df = pd.DataFrame(data)

        return df
def format_duration(seconds):
    return str(datetime.timedelta(seconds=int(seconds)))
def format_seconds(seconds):
    return str(datetime.timedelta(seconds=seconds))
# Función para procesar el archivo TCX y obtener un resumen de laps
def process_tcx_file(file_path):
    tree = ET.parse(file_path)
    root = tree.getroot()

    laps_data = []
    for lap in root.iter('{http://www.garmin.com/xmlschemas/TrainingCenterDatabase/v2}Lap'):
        lap_data = {
            'StartTime': lap.attrib['StartTime'],
            'TotalTimeSeconds': float(lap.find('{http://www.garmin.com/xmlschemas/TrainingCenterDatabase/v2}TotalTimeSeconds').text),
            'DistanceMeters': float(lap.find('{http://www.garmin.com/xmlschemas/TrainingCenterDatabase/v2}DistanceMeters').text) if lap.find('{http://www.garmin.com/xmlschemas/TrainingCenterDatabase/v2}DistanceMeters') is not None else 0,
            'MaximumSpeed': float(lap.find('{http://www.garmin.com/xmlschemas/TrainingCenterDatabase/v2}MaximumSpeed').text) if lap.find('{http://www.garmin.com/xmlschemas/TrainingCenterDatabase/v2}MaximumSpeed') is not None else 0,
            'Calories': int(lap.find('{http://www.garmin.com/xmlschemas/TrainingCenterDatabase/v2}Calories').text) if lap.find('{http://www.garmin.com/xmlschemas/TrainingCenterDatabase/v2}Calories') is not None else 0,
            'AverageHeartRateBpm': int(lap.find('{http://www.garmin.com/xmlschemas/TrainingCenterDatabase/v2}AverageHeartRateBpm/{http://www.garmin.com/xmlschemas/TrainingCenterDatabase/v2}Value').text) if lap.find('{http://www.garmin.com/xmlschemas/TrainingCenterDatabase/v2}AverageHeartRateBpm/{http://www.garmin.com/xmlschemas/TrainingCenterDatabase/v2}Value') is not None else 0,
            'MaximumHeartRateBpm': int(lap.find('{http://www.garmin.com/xmlschemas/TrainingCenterDatabase/v2}MaximumHeartRateBpm/{http://www.garmin.com/xmlschemas/TrainingCenterDatabase/v2}Value').text) if lap.find('{http://www.garmin.com/xmlschemas/TrainingCenterDatabase/v2}MaximumHeartRateBpm/{http://www.garmin.com/xmlschemas/TrainingCenterDatabase/v2}Value') is not None else 0,
            'Intensity': lap.find('{http://www.garmin.com/xmlschemas/TrainingCenterDatabase/v2}Intensity').text if lap.find('{http://www.garmin.com/xmlschemas/TrainingCenterDatabase/v2}Intensity') is not None else '',
            'Cadence': int(lap.find('{http://www.garmin.com/xmlschemas/TrainingCenterDatabase/v2}Cadence').text) if lap.find('{http://www.garmin.com/xmlschemas/TrainingCenterDatabase/v2}Cadence') is not None else 0,
            'TriggerMethod': lap.find('{http://www.garmin.com/xmlschemas/TrainingCenterDatabase/v2}TriggerMethod').text if lap.find('{http://www.garmin.com/xmlschemas/TrainingCenterDatabase/v2}TriggerMethod') is not None else '',
            'AverageSpeed': float(lap.find('.//{http://www.garmin.com/xmlschemas/TrainingCenterDatabase/v2}AvgSpeed').text) if lap.find('.//{http://www.garmin.com/xmlschemas/TrainingCenterDatabase/v2}AvgSpeed') is not None else 0,
            'AverageWatts': float(lap.find('.//{http://www.garmin.com/xmlschemas/TrainingCenterDatabase/v2}Watts').text) if lap.find('.//{http://www.garmin.com/xmlschemas/TrainingCenterDatabase/v2}Watts') is not None else 0,
            'MaxWatts': float(lap.find('.//{http://www.garmin.com/xmlschemas/TrainingCenterDatabase/v2}MaxWatts').text) if lap.find('.//{http://www.garmin.com/xmlschemas/TrainingCenterDatabase/v2}MaxWatts') is not None else 0
        }
        laps_data.append(lap_data)

    # Crear DataFrame con los datos de los laps
    laps_df = pd.DataFrame(laps_data)

    # Crear resumen de laps
    laps_summary = laps_df[['StartTime', 'TotalTimeSeconds', 'DistanceMeters', 'Calories', 'AverageHeartRateBpm', 'MaximumHeartRateBpm', 'Intensity', 'Cadence', 'TriggerMethod', 'AverageSpeed', 'AverageWatts', 'MaxWatts']]

    return laps_summary



"""
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
import re
from dash.dependencies import Input, Output, State
import xml.etree.ElementTree as ET

import plotly.graph_objects as go
import plotly.subplots as sp"""

app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP], suppress_callback_exceptions=True)
server = app.server
app.config.suppress_callback_exceptions = True
app.layout = html.Div(
    [
        dbc.NavbarSimple(
            children=[
                dbc.NavItem(dbc.NavLink("Inicio", href="/")),
                dbc.DropdownMenu(
                    children=[
                        dbc.DropdownMenuItem("Análisis de entrenamiento", href="/analisis-entrenamientos"),
                        dbc.DropdownMenuItem("Zonas de atletas", href="/tabla-zonas"),
                    ],
                    nav=True,
                    in_navbar=True,
                    label="Menú",
                ),
                dbc.NavItem(dbc.NavLink("Login", href="/login")),
            ],
            brand="ZonesTrack",
            brand_href="/",
            color="primary",
            dark=True,
        ),
        dcc.Location(id="url"),
        html.Div(id="page-content", className="container mt-5"),
        dcc.Store(id='intermediate-value'),
        dcc.Store(id='archivo'),
    ]
)


@app.callback(
    [Output("page-content", "children"), Output("url", "pathname")],
    [Input("url", "pathname")],
)
def render_page_content(pathname):
    if pathname == "/":
        return (
            html.Div(
                [
                    html.H1("Bienvenido a la página de inicio"),
                    dbc.Button("Registrate aquí", id="register-button", href="/register"),
                ]
            ),
            pathname,
        )
    elif pathname == "/analisis-entrenamientos":
        analisis_entrenamientos =(
        html.Div([
            html.H1("Análisis de Entrenamiento"),
            html.H2("Elige un atleta:"),
            dcc.Dropdown(
                id='atletas-dropdown',
                options=[{'label': f"{atleta['nombre']} {atleta['apellido']}", 'value': f"{atleta['nombre']} {atleta['apellido']}"} for atleta in juanperez_df.to_dict('records')],
                value=None,  # Valor inicial
                multi=False  # Permite seleccionar múltiples atletas
            ),
            dcc.Dropdown(
                id='tipo-entrenamiento',
                options=[
                    {'label': 'Aguas Abiertas', 'value': 'aguas_abiertas'},
                    {'label': 'Ciclismo', 'value': 'ciclismo'},
                    {'label': 'Correr', 'value': 'correr'}
                ],
                value=None,  # Valor inicial como None
                multi=False
            ),
            dcc.Dropdown(
                id='zonas-analisis',
                options=[
                    {'label': '3 Zonas de Análisis', 'value': 3},
                    {'label': '5 Zonas de Análisis', 'value': 5}
                ],
                value=5,  # Valor predeterminado como 5
                multi=False
            ),
            dcc.Upload(
                id='upload-data',
                children=html.Button('Subir archivo TCX'),
                multiple=False
            ),
            dcc.Loading(
                id='loading',
                type= 'graph',
                children=[]
            )
        ]))
        return analisis_entrenamientos, pathname
    elif pathname == "/tabla-zonas":
        tabla_zonas= html.Div([
            html.Div(id="tabla-zonas", children=[
                dash_table.DataTable(
                    id='atletas-table',
                    columns=[{'name': col, 'id': col, 'deletable': True, 'renamable': False} for col in juanperez_df.columns],
                    data=juanperez_df.to_dict('records'),
                    editable=True,
                    row_deletable=True,
                    style_table={'overflowX': 'scroll'},
                    style_header={'fontWeight': 'bold', 'fontSize': '16px'},
                    style_cell={'textAlign': 'center', 'font_size': '14px', 'minWidth': '100px', 'width': '100px', 'whiteSpace': 'normal'},
                    tooltip_data=[{column: {'value': column_tooltips.get(column, '')} for column in juanperez_df.columns} for _ in juanperez_df.to_dict('records')],
                    tooltip_duration=None  # Muestra el tooltip indefinidamente
                ),
                html.Button('Añadir atleta', id='editing-rows-button', n_clicks=0),
                html.Div(id="confirm-delete-modal-container"),
            ])
        ])
        return tabla_zonas, pathname
    elif pathname == "/login":
        return render_login_page(), pathname
    elif pathname == "/register":
        return render_register_page(), pathname
    else:
        return html.H1("Página no encontrada"), pathname


def validate_user_data(username, email, user_type, password, confirm_password):
    existing_usernames = ["username1", "username2"]  # Ejemplo de lista de nombres de usuario existentes
    errors = []

    # Validar si el nombre de usuario ya está en uso
    if username in existing_usernames:
        errors.append("El nombre de usuario ya está en uso.")

    # Validar que las contraseñas coincidan
    if password != confirm_password:
        errors.append("Las contraseñas no coinciden.")

    # Validar la fortaleza de la contraseña
    if not re.search(r"(?=.*[a-z])(?=.*[A-Z])(?=.*\d).{8,}", password):
        errors.append(
            "La contraseña debe contener al menos 8 caracteres, incluyendo una letra mayúscula, una letra minúscula y un número."
        )

    # Validar el formato del correo electrónico
    if not re.match(r"[^@]+@[^@]+\.[^@]+", email):
        errors.append("El formato del correo electrónico no es válido.")

    return errors


def render_login_page():
    return html.Div(
        [
            html.H2("Inicio de sesión"),
            html.Form(
                [
                    html.Div(
                        [
                            html.Label("Nombre de usuario:"),
                            dcc.Input(id="username-input", type="text", required=True),
                        ],
                        className="form-group",
                    ),
                    html.Div(
                        [
                            html.Label("Contraseña:"),
                            dcc.Input(id="password-input", type="password", required=True),
                        ],
                        className="form-group",
                    ),
                    html.Button("Iniciar sesión", id="login-button", n_clicks=0, className="btn btn-primary"),
                ],
                className="form",
            ),
        ]
    )


def render_register_page():
    return html.Div(
        [
            html.H2("Registro"),
            html.Form(
                [
                    html.Div(
                        [
                            html.Label("Nombre:"),
                            dcc.Input(id="register-firstname-input", type="text", required=True),
                        ],
                        className="form-group",
                    ),
                    html.Div(
                        [
                            html.Label("Apellido(s):"),
                            dcc.Input(id="register-lastname-input", type="text", required=True),
                        ],
                        className="form-group",
                    ),
                    html.Div(
                        [
                            html.Label("Nombre de usuario:"),
                            dcc.Input(id="register-username-input", type="text", required=True),
                        ],
                        className="form-group",
                    ),
                    html.Div(
                        [
                            html.Label("Correo electrónico:"),
                            dcc.Input(id="register-email-input", type="email", required=True),
                        ],
                        className="form-group",
                    ),
                    html.Div(
                        [
                            html.Label("Tipo de usuario:"),
                            dcc.Dropdown(
                                id="register-user-type-input",
                                options=[
                                    {"label": "Deportista", "value": "deportista"},
                                    {"label": "Entrenador", "value": "entrenador"},
                                ],
                                placeholder="Seleccionar tipo de usuario",
                                value=None,
                            ),
                        ],
                        className="form-group",
                    ),
                    html.Div(
                        [
                            html.Label("Contraseña:"),
                            dcc.Input(id="register-password-input", type="password", required=True),
                        ],
                        className="form-group",
                    ),
                    html.Div(
                        [
                            html.Label("Confirmar contraseña:"),
                            dcc.Input(id="register-confirm-password-input", type="password", required=True),
                        ],
                        className="form-group",
                    ),
                    html.Button("Registrarse", id="register-button2", n_clicks=0, className="btn btn-primary"),
                    html.Div(id="register-error-message", className="text-danger"),  # Para mostrar los mensajes de error
                ],
                className="form",
            ),
        ]
    )


@app.callback(
    Output("register-error-message", "children"),
    [Input("register-button2", "n_clicks")],
    [
        State("register-username-input", "value"),
        State("register-email-input", "value"),
        State("register-user-type-input", "value"),
        State("register-password-input", "value"),
        State("register-confirm-password-input", "value")
    ],
    prevent_initial_call=True)

def validate_registration(n_clicks, username, email, user_type, password, confirm_password):
    if n_clicks is not None and n_clicks > 0:
        errors = validate_user_data(username, email, user_type, password, confirm_password)
        if errors:
            return html.Ul([html.Li(error) for error in errors])
        else:
            return "Registro exitoso. ¡Bienvenido!"
    return ""

#tabla zonas
@app.callback(
    Output('atletas-table', 'data'),
    Input('editing-rows-button', 'n_clicks'),
    State('atletas-table', 'data'),
    prevent_initial_call=True
)
def update_table(n_clicks, rows):
    if n_clicks > 0:
        new_atleta = {col: '' for col in juanperez_df.columns}
        juanperez_df.loc[len(juanperez_df)] = [''] * len(juanperez_df.columns)
        rows.append(new_atleta)
    return rows

@app.callback(
    Output('tabla-zonas', 'children'),
    Input('atletas-table', 'data'),
    State('atletas-table', 'data'),
    prevent_initial_call=True
)
def update_juanperez_df(data, rows):
    global juanperez_df
    juanperez_df = pd.DataFrame(rows)
    return html.Div([
        dash_table.DataTable(
            id='atletas-table',
            columns=[{'name': col, 'id': col, 'deletable': True, 'renamable': False} for col in juanperez_df.columns],
            data=juanperez_df.to_dict('records'),
            editable=True,
            row_deletable=True,
            style_table={'overflowX': 'scroll'},
            style_header={'fontWeight': 'bold', 'fontSize': '16px'},
            style_cell={'textAlign': 'center', 'font_size': '14px', 'minWidth': '100px', 'width': '100px', 'whiteSpace': 'normal'},
            tooltip_data=[{column: {'value': column_tooltips.get(column, '')} for column in juanperez_df.columns} for _ in juanperez_df.to_dict('records')],
            tooltip_duration=None  # Muestra el tooltip indefinidamente
        ),
        html.Button('Añadir atleta', id='editing-rows-button', n_clicks=0)
    ])

@app.callback(
    Output('editing-rows-button', 'n_clicks'),
    Input('atletas-table', 'data'))
def reset_n_clicks(rows):
    return 0



#analisis entrenos
@app.callback(
    Output('intermediate-value', 'data'),
    [Input('metricas-entrenamiento', 'selectedData')],
    prevent_initial_call=True)
def update_selection(selection):
    if selection:
        rango_seleccionado = selection['range']['x']
        tiempo_inicio, tiempo_final = rango_seleccionado

        # Aquí debes convertir tiempo_inicio y tiempo_final a tus unidades de tiempo, si es necesario
        return json.dumps({'inicio': tiempo_inicio, 'final': tiempo_final})
    else:
      return json.dumps({'inicio': None, 'final': None})


@app.callback(
    Output('loading', 'children'),
    [Input('upload-data', 'contents'),
     Input('atletas-dropdown', 'value'),
     Input('tipo-entrenamiento', 'value'),
     Input('zonas-analisis', 'value'),
     Input('intermediate-value', 'data')],
    prevent_initial_call=True)
def update_zone_plots2(contents, selected_atleta, tipo_entrenamiento, zonas_analisis, selected_range_json):
    if contents is not None and selected_atleta is not None and tipo_entrenamiento is not None and zonas_analisis is not None:
        try:
            # Parsea el rango seleccionado del json
            df = importar_archivo(contents)
            datos_entrenamiento = df[["Time", "Heart Rate", "Speed", "Watts", 'Distance']]
            datos_entrenamiento['Accumulated Time'] = df['Time'].cumsum()
            datos_entrenamiento['Elapsed Time Str'] = datos_entrenamiento['Accumulated Time'].apply(format_seconds)
            children = []
            if selected_range_json is not None:
                selected_range = json.loads(selected_range_json)
                inicio = selected_range.get('inicio')
                final = selected_range.get('final')
                if inicio != final:
                  datos_entrenamiento = datos_entrenamiento[
                      (datos_entrenamiento['Accumulated Time'] >= inicio) &
                      (datos_entrenamiento['Accumulated Time'] <= final)
                  ]
                selected_range_str = json.dumps(selected_range_json, indent=4)  # Formatear JSON con sangrías para mejor legibilidad
                children.append(html.Pre(selected_range_str, style={'white-space': 'pre-wrap'}))
            tick_indices = np.linspace(start=0, stop=len(datos_entrenamiento) - 1, num=12, dtype=int)
            tick_vals = datos_entrenamiento['Accumulated Time'].iloc[tick_indices]
            tick_texts = datos_entrenamiento['Elapsed Time Str'].iloc[tick_indices]
            nombre, apellido = selected_atleta.split(' ')

            tiempo_total = datos_entrenamiento['Accumulated Time'].iloc[-1]-datos_entrenamiento['Accumulated Time'].iloc[0]
            distancia_total = datos_entrenamiento['Distance'].iloc[-1]-datos_entrenamiento['Distance'].iloc[0]
            velocidad_media = distancia_total / tiempo_total * 3.6 if tiempo_total != 0 else 0
            #tiempo_total = datos_entrenamiento['Elapsed Time Str'].iloc[-1]

            # Si es un entrenamiento de correr, calcular min/km medio
            if tipo_entrenamiento == 'correr':
                min_por_km_medio = tiempo_total / (distancia_total/1000) if distancia_total != 0 else 0
            else:
                min_por_km_medio = None

            # Calcular el pulso medio y pulso máximo si hay datos de frecuencia cardíaca
            if 'Heart Rate' in datos_entrenamiento.columns:
                pulso_medio = datos_entrenamiento['Heart Rate'].mean()
                pulso_maximo = datos_entrenamiento['Heart Rate'].max()
            else:
                pulso_medio = None
                pulso_maximo = None

            # Calcular la potencia media y potencia máxima si hay datos de potencia
            if 'Watts' in datos_entrenamiento.columns:
                potencia_media = datos_entrenamiento['Watts'].mean()
                potencia_maxima = datos_entrenamiento['Watts'].max()
            else:
                potencia_media = None
                potencia_maxima = None

            # Crear el resumen del entrenamiento como una tabla HTML
            resumen_entrenamiento_items = [
                html.Tr([html.Th("Tiempo Total"), html.Td(str(format_seconds(round(tiempo_total))))]),
                html.Tr([html.Th("Distancia Total"), html.Td(str(round(distancia_total, 2))+"m")]),
                html.Tr([html.Th("Velocidad Media"), html.Td(str(round(velocidad_media,2))+"km/h")])
            ]

            if min_por_km_medio is not None:
                resumen_entrenamiento_items.append(html.Tr([html.Th("Min/km Medio"), html.Td(str(format_seconds(round(min_por_km_medio))))]))

            if pulso_medio is not None:
                resumen_entrenamiento_items.append(html.Tr([html.Th("Pulso Medio"), html.Td(str(round(pulso_medio))+"bpm")]))

            if pulso_maximo is not None:
                resumen_entrenamiento_items.append(html.Tr([html.Th("Pulso Máximo"), html.Td(str(pulso_maximo)+"bpm")]))

            if potencia_media is not None:
                resumen_entrenamiento_items.append(html.Tr([html.Th("Potencia Media"), html.Td(str(round(potencia_media))+"w")]))

            if potencia_maxima is not None:
                resumen_entrenamiento_items.append(html.Tr([html.Th("Potencia Máxima"), html.Td(str(potencia_maxima)+"w")]))

            resumen_entrenamiento = html.Table(resumen_entrenamiento_items)

            # Añadir el resumen del entrenamiento al principio de la lista de children
            children.insert(0, html.Div(resumen_entrenamiento))
            tablas = []
            graficos = []
            for atleta in juanperez_df.values:
                if atleta[0] == nombre and atleta[1] == apellido:
                    # Extraer los valores de PC_a, VAM, pmax, PC_c, PAM y FC_max del atleta
                    PC_a = int(atleta[3])
                    VAM = float(atleta[4])
                    pmax = int(atleta[6])
                    PC_c = int(atleta[5])
                    if tipo_entrenamiento == 'aguas_abiertas':
                      FC_max = int(atleta[8])
                    elif tipo_entrenamiento == 'ciclismo':
                      FC_max = int(atleta[9])
                    else:
                      FC_max = int(atleta[10])

            if zonas_analisis == 3:
                zonas = ['Z0', 'Z1', 'Z2', 'Z3', 'MAX']
                fc_zones = [0, 0.55*FC_max, 0.8*FC_max, 0.87*FC_max, FC_max, 240]
                power_zones = [0, 0.6*pmax, 0.75*pmax, 0.85*pmax, pmax, 2000]
                speed_zones = [0, 0.6*VAM, 0.75*VAM, 0.85*VAM, VAM, 40]
                paleta = ['#87CEEB', 'green', 'yellow', 'red', '#AB63FA']
            elif zonas_analisis == 5:
                zonas = ['Z0', 'Z1', 'Z2', 'Z3', 'Z4', 'Z5', 'MAX']
                fc_zones = [0, 0.55*FC_max, 0.72*FC_max, 0.8*FC_max, 0.87*FC_max, 0.93*FC_max, FC_max, 240]
                power_zones = [0, 0.65*pmax, 0.8*pmax, 0.9*pmax, pmax, 1.15*pmax, 1.3*pmax, 2000]
                speed_zones = [0, 0.55*VAM, 0.72*VAM, 0.8*VAM, 0.87*VAM, 0.93*VAM,VAM, 40]
                paleta = ['#87CEEB', '#98FB98', 'green', 'yellow', 'orange', 'red', '#AB63FA']

            if datos_entrenamiento["Heart Rate"].isnull().all():
                children.append(html.H2("No hay datos de Fracuencia cardíaca"))
            else:
                datos_entrenamiento["FC_zona"] = pd.cut(datos_entrenamiento["Heart Rate"], fc_zones, labels=[str(i) for i in range(0,len(zonas))], include_lowest=True)
                duracion_FC = [0] * len(zonas)  # inicializar la lista con ceros para cada zona de FC
                datos_entrenamiento["FC_zona"] = datos_entrenamiento["FC_zona"].fillna(method='ffill')

                for i, row in datos_entrenamiento.iterrows():  # iterar sobre cada fila del dataframe
                    zona = int(row["FC_zona"])
                    duracion = row["Time"]  # obtener la duración en segundos del timelap
                    duracion_FC[zona] += duracion  # sumar la duración a la zona correspondiente

                tiempo_total_FC = [str(datetime.timedelta(seconds=d)) if d != 0 else "" for d in duracion_FC]

                total_duration = sum(duracion_FC)
                porcentaje = [(d / total_duration) * 100 if total_duration != 0 else 0 for d in duracion_FC]

                fig_hr = px.bar(x=zonas, y=duracion_FC, text=tiempo_total_FC)  # Usar color en lugar de marker_color
                fig_hr.update_traces(
                    hovertemplate='%{text} segundos (%{customdata:.2f}%)',
                    customdata=porcentaje,
                    marker_color=paleta
                    )
                fig_hr.update_layout(
                    title='Tiempo total en cada zona de FC',
                    xaxis_title='Zonas de FC',
                    yaxis_title='Duración',
                    yaxis=dict(
                        tickformat="%H:%M:%S"  # Formato de tiempo "hh:mm:ss"
                    ),
                )
                graficos.append(dcc.Graph(id='hr-zone-plot', figure=fig_hr))

                #tabla de FC

                zone_data = []
                for i in range(len(zonas)):
                    zone_data.append({
                        'Zone': zonas[i],
                        'Percentage Lower Bound': str(int(fc_zones[i]/FC_max*100))+"%" if i != 0 else "",
                        'Percentage Upper Bound': str(int(fc_zones[i + 1]/FC_max*100))+"%" if i < (len(fc_zones) - 2) else "",
                        'Lower Bound': int(fc_zones[i]) if i != 0 else "",
                        'Upper Bound': int(fc_zones[i + 1]) if i < (len(fc_zones) - 2) else "",
                        'Time Spent': tiempo_total_FC[i],
                        'Background Color': paleta[i]
                    })
                # Agregar el estilo condicional para el color de fondo
                table = dash_table.DataTable(
                    columns=[
                        {"name": "Zone", "id": "Zone"},
                        {"name": "% Lower Bound", "id": "Percentage Lower Bound"},
                        {"name": "% Upper Bound", "id": "Percentage Upper Bound"},
                        {"name": "Lower Bound", "id": "Lower Bound"},
                        {"name": "Upper Bound", "id": "Upper Bound"},
                        {"name": "Time Spent", "id": "Time Spent"}
                    ],
                    data=zone_data,
                     style_header={
                        'backgroundColor': 'white',
                        'fontWeight': 'bold',
                        'textAlign': 'center',
                        'whiteSpace': 'normal'
                    },
                    style_table={'height': '300px', 'width': '100%', 'overflowY': 'auto'},
                    style_data_conditional=[
                        {
                            'if': {'row_index': i},
                            'backgroundColor': paleta[i],
                            'color': 'black' if paleta[i]!='green' else 'white'  # Color del texto en cada fila
                        }
                        for i in range(len(paleta))
                    ],
                )
                tabla_div = html.Div(
                    children=[
                        html.H3("Tiempo en Zonas de FC"),
                        table
                    ]
                )
                tablas.append(tabla_div)



            if datos_entrenamiento["Watts"].isnull().all() and (tipo_entrenamiento == 'ciclismo' or tipo_entrenamiento == 'correr'):
              children.append(html.H2("No hay datos de Potencia"))

            elif tipo_entrenamiento == 'ciclismo' :
              datos_entrenamiento["Watts"].fillna(0)
              # Cálculo de las zonas de potencia (análogo al cálculo de zonas de FC)
              datos_entrenamiento["potencia_zona"] = pd.cut(datos_entrenamiento["Watts"], power_zones, labels=[str(i) for i in range(0,len(zonas))], include_lowest=True)
              duracion_Pot = [0] * len(zonas)
              datos_entrenamiento["potencia_zona"] = datos_entrenamiento["potencia_zona"].fillna(method='ffill')

              for i, row in datos_entrenamiento.iterrows():
                  zona = int(row["potencia_zona"])
                  duracion = row["Time"]
                  duracion_Pot[zona] += duracion
              tiempo_total_Pot = [str(datetime.timedelta(seconds=d)) if d != 0 else "" for d in duracion_Pot]
              total_duration = sum(duracion_Pot)
              porcentaje = [(d / total_duration) * 100 if total_duration != 0 else 0 for d in duracion_Pot]


              fig_watts = px.bar(x=zonas, y=duracion_Pot, text=tiempo_total_Pot)
              fig_watts.update_traces(
                  hovertemplate='%{text} segundos (%{customdata:.2f}%)',
                  customdata=porcentaje,
                  marker_color=paleta
              )
              fig_watts.update_layout(
                      title='Duración en zonas de Potencia',
                      xaxis_title='Zonas de Potencia',
                      yaxis_title='Duración',
                      yaxis=dict(
                          tickformat="%H:%M:%S"
                      ),
              )
              zone_data = []
              for i in range(len(zonas)):
                  zone_data.append({
                      'Zone': zonas[i],
                          'Percentage Lower Bound': str(int(power_zones[i]/pmax*100))+"%" if i != 0 else "",
                          'Percentage Upper Bound': str(int(power_zones[i + 1]/pmax*100))+"%" if i < (len(fc_zones) - 2) else "",
                          'Lower Bound': int(power_zones[i]) if i != 0 else "",
                          'Upper Bound': int(power_zones[i + 1]) if i < (len(fc_zones) - 2) else "",
                          'Time Spent': tiempo_total_Pot[i],
                          'Background Color': paleta[i]
                      })
                      # Agregar el estilo condicional para el color de fondo
                      table = dash_table.DataTable(
                        columns=[
                          {"name": "Zone", "id": "Zone"},
                          {"name": "% Lower Bound", "id": "Percentage Lower Bound"},
                          {"name": "% Upper Bound", "id": "Percentage Upper Bound"},
                          {"name": "Lower Bound", "id": "Lower Bound"},
                          {"name": "Upper Bound", "id": "Upper Bound"},
                          {"name": "Time Spent", "id": "Time Spent"}
                  ],
                  data=zone_data,
                   style_header={
                        'backgroundColor': 'white',
                        'fontWeight': 'bold',
                        'textAlign': 'center',
                        'whiteSpace': 'normal'
                    },
                    style_table={'height': '300px', 'width': '100%', 'overflowY': 'auto'},

                  style_data_conditional=[
                      {
                          'if': {'row_index': i},
                          'backgroundColor': paleta[i],
                          'color': 'black' if paleta[i]!='green' else 'white'  # Color del texto en cada fila
                      }
                      for i in range(len(paleta))
                  ],
              )
              tabla_div = html.Div(
                    children=[
                        html.H3("Tiempo en Zonas de Potencia"),
                        table
                    ]
                )
              tablas.append(tabla_div)
              graficos.append(dcc.Graph(id='watts-zone-plot', figure=fig_watts))

            elif tipo_entrenamiento == 'correr':
              if datos_entrenamiento["Watts"].isnull().all() != True:
                  datos_entrenamiento["Watts"].fillna(0)
                  datos_entrenamiento["potencia_zona"] = pd.cut(datos_entrenamiento["Watts"], power_zones, labels=[str(i) for i in range(0,len(zonas))], include_lowest=True)
                  duracion_Pot = [0] * len(zonas)
                  datos_entrenamiento["potencia_zona"] = datos_entrenamiento["potencia_zona"].fillna(method='ffill')

                  for i, row in datos_entrenamiento.iterrows():
                      zona = int(row["potencia_zona"])
                      duracion = row["Time"]
                      duracion_Pot[zona] += duracion
                  tiempo_total_Pot = [str(datetime.timedelta(seconds=d)) if d != 0 else "" for d in duracion_Pot]
                  total_duration = sum(duracion_Pot)
                  porcentaje = [(d / total_duration) * 100 if total_duration != 0 else 0 for d in duracion_Pot]


                  fig_watts = px.bar(x=zonas, y=duracion_Pot, text=tiempo_total_Pot)
                  fig_watts.update_traces(
                      hovertemplate='%{text} segundos (%{customdata:.2f}%)',
                      customdata=porcentaje,
                      marker_color=paleta
                  )
                  fig_watts.update_layout(
                      title='Duración en zonas de Potencia',
                      xaxis_title='Zonas de Potencia',
                      yaxis_title='Duración',
                      yaxis=dict(
                          tickformat="%H:%M:%S"
                      ),
                  )
                  zone_data = []
                  for i in range(len(zonas)):
                      zone_data.append({
                          'Zone': zonas[i],
                          'Percentage Lower Bound': str(int(power_zones[i]/pmax*100))+"%" if i != 0 else "",
                          'Percentage Upper Bound': str(int(power_zones[i + 1]/pmax*100))+"%" if i < (len(fc_zones) - 2) else "",
                          'Lower Bound': int(power_zones[i]) if i != 0 else "",
                          'Upper Bound': int(power_zones[i + 1]) if i < (len(fc_zones) - 2) else "",
                          'Time Spent': tiempo_total_Pot[i],
                          'Background Color': paleta[i]
                      })
                  # Agregar el estilo condicional para el color de fondo
                  table = dash_table.DataTable(
                      columns=[
                          {"name": "Zone", "id": "Zone"},
                          {"name": "% Lower Bound", "id": "Percentage Lower Bound"},
                          {"name": "% Upper Bound", "id": "Percentage Upper Bound"},
                          {"name": "Lower Bound", "id": "Lower Bound"},
                          {"name": "Upper Bound", "id": "Upper Bound"},
                          {"name": "Time Spent", "id": "Time Spent"}
                      ],
                      data=zone_data,
                       style_header={
                        'backgroundColor': 'white',
                        'fontWeight': 'bold',
                        'textAlign': 'center',
                        'whiteSpace': 'normal'
                    },
                    style_table={'height': '300px', 'width': '100%', 'overflowY': 'auto'},
                      style_data_conditional=[
                          {
                              'if': {'row_index': i},
                              'backgroundColor': paleta[i],
                              'color': 'black' if paleta[i]!='green' else 'white'  # Color del texto en cada fila
                          }
                          for i in range(len(paleta))
                      ],
                  )
                  tabla_div = html.Div(
                    children=[
                        html.H3("Tiempo en Zonas de Potencia"),
                        table
                    ]
                  )
                  tablas.append(tabla_div)
                  #children.append(html.H3("Tiempo en Zonas de Potencia"))
                  #children.append(table)
                  graficos.append(dcc.Graph(id='watts-zone-plot', figure=fig_watts))

            if tipo_entrenamiento == 'correr' and datos_entrenamiento["Speed"].isnull().all():
              children.append(html.H2("No hay datos de Velocidad"))

            elif tipo_entrenamiento == 'correr':
              datos_entrenamiento["velocidad_zona"] = pd.cut(datos_entrenamiento["Speed"], speed_zones, labels=[str(i) for i in range(0,len(zonas))], include_lowest=True)
              duracion_vel = [0] * len(zonas)
              datos_entrenamiento["velocidad_zona"] = datos_entrenamiento["velocidad_zona"].fillna(method='ffill')


              for i, row in datos_entrenamiento.iterrows():
                  zona = int(row["velocidad_zona"])
                  duracion = row["Time"]
                  duracion_vel[zona] += duracion
              tiempo_total_Vel = [str(datetime.timedelta(seconds=d)) if d != 0 else "" for d in duracion_vel]
              total_duration = sum(duracion_vel)
              porcentaje = [(d / total_duration) * 100 if total_duration != 0 else 0 for d in duracion_vel]

              fig_speed = px.bar(x=zonas, y=duracion_vel, text=tiempo_total_Vel)
              fig_speed.update_traces(
                  hovertemplate='%{text} segundos (%{customdata:.2f}%)',
                  customdata=porcentaje,
                  marker_color=paleta
              )
              fig_speed.update_layout(
                  title='Duración en zonas de Velocidad',
                  xaxis_title='Zonas de Velocidad',
                  yaxis_title='Duración',
                  yaxis=dict(
                      tickformat="%H:%M:%S"
                  ),

              )
              zone_data = []
              for i in range(len(zonas)):
                  zone_data.append({
                      'Zone': zonas[i],
                      'Percentage Lower Bound': str(int(speed_zones[i]/VAM*100))+"%" if i != 0 else "",
                      'Percentage Upper Bound': str(int(speed_zones[i + 1]/VAM*100))+"%" if i < (len(fc_zones) - 2) else "",
                      'Lower Bound': str(datetime.timedelta(seconds=int(3600/speed_zones[i]))) if i != 0 else "",
                      'Upper Bound': str(datetime.timedelta(seconds=int(3600/speed_zones[i + 1]))) if i < (len(fc_zones) - 2) else "",
                      'Time Spent': tiempo_total_Vel[i],
                      'Background Color': paleta[i]
                  })
              # Agregar el estilo condicional para el color de fondo
              table = dash_table.DataTable(
                  columns=[
                      {"name": "Zone", "id": "Zone"},
                      {"name": "% Lower Bound", "id": "Percentage Lower Bound"},
                      {"name": "% Upper Bound", "id": "Percentage Upper Bound"},
                      {"name": "Lower Bound", "id": "Lower Bound"},
                      {"name": "Upper Bound", "id": "Upper Bound"},
                      {"name": "Time Spent", "id": "Time Spent"}
                  ],
                  data=zone_data,
                   style_header={
                        'backgroundColor': 'white',
                        'fontWeight': 'bold',
                        'textAlign': 'center',
                        'whiteSpace': 'normal'
                    },
                    style_table={'height': '300px', 'width': '100%', 'overflowY': 'auto'},
                  style_data_conditional=[
                      {
                          'if': {'row_index': i},
                          'backgroundColor': paleta[i],
                          'color': 'black' if paleta[i]!='green' else 'white' # Color del texto en cada fila
                      }
                      for i in range(len(paleta))
                  ],
              )
              tabla_div = html.Div(
                    children=[
                        html.H3("Tiempo en Zonas de Velocidad"),
                        table
                    ]
              )
              tablas.append(tabla_div)
              #children.append(html.H3("Tiempo en Zonas de Velocidad"))
              #children.append(table)
              graficos.append(dcc.Graph(id='speed-zone-plot', figure=fig_speed))

            fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.1, subplot_titles=["Frecuencia Cardíaca", "Potencia", "Velocidad"])

            # Añadir el trace para Frecuencia Cardíaca
            fig.add_trace(go.Scatter(
                x=datos_entrenamiento['Accumulated Time'],
                y=datos_entrenamiento['Heart Rate'],
                mode='lines',
                name='Frecuencia Cardíaca',
                text=datos_entrenamiento['Elapsed Time Str'],
                hovertemplate='%{y} bpm<br>%{text}<extra></extra>',
                showlegend=False,
            ), row=1, col=1)

            # Añadir el trace para Potencia
            if (tipo_entrenamiento == 'ciclismo' or tipo_entrenamiento == 'correr') and not datos_entrenamiento["Watts"].isnull().all():
                fig.add_trace(go.Scatter(
                    x=datos_entrenamiento['Accumulated Time'],
                    y=datos_entrenamiento['Watts'],
                    mode='lines',
                    name='Potencia',
                    text=datos_entrenamiento['Elapsed Time Str'],
                    hovertemplate='%{y} Watts<br>%{text}<extra></extra>',
                    showlegend=False,
                ), row=2, col=1)

            # Añadir los traces para Velocidad
            if (tipo_entrenamiento == 'correr' or tipo_entrenamiento == 'correr') and not datos_entrenamiento["Speed"].isnull().all():
                fig.add_trace(go.Scatter(
                    x=datos_entrenamiento['Accumulated Time'],
                    y=datos_entrenamiento['Speed'],
                    mode='lines',
                    name='Velocidad',
                    text=[str(datetime.timedelta(seconds=int(3600/velocidad))) if velocidad != 0 else "" for velocidad in datos_entrenamiento["Speed"]],
                    hovertemplate='%{text} Min/km',
                    showlegend=False,
                    ), row=3, col=1)


            shapes_fc = []
            shapes_potencia = []
            shapes_velocidad = []


            # Ajustar los márgenes del eje y para cada métrica
            for i, color in enumerate(paleta):
                if i < len(fc_zones) - 1:
                    min_fc = max(min(datos_entrenamiento['Heart Rate']) - 5, 60)
                    max_fc = max(datos_entrenamiento['Heart Rate']) + 3 if fc_zones[-2] < (max(datos_entrenamiento['Heart Rate']) + 3) else fc_zones[-2]
                    shapes_fc.append({
                        'type': 'rect',
                        'x0': min(datos_entrenamiento['Accumulated Time']),
                        'y0': fc_zones[i],
                        'x1': max(datos_entrenamiento['Accumulated Time']),
                        'y1': fc_zones[i + 1],
                        'fillcolor': color,
                        'opacity': 0.5,
                        'layer': 'below',
                        'line_width': 0,
                    })

                    min_power = max(min(datos_entrenamiento['Watts']), 0)
                    max_power = max(datos_entrenamiento['Watts']) + 3 if power_zones[-2] < (max(datos_entrenamiento['Watts']) + 3) else power_zones[-2]
                    shapes_potencia.append({
                        'type': 'rect',
                        'x0': min(datos_entrenamiento['Accumulated Time']),
                        'y0': power_zones[i],
                        'x1': max(datos_entrenamiento['Accumulated Time']),
                        'y1': power_zones[i + 1],
                        'fillcolor': color,
                        'opacity': 0.5,
                        'layer': 'below',
                        'line_width': 0,
                    })

                    min_speed = max(min(datos_entrenamiento['Speed']) - 2, 0)
                    max_speed = max(datos_entrenamiento['Speed']) + 1 if speed_zones[-2] < (max(datos_entrenamiento['Speed']) + 1) else speed_zones[-2]
                    shapes_velocidad.append({
                        'type': 'rect',
                        'x0': min(datos_entrenamiento['Accumulated Time']),
                        'y0': speed_zones[i],
                        'x1': max(datos_entrenamiento['Accumulated Time']),
                        'y1': speed_zones[i+1],
                        'fillcolor': color,
                        'opacity': 0.5,
                        'layer': 'below',
                        'line_width': 0,
                    })

            # Actualizar las etiquetas de los ejes y rango de la y-axis
            fig.update_yaxes(title_text="bpm", range=[min_fc, max_fc], row=1, col=1)
            fig.update_yaxes(title_text="Watts", range=[min_power, max_power], row=2, col=1)
            fig.update_yaxes(title_text="min/km", range=[min_speed, max_speed], row=3, col=1)



            # Añadir shapes a los gráficos
            for shape in shapes_fc:
                fig.add_shape(shape, row=1, col=1)

            for shape in shapes_potencia:
                fig.add_shape(shape, row=2, col=1)

            for shape in shapes_velocidad:
                fig.add_shape(shape, row=3, col=1)

            # Actualizar la configuración del gráfico
            fig.update_layout(
                xaxis=dict(
                    title='Tiempo Transcurrido',
                    tickvals=tick_vals,
                    ticktext=tick_texts,
                    showgrid=False
                ),
                dragmode='select',
                height=600
            )
            fig.update_xaxes(
                    title_text='Tiempo Transcurrido',
                    tickvals=tick_vals,  # Valores de los ticks basados en índices seleccionados
                    ticktext=tick_texts,  # Textos de los ticks correspondientes
                    showgrid=False,
                    row=3, col=1,
                    range=[min(datos_entrenamiento['Accumulated Time']), max(datos_entrenamiento['Accumulated Time'])]  # Rango específico para Velocidad
                )

            # Añadir el gráfico al layout de la aplicación
            children.append(dcc.Graph(id='metricas-entrenamiento', figure=fig))



            # Inicializar una lista para contener las filas de tablas
            rows = []

            # Iterar sobre las tablas
            for i in range(0, len(tablas), 3):
                # Crear una fila para las tablas
                row_children = []
                for j in range(i, min(i+3, len(tablas))):
                    row_children.append(dbc.Col(tablas[j], width=4))  # Ajusta el ancho según sea necesario
                tabla_row = dbc.Row(row_children, className='mb-4')
                rows.append(tabla_row)

            # Agregar las filas de tablas al layout
            children.extend(rows)

            # Agregar los gráficos a la lista de elementos
            for i in range(len(graficos)):
                # Crear una fila para los gráficos
                children.append(html.Div(
                graficos[i],
                style={'margin': 'auto', 'width': '70%'}
            ))
            return children

        except Exception as e:
            empty_fig = px.bar(x=[0,0,0,0], y=[0,0,0,0])
            empty_fig.update_layout(
                title=str(e),
                xaxis_title='',
                yaxis_title=''
            )
            return dcc.Graph(id='hr-error-plot', figure=empty_fig)
    else:
        return []

if __name__ == "__main__":
    app.run_server(debug=False)

