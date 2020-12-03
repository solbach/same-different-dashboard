import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.express as px
import plotly.graph_objs as go
import pandas as pd


def init_web():
    external_stylesheets = ['assets/style.css']
    app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

    return app


def loading_data():
    data_source = '/home/markus/git/jtl/same-different-experiment/src/TobiiProGlasses/src/data/SD/trial_34.pkl'
    df = pd.read_pickle(data_source)

    return df


def plot_trajectory(df):
    # Plotting
    # fig = px.line_3d(df, x="tobii_x", y="tobii_z", z='tobii_y', width=825, height=675)

    object_left, object_right = get_objects(df)
    trajectory = get_trajectory(df)
    annot_start, annot_stop = get_annotations(df)


    # Create Figure with traces
    fig = go.Figure(data=[object_left, object_right, trajectory, annot_start, annot_stop])

    # Set Parameters of Figure
    zoom_factor = 0.80
    camera = dict(
        up=dict(x=0, y=1, z=0),
        center=dict(x=0, y=0, z=0),
        eye=dict(x=0, y=2 * zoom_factor, z=1 * zoom_factor)
    )

    fig.update_layout(scene_camera=camera,
                      autosize=False,
                      width=825,
                      height=675,
                      font=dict(color="#FFFFFF"),
                      scene=dict(
                          xaxis=dict(range=[-1.7, 1.5], showgrid=True, gridwidth=1, gridcolor='#666666',
                                     backgroundcolor="#222222", zeroline=True, zerolinecolor='#666666'),
                          yaxis=dict(range=[-2, 2], showgrid=True, gridwidth=1, gridcolor='#666666',
                                     backgroundcolor="#222222", zeroline=True, zerolinecolor='#666666'),
                          zaxis=dict(range=[-2.5, 2], showgrid=True, gridwidth=1, gridcolor='#666666',
                                     backgroundcolor="#222222", zeroline=True, zerolinecolor='#666666'),
                          bgcolor='#222222'),
                      paper_bgcolor="#222222",
                      plot_bgcolor="#222222")

    return fig


def get_annotations(df):
    annot_start = go.Scatter3d(
        x=[df.loc[df.first_valid_index(), 'tobii_x']],
        y=[df.loc[df.first_valid_index(), 'tobii_y']],
        z=[df.loc[df.first_valid_index(), 'tobii_z']],
        mode="markers+text",
        showlegend=False,
        text=["Start"],
        marker=dict(
            size=3,
            line=dict(width=1,
                      color='#FFFFFF'),
            color="#c370e0",
            symbol="square",
            opacity=0.8
        ),
        textposition="bottom center"
    )

    annot_stop = go.Scatter3d(
        x=[df.loc[df.last_valid_index(), 'tobii_x']],
        y=[df.loc[df.last_valid_index(), 'tobii_y']],
        z=[df.loc[df.last_valid_index(), 'tobii_z']],
        mode="markers+text",
        showlegend=False,
        text=["Stop"],
        marker=dict(
            size=3,
            line=dict(width=1,
                      color='#FFFFFF'),
            color="#c370e0",
            symbol="x",
            opacity=0.8
        ),
        textposition="bottom center"
    )

    return annot_start, annot_stop


def get_trajectory(df):
    # Trajectory
    trajectory = go.Scatter3d(
        x=df['tobii_x'],
        y=df['tobii_y'],
        z=df['tobii_z'],
        mode='lines',
        line=dict(
            width=2,
            # dash='dot',
            color="#FFFFFF"
        ),
        name='trajectory'
    )
    return trajectory


def get_objects(df):
    obj_left_x = df['object_left_x'].mean()
    obj_left_y = df['object_left_y'].mean()
    obj_left_z = df['object_left_z'].mean()
    obj_right_x = df['object_right_x'].mean()
    obj_right_y = df['object_right_y'].mean()
    obj_right_z = df['object_right_z'].mean()

    object_left = go.Scatter3d(
        x=[obj_left_x],
        y=[obj_left_y],
        z=[obj_left_z],
        mode='markers',
        marker=dict(
            size=5,
            line=dict(width=1,
                      color='#FFFFFF'),
            color="#12c2e9",
            symbol="circle",
            opacity=0.8
        ),
        name='object left'
    )
    object_right = go.Scatter3d(
        x=[obj_right_x],
        y=[obj_right_y],
        z=[obj_right_z],
        mode='markers',
        marker=dict(
            size=5,
            line=dict(width=1,
                      color='#FFFFFF'),
            color="#f64f59",
            symbol="circle",
            opacity=0.8
        ),
        name='object right'
    )
    return object_left, object_right


def create_dashboard(app, df):
    fig = plot_trajectory(df)

    app.layout = html.Div(children=[
        html.H1(children='Same-Different Analysis Dashboard'),

        html.Div(children='''
                Some more text can go here.
            '''),

        dcc.Graph(
            id='trajectory_plot',
            figure=fig
        )
    ])

    return app


if __name__ == '__main__':
    init_web()
    df = loading_data()

    app = init_web()
    app = create_dashboard(app, df)

    app.run_server(debug=True)
