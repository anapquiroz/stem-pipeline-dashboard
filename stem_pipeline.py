import pandas as pd
import plotly.express as px
from dash import Dash, dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import numpy as np
import os

# Load the data
df = pd.read_csv("NCES_IPEDS_RAW_DATA.csv", dtype={"Award Level Code": str})


# Melt year columns into long format
id_vars = ["Citizenship", "CIP Code and Description (2 digit)", "Award Level Code"]
year_cols = [str(y) for y in range(1997, 2024)]
df_melted = df.melt(id_vars=id_vars, value_vars=year_cols, var_name="Year", value_name="Count")
df_melted["Count"] = pd.to_numeric(df_melted["Count"], errors="coerce").fillna(0)

# Clean up columns
citizenship_map = {
    "U.S. citizen or permanent resident": "US",
    "Nonresident alien (temporary visa)": "INTL"
}
award_map = {"17": "BA", "18": "MA", "19": "PhD"}

df_melted["Citizenship Abbr"] = df_melted["Citizenship"].map(citizenship_map).fillna(df_melted["Citizenship"])
df_melted["Award Abbr"] = df_melted["Award Level Code"].astype(str).map(award_map).fillna(df_melted["Award Level Code"].astype(str))
df_melted["CIP Label"] = df_melted["CIP Code and Description (2 digit)"].str.extract(r"\d+\s*[-:]*\s*(.*)")[0].fillna("")
df_melted["Group"] = df_melted["Citizenship Abbr"] + " | " + df_melted["Award Abbr"] + " | " + df_melted["CIP Label"]
df_melted["Year"] = df_melted["Year"].astype(int)

# Ensure all combinations of group and year exist
all_years = list(range(1997, 2024))
df_melted["Group"] = df_melted["Group"].fillna("Unknown")
df_melted["Citizenship Abbr"] = df_melted["Citizenship Abbr"].fillna("Unknown")
df_melted["Award Abbr"] = df_melted["Award Abbr"].fillna("Unknown")
df_melted["CIP Label"] = df_melted["CIP Label"].fillna("Unknown")

all_groups = df_melted["Group"].unique()
index = pd.MultiIndex.from_product([all_groups, all_years], names=["Group", "Year"])
# Include all Award Abbr and Years by expanding for each combination
group_year_index = pd.MultiIndex.from_product([df_melted["Group"].unique(), all_years], names=["Group", "Year"])
df_grouped = df_melted.set_index(["Group", "Year"])["Count"].reindex(group_year_index, fill_value=0).reset_index()
df_grouped = df_grouped.set_index(["Group", "Year"]).reindex(index, fill_value=0).reset_index()
# Ensure label rows come from all years including earliest PhD records
df_labels = (df_melted[df_melted["Count"] > 0]
              .sort_values("Year")
              .drop_duplicates("Group")
              [["Group", "Citizenship Abbr", "Award Abbr", "CIP Label"]])
df_complete = df_grouped.merge(df_labels, on="Group", how="left")

# App setup
app = Dash(__name__, external_stylesheets=[dbc.themes.FLATLY, "https://use.fontawesome.com/releases/v5.15.4/css/all.css"])
server = app.server

app.layout = html.Div(
    [  # begin children list
    html.Div([
        html.H1("The US STEM Talent Pipeline From 1997 to 2023", style={"color": "#003f87"}),
        html.H6("Interactive Dashboard", className="text-muted")
    ], style={"textAlign": "center", "marginBottom": "30px"}),

    html.Div(id="summary-cards", style={"display": "flex", "gap": "30px", "padding": "20px 0", "flexWrap": "wrap", "justifyContent": "center"}),

    html.Div([
        html.Label("Select Citizenship"),
        dcc.Dropdown(id="citizenship-filter", options=[{"label": c, "value": c} for c in sorted(df_complete["Citizenship Abbr"].dropna().unique())], value=[], multi=True),

        html.Label("Select Award Level"),
        dcc.Dropdown(id="award-filter", options=[{"label": a, "value": a} for a in sorted(df_complete["Award Abbr"].dropna().unique())], value=[], multi=True),

        html.Label("Select Field of Study"),
        dcc.Dropdown(id="field-filter", options=[{"label": f, "value": f} for f in sorted(df_complete["CIP Label"].dropna().unique())], value=[], multi=True),

        html.Label("Select Year Range"),
        dcc.RangeSlider(id="year-slider", min=1997, max=2023, value=[1997, 2023], marks={y: str(y) for y in range(1997, 2024, 2)}),

        html.Label("Select Chart Type"),
        dcc.RadioItems(id="chart-type", options=[
            {"label": "Line Chart", "value": "line"},
            {"label": "Bar Chart", "value": "bar"},
            {"label": "Stacked Bar Chart", "value": "stack"}
        ], value="line"),

        html.Br(),
        html.Button("Download CSV", id="download-button", className="btn btn-secondary me-2"),
        html.Button("Reset Filters", id="reset-button", className="btn btn-outline-dark"),
        dcc.Download(id="download-dataframe-csv")
    ], style={"padding": "20px", "backgroundColor": "#e9ecef", "border": "1px solid #adb5bd", "borderRadius": "0.5rem", "marginBottom": "30px", "boxShadow": "0 2px 5px rgba(0, 0, 0, 0.1)"}),

    dcc.Graph(id="line-chart", config={"displaylogo": False}, style={"marginTop": "40px"})
    ], style={"backgroundColor": "#ffffff", "padding": "30px"})

@app.callback(
    Output("download-dataframe-csv", "data"),
    Input("download-button", "n_clicks"),
    State("citizenship-filter", "value"),
    State("award-filter", "value"),
    State("field-filter", "value"),
    State("year-slider", "value"),
    prevent_initial_call=True
)
def download_csv(n_clicks, citizenship, award, field, year_range):
    df_filtered = df_complete.copy()
    if citizenship:
        df_filtered = df_filtered[df_filtered["Citizenship Abbr"].isin(citizenship)]
    if award:
        df_filtered = df_filtered[df_filtered["Award Abbr"].isin(award)]
    if field:
        df_filtered = df_filtered[df_filtered["CIP Label"].isin(field)]
    df_filtered = df_filtered[df_filtered["Year"].between(year_range[0], year_range[1])]
    return dcc.send_data_frame(df_filtered.to_csv, filename="filtered_data.csv")

@app.callback(
    Output("citizenship-filter", "value"),
    Output("award-filter", "value"),
    Output("field-filter", "value"),
    Output("year-slider", "value"),
    Input("reset-button", "n_clicks")
)
def reset_filters(n_clicks):
    return [], [], [], [1997, 2023]

@app.callback(
    Output("line-chart", "figure"),
    Input("citizenship-filter", "value"),
    Input("award-filter", "value"),
    Input("field-filter", "value"),
    Input("year-slider", "value"),
    Input("chart-type", "value")
)
def update_chart(citizenship, award, field, year_range, chart_type):
    df_filtered = df_complete.copy()
    if citizenship:
        df_filtered = df_filtered[df_filtered["Citizenship Abbr"].isin(citizenship)]
    if award:
        df_filtered = df_filtered[df_filtered["Award Abbr"].isin(award)]
    if field:
        df_filtered = df_filtered[df_filtered["CIP Label"].isin(field)]
    df_filtered = df_filtered[df_filtered["Year"].between(year_range[0], year_range[1])]

    if df_filtered.empty:
        fig = px.line()
        fig.update_layout(title="No data available for the selected filters")
        return fig

    if chart_type == "bar":
        fig = px.bar(df_filtered, x="Year", y="Count", color="Group", barmode="group")
    elif chart_type == "stack":
        fig = px.bar(df_filtered, x="Year", y="Count", color="Group", barmode="stack")
    else:
        fig = px.line(df_filtered, x="Year", y="Count", color="Group")

    fig.update_layout(title="The US STEM Talent Pipeline From 1997 to 2023", hovermode="x unified")
    return fig

@app.callback(
    Output("summary-cards", "children"),
    Input("citizenship-filter", "value"),
    Input("award-filter", "value"),
    Input("field-filter", "value"),
    Input("year-slider", "value")
)
def update_summary(citizenship, award, field, year_range):
    df_filtered = df_complete.copy()
    if citizenship:
        df_filtered = df_filtered[df_filtered["Citizenship Abbr"].isin(citizenship)]
    if award:
        df_filtered = df_filtered[df_filtered["Award Abbr"].isin(award)]
    if field:
        df_filtered = df_filtered[df_filtered["CIP Label"].isin(field)]
    df_filtered = df_filtered[df_filtered["Year"].between(year_range[0], year_range[1])]

    total = df_filtered["Count"].sum()
    grouped = df_filtered.groupby("Year")["Count"].sum().sort_index()
    if grouped.empty:
        return []
    peak_year = grouped.idxmax() if not grouped.empty else "N/A"

    return [
        dbc.Card(dbc.CardBody([
            html.H5("Total Degrees"),
            html.H3(f"{int(total):,}"),
            html.P("Total number of degrees awarded", className="text-muted")
        ])),
        dbc.Card(dbc.CardBody([
            html.H5("Peak Year"),
            html.H3(str(peak_year)),
            html.P("Year with the most degrees awarded", className="text-muted")
        ]))
    ]

if __name__ == '__main__':
    import os
    port = int(os.environ.get("PORT", 8050))
    app.run(host="0.0.0.0", port=port, debug=True)























