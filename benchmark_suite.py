import ast
import json
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import gradio as gr
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score

# ============================================================
# CODEX BENCHMARK SUITE
# ============================================================

ALLOWED_NAMES = {
    "np": np,
    "sin": np.sin,
    "cos": np.cos,
    "tan": np.tan,
    "exp": np.exp,
    "sqrt": np.sqrt,
    "log": np.log,
    "abs": np.abs,
}

def safe_eval(expr, x):
    tree = ast.parse(expr, mode="eval")

    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            if node.id not in ALLOWED_NAMES and node.id != "x":
                raise ValueError(f"Unauthorized identifier: {node.id}")

    scope = ALLOWED_NAMES.copy()
    scope["x"] = x

    return eval(
        compile(tree, "<string>", "eval"),
        {"__builtins__": {}},
        scope,
    )

# ============================================================
# EQUATION PLOTTER
# ============================================================

def plot_equation(label, expr):
    try:
        x = np.linspace(-10, 10, 500)
        y = safe_eval(expr, x)

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode="lines",
                name=label
            )
        )

        fig.update_layout(
            title=f"Equation: {label}",
            template="plotly_white",
            xaxis_title="x",
            yaxis_title="y"
        )

        return fig

    except Exception as e:
        fig = go.Figure()
        fig.update_layout(
            title=f"Error: {str(e)}"
        )
        return fig

# ============================================================
# REGRESSION BENCHMARK
# ============================================================

def regression_dashboard():

    np.random.seed(42)

    X_raw = np.linspace(-5, 5, 60)

    y_true = 2 * X_raw**2 - 3 * X_raw + 1

    y = y_true + np.random.normal(0, 5, 60)

    X = X_raw.reshape(-1, 1)

    x_plot = np.linspace(
        -5.5,
        5.5,
        300
    ).reshape(-1, 1)

    fig = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=[
            "Degree 1",
            "Degree 2",
            "Degree 4"
        ]
    )

    scores = []

    for idx, deg in enumerate([1, 2, 4], start=1):

        model = make_pipeline(
            PolynomialFeatures(deg),
            LinearRegression()
        )

        model.fit(X, y)

        pred = model.predict(X)

        r2 = r2_score(y, pred)

        scores.append(
            {
                "Degree": deg,
                "R² Score": round(r2, 4)
            }
        )

        fig.add_trace(
            go.Scatter(
                x=X_raw,
                y=y,
                mode="markers",
                name=f"Data D{deg}"
            ),
            row=1,
            col=idx
        )

        fig.add_trace(
            go.Scatter(
                x=x_plot.flatten(),
                y=model.predict(x_plot),
                mode="lines",
                name=f"Degree {deg}"
            ),
            row=1,
            col=idx
        )

    fig.update_layout(
        title="Codex Regression Benchmark",
        template="plotly_white",
        height=500
    )

    return fig, pd.DataFrame(scores)

# ============================================================
# GRADIENT DESCENT
# ============================================================

def run_gradient_descent():

    w = 3.0
    lr = 0.1

    history = []

    for step in range(31):

        history.append(
            {
                "step": step,
                "w": w,
                "loss": w**2
            }
        )

        w = w - lr * (2 * w)

    df = pd.DataFrame(history)

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=df["step"],
            y=df["loss"],
            mode="lines+markers",
            name="Loss"
        )
    )

    fig.update_layout(
        title="Gradient Descent Convergence",
        template="plotly_white",
        xaxis_title="Step",
        yaxis_title="Loss"
    )

    final_w = round(df.iloc[-1]["w"], 4)

    return fig, f"Final w after 30 steps = {final_w}"

# ============================================================
# EXPORT RESULTS
# ============================================================

def export_results():

    _, scores_df = regression_dashboard()

    gd_fig, gd_result = run_gradient_descent()

    scores_df.to_csv(
        "codex_benchmark_results.csv",
        index=False
    )

    result_json = {
        "gradient_descent": gd_result,
        "regression_scores":
            scores_df.to_dict(
                orient="records"
            )
    }

    with open(
        "codex_benchmark_results.json",
        "w"
    ) as f:
        json.dump(
            result_json,
            f,
            indent=2
        )

    return (
        "Export complete.\n"
        "Files created:\n"
        "- codex_benchmark_results.csv\n"
        "- codex_benchmark_results.json"
    )

# ============================================================
# GRADIO UI

# ============================================================

with gr.Blocks(
    title="Codex Benchmark Suite"
) as demo:

    gr.Markdown(
        """
        # 🚀 Codex Benchmark Suite

        ### Math Equations
        ### Regression Benchmark
        ### Gradient Descent Benchmark
        """
    )

    with gr.Tab("Equation Plotter"):

        label = gr.Textbox(
            label="Equation Label",
            value="Quadratic"
        )

        expr = gr.Textbox(
            label="Expression",
            value="2*x**2 + 3*x - 5"
        )

        plot_btn = gr.Button(
            "Plot Equation"
        )

        eq_plot = gr.Plot()

        plot_btn.click(
            plot_equation,
            inputs=[label, expr],
            outputs=eq_plot
        )

    with gr.Tab("Regression Benchmark"):

        reg_btn = gr.Button(
            "Run Regression Benchmark"
        )

        reg_plot = gr.Plot()

        reg_table = gr.Dataframe()

        reg_btn.click(
            regression_dashboard,
            outputs=[
                reg_plot,
                reg_table
            ]
        )

    with gr.Tab("Gradient Descent"):

        gd_btn = gr.Button(
            "Run Gradient Descent"
        )

        gd_plot = gr.Plot()

        gd_text = gr.Textbox()

        gd_btn.click(
            run_gradient_descent,
            outputs=[
                gd_plot,
                gd_text
            ]
        )

    with gr.Tab("Export Results"):

        export_btn = gr.Button(
            "Export CSV / JSON"
        )

        export_text = gr.Textbox()

        export_btn.click(
            export_results,
            outputs=export_text
        )

demo.launch()
