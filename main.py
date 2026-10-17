
import ast
import math
import tkinter as tk
from tkinter import ttk, messagebox

import sympy as sp
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


# =========================================================
# 1. SAFE MATHEMATICAL EXPRESSION PARSER
# =========================================================

x = sp.Symbol("x")

FUNCTIONS = {
    "sin": sp.sin,
    "cos": sp.cos,
    "tan": sp.tan,
    "asin": sp.asin,
    "acos": sp.acos,
    "atan": sp.atan,
    "sinh": sp.sinh,
    "cosh": sp.cosh,
    "tanh": sp.tanh,
    "exp": sp.exp,
    "ln": sp.log,
    "log": sp.log,
    "sqrt": sp.sqrt,
    "abs": sp.Abs,
    "floor": sp.floor,
    "ceil": sp.ceiling,
}

CONSTANTS = {
    "x": x,
    "pi": sp.pi,
    "e": sp.E,
}


class ExpressionParser(ast.NodeVisitor):
    """Allow only approved mathematical expressions."""

    def visit_Expression(self, node):
        return self.visit(node.body)

    def visit_Constant(self, node):
        if type(node.value) not in (int, float):
            raise ValueError(
                "Only numeric constants are allowed."
            )

        if not math.isfinite(node.value):
            raise ValueError("The number must be finite.")

        return sp.sympify(node.value)

    def visit_Name(self, node):
        if node.id in CONSTANTS:
            return CONSTANTS[node.id]

        raise ValueError(f"Unknown symbol: {node.id}")

    def visit_BinOp(self, node):
        left = self.visit(node.left)
        right = self.visit(node.right)

        operations = {
            ast.Add: lambda a, b: a + b,
            ast.Sub: lambda a, b: a - b,
            ast.Mult: lambda a, b: a * b,
            ast.Div: lambda a, b: a / b,
            ast.Pow: lambda a, b: a ** b,
        }

        operation = operations.get(type(node.op))

        if operation is None:
            raise ValueError(
                "Unsupported mathematical operator."
            )

        return operation(left, right)

    def visit_UnaryOp(self, node):
        value = self.visit(node.operand)

        if isinstance(node.op, ast.UAdd):
            return value

        if isinstance(node.op, ast.USub):
            return -value

        raise ValueError("Unsupported unary operator.")

    def visit_Call(self, node):
        if not isinstance(node.func, ast.Name):
            raise ValueError(
                "Only approved functions are allowed."
            )

        name = node.func.id

        if name not in FUNCTIONS:
            raise ValueError(
                f"Unsupported function: {name}"
            )

        if node.keywords:
            raise ValueError(
                "Keyword arguments are not allowed."
            )

        args = [self.visit(arg) for arg in node.args]

        if name == "log":
            if len(args) not in (1, 2):
                raise ValueError(
                    "log requires one or two arguments."
                )
        elif len(args) != 1:
            raise ValueError(
                f"{name} requires one argument."
            )

        return FUNCTIONS[name](*args)

    def generic_visit(self, node):
        raise ValueError(
            "Unsupported expression element: "
            + type(node).__name__
        )


def parse_function(expression):
    expression = expression.strip()

    if not expression:
        raise ValueError(
            "Enter a mathematical function."
        )

    # Allow x^2 as an alternative to x**2.
    expression = expression.replace("^", "**")

    syntax_tree = ast.parse(expression, mode="eval")
    result = ExpressionParser().visit(syntax_tree)

    if result.has(sp.zoo, sp.nan, sp.oo, -sp.oo):
        raise ValueError(
            "The expression contains an invalid value."
        )

    return result


def evaluate_function(function, value):
    result = function(float(value))

    if isinstance(result, complex):
        if abs(result.imag) > 1e-12:
            raise ValueError(
                "The function returned a complex value."
            )

        result = result.real

    result = float(result)

    if not math.isfinite(result):
        raise ValueError(
            "The function returned an invalid value."
        )

    return result


# =========================================================
# 2. SECANT METHOD
# =========================================================

def secant_method(
    function, x0, x1, tolerance, max_iterations
):
    f0 = evaluate_function(function, x0)
    f1 = evaluate_function(function, x1)

    rows = [
        (0, x0, f0, None),
        (1, x1, f1, None),
    ]

    if abs(f0) <= tolerance:
        return x0, rows, "The first guess is already a root."

    if abs(f1) <= tolerance:
        return x1, rows, "The second guess is already a root."

    for iteration in range(2, max_iterations + 2):
        denominator = f1 - f0

        if abs(denominator) < 1e-14:
            raise ValueError(
                "The denominator is too close to zero. "
                "Try different initial guesses."
            )

        x2 = x1 - f1 * (x1 - x0) / denominator

        f2 = evaluate_function(function, x2)
        error = abs(x2 - x1)

        rows.append((iteration, x2, f2, error))

        if abs(f2) <= tolerance or error <= tolerance:
            return x2, rows, "Converged successfully."

        x0, x1 = x1, x2
        f0, f1 = f1, f2

    return x1, rows, "Maximum iterations reached."


# =========================================================
# 3. TKINTER GUI
# =========================================================

class SecantCalculator:

    def __init__(self, root):
        self.root = root
        self.root.title("Secant Method Calculator")
        self.root.geometry("1200x800")
        self.root.minsize(900, 650)

        self.expression = tk.StringVar(
            value="x^2 - 4"
        )
        self.x0 = tk.StringVar(value="1")
        self.x1 = tk.StringVar(value="3")
        self.tolerance = tk.StringVar(
            value="0.000001"
        )
        self.max_iterations = tk.StringVar(value="100")
        self.status = tk.StringVar(
            value="Enter a function and click Calculate."
        )

        self.last_function = None
        self.last_root = None

        self.build_gui()

    def build_gui(self):
        main = ttk.Frame(self.root, padding=10)
        main.pack(fill="both", expand=True)

        main.columnconfigure(0, weight=1)
        main.columnconfigure(1, weight=2)
        main.rowconfigure(0, weight=1)

        left = ttk.Frame(main, padding=5)
        left.grid(row=0, column=0, sticky="nsew")

        right = ttk.Frame(main, padding=5)
        right.grid(row=0, column=1, sticky="nsew")

        right.columnconfigure(0, weight=1)
        right.rowconfigure(3, weight=1)

        # Function input
        ttk.Label(
            left,
            text="Secant Method Calculator",
            font=("Arial", 16, "bold")
        ).pack(anchor="w", pady=(0, 12))

        ttk.Label(
            left, text="Function f(x):"
        ).pack(anchor="w")

        self.expression_entry = ttk.Entry(
            left,
            textvariable=self.expression,
            font=("Consolas", 13)
        )
        self.expression_entry.pack(
            fill="x", pady=5
        )

        # Calculator buttons
        keypad = ttk.LabelFrame(
            left, text="Function Builder", padding=6
        )
        keypad.pack(fill="x", pady=8)

        buttons = [
            ["7", "8", "9", "/", "("],
            ["4", "5", "6", "*", ")"],
            ["1", "2", "3", "-", "^"],
            ["0", ".", "x", "+", "pi"],
            ["sin", "cos", "tan", "exp", "sqrt"],
            ["ln", "log", "abs", "e", "⌫"],
        ]

        for r, row in enumerate(buttons):
            for c, label in enumerate(row):
                button = ttk.Button(
                    keypad,
                    text=label,
                    command=lambda value=label:
                        self.insert_token(value)
                )

                button.grid(
                    row=r,
                    column=c,
                    sticky="nsew",
                    padx=2,
                    pady=2,
                    ipadx=2,
                    ipady=5
                )

        for c in range(5):
            keypad.columnconfigure(c, weight=1)

        controls = ttk.Frame(keypad)
        controls.grid(
            row=len(buttons),
            column=0,
            columnspan=5,
            sticky="ew",
            pady=(5, 0)
        )

        ttk.Button(
            controls,
            text="Clear",
            command=self.clear_expression
        ).pack(
            side="left", fill="x", expand=True, padx=2
        )

        ttk.Button(
            controls,
            text="Insert x",
            command=lambda: self.insert_token("x")
        ).pack(
            side="left", fill="x", expand=True, padx=2
        )

        ttk.Label(
            left,
            text=(
                "Use explicit multiplication, e.g. 2*x."
            ),
            wraplength=350
        ).pack(anchor="w", pady=4)

        # Initial assumptions
        settings = ttk.LabelFrame(
            left, text="Initial Assumptions", padding=8
        )
        settings.pack(fill="x", pady=8)

        fields = [
            ("First guess (x0)", self.x0),
            ("Second guess (x1)", self.x1),
            ("Tolerance", self.tolerance),
            ("Maximum iterations", self.max_iterations),
        ]

        for r, (label, variable) in enumerate(fields):
            ttk.Label(
                settings, text=label
            ).grid(
                row=r, column=0, sticky="w", pady=4
            )

            ttk.Entry(
                settings,
                textvariable=variable,
                width=16
            ).grid(
                row=r,
                column=1,
                sticky="ew",
                padx=5,
                pady=4
            )

        settings.columnconfigure(1, weight=1)

        ttk.Button(
            left,
            text="Calculate Root",
            command=self.calculate
        ).pack(fill="x", pady=(5, 4), ipady=6)

        ttk.Button(
            left,
            text="Reset",
            command=self.reset
        ).pack(fill="x")

        ttk.Label(
            left,
            textvariable=self.status,
            wraplength=350
        ).pack(anchor="w", pady=10)

        # Iteration table
        ttk.Label(
            right,
            text="Iteration Results",
            font=("Arial", 13, "bold")
        ).grid(
            row=0, column=0, sticky="w", pady=(0, 5)
        )

        table_frame = ttk.Frame(right)
        table_frame.grid(
            row=1, column=0, sticky="ew"
        )
        table_frame.columnconfigure(0, weight=1)

        columns = ("iteration", "xn", "fx", "error")

        self.table = ttk.Treeview(
            table_frame,
            columns=columns,
            show="headings",
            height=9
        )

        headings = [
            ("iteration", "Iteration", 75),
            ("xn", "x_n", 150),
            ("fx", "f(x_n)", 150),
            ("error", "Approx. Error", 150),
        ]

        for column, title, width in headings:
            self.table.heading(column, text=title)
            self.table.column(
                column,
                width=width,
                anchor="center"
            )

        scrollbar = ttk.Scrollbar(
            table_frame,
            orient="vertical",
            command=self.table.yview
        )

        self.table.configure(
            yscrollcommand=scrollbar.set
        )

        self.table.grid(
            row=0, column=0, sticky="ew"
        )
        scrollbar.grid(
            row=0, column=1, sticky="ns"
        )

        # Graph
        ttk.Label(
            right,
            text="Function Graph",
            font=("Arial", 13, "bold")
        ).grid(
            row=2, column=0, sticky="nw", pady=(12, 5)
        )

        self.figure = Figure(
            figsize=(6, 4), dpi=100
        )

        self.axes = self.figure.add_subplot(111)
        self.axes.set_title(
            "Enter a function and calculate"
        )
        self.axes.set_xlabel("x")
        self.axes.set_ylabel("f(x)")
        self.axes.axhline(
            0, color="black", linewidth=0.8
        )
        self.axes.grid(True, alpha=0.3)

        self.canvas = FigureCanvasTkAgg(
            self.figure, master=right
        )

        self.canvas.get_tk_widget().grid(
            row=3, column=0, sticky="nsew"
        )

    # =====================================================
    # 4. FUNCTION BUILDER
    # =====================================================

    def insert_token(self, token):
        entry = self.expression_entry

        if token == "⌫":
            position = entry.index(tk.INSERT)

            if position > 0:
                entry.delete(position - 1, position)

            return

        replacements = {
            "sin": "sin(",
            "cos": "cos(",
            "tan": "tan(",
            "exp": "exp(",
            "sqrt": "sqrt(",
            "ln": "ln(",
            "log": "log(",
            "abs": "abs(",
        }

        text = replacements.get(token, token)

        position = entry.index(tk.INSERT)
        entry.insert(position, text)
        entry.focus_set()

    def clear_expression(self):
        self.expression.set("")
        self.expression_entry.focus_set()

    # =====================================================
    # 5. CALCULATE AND DISPLAY RESULTS
    # =====================================================

    def calculate(self):
        try:
            expression = parse_function(
                self.expression.get()
            )

            numerical_function = sp.lambdify(
                x, expression, modules=["math"]
            )

            x0 = float(self.x0.get())
            x1 = float(self.x1.get())
            tolerance = float(self.tolerance.get())
            max_iterations = int(
                self.max_iterations.get()
            )

            if not all(
                map(math.isfinite, [x0, x1, tolerance])
            ):
                raise ValueError(
                    "Guesses and tolerance must be finite numbers."
                )

            if tolerance <= 0:
                raise ValueError(
                    "Tolerance must be greater than zero."
                )

            if max_iterations < 1:
                raise ValueError(
                    "Maximum iterations must be at least 1."
                )

            if x0 == x1:
                raise ValueError(
                    "The initial guesses must be different."
                )

            root, rows, result_message = secant_method(
                numerical_function,
                x0,
                x1,
                tolerance,
                max_iterations
            )

            self.last_function = numerical_function
            self.last_root = root

            # Clear the old results.
            for item in self.table.get_children():
                self.table.delete(item)

            # Fill the iteration table.
            for iteration, xn, fx, error in rows:
                self.table.insert(
                    "",
                    "end",
                    values=(
                        iteration,
                        f"{xn:.10g}",
                        f"{fx:.6g}",
                        (
                            "—"
                            if error is None
                            else f"{error:.6g}"
                        )
                    )
                )

            self.status.set(
                f"{result_message}\n"
                f"Approximate root: {root:.12g}\n"
                f"Function: f(x) = {expression}"
            )

            self.draw_graph(
                numerical_function, x0, x1, root
            )

        except Exception as error:
            messagebox.showerror(
                "Calculation Error", str(error)
            )
            self.status.set(
                "Calculation failed. Check your inputs."
            )

    # =====================================================
    # 6. GRAPH THE FUNCTION AND APPROXIMATE ROOT
    # =====================================================

    def draw_graph(self, function, x0, x1, root):
        self.axes.clear()

        span = max(
            abs(x1 - x0) * 2,
            abs(root) * 0.2,
            5
        )

        left = min(x0, x1, root) - span
        right = max(x0, x1, root) + span

        points = 600

        xs = [
            left + (right - left) * i / (points - 1)
            for i in range(points)
        ]

        ys = []

        for value in xs:
            try:
                result = evaluate_function(
                    function, value
                )
                ys.append(result)
            except Exception:
                ys.append(float("nan"))

        self.axes.plot(xs, ys, label="f(x)")
        self.axes.axhline(
            0, color="black", linewidth=0.8
        )
        self.axes.axvline(
            root,
            linestyle="--",
            label="Approximate root"
        )
        self.axes.scatter(
            [root],
            [0],
            marker="o",
            label=f"Root ≈ {root:.6g}"
        )

        self.axes.set_title(
            "Secant Method: Function and Root"
        )
        self.axes.set_xlabel("x")
        self.axes.set_ylabel("f(x)")
        self.axes.grid(True, alpha=0.3)
        self.axes.legend()

        self.figure.tight_layout()
        self.canvas.draw()

    # =====================================================
    # 7. RESET THE APPLICATION
    # =====================================================

    def reset(self):
        self.expression.set("x^2 - 4")
        self.x0.set("1")
        self.x1.set("3")
        self.tolerance.set("0.000001")
        self.max_iterations.set("100")

        self.status.set(
            "Enter a function and click Calculate."
        )

        for item in self.table.get_children():
            self.table.delete(item)

        self.last_function = None
        self.last_root = None

        self.axes.clear()
        self.axes.set_title(
            "Enter a function and calculate"
        )
        self.axes.set_xlabel("x")
        self.axes.set_ylabel("f(x)")
        self.axes.axhline(
            0, color="black", linewidth=0.8
        )
        self.axes.grid(True, alpha=0.3)

        self.canvas.draw()


# =========================================================
# 8. START THE APPLICATION
# =========================================================

if __name__ == "__main__":
    root = tk.Tk()
    app = SecantCalculator(root)
    root.mainloop()