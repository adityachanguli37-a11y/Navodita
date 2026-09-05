import ast
import math
import operator
import re
from typing import Optional, Tuple

# Safe allowed operators
ALLOWED_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}

# Safe allowed functions
ALLOWED_FUNCTIONS = {
    "sqrt": math.sqrt,
    "abs": abs,
    "round": round,
}

# Regex to detect math expressions or math questions
MATH_PATTERN = re.compile(
    r"^(?:(?:what\s+is|calculate|evaluate|solve|compute|\s*)\s*)?([0-9\.\s\+\-\*\/\%\(\)\^\,a-zA-Z]+)\??$",
    re.IGNORECASE,
)

OPERATOR_CHARS = set("+-*/%^")


def _eval_ast_node(node: ast.AST) -> float:
    """Recursively evaluate an AST node strictly within allowed math operations."""
    if isinstance(node, ast.Expression):
        return _eval_ast_node(node.body)

    # Support Python Constant and Num
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return float(node.value)
        raise ValueError("Invalid constant type in expression")

    if hasattr(ast, "Num") and isinstance(node, getattr(ast, "Num")):
        return float(node.n)

    if isinstance(node, ast.UnaryOp):
        op_type = type(node.op)
        if op_type not in ALLOWED_OPERATORS:
            raise ValueError(f"Unsupported unary operator: {op_type.__name__}")
        operand = _eval_ast_node(node.operand)
        return ALLOWED_OPERATORS[op_type](operand)

    if isinstance(node, ast.BinOp):
        op_type = type(node.op)
        if op_type not in ALLOWED_OPERATORS:
            raise ValueError(f"Unsupported binary operator: {op_type.__name__}")
        left = _eval_ast_node(node.left)
        right = _eval_ast_node(node.right)

        # Safety guard for exponentiation limit
        if op_type == ast.Pow:
            if abs(right) > 100 or abs(left) > 10000:
                raise ValueError("Exponent or base too large for safe calculation")

        # Zero division checks
        if op_type in (ast.Div, ast.FloorDiv, ast.Mod) and right == 0:
            raise ZeroDivisionError("Division by zero is undefined")

        return ALLOWED_OPERATORS[op_type](left, right)

    # Safe function call evaluation (e.g. sqrt(144))
    if isinstance(node, ast.Call):
        if isinstance(node.func, ast.Name):
            func_name = node.func.id.lower()
            if func_name in ALLOWED_FUNCTIONS:
                args = [_eval_ast_node(arg) for arg in node.args]
                if func_name == "sqrt":
                    if len(args) != 1:
                        raise ValueError("sqrt expects exactly 1 argument")
                    if args[0] < 0:
                        raise ValueError("Cannot calculate square root of negative number")
                    return math.sqrt(args[0])
                elif func_name == "abs":
                    return abs(args[0])
                elif func_name == "round":
                    return round(args[0], int(args[1]) if len(args) > 1 else 0)
        raise ValueError(f"Disallowed function call in expression")

    raise ValueError(f"Disallowed AST element: {type(node).__name__}")


def extract_math_expression(text: str) -> Optional[str]:
    """Check if the text represents or contains an exact mathematical calculation request."""
    if not text or not isinstance(text, str):
        return None
    cleaned = text.strip()
    match = MATH_PATTERN.match(cleaned)
    if not match:
        return None

    candidate = match.group(1).strip()
    
    # Check for functions like sqrt(144)
    has_allowed_func = any(re.search(rf"\b{fn}\s*\(", candidate, re.IGNORECASE) for fn in ALLOWED_FUNCTIONS)
    has_digit = any(c.isdigit() for c in candidate)
    has_operator = any(c in OPERATOR_CHARS for c in candidate)

    # Reject if it contains words other than allowed functions
    non_math_words = [w for w in re.findall(r"\b[a-zA-Z]+\b", candidate.lower()) if w not in ALLOWED_FUNCTIONS]
    if non_math_words:
        return None

    if has_digit and (has_operator or has_allowed_func):
        candidate = candidate.replace("^", "**")
        return candidate

    return None


def evaluate_math_expression(expression: str) -> Tuple[bool, str]:
    """
    Safely evaluate a mathematical expression without using eval().
    Returns (success: bool, result_or_error: str).
    """
    try:
        parsed = ast.parse(expression, mode="eval")
        result = _eval_ast_node(parsed)

        if result.is_integer():
            formatted = str(int(result))
        else:
            formatted = f"{result:.6f}".rstrip("0").rstrip(".")

        return True, formatted
    except ZeroDivisionError:
        return False, "Error: Division by zero is undefined."
    except OverflowError:
        return False, "Error: Calculation resulted in numerical overflow."
    except Exception as e:
        return False, f"Error: Invalid mathematical expression ({str(e)})."


def process_calculator_query(text: str) -> Optional[str]:
    """
    Attempt to process a user query through the safe math engine.
    If the text is an exact calculation query, returns the natural response string, else None.
    """
    expr = extract_math_expression(text)
    if not expr:
        return None

    success, result = evaluate_math_expression(expr)
    if success:
        return f"{expr.strip()} = {result}"
    return result
