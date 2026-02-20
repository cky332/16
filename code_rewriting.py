"""
Code rewriting attacks for watermark robustness testing.

Provides AST-based code transformation functions that preserve semantic
equivalence while changing the token-level representation. Each attack
function takes a code string and returns the rewritten code string.
"""

import ast
import io
import keyword
import tokenize as tokenize_module


# Python built-in names that should never be renamed
import builtins as _builtins_module
_BUILTINS = set(dir(_builtins_module))
_KEYWORDS = set(keyword.kwlist)
_PRESERVE_NAMES = _BUILTINS | _KEYWORDS | {
    "self", "cls", "None", "True", "False",
    "__init__", "__str__", "__repr__", "__len__", "__getitem__",
    "__setitem__", "__delitem__", "__iter__", "__next__", "__call__",
    "__enter__", "__exit__", "__eq__", "__ne__", "__lt__", "__gt__",
    "__le__", "__ge__", "__hash__", "__contains__",
}


def extract_prompt(code: str) -> str:
    """Extract the prompt portion from a HumanEval generation.

    The prompt is everything up to and including the closing triple-quote
    of the docstring plus the trailing newline.
    """
    # Find the closing """ of the docstring
    # The pattern is: def ...:\n    \"\"\"...\"\"\"\n
    # We need to find the SECOND occurrence of triple quotes (closing one)
    delimiters = ['"""', "'''"]
    for delim in delimiters:
        parts = code.split(delim)
        if len(parts) >= 3:
            # prompt = everything up to and including the second delimiter
            prompt = delim.join(parts[:2]) + delim
            remaining = delim.join(parts[2:])
            if remaining.startswith("\n"):
                prompt += "\n"
            return prompt

    # Fallback: return empty string
    return ""


def safe_transform(code: str, transform_fn) -> str:
    """Apply a transform function to code, returning original on failure."""
    try:
        tree = ast.parse(code)
        transformed = transform_fn(tree)
        result = ast.unparse(transformed)
        # Verify result is valid Python
        ast.parse(result)
        return result
    except Exception:
        return code


# ---------------------------------------------------------------------------
# Attack 1: Variable Renaming
# ---------------------------------------------------------------------------

class _VariableRenamer(ast.NodeTransformer):
    """Renames local variables to generic names (var_0, var_1, ...)."""

    def __init__(self, preserve_names: set = None):
        self.preserve = preserve_names or set()
        self.mapping = {}
        self.counter = 0
        # Collect imported names to preserve
        self.imported_names = set()

    def _get_new_name(self, old_name: str) -> str:
        if old_name in self.preserve or old_name in _PRESERVE_NAMES or old_name in self.imported_names:
            return old_name
        if old_name.startswith("_") and old_name != "_":
            return old_name
        if old_name not in self.mapping:
            self.mapping[old_name] = f"var_{self.counter}"
            self.counter += 1
        return self.mapping[old_name]

    def visit_Import(self, node):
        for alias in node.names:
            name = alias.asname if alias.asname else alias.name
            self.imported_names.add(name.split(".")[0])
        return node

    def visit_ImportFrom(self, node):
        for alias in node.names:
            name = alias.asname if alias.asname else alias.name
            self.imported_names.add(name)
        return node

    def visit_FunctionDef(self, node):
        # Don't rename the function name itself
        self.preserve.add(node.name)
        self.generic_visit(node)
        return node

    def visit_ClassDef(self, node):
        self.preserve.add(node.name)
        self.generic_visit(node)
        return node

    def visit_Name(self, node):
        node.id = self._get_new_name(node.id)
        return node

    def visit_arg(self, node):
        node.arg = self._get_new_name(node.arg)
        return node

    def visit_Attribute(self, node):
        # Only rename the value part, NOT the attribute name
        # e.g., in `obj.attr`, rename `obj` but keep `attr`
        self.visit(node.value)
        return node


def rename_variables(code: str) -> str:
    """Rename local variables to generic names (var_0, var_1, ...)."""
    try:
        tree = ast.parse(code)
        renamer = _VariableRenamer()
        # First pass: collect imports
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                renamer.visit(node)
        # Second pass: rename
        tree = ast.parse(code)  # Re-parse to get clean tree
        transformed = renamer.visit(tree)
        ast.fix_missing_locations(transformed)
        result = ast.unparse(transformed)
        ast.parse(result)  # Verify
        return result
    except Exception:
        return code


# ---------------------------------------------------------------------------
# Attack 2: Code Reformatting
# ---------------------------------------------------------------------------

def reformat_code(code: str) -> str:
    """Reformat code via AST round-trip (normalizes whitespace, quotes, etc.)."""
    try:
        tree = ast.parse(code)
        result = ast.unparse(tree)
        ast.parse(result)
        return result
    except Exception:
        return code


# ---------------------------------------------------------------------------
# Attack 3: Comment Removal
# ---------------------------------------------------------------------------

def remove_comments(code: str) -> str:
    """Remove all # comments from code while preserving structure."""
    try:
        tokens = list(tokenize_module.generate_tokens(io.StringIO(code).readline))
        result_tokens = []
        for tok in tokens:
            if tok.type == tokenize_module.COMMENT:
                continue
            result_tokens.append(tok)
        result = tokenize_module.untokenize(result_tokens)
        ast.parse(result)  # Verify
        return result
    except Exception:
        return code


# ---------------------------------------------------------------------------
# Attack 4: Dead Code Insertion
# ---------------------------------------------------------------------------

class _DeadCodeInserter(ast.NodeTransformer):
    """Insert dead code statements into function bodies."""

    def __init__(self):
        self.counter = 0

    def _make_dead_stmt(self):
        var_name = f"_dead_{self.counter}"
        self.counter += 1
        node = ast.Assign(
            targets=[ast.Name(id=var_name, ctx=ast.Store())],
            value=ast.Constant(value=None),
        )
        return node

    def visit_FunctionDef(self, node):
        self.generic_visit(node)
        new_body = []
        for stmt in node.body:
            new_body.append(stmt)
            # Don't insert after docstring
            if isinstance(stmt, ast.Expr) and isinstance(
                getattr(stmt, "value", None), ast.Constant
            ):
                if isinstance(stmt.value.value, str):
                    continue
            new_body.append(self._make_dead_stmt())
        node.body = new_body
        ast.fix_missing_locations(node)
        return node


def insert_dead_code(code: str) -> str:
    """Insert dead code (unused variable assignments) into function bodies."""
    try:
        tree = ast.parse(code)
        transformer = _DeadCodeInserter()
        transformed = transformer.visit(tree)
        ast.fix_missing_locations(transformed)
        result = ast.unparse(transformed)
        ast.parse(result)
        return result
    except Exception:
        return code


# ---------------------------------------------------------------------------
# Attack 5: If/Else Swap
# ---------------------------------------------------------------------------

class _IfElseSwapper(ast.NodeTransformer):
    """Negate if conditions and swap if/else branches."""

    def visit_If(self, node):
        self.generic_visit(node)
        if node.orelse:
            # Negate condition
            node.test = ast.UnaryOp(op=ast.Not(), operand=node.test)
            # Swap branches
            node.body, node.orelse = node.orelse, node.body
            ast.fix_missing_locations(node)
        return node


def swap_if_else(code: str) -> str:
    """Swap if/else branches by negating the condition."""
    try:
        tree = ast.parse(code)
        transformer = _IfElseSwapper()
        transformed = transformer.visit(tree)
        ast.fix_missing_locations(transformed)
        result = ast.unparse(transformed)
        ast.parse(result)
        return result
    except Exception:
        return code


# ---------------------------------------------------------------------------
# Attack 6: Expression Rewriting
# ---------------------------------------------------------------------------

class _ExpressionRewriter(ast.NodeTransformer):
    """Rewrite comparison expressions to equivalent forms."""

    def visit_Compare(self, node):
        self.generic_visit(node)
        # Only handle single comparisons
        if len(node.ops) != 1 or len(node.comparators) != 1:
            return node

        op = node.ops[0]
        # a != b -> not (a == b)
        if isinstance(op, ast.NotEq):
            new_compare = ast.Compare(
                left=node.left,
                ops=[ast.Eq()],
                comparators=node.comparators,
            )
            return ast.UnaryOp(op=ast.Not(), operand=new_compare)
        # a >= b -> not (a < b)
        elif isinstance(op, ast.GtE):
            new_compare = ast.Compare(
                left=node.left,
                ops=[ast.Lt()],
                comparators=node.comparators,
            )
            return ast.UnaryOp(op=ast.Not(), operand=new_compare)
        # a <= b -> not (a > b)
        elif isinstance(op, ast.LtE):
            new_compare = ast.Compare(
                left=node.left,
                ops=[ast.Gt()],
                comparators=node.comparators,
            )
            return ast.UnaryOp(op=ast.Not(), operand=new_compare)
        return node


def rewrite_expressions(code: str) -> str:
    """Rewrite comparison expressions to equivalent forms."""
    try:
        tree = ast.parse(code)
        transformer = _ExpressionRewriter()
        transformed = transformer.visit(tree)
        ast.fix_missing_locations(transformed)
        result = ast.unparse(transformed)
        ast.parse(result)
        return result
    except Exception:
        return code


# ---------------------------------------------------------------------------
# Attack 7: Type Annotation Removal
# ---------------------------------------------------------------------------

class _TypeAnnotationRemover(ast.NodeTransformer):
    """Remove all type annotations from function signatures and bodies."""

    def visit_FunctionDef(self, node):
        self.generic_visit(node)
        node.returns = None
        for arg in node.args.args:
            arg.annotation = None
        for arg in node.args.posonlyargs:
            arg.annotation = None
        for arg in node.args.kwonlyargs:
            arg.annotation = None
        if node.args.vararg:
            node.args.vararg.annotation = None
        if node.args.kwarg:
            node.args.kwarg.annotation = None
        return node

    def visit_AnnAssign(self, node):
        self.generic_visit(node)
        if node.value:
            return ast.Assign(
                targets=[node.target],
                value=node.value,
            )
        # Annotation-only statement (no value) — remove it
        return None


def remove_type_annotations(code: str) -> str:
    """Remove all type annotations from code."""
    try:
        tree = ast.parse(code)
        transformer = _TypeAnnotationRemover()
        transformed = transformer.visit(tree)
        ast.fix_missing_locations(transformed)
        result = ast.unparse(transformed)
        ast.parse(result)
        return result
    except Exception:
        return code


# ---------------------------------------------------------------------------
# Attack 8: Combined Attack
# ---------------------------------------------------------------------------

def combined_attack(code: str) -> str:
    """Apply all attacks in sequence for maximum disruption.

    Note: dead_code insertion is excluded because it adds tokens that
    counterintuitively *increase* z-scores (more green tokens from the
    added code), masking the effect of other attacks.
    """
    code = remove_type_annotations(code)
    code = remove_comments(code)
    code = rename_variables(code)
    code = rewrite_expressions(code)
    code = swap_if_else(code)
    code = reformat_code(code)
    return code


# ---------------------------------------------------------------------------
# Attack 9: Structural Paraphrase (aggressive AST rewriting)
# ---------------------------------------------------------------------------

class _ForToWhile(ast.NodeTransformer):
    """Convert for-range loops to while loops."""

    def visit_For(self, node):
        self.generic_visit(node)
        # Only convert `for var in range(...)` patterns
        if not (
            isinstance(node.iter, ast.Call)
            and isinstance(node.iter.func, ast.Name)
            and node.iter.func.id == "range"
            and isinstance(node.target, ast.Name)
        ):
            return node

        args = node.iter.args
        if len(args) == 1:
            start, stop, step = ast.Constant(value=0), args[0], ast.Constant(value=1)
        elif len(args) == 2:
            start, stop, step = args[0], args[1], ast.Constant(value=1)
        elif len(args) == 3:
            start, stop, step = args[0], args[1], args[2]
        else:
            return node

        var = node.target
        # var = start
        init = ast.Assign(targets=[ast.Name(id=var.id, ctx=ast.Store())], value=start)
        # var < stop
        test = ast.Compare(
            left=ast.Name(id=var.id, ctx=ast.Load()),
            ops=[ast.Lt()],
            comparators=[stop],
        )
        # var += step
        increment = ast.AugAssign(
            target=ast.Name(id=var.id, ctx=ast.Store()),
            op=ast.Add(),
            value=step,
        )
        while_body = list(node.body) + [increment]
        while_node = ast.While(test=test, body=while_body, orelse=node.orelse)
        return [init, while_node]


class _ListCompToLoop(ast.NodeTransformer):
    """Convert list comprehensions to explicit for-append loops."""

    def __init__(self):
        self.counter = 0

    def visit_Assign(self, node):
        self.generic_visit(node)
        if not (
            len(node.targets) == 1
            and isinstance(node.value, ast.ListComp)
        ):
            return node

        comp = node.value
        if len(comp.generators) != 1 or comp.generators[0].ifs:
            return node  # Only handle simple single-loop comprehensions

        gen = comp.generators[0]
        target_name = node.targets[0]
        tmp_name = f"_comp_{self.counter}"
        self.counter += 1

        # tmp = []
        init = ast.Assign(
            targets=[ast.Name(id=tmp_name, ctx=ast.Store())],
            value=ast.List(elts=[], ctx=ast.Load()),
        )
        # tmp.append(elt)
        append_call = ast.Expr(
            value=ast.Call(
                func=ast.Attribute(
                    value=ast.Name(id=tmp_name, ctx=ast.Load()),
                    attr="append",
                    ctx=ast.Load(),
                ),
                args=[comp.elt],
                keywords=[],
            )
        )
        # for target in iter: tmp.append(elt)
        loop = ast.For(target=gen.target, iter=gen.iter, body=[append_call], orelse=[])
        # original_target = tmp
        assign_back = ast.Assign(
            targets=[target_name],
            value=ast.Name(id=tmp_name, ctx=ast.Load()),
        )
        return [init, loop, assign_back]


class _AugAssignSwap(ast.NodeTransformer):
    """Swap augmented assignments: x += 1 -> x = x + 1 (and vice versa)."""

    _OP_MAP = {
        ast.Add: ast.Add,
        ast.Sub: ast.Sub,
        ast.Mult: ast.Mult,
        ast.Div: ast.Div,
        ast.Mod: ast.Mod,
    }

    def visit_AugAssign(self, node):
        self.generic_visit(node)
        # Convert x += val to x = x + val
        if isinstance(node.target, ast.Name):
            return ast.Assign(
                targets=[ast.Name(id=node.target.id, ctx=ast.Store())],
                value=ast.BinOp(
                    left=ast.Name(id=node.target.id, ctx=ast.Load()),
                    op=node.op,
                    right=node.value,
                ),
            )
        return node


def structural_paraphrase(code: str) -> str:
    """Aggressive structural paraphrase via AST transforms.

    Chains multiple structural changes: for-to-while conversion,
    list comprehension expansion, augmented assignment expansion,
    variable renaming, and expression rewriting.
    """
    try:
        tree = ast.parse(code)
        tree = _ForToWhile().visit(tree)
        ast.fix_missing_locations(tree)
        code = ast.unparse(tree)
    except Exception:
        pass

    try:
        tree = ast.parse(code)
        tree = _ListCompToLoop().visit(tree)
        ast.fix_missing_locations(tree)
        code = ast.unparse(tree)
    except Exception:
        pass

    try:
        tree = ast.parse(code)
        tree = _AugAssignSwap().visit(tree)
        ast.fix_missing_locations(tree)
        code = ast.unparse(tree)
    except Exception:
        pass

    # Chain with existing attacks
    code = rename_variables(code)
    code = rewrite_expressions(code)
    code = reformat_code(code)
    return code


# ---------------------------------------------------------------------------
# Attack 10: LLM-based Code Paraphrase
# ---------------------------------------------------------------------------

def llm_paraphrase(code: str) -> str:
    """Paraphrase code using an LLM API.

    Tries OpenAI API first (requires OPENAI_API_KEY env var),
    then falls back to structural_paraphrase if unavailable.
    """
    import os as _os

    api_key = _os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        return structural_paraphrase(code)

    try:
        import openai

        client = openai.OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a code rewriter. Rewrite the given Python code "
                        "to be functionally equivalent but with different coding "
                        "style, variable names, loop structures, and expressions. "
                        "Keep the same function signature and docstring. "
                        "Only output the Python code, no explanations or markdown."
                    ),
                },
                {"role": "user", "content": code},
            ],
            temperature=0.7,
            max_tokens=2048,
        )
        result = response.choices[0].message.content.strip()
        # Strip markdown code fences if present
        if result.startswith("```"):
            lines = result.split("\n")
            lines = lines[1:]  # Remove opening fence
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            result = "\n".join(lines)
        # Verify valid Python
        ast.parse(result)
        return result
    except ImportError:
        return structural_paraphrase(code)
    except Exception:
        return structural_paraphrase(code)


# ---------------------------------------------------------------------------
# Registry of all attacks
# ---------------------------------------------------------------------------

ALL_ATTACKS = {
    "rename_variables": rename_variables,
    "reformat": reformat_code,
    "remove_comments": remove_comments,
    "dead_code": insert_dead_code,
    "swap_if_else": swap_if_else,
    "rewrite_expressions": rewrite_expressions,
    "remove_type_annotations": remove_type_annotations,
    "structural_paraphrase": structural_paraphrase,
    "llm_paraphrase": llm_paraphrase,
    "combined": combined_attack,
}
