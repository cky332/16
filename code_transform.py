"""
Code Transformation Robustness Testing
========================================
Applies various code transformations to watermarked code and tests
whether the watermark can still be detected after transformation.

Transformations:
1. Variable renaming
2. Code reformatting (autopep8/black style)
3. Comment addition/removal
4. Equivalent code transformations
5. Whitespace manipulation
"""

import json
import re
import ast
import argparse
import keyword
import random
from copy import deepcopy


class CodeTransformer:
    """Applies various code transformations while preserving semantics."""

    def __init__(self, seed=42):
        self.rng = random.Random(seed)

    def rename_variables(self, code):
        """Systematically rename all user-defined variables to generic names."""
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return code  # can't parse, return as-is

        # Collect all variable names (assignments and function args)
        var_names = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
                var_names.add(node.id)
            elif isinstance(node, ast.FunctionDef):
                for arg in node.args.args:
                    var_names.add(arg.arg)

        # Filter out builtins and keywords
        builtins_set = set(dir(__builtins__)) if isinstance(__builtins__, dict) else set(dir(__builtins__))
        var_names = {v for v in var_names if v not in builtins_set and not keyword.iskeyword(v)}

        # Create mapping
        mapping = {}
        for i, name in enumerate(sorted(var_names)):
            mapping[name] = f'var_{i}'

        # Apply renaming using regex word boundaries
        result = code
        for old_name, new_name in sorted(mapping.items(), key=lambda x: -len(x[0])):
            result = re.sub(r'\b' + re.escape(old_name) + r'\b', new_name, result)

        return result

    def normalize_whitespace(self, code):
        """Normalize whitespace: consistent indentation, remove trailing spaces."""
        lines = code.split('\n')
        normalized = []
        for line in lines:
            # Remove trailing whitespace
            stripped = line.rstrip()
            # Normalize indentation to 4 spaces
            leading = len(line) - len(line.lstrip())
            if leading > 0:
                indent_level = leading // 4
                remainder = leading % 4
                if remainder >= 2:
                    indent_level += 1
                stripped = '    ' * indent_level + line.lstrip()
            normalized.append(stripped)
        return '\n'.join(normalized)

    def remove_comments(self, code):
        """Remove all comments from code."""
        lines = code.split('\n')
        result = []
        in_docstring = False
        docstring_char = None

        for line in lines:
            stripped = line.strip()

            # Handle docstrings
            if not in_docstring:
                if stripped.startswith('"""') or stripped.startswith("'''"):
                    docstring_char = stripped[:3]
                    if stripped.count(docstring_char) >= 2 and len(stripped) > 3:
                        # Single-line docstring
                        continue
                    in_docstring = True
                    continue
                # Remove inline comments
                # Be careful not to remove # inside strings
                in_string = False
                string_char = None
                new_line = []
                for i, ch in enumerate(line):
                    if ch in ('"', "'") and not in_string:
                        in_string = True
                        string_char = ch
                    elif ch == string_char and in_string:
                        in_string = False
                    elif ch == '#' and not in_string:
                        break
                    new_line.append(ch)
                cleaned = ''.join(new_line).rstrip()
                if cleaned.strip():
                    result.append(cleaned)
                elif not stripped:
                    result.append(line)
            else:
                if docstring_char and docstring_char in stripped:
                    in_docstring = False
                continue

        return '\n'.join(result)

    def add_comments(self, code):
        """Add innocuous comments to code."""
        lines = code.split('\n')
        result = []
        comments = [
            '# process data',
            '# compute result',
            '# check condition',
            '# iterate over elements',
            '# initialize variables',
            '# return output',
            '# handle edge case',
        ]
        comment_idx = 0

        for i, line in enumerate(lines):
            result.append(line)
            stripped = line.strip()
            # Add comment before loops, conditionals, return statements
            if stripped.startswith(('for ', 'while ', 'if ', 'return ')):
                if self.rng.random() < 0.4:
                    indent = len(line) - len(line.lstrip())
                    result.insert(-1, ' ' * indent + comments[comment_idx % len(comments)])
                    comment_idx += 1

        return '\n'.join(result)

    def add_blank_lines(self, code):
        """Add extra blank lines between code blocks."""
        lines = code.split('\n')
        result = []
        for i, line in enumerate(lines):
            result.append(line)
            stripped = line.strip()
            # Add blank line after certain patterns
            if stripped.endswith(':') or stripped.startswith('return '):
                if i < len(lines) - 1 and lines[i + 1].strip():
                    result.append('')
        return '\n'.join(result)

    def remove_blank_lines(self, code):
        """Remove extra blank lines."""
        lines = code.split('\n')
        result = []
        prev_blank = False
        for line in lines:
            if line.strip() == '':
                if not prev_blank:
                    result.append(line)
                prev_blank = True
            else:
                result.append(line)
                prev_blank = False
        return '\n'.join(result)

    def reorder_imports(self, code):
        """Sort import statements alphabetically."""
        lines = code.split('\n')
        import_lines = []
        other_lines = []
        import_section_done = False

        for line in lines:
            if not import_section_done and (line.strip().startswith('import ') or
                                             line.strip().startswith('from ')):
                import_lines.append(line)
            else:
                if import_lines and not line.strip():
                    continue  # skip blank line after imports
                import_section_done = True
                other_lines.append(line)

        if import_lines:
            import_lines.sort()
            return '\n'.join(import_lines + [''] + other_lines)
        return code


def apply_transforms(code, transforms, transformer):
    """Apply a list of transformations to code."""
    result = code
    for transform_name in transforms:
        transform_fn = getattr(transformer, transform_name, None)
        if transform_fn:
            result = transform_fn(result)
    return result


def transform_generations(generations, transformer, transforms, prompts=None):
    """Apply transformations to all generations.

    If prompts is provided, only transform the generated portion (after the prompt),
    preserving the prompt prefix so that watermark detection can still match it.
    """
    transformed = deepcopy(generations)
    for i, gens in enumerate(transformed):
        for j, gen in enumerate(gens):
            if prompts and i < len(prompts):
                prefix = prompts[i]
                if gen.startswith(prefix):
                    # Only transform the generated part, keep prefix intact
                    generated_part = gen[len(prefix):]
                    transformed_part = apply_transforms(generated_part, transforms, transformer)
                    transformed[i][j] = prefix + transformed_part
                else:
                    # Fallback: transform entire text
                    transformed[i][j] = apply_transforms(gen, transforms, transformer)
            else:
                transformed[i][j] = apply_transforms(gen, transforms, transformer)
    return transformed


def load_humaneval_prompts():
    """Load HumanEval prompts from the dataset."""
    try:
        import datasets as hf_datasets
        dataset = hf_datasets.load_dataset("openai_humaneval", split="test")
        return [doc["prompt"].strip() for doc in dataset]
    except Exception as e:
        print(f"Warning: Could not load HumanEval dataset: {e}")
        print("Falling back to transforming entire generations (may cause prefix mismatch)")
        return None


def main():
    parser = argparse.ArgumentParser(description="Code transformation robustness testing")
    parser.add_argument('--input', type=str, default='outputs_sweet/generations.json',
                        help='Path to generations.json')
    parser.add_argument('--output_dir', type=str, default='transformed_generations',
                        help='Output directory for transformed code')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    import os
    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading generations...")
    with open(args.input, 'r') as f:
        generations = json.load(f)

    print("Loading HumanEval prompts...")
    prompts = load_humaneval_prompts()
    if prompts:
        print(f"  Loaded {len(prompts)} prompts (will preserve prompt prefix during transformation)")

    transformer = CodeTransformer(seed=args.seed)

    # Define transformation sets
    transform_sets = {
        'rename_vars': ['rename_variables'],
        'remove_comments': ['remove_comments'],
        'add_comments': ['add_comments'],
        'normalize_ws': ['normalize_whitespace'],
        'add_blanks': ['add_blank_lines'],
        'remove_blanks': ['remove_blank_lines'],
        'combined_light': ['normalize_whitespace', 'remove_blank_lines'],
        'combined_heavy': ['rename_variables', 'remove_comments', 'normalize_whitespace'],
    }

    for name, transforms in transform_sets.items():
        print(f"\nApplying transformation: {name} ({transforms})")
        transformed = transform_generations(generations, transformer, transforms, prompts=prompts)

        output_path = os.path.join(args.output_dir, f'generations_{name}.json')
        with open(output_path, 'w') as f:
            json.dump(transformed, f)
        print(f"  Saved to {output_path}")

        # Show example
        if generations and generations[0]:
            orig = generations[0][0]
            trans = transformed[0][0]
            if orig != trans:
                orig_lines = orig.strip().split('\n')[-3:]
                trans_lines = trans.strip().split('\n')[-3:]
                print(f"  Example (last 3 lines):")
                print(f"    Original: {' | '.join(l.strip() for l in orig_lines)}")
                print(f"    Transformed: {' | '.join(l.strip() for l in trans_lines)}")

    print(f"\nAll transformations saved to {args.output_dir}/")
    print("\nTo test robustness, run detection on each transformed generation file:")
    print("  accelerate launch main.py --load_generations_path <transformed_file> --sweet ...")


if __name__ == '__main__':
    main()
