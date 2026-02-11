import ast
import sys
from pathlib import Path


def is_separator(line: str) -> bool:
    line = line.strip()
    return line.startswith("=====") and line.endswith("=====")


class SignatureOnlyTransformer(ast.NodeTransformer):
    def _strip_body(self, node):
        node.body = [ast.Pass()]
        return node

    def visit_FunctionDef(self, node):
        self.generic_visit(node)
        return self._strip_body(node)

    def visit_AsyncFunctionDef(self, node):
        self.generic_visit(node)
        return self._strip_body(node)


def process_python_chunk(source: str) -> str:
    tree = ast.parse(source)
    tree = SignatureOnlyTransformer().visit(tree)
    ast.fix_missing_locations(tree)
    return ast.unparse(tree)


def main(input_path: str, output_path: str):
    lines = Path(input_path).read_text(
        encoding="utf-8",
        errors="ignore",
    ).splitlines()

    chunks: list[tuple[str | None, list[str]]] = []
    current_header = None
    current_lines: list[str] = []

    for line in lines:
        if is_separator(line):
            if current_header is not None:
                chunks.append((current_header, current_lines))
                current_lines = []
            current_header = line
        else:
            current_lines.append(line)

    if current_header is not None:
        chunks.append((current_header, current_lines))

    output_lines: list[str] = []

    for header, body_lines in chunks:
        output_lines.append(f"# {header}")

        source = "\n".join(body_lines).strip()
        if not source:
            output_lines.append("")
            continue

        try:
            transformed = process_python_chunk(source)
            output_lines.append(transformed)
        except SyntaxError as e:
            # Fallback: keep original if file is unparsable
            output_lines.append("# [UNPARSEABLE FILE — ORIGINAL CONTENT KEPT]")
            output_lines.extend(body_lines)

        output_lines.append("")

    Path(output_path).write_text(
        "\n".join(output_lines),
        encoding="utf-8",
    )


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python script_strip_functions.py input.txt output.txt")
        sys.exit(1)

    main(sys.argv[1], sys.argv[2])
