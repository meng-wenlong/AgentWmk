import json
import re
from io import StringIO
import tokenize
from typing import Optional, List, Tuple, Dict, Any
from dataclasses import dataclass, field
from tqdm import tqdm
from pathlib import Path

try:
    from tree_sitter import Language, Parser
    TREE_SITTER_AVAILABLE = True
except ImportError:
    TREE_SITTER_AVAILABLE = False
    print("Warning: tree-sitter not installed. Some features may be limited.")


@dataclass
class CodeSample:
    code: str
    expressions: list[str] = field(default_factory=list)
    identifiers: list[str] = field(default_factory=list)
    comment: Optional[str] = None  # Optional comment for coprotector method

    @classmethod
    def from_dict(cls, data: dict) -> 'CodeSample':
        """Create CodeSample from dictionary (typical JSON input)."""
        return cls(
            code=data.get("code", ""),
            expressions=data.get("split_expressions", []),
            identifiers=data.get("identifiers", []),
            comment=data.get("comment", None)
        )

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        result = {
            "code": self.code,
            "split_expressions": self.expressions,
            "identifiers": self.identifiers
        }
        if self.comment:
            result["comment"] = self.comment
        return result

    def is_valid(self) -> bool:
        """Check if sample has minimum required data."""
        return bool(self.code and (self.expressions or self.identifiers))


class DataLoader:
    """Handles loading and validation of data files."""

    @staticmethod
    def load_jsonl(filepath: str, max_samples: Optional[int] = None) -> List[CodeSample]:
        """
        Load samples from JSONL file.

        Args:
            filepath: Path to JSONL file
            max_samples: Maximum number of samples to load

        Returns:
            List of CodeSample objects
        """
        samples = []

        with open(filepath, 'r') as f:
            for line_num, line in enumerate(f, 1):
                if max_samples and len(samples) >= max_samples:
                    break

                try:
                    data = json.loads(line.strip())
                    sample = CodeSample.from_dict(data)

                    if sample.is_valid():
                        samples.append(sample)
                    else:
                        print(f"Warning: Invalid sample at line {line_num}")

                except json.JSONDecodeError as e:
                    print(f"Warning: JSON error at line {line_num}: {e}")

        return samples

    @staticmethod
    def save_results(results: dict, filepath: str) -> None:
        """Save detection results to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {filepath}")

    @staticmethod
    def save_jsonl(samples: List[CodeSample], filepath: str) -> None:
        """Save CodeSample list to JSONL file."""
        with open(filepath, 'w') as f:
            for sample in samples:
                f.write(json.dumps(sample.to_dict()) + '\n')
        print(f"Saved {len(samples)} samples to {filepath}")


# Tree-sitter utility functions
def get_parser(language: str, tree_sitter_path: Optional[str] = None) -> Optional['Parser']:
    """Initialize tree-sitter parser for the specified language."""
    if not TREE_SITTER_AVAILABLE:
        return None

    try:
        # Use the new tree-sitter API with language modules
        if language == "python":
            import tree_sitter_python as tspython
            lang = Language(tspython.language())
        elif language == "java":
            import tree_sitter_java as tsjava
            lang = Language(tsjava.language())
        else:
            # Try to load from file if provided
            if tree_sitter_path:
                Language.build_library(
                    f'build/my-languages-{language}.so',
                    [tree_sitter_path]
                )
                lang = Language(f'build/my-languages-{language}.so', language)
            else:
                print(f"Unsupported language: {language}")
                return None

        parser = Parser(lang)
        return parser
    except Exception as e:
        print(f"Error initializing parser: {e}")
        return None


def get_language_types(lang: str) -> Dict[str, List[str]]:
    """Get language-specific node type definitions."""
    types = {
        "java": {
            "str": ["character_literal", "string_literal"],
            "num": ["decimal_integer_literal", "decimal_floating_point_literal"],
            "identifier": ["variable_declarator", "formal_parameter", "enhanced_for_statement"],
            "statement": ["binary_expression", "assignment_expression",
                         "method_invocation", "local_variable_declaration",
                         "literal", "return_statement", "object_creation_expression",
                         "field_access", "array_creation_expression"]
        },
        "python": {
            "str": ["string"],
            "num": ["integer", "float"],
            "identifier": ["assignment", "argument_list"],
            "statement": ["binary_expression", "binary_operator", "comparison_operator",
                         "assignment_expression", "call", "subscript",
                         "literal", "expression_statement", "return_statement",
                         "attribute", "keyword_argument"]
        },
        "cpp": {
            "str": ["string_literal", "char_literal"],
            "num": ["number_literal"],
            "identifier": ["function_declarator", "pointer_declarator", "declaration"],
            "statement": ["binary_expression", "assignment_expression",
                         "call_expression", "return_statement"]
        }
    }
    return types.get(lang, types["python"])


def remove_comments_and_docstrings(source: str, lang: str) -> str:
    """Remove comments and docstrings from source code."""
    if lang == 'python':
        try:
            io_obj = StringIO(source)
            out = ""
            prev_toktype = tokenize.INDENT
            last_lineno = -1
            last_col = 0

            for tok in tokenize.generate_tokens(io_obj.readline):
                token_type = tok[0]
                token_string = tok[1]
                start_line, start_col = tok[2]
                end_line, end_col = tok[3]

                if start_line > last_lineno:
                    last_col = 0
                if start_col > last_col:
                    out += (" " * (start_col - last_col))

                if token_type == tokenize.COMMENT:
                    pass
                elif token_type == tokenize.STRING:
                    if prev_toktype != tokenize.INDENT:
                        if prev_toktype != tokenize.NEWLINE:
                            if start_col > 0:
                                out += token_string
                else:
                    out += token_string

                prev_toktype = token_type
                last_col = end_col
                last_lineno = end_line

            return '\n'.join([x for x in out.split('\n') if x.strip()])
        except:
            return source

    else:  # Java, C++, etc.
        def replacer(match):
            s = match.group(0)
            return " " if s.startswith('/') else s

        pattern = re.compile(
            r'//.*?$|/\*.*?\*/|\'(?:\\.|[^\\\'])*\'|"(?:\\.|[^\\"])*"',
            re.DOTALL | re.MULTILINE
        )
        result = re.sub(pattern, replacer, source)
        return '\n'.join([x for x in result.split('\n') if x.strip()])


def tree_to_token_index(root_node) -> List[Tuple]:
    """Convert tree to token indices."""
    if (len(root_node.children) == 0 or root_node.type == 'string') and root_node.type != 'comment':
        return [(root_node.start_byte, root_node.end_byte)]
    else:
        code_tokens = []
        for child in root_node.children:
            code_tokens += tree_to_token_index(child)
        return code_tokens


def index_to_code_token(index: Tuple[int, int], code: str) -> str:
    """Convert byte index to code token."""
    start_byte, end_byte = index
    lines = code.split('\n')

    # Convert byte positions to line/column
    current_byte = 0
    start_line = start_col = end_line = end_col = 0

    for line_num, line in enumerate(lines):
        line_bytes = len(line) + 1  # +1 for newline
        if current_byte <= start_byte < current_byte + line_bytes:
            start_line = line_num
            start_col = start_byte - current_byte
        if current_byte <= end_byte <= current_byte + line_bytes:
            end_line = line_num
            end_col = end_byte - current_byte
            break
        current_byte += line_bytes

    # Extract token
    if start_line == end_line:
        return lines[start_line][start_col:end_col]
    else:
        result = lines[start_line][start_col:]
        for i in range(start_line + 1, end_line):
            result += ' ' + lines[i]
        result += ' ' + lines[end_line][:end_col]
        return result


def rewrite_num_str_func(node, source: bytes, types: Dict, offset: int = 0) -> Tuple[bytes, int]:
    """Rewrite numbers and strings with placeholder tokens."""
    start_byte = node.start_byte + offset
    end_byte = node.end_byte + offset

    if node.type in types["str"]:
        replacement = b'__str__'
        source = source[:start_byte] + replacement + source[end_byte:]
        offset += len(replacement) - (end_byte - start_byte)
    elif node.type in types["num"]:
        if node.text != b"0":
            replacement = b'__num__'
            source = source[:start_byte] + replacement + source[end_byte:]
            offset += len(replacement) - (end_byte - start_byte)

    for child in node.children:
        source, offset = rewrite_num_str_func(child, source, types, offset)

    return source, offset


def rewrite_variables_func(node, source: bytes, types: Dict, variables: List[str] = None,
                           offset: int = 0) -> Tuple[bytes, int, List[str]]:
    """Rewrite variable identifiers with placeholder tokens."""
    if variables is None:
        variables = []

    start_byte = node.start_byte + offset
    end_byte = node.end_byte + offset

    try:
        token = source[start_byte:end_byte].decode("utf8")
    except:
        token = ""

    if node.type == 'identifier' and node.type != token:
        if token in variables:
            replacement = b'__identifier__'
            source = source[:start_byte] + replacement + source[end_byte:]
            offset += len(replacement) - (end_byte - start_byte)
        elif node.parent and node.parent.type in types["identifier"]:
            replacement = b'__identifier__'
            source = source[:start_byte] + replacement + source[end_byte:]
            offset += len(replacement) - (end_byte - start_byte)
            variables.append(token)

    for child in node.children:
        source, offset, variables = rewrite_variables_func(child, source, types, variables, offset)

    return source, offset, variables


def split_statements(node, types: Dict) -> List[Tuple[str, Tuple[int, int]]]:
    """Split code into statement-level expressions."""
    start_byte = node.start_byte
    end_byte = node.end_byte
    expressions = []

    if node.type in types["statement"]:
        try:
            text = node.text.decode("utf-8")
            expressions.append([text, (start_byte, end_byte)])
        except:
            pass

    for child in node.children:
        expressions.extend(split_statements(child, types))

    return expressions


def replace_with_tokens(expression: Tuple[str, Tuple[int, int]],
                        code: str, tokens_index: List) -> str:
    """Replace expression with token representation."""
    expression_text, idx = expression
    expression_token_idxes = []

    for t in tokens_index:
        if t[0] >= idx[0] and t[1] <= idx[1]:
            expression_token_idxes.append(t)
        elif t[0] < idx[0]:
            continue
        elif t[1] > idx[1]:
            break

    code_tokens = [index_to_code_token(x, code) for x in expression_token_idxes]
    code_tokens = [c for c in code_tokens if c]

    return " ".join(code_tokens)


def apply_rule_inversion(split_samples: List[List[Tuple[str, int]]]) -> List[List[str]]:
    """Apply rule inversion to split samples."""
    inverted_samples = []

    for statements in split_samples:
        sample = []
        for idx, (code0, stat_idx0) in enumerate(statements[:-1]):
            for code1, stat_idx1 in statements[idx + 1:]:
                if stat_idx1 != stat_idx0:
                    break
                elif code1 != code0:
                    code0 = code0.replace(code1, "__value__")
            sample.append(code0)

        if statements:
            sample.append(statements[-1][0])

        inverted_samples.append(sample)

    return inverted_samples


def preprocess_from_list(
    code_list: List[str],
    lang: str = "python",
    tree_sitter_path: Optional[str] = None,
    remove_comments: bool = True,
    rewrite_variables: bool = True,
    rewrite_num_str: bool = True,
    granularity: str = "statement",
    rule_inversion: bool = False,
    max_samples: Optional[int] = None
) -> List[CodeSample]:
    """
    Preprocess code samples from a list into expression format for DeCoMa detection.

    Args:
        code_list: List of code strings to process
        lang: Programming language (python, java, cpp)
        tree_sitter_path: Path to tree-sitter library (optional)
        remove_comments: Whether to remove comments and docstrings
        rewrite_variables: Whether to rewrite variable names
        rewrite_num_str: Whether to rewrite numbers and strings
        granularity: Split granularity ("statement" or "identifier")
        rule_inversion: Whether to apply rule inversion
        max_samples: Maximum number of samples to process

    Returns:
        List of processed CodeSample objects
    """

    # Apply max_samples limit if specified
    samples = code_list[:max_samples] if max_samples else code_list
    print(f"Processing {len(samples)} samples")

    # Initialize parser if using tree-sitter features
    parser = None
    if TREE_SITTER_AVAILABLE and (rewrite_variables or rewrite_num_str or granularity == "statement"):
        parser = get_parser(lang, tree_sitter_path)
        if not parser:
            print("Warning: Could not initialize parser. Some features will be disabled.")

    types = get_language_types(lang)
    processed_samples = []

    # Process each sample
    for idx, original_code in enumerate(tqdm(samples, desc="Processing samples")):
        try:
            # Keep original code for reference
            code = original_code
            # Remove comments if requested
            if remove_comments:
                code = remove_comments_and_docstrings(code, lang)

            variables = []
            expressions = []

            # Process with tree-sitter if available
            if parser:
                try:
                    # Parse code
                    tree = parser.parse(bytes(code, "utf8"))
                    root_node = tree.root_node

                    # Rewrite variables
                    if rewrite_variables:
                        code_bytes, _, vars_list = rewrite_variables_func(
                            root_node, bytes(code, "utf8"), types, [], 0
                        )
                        code = code_bytes.decode("utf8")
                        variables = vars_list
                        # Re-parse after rewriting
                        tree = parser.parse(bytes(code, "utf8"))
                        root_node = tree.root_node

                    # Rewrite numbers and strings
                    if rewrite_num_str:
                        code_bytes, _ = rewrite_num_str_func(
                            root_node, bytes(code, "utf8"), types, 0
                        )
                        code = code_bytes.decode("utf8")
                        # Re-parse after rewriting
                        tree = parser.parse(bytes(code, "utf8"))
                        root_node = tree.root_node

                    # Split into statements
                    if granularity == "statement":
                        statement_list = split_statements(root_node, types)
                        tokens_index = tree_to_token_index(root_node)

                        # Group statements
                        grouped_statements = []
                        prev_idx = [0, 0]
                        idx = -1

                        for stmt in statement_list:
                            expr_text = replace_with_tokens(stmt, code, tokens_index)
                            if stmt[1][0] >= prev_idx[0] and stmt[1][1] <= prev_idx[1]:
                                grouped_statements.append([expr_text, idx])
                            else:
                                idx += 1
                                prev_idx = stmt[1]
                                grouped_statements.append([expr_text, idx])

                        # Apply rule inversion if requested
                        if rule_inversion and grouped_statements:
                            expressions = apply_rule_inversion([grouped_statements])[0]
                        else:
                            expressions = [s[0] for s in grouped_statements]
                    else:
                        # Default: use full code as single expression
                        expressions = [code]

                except Exception as e:
                    print(f"Error processing sample {idx}: {e}")
                    expressions = [code]
            else:
                # No parser available, use basic processing
                expressions = [code]

            # Create CodeSample
            sample = CodeSample(
                code=original_code,  # Original code
                expressions=expressions,
                identifiers=variables
            )

            # Always append sample, even if invalid (to maintain index alignment)
            processed_samples.append(sample)

        except Exception as e:
            print(f"Error processing sample {idx}: {e}")
            # Create a placeholder sample for failed processing to maintain index alignment
            placeholder = CodeSample(
                code=original_code if 'original_code' in locals() else "",
                expressions=[],
                identifiers=[]
            )
            processed_samples.append(placeholder)

    # Return processed samples
    valid_count = sum(1 for s in processed_samples if s.is_valid())
    print(f"Processed {len(processed_samples)} samples ({valid_count} valid, {len(processed_samples) - valid_count} invalid/failed)")
    return processed_samples


def preprocess(
    input_path: str,
    output_path: str,
    lang: str = "python",
    tree_sitter_path: Optional[str] = None,
    remove_comments: bool = True,
    rewrite_variables: bool = True,
    rewrite_num_str: bool = True,
    granularity: str = "statement",
    rule_inversion: bool = False,
    max_samples: Optional[int] = None
) -> None:
    """
    Preprocess code samples from file into expression format for DeCoMa detection.

    Args:
        input_path: Path to input file (JSONL format)
        output_path: Path to output file (JSONL format)
        lang: Programming language (python, java, cpp)
        tree_sitter_path: Path to tree-sitter library (optional)
        remove_comments: Whether to remove comments and docstrings
        rewrite_variables: Whether to rewrite variable names
        rewrite_num_str: Whether to rewrite numbers and strings
        granularity: Split granularity ("statement" or "identifier")
        rule_inversion: Whether to apply rule inversion
        max_samples: Maximum number of samples to process
    """

    # Load input samples
    print(f"Loading samples from {input_path}")
    samples = []
    with open(input_path, 'r') as f:
        for i, line in enumerate(f):
            if max_samples and i >= max_samples:
                break
            try:
                data = json.loads(line.strip())
                # Handle different input formats
                if "func" in data:
                    code = data["func"]
                elif "code" in data:
                    code = data["code"]
                else:
                    code = data.get("content", "")
                samples.append(code)
            except Exception as e:
                print(f"Error loading line {i+1}: {e}")
                continue

    print(f"Loaded {len(samples)} samples from file")

    # Use the list-based function
    processed_samples = preprocess_from_list(
        samples,
        lang=lang,
        tree_sitter_path=tree_sitter_path,
        remove_comments=remove_comments,
        rewrite_variables=rewrite_variables,
        rewrite_num_str=rewrite_num_str,
        granularity=granularity,
        rule_inversion=rule_inversion
    )

    # Save results
    DataLoader.save_jsonl(processed_samples, output_path)
    print(f"Preprocessing complete. Output saved to {output_path}")


# Convenience function for quick preprocessing
def preprocess_file(input_file: str, output_file: str, **kwargs) -> None:
    """
    Quick preprocessing of a file with default settings.

    Args:
        input_file: Input JSONL file path
        output_file: Output JSONL file path
        **kwargs: Additional arguments to pass to preprocess()
    """
    default_args = {
        "lang": "python",
        "remove_comments": True,
        "rewrite_variables": True,
        "rewrite_num_str": True,
        "granularity": "statement",
        "rule_inversion": False
    }
    default_args.update(kwargs)

    preprocess(input_file, output_file, **default_args)