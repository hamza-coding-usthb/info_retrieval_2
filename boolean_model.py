import re

def validate_boolean_query(query):
    print(query)
    print("did it reach here 111111111111111")
    """Validate the logical syntax of a Boolean query."""
    valid_terms = r"[A-Za-z0-9]+"
    valid_operators = r"(AND|OR|NOT)"
    valid_pattern = re.compile(f"^(NOT )?{valid_terms}( ({valid_operators}) (NOT )?{valid_terms})*$")
    return bool(valid_pattern.match(query.strip()))

def infix_to_postfix(query):
    """
    Convert an infix Boolean query to postfix notation using the shunting-yard algorithm.
    :param query: The Boolean query string in infix notation.
    :return: A list of tokens in postfix notation.
    """
    print("did it reach here 22222222222222")
    precedence = {"NOT": 3, "AND": 2, "OR": 1}
    output = []
    operators = []

    tokens = query.split()
    for token in tokens:
        if token in precedence:
            while (operators and operators[-1] != "(" and precedence[operators[-1]] >= precedence[token]):
                output.append(operators.pop())
            operators.append(token)
        elif token == "(":
            operators.append(token)
        elif token == ")":
            while operators and operators[-1] != "(":
                output.append(operators.pop())
            operators.pop()  # Remove the "("
        else:
            output.append(token)

    while operators:
        output.append(operators.pop())

    return output


def evaluate_boolean_query_with_pandas(query, term_to_docs):
    """
    Evaluate a Boolean query using a stack-based approach with proper handling of NOT.
    Converts infix to postfix notation before evaluation.
    :param query: The Boolean query string.
    :param term_to_docs: A dictionary mapping terms to sets of document IDs.
    :return: A set of document IDs that match the query.
    """
    print("did it reach here 3333333333333333333")
    print(f"Evaluating query: {query}")
    postfix_query = infix_to_postfix(query)
    print(f"Postfix notation: {postfix_query}")

    stack = []

    # All documents in the dataset
    all_docs = set.union(*term_to_docs.values()) if term_to_docs else set()
    print(f"All documents: {all_docs}")

    for token in postfix_query:
        print(f"Processing token: {token}")
        if token == "AND":
            if len(stack) < 2:
                raise ValueError(f"Invalid query: insufficient operands for 'AND'. Stack: {stack}")
            right = stack.pop()
            left = stack.pop()
            result = left & right
            print(f"AND operation: {left} & {right} = {result}")
            stack.append(result)
        elif token == "OR":
            if len(stack) < 2:
                raise ValueError(f"Invalid query: insufficient operands for 'OR'. Stack: {stack}")
            right = stack.pop()
            left = stack.pop()
            result = left | right
            print(f"OR operation: {left} | {right} = {result}")
            stack.append(result)
        elif token == "NOT":
            if not stack:
                raise ValueError(f"Invalid query: insufficient operand for 'NOT'. Stack: {stack}")
            operand = stack.pop()
            result = all_docs - operand
            print(f"NOT operation: {all_docs} - {operand} = {result}")
            stack.append(result)
        else:
            doc_set = term_to_docs.get(token, set())
            print(f"Term: {token}, Document Set: {doc_set}")
            stack.append(doc_set)

        print(f"Stack after token '{token}': {stack}")

    if len(stack) != 1:
        raise ValueError(f"Invalid query: final stack has more than one element. Stack: {stack}")

    final_result = stack[-1]
    print(f"Final result: {final_result}")
    return final_result
