_NO_CONTEXT_PROMPT = """Context:
{context}

Question:
What is the temporal relation between the {source} and the {target}?

Options:
<, in case the {source} happens before the {target}
>, in case the {source} happens after the {target}
=, in case the {source} happens the same time as the {target}
-, in case the {source} happens not related to the {target}

Answer:
"""
