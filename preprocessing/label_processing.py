import ast


DIAGNOSTIC_SUPERCLASSES = ["NORM", "MI", "STTC", "CD", "HYP"]


def parse_scp_codes(value):
    """Convert the scp_codes string into a Python dictionary."""
    if isinstance(value, dict):
        return value
    try:
        return ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return {}


def add_diagnostic_superclass_labels(dataframe, scp_statements):
    """
    Add one binary column per diagnostic superclass.

    PTB-XL is multi-label: one ECG can belong to several diagnostic
    superclasses at the same time.
    """
    dataframe = dataframe.copy()

    for superclass in DIAGNOSTIC_SUPERCLASSES:
        dataframe[superclass] = 0.0

    for index, row in dataframe.iterrows():
        scp_codes = parse_scp_codes(row["scp_codes"])

        for code in scp_codes.keys():
            if code not in scp_statements.index:
                continue

            statement = scp_statements.loc[code]

            if "diagnostic" in statement and statement["diagnostic"] != 1:
                continue

            superclass = statement.get("diagnostic_class")
            if superclass in DIAGNOSTIC_SUPERCLASSES:
                dataframe.at[index, superclass] = 1.0

    return dataframe, DIAGNOSTIC_SUPERCLASSES
