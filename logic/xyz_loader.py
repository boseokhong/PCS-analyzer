# logic/xyz_loader.py

def parse_xyz(file_path):
    """
    Supports:
    1) Standard XYZ format:
       Line 1: atom count (integer)
       Line 2: comment
       Line 3+: element x y z

    2) Headerless XYZ format:
       element x y z
       element x y z
       ...

    Returns:
        List of tuples: [(element, x, y, z), ...]
    """
    atom_data = []

    with open(file_path, 'r', encoding='utf-8') as f:
        lines = [ln.strip() for ln in f if ln.strip()]

    if not lines:
        return atom_data

    # --- detect standard XYZ header ---
    start_index = 0

    try:
        # If first line is integer, assume standard XYZ
        int(lines[0])
        start_index = 2
    except ValueError:
        # Headerless format
        start_index = 0

    for line in lines[start_index:]:
        parts = line.split()
        if len(parts) >= 4:
            try:
                atom_data.append((
                    parts[0],
                    float(parts[1]),
                    float(parts[2]),
                    float(parts[3])
                ))
            except ValueError:
                continue

    return atom_data