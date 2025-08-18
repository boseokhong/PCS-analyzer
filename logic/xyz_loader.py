# logic/xyz_loader.py

def parse_xyz(file_path):
    atom_data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    for line in lines[2:]:
        parts = line.split()
        if len(parts)>=4:
            atom_data.append((parts[0], float(parts[1]), float(parts[2]), float(parts[3])))
    return atom_data
