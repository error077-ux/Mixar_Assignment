import trimesh

def identify_mock_seams(mesh):
    seams = []
    faces = mesh.faces
    for face in faces:
        edges = [(face[i], face[(i + 1) % 3]) for i in range(3)]
        for v1, v2 in edges:
            if abs(v1 - v2) % 5 == 0:
                seams.append((v1, v2))
    return seams

def encode_seams(seams):
    return [f"SEAM_{v1}_{v2}" for v1, v2 in seams]

def decode_tokens(tokens):
    edges = []
    for token in tokens:
        parts = token.split("_")
        v1, v2 = int(parts[1]), int(parts[2])
        edges.append((v1, v2))
    return edges

def main():
    print("=== Seam Tokenization Prototype ===")
    mesh = trimesh.load("meshes/branch.obj", process=False)

    seams = identify_mock_seams(mesh)
    print(f"Detected {len(seams)} seam-like edges.")

    tokens = encode_seams(seams)
    print("Example tokens:", tokens[:10])

    decoded_edges = decode_tokens(tokens)
    print("Decoded edges (sample):", decoded_edges[:5])

    with open("seam_tokens.txt", "w") as f:
        for t in tokens:
            f.write(t + "\n")
    print("Token list saved as seam_tokens.txt")

if __name__ == "__main__":
    main()
