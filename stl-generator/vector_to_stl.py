import numpy as np
from stl import mesh
import csv
import sys


def create_vector_mesh(x, y, z, px, py, pz):
    # Создаем вершины для стрелки
    # Основание стрелки (цилиндр)
    radius = 0.1
    height = np.sqrt(px**2 + py**2 + pz**2) * 0.8  # 80% от длины вектора

    # Создаем цилиндр
    theta = np.linspace(0, 2 * np.pi, 32)
    z_cylinder = np.linspace(0, height, 2)
    theta_grid, z_grid = np.meshgrid(theta, z_cylinder)

    x_cylinder = radius * np.cos(theta_grid)
    y_cylinder = radius * np.sin(theta_grid)
    z_cylinder = z_grid

    # Создаем наконечник стрелки (конус)
    cone_height = height * 0.2  # 20% от длины вектора
    cone_radius = radius * 2

    theta_cone = np.linspace(0, 2 * np.pi, 32)
    z_cone = np.linspace(0, cone_height, 2)
    theta_grid_cone, z_grid_cone = np.meshgrid(theta_cone, z_cone)

    x_cone = (cone_radius * (1 - z_grid_cone / cone_height)) * np.cos(theta_grid_cone)
    y_cone = (cone_radius * (1 - z_grid_cone / cone_height)) * np.sin(theta_grid_cone)
    z_cone = z_grid_cone + height

    # Объединяем вершины
    vertices = np.vstack(
        (
            np.column_stack(
                (x_cylinder.flatten(), y_cylinder.flatten(), z_cylinder.flatten())
            ),
            np.column_stack((x_cone.flatten(), y_cone.flatten(), z_cone.flatten())),
        )
    )

    # Создаем грани
    faces = []
    n_theta = 32

    # Грани цилиндра
    for i in range(n_theta):
        i_next = (i + 1) % n_theta
        faces.extend(
            [[i, i + n_theta, i_next], [i_next, i + n_theta, i_next + n_theta]]
        )

    # Грани конуса
    base_idx = 2 * n_theta
    for i in range(n_theta):
        i_next = (i + 1) % n_theta
        faces.extend(
            [
                [base_idx + i, base_idx + i_next, base_idx + n_theta + i],
                [
                    base_idx + i_next,
                    base_idx + n_theta + i_next,
                    base_idx + n_theta + i,
                ],
            ]
        )

    # Создаем меш
    vector_mesh = mesh.Mesh(np.zeros(len(faces), dtype=mesh.Mesh.dtype))
    for i, face in enumerate(faces):
        for j in range(3):
            vector_mesh.vectors[i][j] = vertices[face[j]]

    # Поворачиваем вектор в нужном направлении
    if not (px == 0 and py == 0 and pz == 0):
        # Нормализуем вектор направления
        direction = np.array([px, py, pz])
        direction = direction / np.linalg.norm(direction)

        # Находим угол поворота
        z_axis = np.array([0, 0, 1])
        rotation_axis = np.cross(z_axis, direction)
        if np.linalg.norm(rotation_axis) > 0:
            rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
            rotation_angle = np.arccos(np.dot(z_axis, direction))

            # Применяем поворот
            for i in range(len(vector_mesh.vectors)):
                for j in range(3):
                    vector_mesh.vectors[i][j] = np.dot(
                        rotation_matrix(rotation_axis, rotation_angle),
                        vector_mesh.vectors[i][j],
                    )

    # Смещаем вектор в начальную точку
    vector_mesh.vectors += np.array([x, y, z])

    return vector_mesh


def rotation_matrix(axis, theta):
    """
    Создает матрицу поворота вокруг оси на заданный угол
    """
    axis = np.asarray(axis)
    axis = axis / np.sqrt(np.dot(axis, axis))
    a = np.cos(theta / 2.0)
    b, c, d = -axis * np.sin(theta / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return np.array(
        [
            [aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
            [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
            [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc],
        ]
    )


def main():
    if len(sys.argv) != 2:
        print("Usage: python vector_to_stl.py input.csv")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = "vectors.stl"

    try:
        # Список для хранения всех мешей
        meshes = []

        with open(input_file, "r") as f:
            reader = csv.reader(f)
            next(reader, None)
            for row in reader:
                if len(row) != 7:
                    print(f"Invalid row format: {row}")
                    continue

                N = int(row[0])
                x, y, z = map(float, row[1:4])
                px, py, pz = map(float, row[4:7])

                # Создаем меш для текущего вектора
                vector_mesh = create_vector_mesh(x, y, z, px, py, pz)
                meshes.append(vector_mesh)
                print(f"Processed vector {N}")

        if meshes:
            # Объединяем все меши
            combined_mesh = mesh.Mesh(np.concatenate([m.data for m in meshes]))
            # Сохраняем в один STL файл
            combined_mesh.save(output_file)
            print(f"Created {output_file} with {len(meshes)} vectors")
        else:
            print("No valid vectors found in the input file")

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
