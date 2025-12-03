import argparse
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp


def read_doses_from_file(filename):
    """
    Считывает размеры сетки и дозы из текстового файла.
    Формат:
    первая строка: Nx Ny Nz
    далее Nx*Ny*Nz чисел (по одному или по несколько в строке).
    """
    with open(filename, 'r', encoding='utf-8') as f:
        first_line = f.readline().strip().split()
        if len(first_line) != 3:
            raise ValueError("Первая строка должна содержать 3 целых числа: Nx Ny Nz")

        Nx, Ny, Nz = map(int, first_line)
        total_voxels = Nx * Ny * Nz

        doses_list = []
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            for p in parts:
                doses_list.append(float(p))

        if len(doses_list) != total_voxels:
            raise ValueError(
                f"Ожидалось {total_voxels} значений доз, "
                f"а прочитано {len(doses_list)}."
            )

    doses = np.array(doses_list, dtype=float)
    return Nx, Ny, Nz, doses


def local_histogram(chunk, bin_edges):
    hist, _ = np.histogram(chunk, bins=bin_edges)
    return hist


def compute_dvh_parallel(doses, delta_D, n_processes=None):
    D_max = doses.max()

    # Границы бинов: 0, ΔD, 2ΔD, ..., до значения >= D_max
    bin_edges = np.arange(0.0, D_max + delta_D * 1.0001, delta_D)
    if len(bin_edges) < 2:
        raise ValueError("Слишком большой шаг ΔD: получается меньше двух бинов.")

    # Значения D соответствуют левым границам интервалов
    D_values = bin_edges[:-1]

    total_voxels = doses.size

    if n_processes is None or n_processes <= 0:
        n_processes = mp.cpu_count()

    # Делим массив доз на n_processes приблизительно равных кусков
    chunks = np.array_split(doses, n_processes)

    with mp.Pool(processes=n_processes) as pool:
        local_hists = pool.starmap(local_histogram, [(chunk, bin_edges) for chunk in chunks])

    # Складываем локальные гистограммы, получаем общую
    global_hist = np.sum(local_hists, axis=0)  # длина = len(bin_edges) - 1

    # Количество вокселей с дозой >= D: накопленная сумма справа налево
    tail_counts = np.cumsum(global_hist[::-1])[::-1]

    # Переводим в проценты объёма
    V_values = tail_counts / total_voxels * 100.0

    return D_values, V_values


def parse_args():
    parser = argparse.ArgumentParser(
        description="Расчёт DVH (dose–volume histogram) с параллельной обработкой."
    )
    parser.add_argument(
        "input_file",
        help="Путь к входному файлу с дозами (формат: Nx Ny Nz, далее дозы).",
    )
    parser.add_argument(
        "-d", "--delta",
        type=float,
        required=True,
        help="Шаг по дозе ΔD (например, 0.1).",
    )
    parser.add_argument(
        "-o", "--output",
        help="Файл для вывода таблицы D V(D). Если не указан, выводится в stdout.",
    )
    parser.add_argument(
        "-p", "--processes",
        type=int,
        default=None,
        help="Число процессов для параллельной обработки (по умолчанию = числу ядер).",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Не строить график, только вывести/сохранить таблицу.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    delta_D = args.delta
    if delta_D <= 0:
        raise ValueError("Шаг ΔD должен быть положительным числом.")

    Nx, Ny, Nz, doses = read_doses_from_file(args.input_file)
    total_voxels = Nx * Ny * Nz

    print(f"Файл: {args.input_file}")
    print(f"Размеры сетки: Nx={Nx}, Ny={Ny}, Nz={Nz}")
    print(f"Общее число кубиков: {total_voxels}")
    print(f"Минимальная доза: {doses.min():.4f}")
    print(f"Максимальная доза: {doses.max():.4f}")
    print(f"Шаг по дозе ΔD = {delta_D}")

    D_values, V_values = compute_dvh_parallel(doses, delta_D, args.processes)

    # Вывод таблицы
    lines = ["# D\tV(D)_percent"]
    for D, V in zip(D_values, V_values):
        lines.append(f"{D:.6f}\t{V:.6f}")

    text_table = "\n".join(lines)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(text_table + "\n")
        print(f"\nРезультаты сохранены в файл: {args.output}")
    else:
        print("\nТаблица D  V(D) (в процентах):")
        print(text_table)

    # График
    if not args.no_plot:
        plt.figure()
        plt.plot(D_values, V_values, marker='o')
        plt.xlabel("D (доза)")
        plt.ylabel("V(D), % объёма")
        plt.title("График зависимости доза–объём (DVH, параллельный расчёт)")
        plt.grid(True)
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
