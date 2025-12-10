import argparse
import multiprocessing as mp

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

def read_doses_from_file(filename):

    with open(filename, 'r', encoding='utf-8') as f:
        first_line = f.readline().strip().split()
        if len(first_line) != 3:
            raise ValueError("Первая строка должна содержать 3 целых числа: Nx Ny Nz")

        Nx, Ny, Nz = map(int, first_line)
        total_voxels = Nx * Ny * Nz

        doses = []
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            for p in parts:
                doses.append(float(p))

        if len(doses) != total_voxels:
            raise ValueError(
                f"Ожидалось {total_voxels} значений доз, "
                f"а прочитано {len(doses)}."
            )

    return Nx, Ny, Nz, doses

def local_histogram_pure(chunk, delta_D, num_bins):

    hist = [0] * num_bins

    for d in chunk:
        if d < 0:
            idx = 0
        else:
            idx = int(d / delta_D)
            if idx >= num_bins:
                idx = num_bins - 1
        hist[idx] += 1

    return hist

def compute_dvh_parallel_pure(doses, delta_D, n_processes=None):

    if not doses:
        raise ValueError("Список доз пуст")

    D_max = max(doses)
    if delta_D <= 0:
        raise ValueError("ΔD должен быть > 0")
    num_bins = int(D_max // delta_D) + 1

    total_voxels = len(doses)

    if n_processes is None or n_processes <= 0:
        n_processes = mp.cpu_count()

    chunk_size = (total_voxels + n_processes - 1) // n_processes
    chunks = [doses[i:i + chunk_size] for i in range(0, total_voxels, chunk_size)]

    with mp.Pool(processes=n_processes) as pool:
        local_hists = pool.starmap(
            local_histogram_pure,
            [(chunk, delta_D, num_bins) for chunk in chunks]
        )

    global_hist = [0] * num_bins
    for h in local_hists:
        for i in range(num_bins):
            global_hist[i] += h[i]

    tail_counts = [0] * num_bins
    running = 0
    for i in range(num_bins - 1, -1, -1):
        running += global_hist[i]
        tail_counts[i] = running

    D_values = [i * delta_D for i in range(num_bins)]
    V_values = [tail_counts[i] / total_voxels * 100.0 for i in range(num_bins)]

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
    print(f"Минимальная доза: {min(doses):.4f}")
    print(f"Максимальная доза: {max(doses):.4f}")
    print(f"Шаг по дозе ΔD = {delta_D}")
    if args.processes:
        print(f"Число процессов: {args.processes}")
    else:
        print("Число процессов: по умолчанию (число ядер)")

    import time
    t0 = time.time()
    D_values, V_values = compute_dvh_parallel_pure(doses, delta_D, args.processes)
    t1 = time.time()
    print(f"\nВремя вычислений DVH (без учёта чтения файла и графика): {t1 - t0:.3f} c")

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

    if (not args.no_plot) and (plt is not None):
        plt.figure()
        plt.plot(D_values, V_values, marker='o')
        plt.xlabel("D (доза)")
        plt.ylabel("V(D), % объёма")
        plt.title("DVH (без NumPy, параллельный расчёт)")
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    elif (not args.no_plot) and (plt is None):
        print("\nmatplotlib не установлен, график построить нельзя.")


if __name__ == "__main__":
    main()
