"""Cells of the main grid: every base, teacher and seed, and the teacher draw each one uses."""


def student_name(base: str, teacher: str, seed: int) -> str:
    return f"{base}__{teacher}__s{seed}"


def main_cells(main: dict) -> list[dict]:
    rows = []
    for base, model in main["bases"].items():
        for position, seed in enumerate(main["seeds"]):
            for teacher in main["teachers"]:
                rows.append(
                    {
                        "base": base,
                        "model": model,
                        "teacher": teacher,
                        "seed": seed,
                        "rep": main["draws"][base][position],
                        "student": student_name(base, teacher, seed),
                    }
                )
    return rows


def all_reps(main: dict) -> list[int]:
    reps = sorted(r for draws in main["draws"].values() for r in draws)
    if len(reps) != len(set(reps)):
        raise ValueError(f"main_grid.draws gives two runs the same draw: {main['draws']}")
    return reps

