"""build_ascent_small_mrg_table.py

Собирает таблицу параметров малого MRG-аксона по ASCENT SMALL_MRG_INTERPOLATION
для диаметров 1.0-4.0 мкм с шагом 0.5.

Источник:
  - ASCENT v1.5.0, config/system/fiber_z.json
  - ASCENT docs, Running_ASCENT/Info.md, раздел SMALL_MRG_INTERPOLATION
  - формулы из Figure A документации ASCENT

Важно:
  ASCENT официально поддерживает SMALL_MRG_INTERPOLATION для диаметров
  от 1.011 до 16.0 мкм. Строка для 1.0 мкм в таблице помечается как
  экстраполяция вне поддержанного диапазона.
"""

from math import exp
from pathlib import Path

import pandas as pd


DIAMETERS_UM = [1.5 + 0.5 * i for i in range(6)]


def ascent_small_mrg_row(diameter_um: float) -> dict:
    """Средние параметры без случайной компоненты из документации ASCENT."""

    d = float(diameter_um)

    # ASCENT SMALL_MRG_INTERPOLATION: официально поддержан диапазон 1.011-16.0 мкм.
    supported = 1.011 <= d <= 16.0

    # Figure A panel D: g-ratio как функция диаметра миелинизированного волокна D.
    g_ratio = 0.020 * (d - 2.39) + 0.55

    # Internodal axon diameter d_a.
    axon_d_um = g_ratio * d

    # Figure A panel B: dn/da.
    node_to_axon_ratio = -0.011 * (axon_d_um - 7.15) + 0.40
    node_d_um = node_to_axon_ratio * axon_d_um

    # ASCENT config/system/fiber_z.json.
    delta_z_um = -3.22 * d**2 + 148.0 * d - 128.0
    paranodal_length_1_um = 3.0
    node_length_um = 1.0
    flut_length_um = -0.171 * d**2 + 6.48 * d - 0.935
    stin_length_um = (delta_z_um - node_length_um - (2.0 * paranodal_length_1_um) - (2.0 * flut_length_um)) / 6.0

    # Figure A panel C: ln(nl).
    nl_cont = exp(0.5 * (axon_d_um - 1.75) + 3.2)
    nl_rounded = int(round(nl_cont))

    return {
        "fiber_diameter_um": d,
        "supported_by_ascent": int(supported),
        "note": "extrapolated below ASCENT lower bound" if not supported else "",
        "delta_z_um": delta_z_um,
        "node_length_um": node_length_um,
        "paranodal_length_1_um": paranodal_length_1_um,
        "paranodal_length_2_um": flut_length_um,
        "inter_length_um": stin_length_um,
        "g_ratio": g_ratio,
        "internodal_axon_diameter_um": axon_d_um,
        "node_to_axon_ratio": node_to_axon_ratio,
        "node_diameter_um": node_d_um,
        "MYSA_diameter_um": node_d_um,
        "FLUT_diameter_um": axon_d_um,
        "STIN_diameter_um": axon_d_um,
        "myelin_lamellae_continuous": nl_cont,
        "myelin_lamellae_rounded": nl_rounded,
        # Из документации ASCENT для малого MRG.
        "gnabar_S_per_cm2": 2.333,
        "gkbar_S_per_cm2": 0.116,
        # Оставшиеся из оригинального MRG, если не переопределены ASCENT.
        "gnapbar_S_per_cm2": 0.01,
        "gl_S_per_cm2": 0.007,
        "ena_mV": 50.0,
        "ek_mV": -90.0,
        "el_mV": -90.0,
        "v_init_mV": -80.0,
    }


def main():
    rows = [ascent_small_mrg_row(d) for d in DIAMETERS_UM]
    df = pd.DataFrame(rows)

    out_dir = Path(__file__).resolve().parent / "data" / "ascent_small_mrg"
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / "ascent_small_mrg_table_1p0_to_4p0_um.csv"
    df.to_csv(csv_path, index=False)

    readme_lines = [
        "ASCENT small MRG parameter table",
        "===============================",
        "",
        "Source files:",
        "- ascent_repo/config/system/fiber_z.json",
        "- ascent_repo/docs/source/Running_ASCENT/Info.md",
        "- ascent_repo/docs/source/uploads/small_MRG_interp_v1_figures/FigureA.png",
        "",
        "Implemented formulas:",
        "- delta_z = -3.22*D^2 + 148*D - 128",
        "- L_FLUT = -0.171*D^2 + 6.48*D - 0.935",
        "- L_STIN = (delta_z - node_length - 2*L_MYSA - 2*L_FLUT)/6",
        "- g_ratio = 0.020*(D - 2.39) + 0.55",
        "- d_a = g_ratio * D",
        "- d_n/d_a = -0.011*(d_a - 7.15) + 0.40",
        "- d_n = (d_n/d_a) * d_a",
        "- ln(nl) = 0.5*(d_a - 1.75) + 3.2",
        "- nl = exp(0.5*(d_a - 1.75) + 3.2)",
        "",
        "Notes:",
        "- D is the full fiber diameter (axon + myelin), in um.",
        "- node_length = 1 um and paranodal_length_1 (MYSA) = 3 um are fixed in ASCENT.",
        "- ASCENT SMALL_MRG_INTERPOLATION is officially valid only for 1.011-16.0 um.",
        "- In this table we keep only diameters from 1.5 to 4.0 um, so all rows are within the supported range.",
        "- ASCENT small-fiber model documentation states that gnabar is reduced to 2.333 S/cm2 and gkbar is increased to 0.116 S/cm2.",
        "- gnapbar, gl, ena, ek, el, v_init are kept here from the original MRG description unless otherwise specified in ASCENT docs.",
    ]
    (out_dir / "README_ascent_small_mrg_table.txt").write_text("\n".join(readme_lines) + "\n", encoding="utf-8")

    print(f"Saved CSV: {csv_path}")
    print(f"Saved README: {out_dir / 'README_ascent_small_mrg_table.txt'}")
    print(df.to_string(index=False))


main()
