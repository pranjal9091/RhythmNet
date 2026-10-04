"""Exploratory Data Analysis (EDA) and dataset auditing package."""

from ecg_arrhythmia.eda.distribution import (
    analyze_class_distributions,
    plot_class_distribution,
)
from ecg_arrhythmia.eda.patient_audit import (
    analyze_minority_patient_coverage,
    analyze_patient_contributions,
    plot_record_class_heatmap,
)
from ecg_arrhythmia.eda.rr_analysis import (
    compute_rr_intervals,
    compute_rr_statistics,
    plot_rr_distribution_by_class,
    plot_rr_distribution_by_partition,
)
from ecg_arrhythmia.eda.statistics import (
    compare_partition_distributions,
    plot_partition_comparison,
    run_data_quality_audit,
)
from ecg_arrhythmia.eda.waveform_analysis import (
    compute_morphology_statistics,
    plot_waveform_overlays,
    plot_waveforms_by_class,
)

__all__ = [
    "analyze_class_distributions",
    "plot_class_distribution",
    "analyze_patient_contributions",
    "analyze_minority_patient_coverage",
    "plot_record_class_heatmap",
    "plot_waveforms_by_class",
    "plot_waveform_overlays",
    "compute_morphology_statistics",
    "compute_rr_intervals",
    "compute_rr_statistics",
    "plot_rr_distribution_by_class",
    "plot_rr_distribution_by_partition",
    "run_data_quality_audit",
    "compare_partition_distributions",
    "plot_partition_comparison",
]
