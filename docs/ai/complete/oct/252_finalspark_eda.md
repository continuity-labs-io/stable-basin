Write src/data/ephys/finalspark_protocol.py. Read only
data/ephys/finalspark/fs369data/fs369_package.hdf5. Do not touch the raw file.
Print a short report. Do not plot. Do not write new data files.

TASK 1 — row semantics (everything else depends on this)
The README says the stimulations table has "one row per stimulation trigger,"
but nb_stim_pulse is 2 or 10, so a row is either one pulse or one whole train.
Decide empirically: sort by electrode then time_of_stim, take gaps within each
electrode, and test whether rows cluster into groups whose size matches that
row's nb_stim_pulse. Report:
  - median group size for nb_stim_pulse==2 rows and for ==10 rows
  - the fraction of groups whose size equals nb_stim_pulse
Verdict line: "ROWS ARE PULSES" if those match, else "ROWS ARE TRIGGERS".

TASK 2 — resolve pulse_train_period units
pulse_train_period takes values 1.0 and 10000.0 with no stated unit.
If TASK 1 says rows are pulses: for each train, compute the observed median
gap between consecutive rows in seconds, group by the train's
pulse_train_period value, and print observed_seconds / pulse_train_period for
each. Report the implied unit (ratio near 1e-6 means microseconds, 1e-3 means
milliseconds, 1 means seconds).
If rows are triggers: the within-train interval is not observable here. Say so
explicitly, print the observed gap distribution between consecutive triggers on
the same electrode grouped by pulse_train_period, and stop. Do not guess.

TASK 3 — is the dose axis confounded with age? (most important)
The culture was recorded 2025-02-19 and died 2025-03-12. Stimulation ran
2025-02-20 to 2025-03-10. If the operator raised amplitude as the culture
declined, then a1 is confounded with age and the dose axis is unusable.
  - Bin time_of_stim by day since first stimulation.
  - For each day print: n_stims, n_electrodes, median a1, IQR of a1, the set of
    d1 values, the set of nb_stim_pulse values.
  - Print Spearman correlation of a1 against days-since-first-stim, with p.
Verdict line: "DOSE CONFOUNDED WITH AGE" if |rho| >= 0.3, else "DOSE AXIS
USABLE".

TASK 4 — parameter sets and coverage
Define a parameter set as the tuple (a1, a2, d1, d2, nb_stim_pulse,
pulse_train_period, stim_polarity, stim_shape).
  - How many distinct sets exist.
  - Top 15 by count, with n_stims, n_electrodes, first and last timestamp.
  - For each set, number of distinct days it appears on.
Verdict line: name the one set with the widest coverage across days and
electrodes. That is the candidate for a repeatability test.

TASK 5 — unstimulated epochs
  - Pre-stimulation baseline: duration between first event and first stim.
  - All gaps between consecutive stimulations, pooled across electrodes, longer
    than 6 hours: print start, end, duration, sorted descending, top 10.
Verdict line: total hours of recording with no stimulation anywhere.

OUTPUT
End with a five-line summary, one line per task, each starting with its verdict
word. Nothing else.
