# Data

The dataset is **not included** in this repository (about 1.07 GB).

Place the preprocessed CSV here so that the notebooks in `notebooks/` can load it via `../data/car_sales_preprocessed_2.csv`:

```
data/
└── car_sales_preprocessed_2.csv
```

Expected shape: 761,973 rows x 236 columns (as printed in the notebook outputs):

- `quarter` (1-4), `Brand_LE`, `Model_LE`, `Body Type_LE` (label-encoded integer IDs)
- `Year_2022`, `Year_2023` (boolean dummies; rows where both are False belong to a third, dropped year category)
- `Gov_*` (26 one-hot governorate columns, Arabic names)
- `TU_*` (one-hot traffic-unit columns, Arabic names)

Source: [TODO: confirm source]. License: [TODO: confirm license].
