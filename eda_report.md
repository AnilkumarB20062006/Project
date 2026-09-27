## Task 1 — Profiling

Shape: (891, 15)

```
```

**df.describe():**

|        |   survived |     pclass | sex   |      age |      sibsp |      parch |     fare | embarked   | class   | who   |   adult_male | deck   | embark_town   | alive   |   alone |
|:-------|-----------:|-----------:|:------|---------:|-----------:|-----------:|---------:|:-----------|:--------|:------|-------------:|:-------|:--------------|:--------|--------:|
| count  | 891        | 891        | 891   | 714      | 891        | 891        | 891      | 889        | 891     | 891   |          891 | 203    | 889           | 891     |     891 |
| unique | nan        | nan        | 2     | nan      | nan        | nan        | nan      | 3          | 3       | 3     |            2 | 7      | 3             | 2       |       2 |
| top    | nan        | nan        | male  | nan      | nan        | nan        | nan      | S          | Third   | man   |            1 | C      | Southampton   | no      |       1 |
| freq   | nan        | nan        | 577   | nan      | nan        | nan        | nan      | 644        | 491     | 537   |          537 | 59     | 644           | 549     |     537 |
| mean   |   0.383838 |   2.30864  | nan   |  29.6991 |   0.523008 |   0.381594 |  32.2042 | nan        | nan     | nan   |          nan | nan    | nan           | nan     |     nan |
| std    |   0.486592 |   0.836071 | nan   |  14.5265 |   1.10274  |   0.806057 |  49.6934 | nan        | nan     | nan   |          nan | nan    | nan           | nan     |     nan |
| min    |   0        |   1        | nan   |   0.42   |   0        |   0        |   0      | nan        | nan     | nan   |          nan | nan    | nan           | nan     |     nan |
| 25%    |   0        |   2        | nan   |  20.125  |   0        |   0        |   7.9104 | nan        | nan     | nan   |          nan | nan    | nan           | nan     |     nan |
| 50%    |   0        |   3        | nan   |  28      |   0        |   0        |  14.4542 | nan        | nan     | nan   |          nan | nan    | nan           | nan     |     nan |
| 75%    |   1        |   3        | nan   |  38      |   1        |   0        |  31      | nan        | nan     | nan   |          nan | nan    | nan           | nan     |     nan |
| max    |   1        |   3        | nan   |  80      |   8        |   6        | 512.329  | nan        | nan     | nan   |          nan | nan    | nan           | nan     |     nan |

**Missing values (columns with any):**

|             |   missing_count |   missing_pct |
|:------------|----------------:|--------------:|
| deck        |             688 |         77.22 |
| age         |             177 |         19.87 |
| embarked    |               2 |          0.22 |
| embark_town |               2 |          0.22 |

## Task 2 — Missing-value handling (threshold rule)


| Column | % missing | Rule bucket | Action | Justification |
|---|---|---|---|---|
| deck | 77.22% | > 30% | Drop column | Too sparse to impute reliably; not needed by any required downstream task (correlation matrix / models don't require it). |
| age | 19.87% | 5%-30% | Impute (median) | Enough missingness that dropping would waste ~20% of rows; median is robust to age's right skew. |
| embarked | 0.22% | < 5% | Drop rows (2) | Missingness is negligible; dropping costs almost nothing and avoids guessing a categorical value. |
| embark_town | 0.22% | < 5% | Drop rows (same 2) | Same underlying rows as `embarked`. |

Cleaned shape: (889, 14) (saved to titanic_cleaned.csv)


## Task 3 — Univariate analysis (age, fare)

- **age**: IQR outlier bounds = [2.50, 54.50] -> **65 outliers** (chart: `charts/univariate_age.png`)
- **fare**: IQR outlier bounds = [-26.76, 65.66] -> **114 outliers** (chart: `charts/univariate_fare.png`)

**fare** — mean=32.10, median=14.45, mode=8.05.
Since mean (32.10) > median (14.45) > mode (8.05), the ordering mean > median > mode indicates fare is **right-skewed** (a long tail of expensive first-class fares pulls the mean above the median).

## Task 4 — Bivariate analysis

**Survival rate by sex:** male=0.189, female=0.740
**Survival rate by pclass:** 1=0.626, 2=0.473, 3=0.242

**Survival rate by sex & pclass:**

| sex    |   pclass |   survival_rate |
|:-------|---------:|----------------:|
| female |        1 |        0.967391 |
| female |        2 |        0.921053 |
| female |        3 |        0.5      |
| male   |        1 |        0.368852 |
| male   |        2 |        0.157407 |
| male   |        3 |        0.135447 |

**Correlation heatmap:** `charts/correlation_heatmap.png`

**Two strongest correlations (by |r|):**
- `pclass` vs `fare`: r = -0.548
- `sibsp` vs `parch`: r = 0.415

Interpretation: `pclass` and `fare` show the strongest relationship (r=-0.55) — pclass is numbered 1 (top) to 3 (bottom), so the negative sign means higher-numbered (cheaper, lower) classes go with lower fares, exactly as expected from how ticket pricing maps to cabin class. `sibsp` and `parch` (r=0.41) are positively correlated because both count family members aboard — passengers traveling with a spouse/sibling were also more likely to be traveling with a parent/child, i.e. larger family groups moved together on both counts.

## Task 5 — Multivariate data story

**Chart 1** (`story_1_class_sex.png`): Women survived at a far higher rate than men in every class, and the gap barely narrows even in 3rd class — sex was the single biggest factor in who lived, consistent with a 'women and children first' evacuation norm.
**Chart 2** (`story_2_age_survival.png`): Median age is similar across both groups, but survivors skew slightly younger with a visible cluster of young children — supporting the idea that the youngest passengers were prioritized, even though age alone is a weak predictor on its own.
**Chart 3** (`story_3_fare_age_scatter.png`): Survivors (orange) concentrate at higher fares regardless of age, while non-survivors dominate the low-fare band — fare (a proxy for class and deck location) separates outcomes more cleanly than age does.
**Chart 4** (`story_4_embarked.png`): Passengers who boarded at Cherbourg ('C') had a notably better survival ratio than those from Southampton ('S') — largely because Cherbourg had a higher share of 1st-class passengers, echoing the class effect seen above rather than a location effect on its own.
**Chart 5** (`story_5_pairplot.png`): Viewed together, class and fare separate the two survival groups more sharply than any single variable — reinforcing that 'who could afford a better class/cabin' is the dominant multivariate story behind survival, with sex (shown in Chart 1) layered on top of it.

## Task 6 — Standardization sanity check (EDA only)

**Before standardization:**
|      |    age |   fare |
|:-----|-------:|-------:|
| mean | 29.315 | 32.097 |
| std  | 12.985 | 49.698 |

**After standardization (z-score):**
|      |   age |   fare |
|:-----|------:|-------:|
| mean |     0 |      0 |
| std  |     1 |      1 |

After transforming with z = (x - mean) / std, both `age` and `fare` have mean ≈ 0 and standard deviation ≈ 1, confirming the manual z-score computation is correct. This is a pure EDA-stage sanity check — the actual modeling pipeline in `02_modeling.py` fits its own `StandardScaler` on the training split only (see Task 8), so no information from this full-dataset check leaks into the model.