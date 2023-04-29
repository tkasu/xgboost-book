import polars as pl


def clean(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns(
        [
            pl.col("Q2").str.slice(0, 2).str.parse_int(radix=10).alias("age"),
            pl.col("Q4")
            .map_dict(
                {
                    "No formal education past high school": 12,
                    "Some college/university study without earning a bachelor’s degree": 13,
                    "Bachelor’s degree": 16,
                    "Master’s degree": 18,
                    "Professional degree": 19,
                    "Doctoral degree": 20,
                    "I prefer not to answer": None,
                },
                default=None,
                return_dtype=pl.datatypes.UInt8,
            )
            .alias("education"),
            pl.col("Q5")
            .map_dict(
                {
                    "Computer science (software engineering, etc.)": "cs",
                    "Engineering (non-computer focused)": "eng",
                    "Mathematics or statistics": "stat",
                },
                default="other",
                return_dtype=pl.datatypes.Utf8,
            )
            .alias("major"),
            (
                pl.col("Q8")
                .str.replace(r"\+", "")
                .str.split(by="-")
                .arr.last()
                .str.rstrip()
                .str.parse_int(radix=10)
            ).alias("years_exp"),
            (
                pl.col("Q9")
                .str.split(by="-")
                .arr.last()
                .str.replace(r"\+", "")
                .str.replace(r"\,", "")
                .str.replace(
                    "I do not wish to disclose my approximate yearly compensation", "0"
                )
                .str.parse_int(radix=10)
                .fill_null(0)
                .mul(1_000)
            ).alias("compensation"),
            (
                pl.col("Q16_Part_1")
                .str.replace(r"Python", "1")
                .str.parse_int(radix=10)
                .fill_null(0)
            ).alias("python"),
            (
                pl.col("Q16_Part_2")
                .str.replace(r"R+", "1")
                .str.parse_int(radix=10)
                .fill_null(0)
            ).alias("r"),
            (
                pl.col("Q16_Part_3")
                .str.replace(r"SQL", "1")
                .str.parse_int(radix=10)
                .fill_null(0)
            ).alias("sql"),
        ]
    ).select(
        "Q1",
        "Q3",
        "age",
        "education",
        "major",
        "years_exp",
        "compensation",
        "python",
        "r",
        "sql",
    )
