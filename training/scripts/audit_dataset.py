import hashlib
from pathlib import Path
from collections import defaultdict

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

MANIFEST = (
    PROJECT_ROOT
    / "ml/data/manifests/unified_manifest.csv"
)

REPORT_DIR = (
    PROJECT_ROOT
    / "ml/artifacts/reports/dataset_audit"
)

REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


def file_hash(path, chunk_size=1024 * 1024):

    h = hashlib.sha256()

    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)

            if not chunk:
                break

            h.update(chunk)

    return h.hexdigest()


def main():

    print("=" * 70)
    print("TCS DATASET AUDIT")
    print("=" * 70)

    df = pd.read_csv(MANIFEST)

    print()
    print("Total images:", len(df))

    print()
    print("Splits:")
    print(df["split"].value_counts())

    print()
    print("Labels:")
    print(df["binary_label"].value_counts())

    records = []

    for i, row in df.iterrows():

        path = Path(str(row["image_path"]))

        if not path.exists():
            records.append({
                "index": i,
                "path": str(path),
                "split": row["split"],
                "hash": None,
                "exists": False,
            })
            continue

        try:
            digest = file_hash(path)

            records.append({
                "index": i,
                "path": str(path),
                "split": row["split"],
                "hash": digest,
                "exists": True,
            })

        except Exception as e:

            print(
                f"Hash error: {path} -> {e}"
            )

    hashes = pd.DataFrame(records)

    hashes.to_csv(
        REPORT_DIR / "file_hashes.csv",
        index=False,
    )

    missing = hashes[
        ~hashes["exists"]
    ]

    print()
    print("Missing files:", len(missing))

    if len(missing):

        missing.to_csv(
            REPORT_DIR / "missing_files.csv",
            index=False,
        )

        print(
            missing[
                ["split", "path"]
            ].to_string(index=False)
        )

    valid = hashes[
        hashes["exists"] &
        hashes["hash"].notna()
    ]

    groups = (
        valid
        .groupby("hash")
        .agg(
            count=("hash", "size"),
            splits=("split", lambda x:
                ",".join(sorted(set(x)))
            ),
            paths=("path", lambda x:
                " || ".join(x)
            ),
        )
        .reset_index()
    )

    duplicate_groups = groups[
        groups["count"] > 1
    ]

    cross_split = duplicate_groups[
        duplicate_groups["splits"].str.contains(",")
    ]

    print()
    print("Duplicate hash groups:", len(
        duplicate_groups
    ))

    print(
        "Cross-split duplicate groups:",
        len(cross_split)
    )

    duplicate_groups.to_csv(
        REPORT_DIR / "duplicate_groups.csv",
        index=False,
    )

    cross_split.to_csv(
        REPORT_DIR / "cross_split_duplicates.csv",
        index=False,
    )

    print()
    print("=" * 70)
    print("MVTec PROTOCOL AUDIT")
    print("=" * 70)

    paths = (
        df["image_path"]
        .astype(str)
        .str.replace("\\", "/", regex=False)
    )

    mvtec = df[
        paths.str.contains(
            "/mvtec/",
            case=False,
            regex=False,
        )
    ].copy()

    mvtec["normalized_path"] = (
        mvtec["image_path"]
        .astype(str)
        .str.replace(
            "\\",
            "/",
            regex=False,
        )
    )

    test_train = mvtec[
        (mvtec["split"] == "test") &
        (
            mvtec["normalized_path"]
            .str.contains(
                "/train/",
                case=False,
                regex=False,
            )
        )
    ]

    print(
        "MVTec test samples originating from "
        "official train directories:",
        len(test_train)
    )

    if len(test_train):

        test_train[
            [
                "image_path",
                "binary_label",
                "split",
            ]
        ].to_csv(
            REPORT_DIR
            / "mvtec_train_inside_test.csv",
            index=False,
        )

        print()
        print(
            test_train[
                [
                    "image_path",
                    "binary_label",
                ]
            ].to_string(index=False)
        )

    print()
    print("=" * 70)
    print("AUDIT COMPLETE")
    print("=" * 70)

    print(
        f"\nReports saved to:\n{REPORT_DIR}"
    )


if __name__ == "__main__":
    main()
