import pyarrow.parquet as pq
import pyarrow as pa
from pathlib import Path

# -----------------------------
# Function: split large Parquet into smaller files by row count
# -----------------------------
def split_parquet_streaming(input_parquet, rows_per_file=500_000, output_dir=None):
    """
    Split a large Parquet file into multiple smaller Parquet files,
    each containing up to 'rows_per_file' rows. Last file may be smaller.
    This function streams the input and never loads the full table into memory.
    """
    input_parquet = Path(input_parquet)
    if output_dir is None:
        output_dir = input_parquet.parent / f"{input_parquet.stem}_split"
    output_dir.mkdir(exist_ok=True)

    pf = pq.ParquetFile(input_parquet)
    buffer = []
    file_idx = 0
    total_rows_written = 0

    for batch in pf.iter_batches(batch_size=rows_per_file):
        table = pa.Table.from_batches([batch])
        buffer.append(table)

        # Check if we have enough rows to write a new file
        buffered_rows = sum(t.num_rows for t in buffer)
        while buffered_rows >= rows_per_file:
            chunk_tables = []
            rows_to_take = rows_per_file
            while rows_to_take > 0 and buffer:
                t = buffer.pop(0)
                if t.num_rows <= rows_to_take:
                    chunk_tables.append(t)
                    rows_to_take -= t.num_rows
                else:
                    # split table
                    chunk_tables.append(t.slice(0, rows_to_take))
                    buffer.insert(0, t.slice(rows_to_take))
                    rows_to_take = 0
            out_file = output_dir / f"{input_parquet.stem}_part{file_idx}.parquet"
            pq.write_table(pa.concat_tables(chunk_tables), out_file)
            print(f"Wrote {rows_per_file} rows to {out_file}")
            file_idx += 1
            total_rows_written += rows_per_file
            buffered_rows -= rows_per_file

    # Write any remaining rows
    if buffer:
        out_file = output_dir / f"{input_parquet.stem}_part{file_idx}.parquet"
        pq.write_table(pa.concat_tables(buffer), out_file)
        remaining_rows = sum(t.num_rows for t in buffer)
        print(f"Wrote remaining {remaining_rows} rows to {out_file}")
        total_rows_written += remaining_rows

    print(f"Done. Total rows written: {total_rows_written}")


# -----------------------------
# Main script
# -----------------------------
if __name__ == "__main__":
    # -------- User settings --------
    original_parquet_file = Path("/home/esamouil/analysis/data_stor/local_commissioning_data/004_dream_cc_2026/Z220126_IBM1_0008.parquet")  # <- change this
    output_folder = Path("/home/esamouil/analysis/data_stor/local_commissioning_data/004_dream_cc_2026/Z220126_IBM1_0008_parts")                 # <- change this
    lines_per_file = 6_000_000                                       # <- number of rows per small file

    # Run the split
    split_parquet_streaming(
        input_parquet=original_parquet_file,
        rows_per_file=lines_per_file,
        output_dir=output_folder
    )