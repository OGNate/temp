import s3fs
import time
import uuid
import os
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field

@dataclass
class Metric:
    operation: str
    size_bytes: int
    duration_sec: float
    rows: int = 0
    parallel: bool = False
    @property
    def speed_mbps(self): return (self.size_bytes / 1024 / 1024) / self.duration_sec if self.duration_sec > 0 else 0

@dataclass
class BenchmarkReport:
    metrics: list = field(default_factory=list)
    
    def add(self, m): self.metrics.append(m)
    
    def generate(self):
        lines = ["\n" + "="*70, "S3 PARQUET BENCHMARK REPORT", "="*70 + "\n"]
        for op, parallel in [("write", False), ("read", False), ("write", True), ("read", True)]:
            subset = [m for m in self.metrics if m.operation == op and m.parallel == parallel]
            if not subset: continue
            label = f"{'Parallel ' if parallel else 'Sequential '}{op.upper()}S"
            lines += [f"\n{label}", "-"*60]
            total_bytes, total_time, total_rows = 0, 0, 0
            for m in subset:
                lines.append(f"  {m.rows:>10,} rows | {m.size_bytes/1024/1024:>8.2f} MB | {m.duration_sec:>6.3f}s | {m.speed_mbps:>8.2f} MB/s")
                total_bytes += m.size_bytes; total_time += m.duration_sec; total_rows += m.rows
            avg_speed = (total_bytes/1024/1024)/total_time if total_time > 0 else 0
            lines += [f"  {total_rows:>10,} rows | {total_bytes/1024/1024:>8.2f} MB | {total_time:>6.3f}s | {avg_speed:>8.2f} MB/s avg"]
        lines += ["\n" + "="*70]
        return "\n".join(lines)

class S3ParquetBenchmark:
    # (rows, label) - approximate sizes after parquet compression
    SIZES = [(1_000, "1K"), (10_000, "10K"), (100_000, "100K"), (500_000, "500K"), (1_000_000, "1M")]
    
    def __init__(self, access_key, secret_key, endpoint_url, bucket):
        self.fs = s3fs.S3FileSystem(key=access_key, secret=secret_key, endpoint_url=endpoint_url)
        self.bucket = bucket
        self.temp_dir = f"{bucket}/benchmark-{uuid.uuid4().hex[:8]}"
        self.report = BenchmarkReport()
        self.files = []

    def _generate_dataframe(self, rows):
        return pd.DataFrame({
            'id': np.arange(rows),
            'timestamp': pd.date_range('2020-01-01', periods=rows, freq='s'),
            'value_a': np.random.randn(rows),
            'value_b': np.random.randn(rows),
            'category': np.random.choice(['A', 'B', 'C', 'D'], rows),
            'text': [f'record_{i}' for i in range(rows)]
        })
    
    def _write_parquet(self, path, df):
        start = time.perf_counter()
        with self.fs.open(path, 'wb') as f:
            df.to_parquet(f, engine='pyarrow', compression='snappy')
        duration = time.perf_counter() - start
        size = self.fs.size(path)
        return duration, size
    
    def _read_parquet(self, path):
        size = self.fs.size(path)
        start = time.perf_counter()
        with self.fs.open(path, 'rb') as f:
            df = pd.read_parquet(f, engine='pyarrow')
        duration = time.perf_counter() - start
        return duration, size, len(df)

    def setup(self):
        print(f"Creating temp directory: {self.temp_dir}")
        self.fs.mkdirs(self.temp_dir, exist_ok=True)

    def run_sequential_writes(self):
        print("\nRunning sequential parquet writes...")
        for rows, label in self.SIZES:
            path = f"{self.temp_dir}/seq_{label}.parquet"
            df = self._generate_dataframe(rows)
            duration, size = self._write_parquet(path, df)
            self.files.append(path)
            self.report.add(Metric("write", size, duration, rows))
            print(f"  Wrote {label} rows ({size/1024/1024:.2f}MB): {duration:.3f}s ({size/1024/1024/duration:.2f} MB/s)")

    def run_sequential_reads(self):
        print("\nRunning sequential parquet reads...")
        for path in self.files:
            duration, size, rows = self._read_parquet(path)
            self.report.add(Metric("read", size, duration, rows))
            print(f"  Read {rows:,} rows ({size/1024/1024:.2f}MB): {duration:.3f}s ({size/1024/1024/duration:.2f} MB/s)")

    def run_parallel_writes(self, workers=4):
        print(f"\nRunning parallel parquet writes ({workers} workers)...")
        tasks = [(f"{self.temp_dir}/par_{i}_{label}.parquet", rows)
                 for i in range(workers) for rows, label in self.SIZES[:3]]
        
        def write_task(args):
            path, rows = args
            df = self._generate_dataframe(rows)
            duration, size = self._write_parquet(path, df)
            return path, size, duration, rows
        
        with ThreadPoolExecutor(max_workers=workers) as ex:
            for path, size, duration, rows in [f.result() for f in as_completed([ex.submit(write_task, t) for t in tasks])]:
                self.files.append(path)
                self.report.add(Metric("write", size, duration, rows, parallel=True))
        print(f"  Completed {len(tasks)} parallel writes")

    def run_parallel_reads(self, workers=4):
        print(f"\nRunning parallel parquet reads ({workers} workers)...")
        parallel_files = [f for f in self.files if "/par_" in f]
        
        def read_task(path):
            return self._read_parquet(path)
        
        with ThreadPoolExecutor(max_workers=workers) as ex:
            for duration, size, rows in [f.result() for f in as_completed([ex.submit(read_task, p) for p in parallel_files])]:
                self.report.add(Metric("read", size, duration, rows, parallel=True))
        print(f"  Completed {len(parallel_files)} parallel reads")

    def cleanup(self):
        print(f"\nCleaning up {self.temp_dir}...")
        self.fs.rm(self.temp_dir, recursive=True)
        print("  Cleanup complete")

    def run(self):
        self.setup()
        self.run_sequential_writes()
        self.run_sequential_reads()
        self.run_parallel_writes()
        self.run_parallel_reads()
        self.cleanup()
        report = self.report.generate()
        print(report)
        return report

if __name__ == "__main__":
    ACCESS_KEY = os.environ.get("S3_ACCESS_KEY", "minioadmin")
    SECRET_KEY = os.environ.get("S3_SECRET_KEY", "minioadmin")
    ENDPOINT_URL = os.environ.get("S3_ENDPOINT_URL", "http://localhost:9000")
    BUCKET = os.environ.get("S3_BUCKET", "test-bucket")
    
    benchmark = S3ParquetBenchmark(ACCESS_KEY, SECRET_KEY, ENDPOINT_URL, BUCKET)
    benchmark.run()
