import s3fs
import time
import uuid
import os
import io
import psutil
import threading
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from collections import defaultdict

@dataclass
class Metric:
    operation: str
    size_bytes: int
    total_duration: float
    rows: int = 0
    parallel: bool = False
    workers: int = 1
    ttfb: float = 0  # Time to first byte
    serialization_time: float = 0  # Time to serialize/deserialize
    transfer_time: float = 0  # Actual network transfer time
    retries: int = 0
    error: Optional[str] = None
    compression_ratio: float = 1.0  # Original size / compressed size
    
    @property
    def throughput_mbps(self): 
        return (self.size_bytes / 1024 / 1024) / self.transfer_time if self.transfer_time > 0 else 0
    
    @property
    def iops(self):
        return 1 / self.total_duration if self.total_duration > 0 else 0

@dataclass
class ResourceSnapshot:
    timestamp: float
    cpu_percent: float
    memory_percent: float
    memory_mb: float

class ResourceMonitor:
    def __init__(self, interval: float = 0.1):
        self.interval = interval
        self.snapshots: List[ResourceSnapshot] = []
        self._stop = False
        self._thread: Optional[threading.Thread] = None
        self.process = psutil.Process()
    
    def start(self):
        self._stop = False
        self.snapshots = []
        self._thread = threading.Thread(target=self._monitor)
        self._thread.start()
    
    def stop(self) -> Dict:
        self._stop = True
        if self._thread:
            self._thread.join()
        if not self.snapshots:
            return {"cpu_avg": 0, "cpu_max": 0, "mem_avg_mb": 0, "mem_max_mb": 0}
        return {
            "cpu_avg": np.mean([s.cpu_percent for s in self.snapshots]),
            "cpu_max": max(s.cpu_percent for s in self.snapshots),
            "mem_avg_mb": np.mean([s.memory_mb for s in self.snapshots]),
            "mem_max_mb": max(s.memory_mb for s in self.snapshots),
        }
    
    def _monitor(self):
        while not self._stop:
            try:
                mem = self.process.memory_info()
                self.snapshots.append(ResourceSnapshot(
                    timestamp=time.perf_counter(),
                    cpu_percent=self.process.cpu_percent(),
                    memory_percent=self.process.memory_percent(),
                    memory_mb=mem.rss / 1024 / 1024
                ))
            except: pass
            time.sleep(self.interval)

class BenchmarkReport:
    def __init__(self):
        self.metrics: List[Metric] = []
        self.resource_stats: Dict[str, Dict] = {}
        self.concurrency_results: Dict[int, Dict] = {}
    
    def add(self, m: Metric): 
        self.metrics.append(m)
    
    def add_resource_stats(self, phase: str, stats: Dict):
        self.resource_stats[phase] = stats
    
    def add_concurrency_result(self, workers: int, stats: Dict):
        self.concurrency_results[workers] = stats
    
    def _calc_percentiles(self, values: List[float]) -> Dict:
        if not values:
            return {"p50": 0, "p95": 0, "p99": 0, "min": 0, "max": 0, "std": 0}
        return {
            "p50": np.percentile(values, 50),
            "p95": np.percentile(values, 95),
            "p99": np.percentile(values, 99),
            "min": min(values),
            "max": max(values),
            "std": np.std(values)
        }
    
    def generate(self) -> str:
        lines = ["\n" + "="*80, "S3 PARQUET BENCHMARK REPORT", "="*80]
        
        # Summary by operation type
        for op, parallel in [("write", False), ("read", False), ("write", True), ("read", True)]:
            subset = [m for m in self.metrics if m.operation == op and m.parallel == parallel and m.error is None]
            if not subset: continue
            
            label = f"{'PARALLEL ' if parallel else 'SEQUENTIAL '}{op.upper()}S"
            lines += [f"\n{label}", "-"*80]
            
            # Individual operations
            lines.append(f"  {'Rows':>10} | {'Size MB':>8} | {'Total':>7} | {'Serial':>7} | {'Transfer':>7} | {'TTFB':>6} | {'MB/s':>8} | {'IOPS':>6}")
            lines.append(f"  {'-'*10} | {'-'*8} | {'-'*7} | {'-'*7} | {'-'*7} | {'-'*6} | {'-'*8} | {'-'*6}")
            
            for m in subset:
                lines.append(
                    f"  {m.rows:>10,} | {m.size_bytes/1024/1024:>8.2f} | "
                    f"{m.total_duration:>6.3f}s | {m.serialization_time:>6.3f}s | "
                    f"{m.transfer_time:>6.3f}s | {m.ttfb:>5.3f}s | "
                    f"{m.throughput_mbps:>8.2f} | {m.iops:>6.2f}"
                )
            
            # Latency percentiles
            latencies = [m.total_duration for m in subset]
            ttfbs = [m.ttfb for m in subset]
            percs = self._calc_percentiles(latencies)
            ttfb_percs = self._calc_percentiles(ttfbs)
            
            lines += [
                f"\n  Latency Percentiles (seconds):",
                f"    Total:  P50={percs['p50']:.3f} | P95={percs['p95']:.3f} | P99={percs['p99']:.3f} | StdDev={percs['std']:.3f}",
                f"    TTFB:   P50={ttfb_percs['p50']:.3f} | P95={ttfb_percs['p95']:.3f} | P99={ttfb_percs['p99']:.3f}"
            ]
            
            # Aggregates
            total_bytes = sum(m.size_bytes for m in subset)
            total_time = sum(m.total_duration for m in subset)
            total_transfer = sum(m.transfer_time for m in subset)
            total_retries = sum(m.retries for m in subset)
            avg_compression = np.mean([m.compression_ratio for m in subset if m.compression_ratio > 0])
            
            lines += [
                f"\n  Aggregates:",
                f"    Total Data: {total_bytes/1024/1024:.2f} MB | Total Time: {total_time:.3f}s",
                f"    Avg Throughput: {(total_bytes/1024/1024)/total_transfer:.2f} MB/s" if total_transfer > 0 else "",
                f"    Total IOPS: {len(subset)/total_time:.2f}" if total_time > 0 else "",
                f"    Retries: {total_retries}",
                f"    Avg Compression Ratio: {avg_compression:.2f}x" if op == "write" else ""
            ]
        
        # Concurrency scaling results
        if self.concurrency_results:
            lines += [f"\n{'CONCURRENCY SCALING':}", "-"*80]
            lines.append(f"  {'Workers':>8} | {'Write MB/s':>12} | {'Read MB/s':>12} | {'Write IOPS':>12} | {'Read IOPS':>12}")
            lines.append(f"  {'-'*8} | {'-'*12} | {'-'*12} | {'-'*12} | {'-'*12}")
            for workers in sorted(self.concurrency_results.keys()):
                r = self.concurrency_results[workers]
                lines.append(
                    f"  {workers:>8} | {r.get('write_throughput', 0):>12.2f} | "
                    f"{r.get('read_throughput', 0):>12.2f} | {r.get('write_iops', 0):>12.2f} | "
                    f"{r.get('read_iops', 0):>12.2f}"
                )
        
        # Resource usage
        if self.resource_stats:
            lines += [f"\n{'RESOURCE USAGE':}", "-"*80]
            lines.append(f"  {'Phase':<30} | {'CPU Avg %':>10} | {'CPU Max %':>10} | {'Mem Avg MB':>12} | {'Mem Max MB':>12}")
            lines.append(f"  {'-'*30} | {'-'*10} | {'-'*10} | {'-'*12} | {'-'*12}")
            for phase, stats in self.resource_stats.items():
                lines.append(
                    f"  {phase:<30} | {stats['cpu_avg']:>10.1f} | {stats['cpu_max']:>10.1f} | "
                    f"{stats['mem_avg_mb']:>12.1f} | {stats['mem_max_mb']:>12.1f}"
                )
        
        # Errors
        errors = [m for m in self.metrics if m.error]
        if errors:
            lines += [f"\n{'ERRORS':}", "-"*80]
            for m in errors:
                lines.append(f"  {m.operation} ({m.rows} rows): {m.error}")
        
        lines += ["\n" + "="*80]
        return "\n".join(filter(None, lines))


class S3ParquetBenchmark:
    SIZES = [(1_000, "1K"), (10_000, "10K"), (100_000, "100K"), (500_000, "500K"), (1_000_000, "1M")]
    CONCURRENCY_LEVELS = [1, 2, 4, 8, 16]
    MAX_RETRIES = 3
    
    def __init__(self, access_key: str, secret_key: str, endpoint_url: str, bucket: str):
        self.fs = s3fs.S3FileSystem(key=access_key, secret=secret_key, endpoint_url=endpoint_url)
        self.bucket = bucket
        self.temp_dir = f"{bucket}/benchmark-{uuid.uuid4().hex[:8]}"
        self.report = BenchmarkReport()
        self.resource_monitor = ResourceMonitor()
        self.files: List[str] = []
        self.file_data: Dict[str, pd.DataFrame] = {}  # For cold vs warm comparison

    def _generate_dataframe(self, rows: int) -> pd.DataFrame:
        return pd.DataFrame({
            'id': np.arange(rows),
            'timestamp': pd.date_range('2020-01-01', periods=rows, freq='s'),
            'value_a': np.random.randn(rows),
            'value_b': np.random.randn(rows),
            'category': np.random.choice(['A', 'B', 'C', 'D'], rows),
            'text': [f'record_{i}' for i in range(rows)]
        })
    
    def _write_parquet(self, path: str, df: pd.DataFrame, track_metrics: bool = True) -> Metric:
        retries = 0
        error = None
        
        # Measure serialization time separately
        serial_start = time.perf_counter()
        buffer = io.BytesIO()
        df.to_parquet(buffer, engine='pyarrow', compression='snappy')
        serialization_time = time.perf_counter() - serial_start
        
        parquet_bytes = buffer.getvalue()
        compressed_size = len(parquet_bytes)
        original_size = df.memory_usage(deep=True).sum()
        compression_ratio = original_size / compressed_size if compressed_size > 0 else 1
        
        # Measure transfer time
        transfer_start = time.perf_counter()
        ttfb = 0
        
        for attempt in range(self.MAX_RETRIES):
            try:
                buffer.seek(0)
                with self.fs.open(path, 'wb') as f:
                    ttfb = time.perf_counter() - transfer_start
                    f.write(parquet_bytes)
                break
            except Exception as e:
                retries += 1
                if attempt == self.MAX_RETRIES - 1:
                    error = str(e)
        
        transfer_time = time.perf_counter() - transfer_start
        total_duration = serialization_time + transfer_time
        
        return Metric(
            operation="write",
            size_bytes=compressed_size,
            total_duration=total_duration,
            rows=len(df),
            ttfb=ttfb,
            serialization_time=serialization_time,
            transfer_time=transfer_time,
            retries=retries,
            error=error,
            compression_ratio=compression_ratio
        )
    
    def _read_parquet(self, path: str, columns: List[str] = None, filters=None) -> Metric:
        retries = 0
        error = None
        size = self.fs.size(path)
        
        transfer_start = time.perf_counter()
        ttfb = 0
        buffer = io.BytesIO()
        
        for attempt in range(self.MAX_RETRIES):
            try:
                with self.fs.open(path, 'rb') as f:
                    ttfb = time.perf_counter() - transfer_start
                    buffer.write(f.read())
                break
            except Exception as e:
                retries += 1
                if attempt == self.MAX_RETRIES - 1:
                    error = str(e)
        
        transfer_time = time.perf_counter() - transfer_start
        
        # Measure deserialization separately
        buffer.seek(0)
        deserial_start = time.perf_counter()
        df = pd.read_parquet(buffer, engine='pyarrow', columns=columns, filters=filters)
        deserialization_time = time.perf_counter() - deserial_start
        
        total_duration = transfer_time + deserialization_time
        
        return Metric(
            operation="read",
            size_bytes=size,
            total_duration=total_duration,
            rows=len(df),
            ttfb=ttfb,
            serialization_time=deserialization_time,
            transfer_time=transfer_time,
            retries=retries,
            error=error
        )

    def setup(self):
        print(f"Creating temp directory: {self.temp_dir}")
        self.fs.mkdirs(self.temp_dir, exist_ok=True)

    def run_sequential_writes(self):
        print("\nRunning sequential parquet writes...")
        self.resource_monitor.start()
        
        for rows, label in self.SIZES:
            path = f"{self.temp_dir}/seq_{label}.parquet"
            df = self._generate_dataframe(rows)
            metric = self._write_parquet(path, df)
            self.files.append(path)
            self.report.add(metric)
            print(f"  Wrote {label} rows: {metric.total_duration:.3f}s "
                  f"(serial: {metric.serialization_time:.3f}s, transfer: {metric.transfer_time:.3f}s, "
                  f"{metric.throughput_mbps:.2f} MB/s, compression: {metric.compression_ratio:.1f}x)")
        
        stats = self.resource_monitor.stop()
        self.report.add_resource_stats("Sequential Writes", stats)

    def run_sequential_reads(self):
        print("\nRunning sequential parquet reads (cold)...")
        self.resource_monitor.start()
        
        # Clear any caching
        self.fs.invalidate_cache()
        
        for path in [f for f in self.files if "/seq_" in f]:
            metric = self._read_parquet(path)
            metric.parallel = False
            self.report.add(metric)
            print(f"  Read {metric.rows:,} rows: {metric.total_duration:.3f}s "
                  f"(TTFB: {metric.ttfb:.3f}s, transfer: {metric.transfer_time:.3f}s, "
                  f"deserial: {metric.serialization_time:.3f}s, {metric.throughput_mbps:.2f} MB/s)")
        
        stats = self.resource_monitor.stop()
        self.report.add_resource_stats("Sequential Reads (Cold)", stats)
        
        # Warm reads
        print("\nRunning sequential parquet reads (warm)...")
        self.resource_monitor.start()
        
        for path in [f for f in self.files if "/seq_" in f]:
            metric = self._read_parquet(path)
            metric.operation = "read_warm"
            print(f"  Warm read {metric.rows:,} rows: {metric.total_duration:.3f}s ({metric.throughput_mbps:.2f} MB/s)")
        
        stats = self.resource_monitor.stop()
        self.report.add_resource_stats("Sequential Reads (Warm)", stats)

    def run_column_projection_reads(self):
        print("\nRunning column projection reads (subset of columns)...")
        
        for path in [f for f in self.files if "/seq_" in f][-2:]:  # Just larger files
            # Full read
            metric_full = self._read_parquet(path)
            # Projected read (2 columns only)
            metric_proj = self._read_parquet(path, columns=['id', 'value_a'])
            
            print(f"  Full read: {metric_full.total_duration:.3f}s | "
                  f"Projected (2 cols): {metric_proj.total_duration:.3f}s | "
                  f"Speedup: {metric_full.total_duration/metric_proj.total_duration:.2f}x")

    def run_parallel_operations(self, workers: int, rows_per_file: int = 100_000, num_files: int = 8):
        """Run parallel writes and reads with specified worker count."""
        print(f"\n  Testing with {workers} workers...")
        
        paths = [f"{self.temp_dir}/conc_{workers}w_{i}.parquet" for i in range(num_files)]
        
        # Parallel writes
        def write_task(path):
            df = self._generate_dataframe(rows_per_file)
            return self._write_parquet(path, df)
        
        self.resource_monitor.start()
        write_start = time.perf_counter()
        
        with ThreadPoolExecutor(max_workers=workers) as ex:
            write_metrics = list(ex.map(write_task, paths))
        
        write_total_time = time.perf_counter() - write_start
        write_stats = self.resource_monitor.stop()
        
        for m in write_metrics:
            m.parallel = True
            m.workers = workers
            self.report.add(m)
        self.files.extend(paths)
        
        total_write_bytes = sum(m.size_bytes for m in write_metrics)
        write_throughput = (total_write_bytes / 1024 / 1024) / write_total_time
        write_iops = len(write_metrics) / write_total_time
        
        # Parallel reads
        self.fs.invalidate_cache()
        
        def read_task(path):
            return self._read_parquet(path)
        
        self.resource_monitor.start()
        read_start = time.perf_counter()
        
        with ThreadPoolExecutor(max_workers=workers) as ex:
            read_metrics = list(ex.map(read_task, paths))
        
        read_total_time = time.perf_counter() - read_start
        read_stats = self.resource_monitor.stop()
        
        for m in read_metrics:
            m.parallel = True
            m.workers = workers
            self.report.add(m)
        
        total_read_bytes = sum(m.size_bytes for m in read_metrics)
        read_throughput = (total_read_bytes / 1024 / 1024) / read_total_time
        read_iops = len(read_metrics) / read_total_time
        
        self.report.add_resource_stats(f"Parallel Writes ({workers} workers)", write_stats)
        self.report.add_resource_stats(f"Parallel Reads ({workers} workers)", read_stats)
        
        return {
            "write_throughput": write_throughput,
            "read_throughput": read_throughput,
            "write_iops": write_iops,
            "read_iops": read_iops
        }

    def run_concurrency_scaling(self):
        print("\nRunning concurrency scaling tests...")
        
        for workers in self.CONCURRENCY_LEVELS:
            results = self.run_parallel_operations(workers)
            self.report.add_concurrency_result(workers, results)
            print(f"    Write: {results['write_throughput']:.2f} MB/s, {results['write_iops']:.2f} IOPS | "
                  f"Read: {results['read_throughput']:.2f} MB/s, {results['read_iops']:.2f} IOPS")

    def cleanup(self):
        print(f"\nCleaning up {self.temp_dir}...")
        try:
            self.fs.rm(self.temp_dir, recursive=True)
            print("  Cleanup complete")
        except Exception as e:
            print(f"  Cleanup error: {e}")

    def run(self):
        try:
            self.setup()
            self.run_sequential_writes()
            self.run_sequential_reads()
            self.run_column_projection_reads()
            self.run_concurrency_scaling()
        finally:
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
