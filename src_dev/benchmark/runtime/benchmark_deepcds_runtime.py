"""DeepCDS inference runtime and memory benchmark.

Times DeepCDS prediction (both strands, exactly as src_dev/inference/predict_with_deepcds.py does it) on GPU and CPU
for every test-set read length and a range of batch sizes, and records peak GPU memory and peak RAM.

Model per read length: the largest model, DeepCDS S+I (6 transformer-encoder blocks, vs 4 for N and S), on the
typical-error reads for 75-300 bp; DeepCDS N on the error-free reads for 700 and 1000 bp (the error sets stop at 300 bp).

Reads: the public test reads (FTP), genomes taken in a fixed random order (seed 42) until enough reads are pooled,
then N_MAX reads sampled (seed 42). Each run uses the first --n_reads_{gpu,cpu} of these.

Each (device, read length, batch size) runs in its own process, so peak RSS and GPU memory belong to that run only:
model load (timed separately), one warm-up batch, then --repeats timed passes over the reads. Timed = encoding,
model forward, CRF decoding and GFF formatting for both strands; FASTA reading, model loading and file writing are
not. Inference is fp32, as in the released tool. Reads > 300 bp use the sliding window with --stride_aa (default 70
codons, as for the test-set predictions in predict_with_DeepCDS.py; the released tool's default is 50).

GPU: each GPU run waits until no other process is on the GPU, and is redone if another process appears on it while
it runs (other_gpu_procs = 0 for every kept row).
GPU memory: peak of the process's total device memory as reported by nvidia-smi (polled every 0.1 s; includes the
CUDA context, i.e. what a user needs free), plus PyTorch's own max_memory_reserved / max_memory_allocated.
RAM: peak resident set size of the process (ru_maxrss).

Outputs: OUT/reads/{set}_{L}bp.fasta, OUT/runtime_benchmark.tsv (one row per run; re-running skips finished rows).
Usage: python benchmark_deepcds_runtime.py [--gpu 0] [--cpu_threads 32] [--devices cuda cpu]
"""
import argparse
import json
import os
import random
import resource
import socket
import statistics
import subprocess
import sys
import threading
import time
import urllib.request
import gzip
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
INFERENCE_DIR = os.path.abspath(f"{HERE}/../../inference")
FTP = "ftp://ftp.healthtech.dtu.dk/public/deepcds_1.0_server_datasets/test"
ACC_FILE = "/tmp/nrt204/homology_audit_work/staged/genome_partitions/test_partition_accessions.txt"

# read length -> (FTP read set, model)
RUNS = {75: ("typical_error", "SI"), 100: ("typical_error", "SI"), 150: ("typical_error", "SI"),
        300: ("typical_error", "SI"), 700: ("error_free", "none"), 1000: ("error_free", "none")}
MODEL_FILES = {"none": "deepcds", "S": "deepcds_S", "SI": "deepcds_SI"}
MODEL_NAMES = {"none": "DeepCDS N", "S": "DeepCDS S", "SI": "DeepCDS S+I"}
N_MAX = 100_000
TRAINED_WINDOW_NT = 300
COLUMNS = ["device", "hardware", "cpu_threads", "model", "read_set", "read_length", "batch_size", "n_reads", "repeats",
           "status", "model_load_s", "time_median_s", "time_min_s", "time_max_s", "reads_per_s", "kbp_per_s",
           "peak_gpu_mem_mib", "torch_max_reserved_mib", "torch_max_allocated_mib", "peak_rss_mib", "other_gpu_procs", "stride_aa"]


# ── read sets ──────────────────────────────────────────────────────────────────

def read_fasta_gz(path):
    recs, name, seq = [], None, []
    with gzip.open(path, "rt") as fh:
        for line in fh:
            if line[0] == ">":
                if name is not None:
                    recs.append((name, "".join(seq)))
                name, seq = line[1:].split()[0], []
            else:
                seq.append(line.strip())
    if name is not None:
        recs.append((name, "".join(seq)))
    return recs


def download(url, dst):
    if not os.path.exists(dst):
        urllib.request.urlretrieve(url, dst + ".part")
        os.replace(dst + ".part", dst)
    return dst


def prepare_reads(out, length, read_set):
    """N_MAX reads of one test set: genomes in seeded random order until the pool holds N_MAX reads."""
    path = f"{out}/reads/{read_set}_{length}bp.fasta"
    if os.path.exists(path):
        return path
    dl = f"{out}/downloads/{read_set}_{length}bp"
    os.makedirs(dl, exist_ok=True)
    accs = open(ACC_FILE).read().split()
    random.Random(42).shuffle(accs)
    pool = []
    with ThreadPoolExecutor(8) as ex:
        for i in range(0, len(accs), 8):
            files = ex.map(lambda a: download(f"{FTP}/{read_set}/{length}bp/{a}.fasta.gz", f"{dl}/{a}.fasta.gz"),
                           accs[i:i + 8])
            for f in files:
                pool += read_fasta_gz(f)
            if len(pool) >= N_MAX:
                break
    reads = random.Random(42).sample(pool, min(N_MAX, len(pool)))
    os.makedirs(f"{out}/reads", exist_ok=True)
    with open(path + ".part", "w") as fh:
        for name, seq in reads:
            fh.write(f">{name}\n{seq}\n")
    os.replace(path + ".part", path)
    print(f"{read_set} {length} bp: {len(reads):,} reads sampled from {len(pool):,} ({i + 8} genomes downloaded)",
          flush=True)
    return path


# ── worker: one (device, read length, batch size) ──────────────────────────────

def gpu_procs(gpu):
    """{pid: MiB} of the compute processes on one (physical) GPU."""
    q = subprocess.run(["nvidia-smi", "-i", str(gpu), "--query-compute-apps=pid,used_memory",
                        "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10).stdout
    return {int(p): int(m) for p, m in (l.split(",") for l in q.splitlines() if "," in l)}


class GpuMemPoller(threading.Thread):
    """Peak device memory of this process according to nvidia-smi, and any other process seen on the same GPU."""

    def __init__(self, gpu):
        super().__init__(daemon=True)
        self.gpu, self.peak, self.others, self.stop = gpu, 0, set(), threading.Event()

    def run(self):
        pid = os.getpid()
        while not self.stop.is_set():
            try:
                procs = gpu_procs(self.gpu)
                self.peak = max(self.peak, procs.get(pid, 0))
                self.others |= set(procs) - {pid}
            except Exception:
                pass
            self.stop.wait(0.1)


def worker(a):
    import torch
    sys.path.insert(0, INFERENCE_DIR)
    import predict_with_deepcds as dc

    device = torch.device(a.device)
    if a.device == "cpu":
        torch.set_num_threads(a.cpu_threads)
    poller = None
    if a.device == "cuda":
        poller = GpuMemPoller(a.gpu)
        poller.start()

    name = MODEL_FILES[a.error_model]
    t0 = time.perf_counter()
    model, mapping = dc.load_model(ckpt_path=f"{a.models_dir}/{name}.pth",
                                   label_mapping_path=f"{INFERENCE_DIR}/configs/{name}/label_mapping.pkl",
                                   hyperparams_path=f"{INFERENCE_DIR}/configs/{name}/hyperparameters.yaml",
                                   device=device, esm2_model="facebook/esm2_t6_8M_UR50D",
                                   label_classes=6 if a.error_model == "SI" else 4)
    if a.device == "cuda":
        torch.cuda.synchronize()
    load_s = time.perf_counter() - t0
    label_lut = dc.build_label_lut(mapping)
    dtype = torch.float32

    reads = []
    with open(a.reads) as fh:
        for line in fh:
            if line[0] == ">":
                nm = line[1:].strip()
            else:
                reads.append((nm, line.strip()))
                if len(reads) == a.n_reads:
                    break

    def predict(recs):
        """Both strands, as in predict_with_deepcds.main()."""
        names = [n for n, _ in recs]
        bufs = {n: dc.io.StringIO() for n in names}
        lens = {n: len(s) for n, s in recs}
        for strand in ("+", "-"):
            seqs = [s if strand == "+" else dc.reverse_complement(s) for _, s in recs]
            kw = {} if strand == "+" else {"strand": "-", "seq_lengths": lens}
            short = [i for i, s in enumerate(seqs) if len(s) <= TRAINED_WINDOW_NT]
            long_ = [i for i, s in enumerate(seqs) if len(s) > TRAINED_WINDOW_NT]
            if short:
                dc.run_direct_inference(model, [names[i] for i in short], [seqs[i] for i in short], label_lut, device,
                                        dtype, a.batch_size, bufs, 60, **kw)
            if long_:
                dc.run_sliding_window(model, [names[i] for i in long_], [seqs[i] for i in long_], label_lut, device,
                                      dtype, a.batch_size, a.stride_aa, bufs, 60, **kw)
        if a.device == "cuda":
            torch.cuda.synchronize()

    res = {"status": "ok", "model_load_s": round(load_s, 2)}
    try:
        predict(reads[:a.batch_size])  # warm-up
        times = []
        for _ in range(a.repeats):
            t0 = time.perf_counter()
            predict(reads)
            times.append(time.perf_counter() - t0)
        med = statistics.median(times)
        res |= {"time_median_s": round(med, 3), "time_min_s": round(min(times), 3), "time_max_s": round(max(times), 3),
                "reads_per_s": round(len(reads) / med, 1),
                "kbp_per_s": round(sum(len(s) for _, s in reads) / med / 1000, 1)}
    except torch.cuda.OutOfMemoryError:
        res["status"] = "OOM"
    if a.device == "cuda":
        poller.stop.set()
        poller.join()
        res |= {"peak_gpu_mem_mib": poller.peak, "other_gpu_procs": len(poller.others),
                "torch_max_reserved_mib": round(torch.cuda.max_memory_reserved() / 2 ** 20),
                "torch_max_allocated_mib": round(torch.cuda.max_memory_allocated() / 2 ** 20)}
    res["peak_rss_mib"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
    res["n_reads"] = len(reads)
    print("RESULT " + json.dumps(res), flush=True)


# ── driver ─────────────────────────────────────────────────────────────────────

def hardware(device, gpu):
    if device == "cuda":
        return subprocess.run(["nvidia-smi", "-i", str(gpu), "--query-gpu=name,memory.total", "--format=csv,noheader"],
                              capture_output=True, text=True).stdout.strip()
    for line in open("/proc/cpuinfo"):
        if line.startswith("model name"):
            return line.split(":", 1)[1].strip()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="/tmp/nrt204/runtime_benchmark_work")
    p.add_argument("--models_dir", default="/home/nrt204/deep-cds/models")
    p.add_argument("--devices", nargs="+", default=["cuda", "cpu"], choices=["cuda", "cpu"])
    p.add_argument("--lengths", nargs="+", type=int, default=list(RUNS))
    p.add_argument("--gpu", type=int, default=0, help="GPU index to run on")
    p.add_argument("--cpu_threads", type=int, default=32)
    p.add_argument("--batch_sizes_gpu", nargs="+", type=int, default=[32, 128, 512, 2048])
    p.add_argument("--batch_sizes_cpu", nargs="+", type=int, default=[32, 128, 512])
    p.add_argument("--n_reads_gpu", type=int, default=100_000)
    p.add_argument("--n_reads_cpu", type=int, default=10_000)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--stride_aa", type=int, default=70, help="sliding-window stride (codons) for reads > 300 bp")
    # worker mode (internal)
    p.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--device", help=argparse.SUPPRESS)
    p.add_argument("--reads", help=argparse.SUPPRESS)
    p.add_argument("--error_model", help=argparse.SUPPRESS)
    p.add_argument("--batch_size", type=int, help=argparse.SUPPRESS)
    p.add_argument("--n_reads", type=int, help=argparse.SUPPRESS)
    a = p.parse_args()
    if a.worker:
        return worker(a)

    os.makedirs(a.out, exist_ok=True)
    tsv = f"{a.out}/runtime_benchmark.tsv"
    done = set()
    if os.path.exists(tsv):
        for line in open(tsv).readlines()[1:]:
            r = dict(zip(COLUMNS, line.rstrip("\n").split("\t")))
            done.add((r["device"], int(r["read_length"]), int(r["batch_size"])))
    else:
        with open(tsv, "w") as fh:
            fh.write("\t".join(COLUMNS) + "\n")
    print(f"host {socket.gethostname()}; GPU {a.gpu}: {hardware('cuda', a.gpu)}; CPU: {hardware('cpu', a.gpu)}, "
          f"{a.cpu_threads} threads", flush=True)

    for length in a.lengths:
        read_set, em = RUNS[length]
        reads = prepare_reads(a.out, length, read_set)
        for device in a.devices:
            for bs in (a.batch_sizes_gpu if device == "cuda" else a.batch_sizes_cpu):
                if (device, length, bs) in done:
                    continue
                n = a.n_reads_gpu if device == "cuda" else a.n_reads_cpu
                env = os.environ | {"CUDA_VISIBLE_DEVICES": str(a.gpu) if device == "cuda" else ""}
                if device == "cpu":
                    env |= {"OMP_NUM_THREADS": str(a.cpu_threads), "MKL_NUM_THREADS": str(a.cpu_threads)}
                cmd = [sys.executable, os.path.abspath(__file__), "--worker", "--device", device, "--reads", reads,
                       "--error_model", em, "--batch_size", str(bs), "--n_reads", str(n), "--repeats", str(a.repeats),
                       "--cpu_threads", str(a.cpu_threads), "--models_dir", a.models_dir, "--stride_aa", str(a.stride_aa)]
                if device == "cuda":
                    cmd += ["--gpu", str(a.gpu)]
                while True:
                    if device == "cuda":
                        while gpu_procs(a.gpu):  # wait until the GPU is free
                            time.sleep(60)
                    proc = subprocess.run(cmd, env=env, capture_output=True, text=True)
                    line = [l for l in proc.stdout.splitlines() if l.startswith("RESULT ")]
                    res = json.loads(line[-1][7:]) if line else {"status": f"failed (exit {proc.returncode})"}
                    if not line:
                        print(proc.stderr[-3000:], flush=True)
                    if not res.get("other_gpu_procs"):
                        break
                    print(f"{device} {length} bp bs {bs}: another process used GPU {a.gpu} during the run; redoing it",
                          flush=True)
                row = {"device": device, "hardware": hardware(device, a.gpu),
                       "cpu_threads": a.cpu_threads if device == "cpu" else "", "model": MODEL_NAMES[em],
                       "read_set": read_set, "read_length": length, "batch_size": bs, "repeats": a.repeats,
                       "n_reads": n, "stride_aa": a.stride_aa if length > TRAINED_WINDOW_NT else ""} | res
                with open(tsv, "a") as fh:
                    fh.write("\t".join(str(row.get(c, "")) for c in COLUMNS) + "\n")
                print(f"{device} {length} bp bs {bs}: {res.get('status')}, {res.get('reads_per_s', '-')} reads/s, "
                      f"GPU {res.get('peak_gpu_mem_mib', '-')} MiB, RSS {res.get('peak_rss_mib')} MiB", flush=True)


if __name__ == "__main__":
    main()
