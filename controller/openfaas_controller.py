import subprocess
import threading
import time
import json
import queue
import yaml
import random
import urllib.request
import urllib.error

class FunctionConfig:
    def __init__(self, name, namespace, memory, cpu, gpu, replicas, concurrency, labels=None, annotations=None):
        self.name = name
        self.namespace = namespace
        self.memory = memory
        self.cpu = cpu
        self.gpu = gpu
        self.replicas = replicas
        self.concurrency = concurrency
        self.labels = labels or {}
        self.annotations = annotations or {}

    def to_dict(self):
        meta_labels = dict(self.labels)
        meta_labels["smoe/managed"] = "true"
        meta = {
            "name": self.name,
            "namespace": self.namespace,
            "labels": meta_labels,
            "annotations": dict(self.annotations),
        }
        resources = {
            "limits": {
                "memory": self.memory,
                "cpu": self.cpu,
            },
            "requests": {
                "memory": self.memory,
                "cpu": self.cpu,
            },
        }
        if self.gpu > 0:
            resources["limits"]["nvidia.com/gpu"] = self.gpu
        spec = {
            "name": self.name,
            "replicas": int(self.replicas),
            "labels": meta_labels,
            "annotations": dict(self.annotations),
            "limits": {
                "memory": self.memory,
                "cpu": self.cpu,
            },
            "requests": {
                "memory": self.memory,
                "cpu": self.cpu,
            },
            "environment": {
                "max_inflight": str(int(self.concurrency)),
            },
        }
        out = {"apiVersion": "openfaas.com/v1", "kind": "Function", "metadata": meta, "spec": spec}
        return out

    def diff_key(self):
        return (
            self.memory,
            self.cpu,
            self.gpu,
            int(self.replicas),
            int(self.concurrency),
            tuple(sorted(self.labels.items())),
            tuple(sorted(self.annotations.items())),
        )

class ManifestCache:
    def __init__(self):
        self.current = {}
        self.lock = threading.Lock()

    def get(self, name):
        with self.lock:
            return self.current.get(name)

    def update(self, name, cfg):
        with self.lock:
            self.current[name] = cfg

    def changed(self, name, cfg):
        with self.lock:
            old = self.current.get(name)
            if old is None:
                return True
            return old.diff_key() != cfg.diff_key()

class MetricsClient:
    def __init__(self, base_url=None, timeout=1.5):
        self.base_url = base_url
        self.timeout = timeout

    def _fetch(self, path):
        if not self.base_url:
            return None
        url = self.base_url.rstrip("/") + "/" + path.lstrip("/")
        req = urllib.request.Request(url)
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                data = resp.read()
                try:
                    return json.loads(data.decode("utf-8"))
                except Exception:
                    return None
        except urllib.error.URLError:
            return None

    def query_latency(self, name):
        data = self._fetch("metrics/latency/%s" % name)
        if data is None:
            return None
        p95 = data.get("p95")
        p99 = data.get("p99")
        return {"p95": p95, "p99": p99}

    def query_concurrency(self, name):
        data = self._fetch("metrics/concurrency/%s" % name)
        if data is None:
            return None
        cur = data.get("inflight")
        peak = data.get("peak")
        return {"inflight": cur, "peak": peak}

class ApplyTask:
    def __init__(self, cfg, dry_run=False):
        self.cfg = cfg
        self.dry_run = dry_run
        self.timestamp = time.time()
        self.result = None
        self.error = None

class OpenFaaSController:
    def __init__(self, namespace="openfaas-fn", metrics_url=None, kubectl_cmd="kubectl", dry_run=False, max_parallel=4, debounce_seconds=0.2):
        self.namespace = namespace
        self.metrics = MetricsClient(metrics_url)
        self.kubectl_cmd = kubectl_cmd
        self.dry_run = dry_run
        self.cache = ManifestCache()
        self.queue = queue.Queue()
        self.threads = []
        self.max_parallel = max_parallel
        self.debounce_seconds = debounce_seconds
        self.closed = False
        for _ in range(max_parallel):
            t = threading.Thread(target=self._worker_loop, daemon=True)
            t.start()
            self.threads.append(t)

    def build_name(self, model, layer, expert_index):
        return "%s-l%d-e%d" % (model, layer, expert_index)

    def normalize_resources(self, conf, max_mem_gb=10.0, max_cpu=6.0):
        mem_unit = max(0.1, float(conf.get("mem", 1.0)))
        gpu_share = max(0.0, float(conf.get("gpu", 0.0)))
        replicas = max(1, int(conf.get("replicas", 1)))
        concurrency = max(1, int(conf.get("concurrency", 1)))
        mem_gb = min(max_mem_gb, max_mem_gb * mem_unit)
        cpu = max(0.1, max_cpu * (0.2 + 0.8 * mem_unit))
        mem_str = "%.0fMi" % (mem_gb * 1024.0)
        cpu_str = "%.3f" % cpu
        gpu_int = int(round(gpu_share))
        return mem_str, cpu_str, gpu_int, replicas, concurrency

    def build_function_config(self, name, base_labels, base_ann, expert_conf):
        mem, cpu, gpu, replicas, concurrency = self.normalize_resources(expert_conf)
        labels = dict(base_labels)
        annotations = dict(base_ann)
        annotations["smoe/last_update"] = str(time.time())
        labels["smoe/tier"] = "expert"
        labels["smoe/model"] = expert_conf.get("model", "moe")
        labels["smoe/layer"] = str(expert_conf.get("layer", -1))
        labels["smoe/index"] = str(expert_conf.get("index", -1))
        cfg = FunctionConfig(name, self.namespace, mem, cpu, gpu, replicas, concurrency, labels=labels, annotations=annotations)
        return cfg

    def apply_expert_configs(self, model_name, layer_actions):
        base_labels = {"smoe/controller": "smoe"}
        base_ann = {}
        tasks = []
        for layer, experts in layer_actions.items():
            for idx, conf in enumerate(experts):
                name = self.build_name(model_name, layer, idx)
                conf["model"] = model_name
                conf["layer"] = layer
                conf["index"] = idx
                cfg = self.build_function_config(name, base_labels, base_ann, conf)
                if not self.cache.changed(name, cfg):
                    continue
                task = ApplyTask(cfg, dry_run=self.dry_run)
                self.queue.put(task)
                tasks.append(task)
        applied = 0
        failed = 0
        for task in tasks:
            while task.result is None and task.error is None:
                time.sleep(0.01)
            if task.error is not None:
                failed += 1
            else:
                applied += 1
                self.cache.update(task.cfg.name, task.cfg)
        return {"applied": applied, "failed": failed, "queued": len(tasks)}

    def _worker_loop(self):
        while not self.closed:
            try:
                task = self.queue.get(timeout=0.2)
            except queue.Empty:
                continue
            now = time.time()
            wait = self.debounce_seconds - (now - task.timestamp)
            if wait > 0:
                time.sleep(wait)
            try:
                result = self._apply_cfg(task.cfg, task.dry_run)
                task.result = result
            except Exception as e:
                task.error = str(e)
            finally:
                self.queue.task_done()

    def _apply_cfg(self, cfg, dry_run):
        manifest = cfg.to_dict()
        data = yaml.safe_dump(manifest)
        if dry_run:
            return {"dry_run": True, "name": cfg.name}
        cmd = [self.kubectl_cmd, "apply", "-f", "-"]
        p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = p.communicate(input=data.encode("utf-8"), timeout=5.0)
        code = p.returncode
        result = {"code": code, "stdout": out.decode("utf-8", "ignore"), "stderr": err.decode("utf-8", "ignore")}
        return result

    def query_runtime_state(self, name):
        lat = self.metrics.query_latency(name)
        conc = self.metrics.query_concurrency(name)
        merged = {"name": name}
        if lat:
            merged.update(lat)
        if conc:
            merged.update(conc)
        return merged

    def shutdown(self):
        self.closed = True
        for t in self.threads:
            t.join(timeout=0.2)
