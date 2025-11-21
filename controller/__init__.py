from .openfaas_controller import OpenFaaSController, FunctionConfig

__all__ = ["OpenFaaSController", "FunctionConfig", "create_controller"]

def create_controller(namespace="openfaas-fn", metrics_url=None, kubectl_cmd="kubectl", dry_run=False, max_parallel=4, debounce_seconds=0.2):
    ctrl = OpenFaaSController(
        namespace=namespace,
        metrics_url=metrics_url,
        kubectl_cmd=kubectl_cmd,
        dry_run=dry_run,
        max_parallel=max_parallel,
        debounce_seconds=debounce_seconds,
    )
    return ctrl
