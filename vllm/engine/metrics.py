import time
from dataclasses import dataclass
from typing import Counter as CollectionsCounter
from typing import Dict, List, Protocol, Union

import numpy as np
from prometheus_client import (REGISTRY, Counter, Gauge, Histogram, Info,
                               disable_created_metrics)

from vllm.logger import init_logger

logger = init_logger(__name__)

disable_created_metrics()

LABEL_NAME_FINISHED_REASON = "finished_reason"

# The begin-* and end* here are used by the documentation generator
# to extract the metrics definitions.


# begin-metrics-definitions
class Metrics:

    def __init__(self, labelnames: List[str], max_model_len: int):
        # Unregister any existing vLLM collectors
        for collector in list(REGISTRY._collector_to_names):
            if hasattr(collector, "_name") and "vllm" in collector._name:
                REGISTRY.unregister(collector)

        # Config Information
        self.info_cache_config = Info(
            name='vllm:cache_config',
            documentation='information of cache_config')

        # System stats
        self.gauge_scheduler_running = Gauge(
            name="vllm:num_requests_running",
            documentation="Number of requests currently running on GPU.",
            labelnames=labelnames)
        self.gauge_scheduler_swapped = Gauge(
            name="vllm:num_requests_swapped",
            documentation="Number of requests swapped to CPU.",
            labelnames=labelnames)
        self.gauge_scheduler_waiting = Gauge(
            name="vllm:num_requests_waiting",
            documentation="Number of requests waiting to be processed.",
            labelnames=labelnames)
        self.gauge_gpu_cache_usage = Gauge(
            name="vllm:gpu_cache_usage_perc",
            documentation="GPU KV-cache usage. 1 means 100 percent usage.",
            labelnames=labelnames)
        self.gauge_cpu_cache_usage = Gauge(
            name="vllm:cpu_cache_usage_perc",
            documentation="CPU KV-cache usage. 1 means 100 percent usage.",
            labelnames=labelnames)
        
        # Iteration-level stats.
        self.counter_prompt_tokens = Counter(
            name="vllm:prompt_tokens_total",
            documentation="Number of prefill tokens processed.",
            labelnames=labelnames)
        self.counter_generation_tokens = Counter(
            name="vllm:generation_tokens_total",
            documentation="Number of generation tokens processed.",
            labelnames=labelnames)
        self.counter_num_prefill_iterations = Counter(
            name="vllm:prefill_iterations_total",
            documentation="Number of prefill iterations. "
                "Iterations with chunked prefill are counted here.",
            labelnames=labelnames),
        self.counter_num_decode_iterations = Counter(
            name="vllm:decode_iterations_total",
            documentation="Number of decode iterations.",
            labelnames=labelnames),
        self.histogram_iteration_num_tokens = Histogram(
            name="vllm:iteration_num_tokens_total",
            documentation="Histogram of number of total tokens per iteration.",
            labelnames=labelnames,
            buckets=[1, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192] # get batched_max_num_tokens
        )
        self.histogram_time_to_first_token = Histogram(
            name="vllm:time_to_first_token_seconds",
            documentation="Histogram of time to first token in seconds.",
            labelnames=labelnames,
            buckets=[
                0.001, 0.005, 0.01, 0.02, 0.04, 0.06, 0.08, 0.1, 0.25, 0.5,
                0.75, 1.0, 2.5, 5.0, 7.5, 10.0
            ])
        self.histogram_time_per_output_token = Histogram(
            name="vllm:time_per_output_token_seconds",
            documentation="Histogram of time per output token in seconds.",
            labelnames=labelnames,
            buckets=[
                0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.75,
                1.0, 2.5
            ])

        # Request-level data.
        #   Latency breakdown
        self.histogram_e2e_time_request = Histogram(
            name="vllm:e2e_request_latency_seconds",
            documentation="Histogram of end to end request latency in seconds.",
            labelnames=labelnames,
            buckets=[1.0, 2.5, 5.0, 10.0, 15.0, 20.0, 30.0, 40.0, 50.0, 60.0])
        self.histogram_queue_time_request = Histogram(
            name="vllm:request_queue_time_seconds",
            documentation=
            "Histogram of time spent in WAITING phase for request.",
            labelnames=labelnames,
            buckets=[0.1, 1.0, 2.5, 5.0, 10.0, 20.0, 50.0, 100])
        self.histogram_inference_time_request = Histogram(
            name="vllm:request_inference_time_seconds",
            documentation=
            "Histogram of time spent in RUNNING phase for request.",
            labelnames=labelnames,
            buckets=[0.1, 1.0, 2.5, 5.0, 10.0, 20.0, 50.0, 100])
        self.histogram_decode_time_request = Histogram(
            name="vllm:request_decode_time_seconds",
            documentation=
            "Histogram of time spent in DECODE phase for request.",
            labelnames=labelnames,
            buckets=[0.1, 1.0, 2.5, 5.0, 10.0, 20.0, 50.0, 100])
        self.histogram_prefill_time_request = Histogram(
            name="vllm:request_prefill_time_seconds",
            documentation=
            "Histogram of time spent in PREFILL phase for request.",
            labelnames=labelnames,
            buckets=[0.1, 1.0, 2.5, 5.0, 10.0, 20.0, 50.0, 100])
        #   Metadata.
        self.counter_request_success = Counter(
            name="vllm:request_success",
            documentation="Count of successfully processed requests.",
            labelnames=labelnames + [LABEL_NAME_FINISHED_REASON])
        self.histogram_num_prompt_tokens_request = Histogram(
            name="vllm:request_num_prompt_tokens",
            documentation="Histogram of number of prompt tokens for requests.",
            labelnames=labelnames,
            buckets=build_1_2_5_buckets(max_model_len))
        self.histogram_num_generation_tokens_request = Histogram(
            name="vllm:request_num_generation_tokens",
            documentation=
            "Histogram of number of generation tokens for requests.",
            labelnames=labelnames,
            buckets=build_1_2_5_buckets(max_model_len))
        self.histogram_max_num_generation_tokens_request = Histogram(
            name="vllm:request_max_num_generation_tokens",
            documentation=
            "Histogram of maximum number of requested generation tokens.",
            labelnames=labelnames,
            buckets=build_1_2_5_buckets(max_model_len))
        self.histogram_n_request = Histogram(
            name="vllm:request_params_n",
            documentation="Histogram of the n request parameter.",
            labelnames=labelnames,
            buckets=[1, 2, 5, 10, 20])
        self.histogram_best_of_request = Histogram(
            name="vllm:request_params_best_of",
            documentation="Histogram of the best_of request parameter.",
            labelnames=labelnames,
            buckets=[1, 2, 5, 10, 20])
        
        # Legacy metrics
        self.gauge_avg_prompt_throughput = Gauge(
            name="vllm:avg_prompt_throughput_toks_per_s",
            documentation="Average prefill throughput in tokens/s.",
            labelnames=labelnames,
        )
        self.gauge_avg_generation_throughput = Gauge(
            name="vllm:avg_generation_throughput_toks_per_s",
            documentation="Average generation throughput in tokens/s.",
            labelnames=labelnames,
        )
        

# end-metrics-definitions


def build_1_2_5_buckets(max_value: int):
    """
    Builds a list of buckets with increasing powers of 10 multiplied by 
    mantissa values (1, 2, 5) until the value exceeds the specified maximum.

    Example:
    >>> build_1_2_5_buckets(100)
    [1, 2, 5, 10, 20, 50, 100]
    """
    mantissa_lst = [1, 2, 5]
    exponent = 0
    buckets = []
    while True:
        for m in mantissa_lst:
            value = m * 10**exponent
            if value <= max_value:
                buckets.append(value)
            else:
                return buckets
        exponent += 1


@dataclass
class Stats:
    """Created by LLMEngine for use by StatLogger."""
    now: float
    prompt_run: bool
    decode_run: bool

    # System stats (should have _sys suffix).
    num_running_sys: int
    num_waiting_sys: int
    num_swapped_sys: int
    gpu_cache_usage_sys: float
    cpu_cache_usage_sys: float

    # Iteration stats (should have _iter suffix).
    num_prompt_tokens_iter: int
    num_generation_tokens_iter: int
    time_to_first_tokens_iter: List[float]
    time_per_output_tokens_iter: List[float]

    # Request Stats (should have _requests suffix).
    #   Latency
    time_e2e_requests: List[float]
    time_queue_requests: List[float]
    time_inference_requests: List[float]
    time_prefill_requests: List[float]
    time_decode_requests: List[float]
    #   Metadata
    num_prompt_tokens_requests: List[int]
    num_generation_tokens_requests: List[int]
    max_num_generation_tokens_requests: List[int]
    best_of_requests: List[int]
    n_requests: List[int]
    finished_reasons_requests: CollectionsCounter[str]


class SupportsMetricsInfo(Protocol):

    def metrics_info(self) -> Dict[str, str]:
        ...


class StatLogger:
    """StatLogger is used LLMEngine to log to Promethus and Stdout."""

    def __init__(self, local_interval: float, labels: Dict[str, str],
                 max_model_len: int) -> None:
        # Metadata for logging locally.
        self.last_local_log = time.time()
        self.local_interval = local_interval

        # Tracked stats over current local logging interval.
        self.num_prompt_tokens: List[int] = []
        self.num_generation_tokens: List[int] = []

        # Prometheus metrics
        self.labels = labels
        self.metrics = Metrics(labelnames=list(labels.keys()),
                               max_model_len=max_model_len)

    def info(self, type: str, obj: SupportsMetricsInfo) -> None:
        if type == "cache_config":
            self.metrics.info_cache_config.info(obj.metrics_info())

    def _get_throughput(self, tracked_stats: List[int], now: float) -> float:
        return float(np.sum(tracked_stats) / (now - self.last_local_log))

    def _local_interval_elapsed(self, now: float) -> bool:
        elapsed_time = now - self.last_local_log
        return elapsed_time > self.local_interval

    def _log_prometheus(self, stats: Stats) -> None:
        # System state data.
        self._log_gauge(self.metrics.gauge_scheduler_running,
                        stats.num_running_sys)
        self._log_gauge(self.metrics.gauge_scheduler_swapped,
                        stats.num_swapped_sys)
        self._log_gauge(self.metrics.gauge_scheduler_waiting,
                        stats.num_waiting_sys)
        self._log_gauge(self.metrics.gauge_gpu_cache_usage,
                        stats.gpu_cache_usage_sys)
        self._log_gauge(self.metrics.gauge_cpu_cache_usage,
                        stats.cpu_cache_usage_sys)
        
        # Iteration level data.
        self._log_counter(self.metrics.counter_prompt_tokens,
                          stats.num_prompt_tokens_iter)
        self._log_counter(self.metrics.counter_generation_tokens,
                          stats.num_generation_tokens_iter)
        self._log_counter(self.metrics.counter_num_prefill_iterations,
                          (1 if stats.prompt_run else 0))
        self._log_counter(self.metrics.counter_num_decode_iterations,
                          (1 if stats.decode_run else 0))
        self._log_histogram(self.metrics.histogram_iteration_num_tokens,
                            [stats.num_prompt_tokens_iter + stats.num_generation_tokens_iter])
        self._log_histogram(self.metrics.histogram_time_to_first_token,
                            stats.time_to_first_tokens_iter)
        self._log_histogram(self.metrics.histogram_time_per_output_token,
                            stats.time_per_output_tokens_iter)

        # Request level data.
        #   Latency.
        self._log_histogram(self.metrics.histogram_e2e_time_request,
                            stats.time_e2e_requests)
        self._log_histogram(self.metrics.histogram_queue_time_request,
                            stats.time_queue_requests)
        self._log_histogram(self.metrics.histogram_inference_time_request,
                            stats.time_inference_requests)
        self._log_histogram(self.metrics.histogram_prefill_time_request,
                            stats.time_decode_requests)
        self._log_histogram(self.metrics.histogram_decode_time_request,
                            stats.time_prefill_requests)
        
        #   Metadata.
        self._log_counter_labels(self.metrics.counter_request_success, 
                                 stats.finished_reasons_requests,
                                 LABEL_NAME_FINISHED_REASON)
        self._log_histogram(self.metrics.histogram_num_prompt_tokens_request,
                            stats.num_prompt_tokens_requests)
        self._log_histogram(self.metrics.histogram_num_generation_tokens_request,
                            stats.num_generation_tokens_requests)
        self._log_histogram(self.metrics.histogram_max_num_generation_tokens_request,
                            stats.max_num_generation_tokens_requests)
        self._log_histogram(self.metrics.histogram_n_request, 
                            stats.n_requests)
        self._log_histogram(self.metrics.histogram_best_of_request, 
                            stats.best_of_requests)
        


    def _log_gauge(self, gauge: Gauge, data: Union[int, float]) -> None:
        # Convenience function for logging to gauge.
        gauge.labels(**self.labels).set(data)

    def _log_counter(self, counter: Counter, data: Union[int, float]) -> None:
        # Convenience function for logging to counter.
        counter.labels(**self.labels).inc(data)
    
    def _log_counter_labels(self, counter: Counter,
                            data: CollectionsCounter,
                            label_key: str) -> None:
        # Convenience function for collection counter of labels.
        for label, count in data.items():
            counter.labels(**{**self.labels, label_key: label}).inc(count)

    def _log_histogram(self, histogram: Histogram,
                       data: Union[List[int], List[float]]) -> None:
        # Convenience function for logging list to histogram.
        for datum in data:
            histogram.labels(**self.labels).observe(datum)

    def _log_prometheus_interval(self, prompt_throughput: float,
                                 generation_throughput: float) -> None:
        # Logs metrics to prometheus that are computed every logging_interval.
        # Support legacy gauge metrics that make throughput calculations on
        # the vLLM side. Moving forward, we should use counters like
        # counter_prompt_tokens, counter_generation_tokens
        # Which log raw data and calculate summaries using rate() on the
        # grafana/prometheus side. See
        # https://github.com/vllm-project/vllm/pull/2316#discussion_r1464204666
        self.metrics.gauge_avg_prompt_throughput.labels(
            **self.labels).set(prompt_throughput)
        self.metrics.gauge_avg_generation_throughput.labels(
            **self.labels).set(generation_throughput)

    def log(self, stats: Stats) -> None:
        """Called by LLMEngine.
           Logs to prometheus and tracked stats every iteration.
           Logs to Stdout every self.local_interval seconds."""

        # Log to prometheus.
        self._log_prometheus(stats)

        # Save tracked stats for token counters.
        self.num_prompt_tokens.append(stats.num_prompt_tokens)
        self.num_generation_tokens.append(stats.num_generation_tokens)

        # Log locally every local_interval seconds.
        if self._local_interval_elapsed(stats.now):
            # Compute summary metrics for tracked stats (and log them
            # to promethus if applicable).
            prompt_throughput = self._get_throughput(self.num_prompt_tokens,
                                                     now=stats.now)
            generation_throughput = self._get_throughput(
                self.num_generation_tokens, now=stats.now)
            self._log_prometheus_interval(
                prompt_throughput=prompt_throughput,
                generation_throughput=generation_throughput)

            # Log to stdout.
            logger.info(
                f"Avg prompt throughput: {prompt_throughput:.1f} tokens/s, "
                f"Avg generation throughput: "
                f"{generation_throughput:.1f} tokens/s, "
                f"Running: {stats.num_running} reqs, "
                f"Swapped: {stats.num_swapped} reqs, "
                f"Pending: {stats.num_waiting} reqs, "
                f"GPU KV cache usage: {stats.gpu_cache_usage * 100:.1f}%, "
                f"CPU KV cache usage: {stats.cpu_cache_usage * 100:.1f}%")

            # Reset tracked stats for next interval.
            self.num_prompt_tokens = []
            self.num_generation_tokens = []
            self.last_local_log = stats.now
