# Copyright 2024-2025 The vLLM Production Stack Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import abc
import enum
import time
from typing import Dict, List, Optional, Tuple

from fastapi import Request
from uhashring import HashRing

from vllm_router.log import init_logger
from vllm_router.service_discovery import EndpointInfo
from vllm_router.stats.engine_stats import EngineStats
from vllm_router.stats.request_stats import RequestStats
from vllm_router.utils import SingletonABCMeta

logger = init_logger(__name__)


class RoutingLogic(str, enum.Enum):
    ROUND_ROBIN = "roundrobin"
    SESSION_BASED = "session"
    CACHE_AWARE_LOAD_BALANCING = "cache_aware_load_balancing"


class RoutingInterface(metaclass=SingletonABCMeta):
    def __init__(self):
        if not hasattr(self, "_initialized"):
            self._request_counts = {}
            self._initialized = True

    def _update_and_print_stats(self, endpoint_url: str):
        """更新并打印路由统计信息"""
        self._request_counts[endpoint_url] = self._request_counts.get(endpoint_url, 0) + 1
        total_requests = sum(self._request_counts.values())
        logger.info(f"路由统计信息:")
        for url, count in self._request_counts.items():
            percentage = (count / total_requests) * 100 if total_requests > 0 else 0
            logger.info(f"  {url}: {count} 个请求 ({percentage:.2f}%)")
        logger.info(f"总请求数: {total_requests}")

    @abc.abstractmethod
    def route_request(
        self,
        endpoints: List[EndpointInfo],
        engine_stats: Dict[str, EngineStats],
        request_stats: Dict[str, RequestStats],
        request: Request,
    ) -> str:
        """
        Route the request to the appropriate engine URL

        Args:
            endpoints (List[EndpointInfo]): The list of engine URLs
            engine_stats (Dict[str, EngineStats]): The engine stats indicating
                the 'physical' load of each engine
            request_stats (Dict[str, RequestStats]): The request stats
                indicating the request-level performance of each engine
            request (Request): The incoming request
        """
        raise NotImplementedError


class RoundRobinRouter(RoutingInterface):
    # TODO (ApostaC): when available engines in the endpoints changes, the
    # algorithm may not be "perfectly" round-robin.
    def __init__(self):
        super().__init__()
        if hasattr(self, "_initialized"):
            return
        self.req_id = 0
        self._initialized = True

    def route_request(
        self,
        endpoints: List[EndpointInfo],
        engine_stats: Dict[str, EngineStats],
        request_stats: Dict[str, RequestStats],
        request: Request,
    ) -> str:
        """
        Route the request to the appropriate engine URL using a simple
        round-robin algorithm

        Args:
            endpoints (List[EndpointInfo]): The list of engine URLs
            engine_stats (Dict[str, EngineStats]): The engine stats indicating
                the 'physical' load of each engine
            request_stats (Dict[str, RequestStats]): The request stats
                indicating the request-level performance of each engine
            request (Request): The incoming request
        """
        len_engines = len(endpoints)
        chosen = sorted(endpoints, key=lambda e: e.url)[self.req_id % len_engines]
        self.req_id += 1
        self._update_and_print_stats(chosen.url)
        return chosen.url


class SessionRouter(RoutingInterface):
    """
    Route the request to the appropriate engine URL based on the session key
    in the request headers
    """

    def __init__(self, session_key: str = None):
        super().__init__()
        if hasattr(self, "_initialized"):
            return
        if session_key is None:
            raise ValueError("SessionRouter must be initialized with a session_key")
        self.session_key = session_key
        self.hash_ring = HashRing()
        self._initialized = True

    def _qps_routing(
        self, endpoints: List[EndpointInfo], request_stats: Dict[str, RequestStats]
    ) -> str:
        """
        Route the request to the appropriate engine URL based on the QPS of
        each engine

        Args:
            request_stats (Dict[str, RequestStats]): The request stats
                indicating the request-level performance of each engine
        """
        lowest_qps = float("inf")
        ret = None
        for info in endpoints:
            url = info.url
            if url not in request_stats:
                return url  # This engine does not have any requests
            request_stat = request_stats[url]
            if request_stat.qps < lowest_qps:
                lowest_qps = request_stat.qps
                ret = url
        return ret

    def _update_hash_ring(self, endpoints: List["EndpointInfo"]):
        """
        Update the hash ring with the current list of endpoints.
        """
        # Extract endpoint URLs
        endpoint_urls = [endpoint.url for endpoint in endpoints]

        # Get the current nodes in the hash ring
        current_nodes = set(self.hash_ring.get_nodes())

        # Convert the new endpoint URLs to a set for easy comparison
        new_nodes = set(endpoint_urls)

        # Remove nodes that are no longer in the list
        for node in current_nodes - new_nodes:
            self.hash_ring.remove_node(node)

        # Add new nodes that are not already in the hash ring
        for node in new_nodes - current_nodes:
            self.hash_ring.add_node(node)

    def route_request(
        self,
        endpoints: List[EndpointInfo],
        engine_stats: Dict[str, EngineStats],
        request_stats: Dict[str, RequestStats],
        request: Request,
    ) -> str:
        """
        Route the request to the appropriate engine URL by the 'session id' in
        the request headers.
        If there is no session id in the request header, it will pick a server
        with lowest qps

        Args:
            endpoints (List[EndpointInfo]): The list of engine URLs
            engine_stats (Dict[str, EngineStats]): The engine stats indicating
                the 'physical' load of each engine
            request_stats (Dict[str, RequestStats]): The request stats
                indicating the request-level performance of each engine
            request (Request): The incoming request
        """
        session_id = request.headers.get(self.session_key, None)
        logger.debug(f"Got session id: {session_id}")

        # Update the hash ring with the current list of endpoints
        self._update_hash_ring(endpoints)

        if session_id is None:
            # Route based on QPS if no session ID is present
            url = self._qps_routing(endpoints, request_stats)
        else:
            # Use the hash ring to get the endpoint for the session ID
            url = self.hash_ring.get_node(session_id)

        self._update_and_print_stats(url)
        return url


class CacheAwareLoadBalancingRouter(RoutingInterface):
    """
    结合负载均衡和KV Cache命中率感知的路由算法
    
    该算法考虑三个关键因素：
    1. 引擎负载（排队请求数、运行请求数）
    2. 预估的KV缓存命中率（针对特定会话）
    """
    
    # KV缓存配置常量
    DEFAULT_KV_CACHE_SIZE_PER_REQUEST_GB = 0.25  # 每个请求占用的KV缓存大小（GB）
    DEFAULT_GPU_MEMORY_SIZE_GB = 50  # 单个GPU可用显存大小（GB）
    DEFAULT_BLOCK_REUSE_TIMEOUT = 40  # KV缓存块复用超时（秒）
    
    def __init__(self, session_key: str = None, kv_cache_size_per_request_gb: float = None, 
                 gpu_memory_size_gb: float = None, block_reuse_timeout: int = None):
        super().__init__()
        if hasattr(self, "_initialized"):
            return
            
        if session_key is None:
            raise ValueError("CacheAwareLoadBalancingRouter must be initialized with a session_key")
            
        self.session_key = session_key
        self.kv_cache_size_per_request_gb = kv_cache_size_per_request_gb or self.DEFAULT_KV_CACHE_SIZE_PER_REQUEST_GB
        self.gpu_memory_size_gb = gpu_memory_size_gb or self.DEFAULT_GPU_MEMORY_SIZE_GB
        self.block_reuse_timeout = block_reuse_timeout or self.DEFAULT_BLOCK_REUSE_TIMEOUT
        
        # 会话状态跟踪
        self.session_last_time: Dict[str, float] = {}  # 每个会话最后访问时间，为了防止dict过大，要定期清理或限制大小
        self.session_to_engine: Dict[str, str] = {}    # 会话到引擎的映射，为了防止dict过大，要定期清理或限制大小
        
        # 引擎访问记录跟踪
        self.engine_access_history: Dict[str, List[Tuple[str, float]]] = {}  # 每个引擎的会话访问历史 {engine_url: [(session_id, timestamp), ...]}
        self.max_history_per_engine = 1000  # 每个引擎最多保留的历史记录数
        
        # 每个引擎可以容纳的最大会话数
        self.max_sessions_per_engine = int(self.gpu_memory_size_gb / self.kv_cache_size_per_request_gb)
        
        self._initialized = True
    
    def _count_intervening_sessions(self, session_id: str, engine_url: str) -> int:
        """
        计算当前会话的上一次访问到现在，有多少个不同的会话访问了同一个引擎
        
        返回：两次访问之间发生的独立会话数
        """
        if engine_url not in self.engine_access_history:
            return self.max_sessions_per_engine  # 找不到历史记录，假设最大数量
            
        history = self.engine_access_history[engine_url]
        if not history:
            return self.max_sessions_per_engine
            
        # 找到当前会话的上一次访问记录
        last_session_time = 0
        current_time = time.time()
        
        for i in range(len(history) - 1, -1, -1):
            s_id, timestamp = history[i]
            if s_id == session_id:
                last_session_time = timestamp
                break
                
        # 如果找不到上一次访问记录，或者间隔太久，当作最大会话数处理
        if last_session_time == 0 or (current_time - last_session_time > self.block_reuse_timeout):
            return self.max_sessions_per_engine
            
        # 统计在上一次访问之后到现在之间访问的独立会话数
        intervening_sessions = set()
        
        for s_id, timestamp in history:
            if timestamp > last_session_time and s_id != session_id:
                intervening_sessions.add(s_id)
                
        return len(intervening_sessions)
    
    def _predict_cache_hit_rate(self, session_id: str, engine_url: str, 
                               engine_stats: Dict[str, EngineStats]) -> float:
        """
        预测特定会话在特定引擎上的KV缓存命中率
        
        基于以下因素：
        1. 会话最后访问时间（时间间隔越短，命中率越高）
        2. 上次访问到现在间隔的独立会话数（中间访问的会话越多，命中率越低）
        
        返回值：预估的命中率（0.0到1.0之间）
        """
        current_time = time.time()
        
        # 如果会话首次访问或未找到记录，返回0（无命中）
        if session_id not in self.session_last_time:
            return 0.0
            
        # 如果当前引擎与上次使用的引擎不同，返回0（无命中）
        if self.session_to_engine.get(session_id) != engine_url:
            return 0.0
            
        # 计算距离上次访问的时间间隔
        time_since_last_request = current_time - self.session_last_time[session_id]
        
        # 估计KV缓存命中率
        # 1. 时间因素：以线性衰减模型估计基于时间的缓存命中率
        time_factor = min(1.0, time_since_last_request / self.block_reuse_timeout)
        time_based_hit_rate = max(0.0, 1.0 - time_factor)
        
        # 2. 计算中间会话数因素
        intervening_sessions = self._count_intervening_sessions(session_id, engine_url)
        
        # 根据中间会话数与最大容量的比例估计缓存逐出概率
        # 使用非线性公式，当中间会话数较少时，命中率仍然很高
        session_ratio = min(1.0, intervening_sessions / self.max_sessions_per_engine)
        session_based_hit_rate = max(0.0, 1.0 - pow(session_ratio, 0.7))  # 使用幂函数使曲线更平滑
        
        # 综合考虑各因素
        # 时间因素和会话数因素哪个更低，就以哪个为主
        final_hit_rate = min(time_based_hit_rate, session_based_hit_rate)
        
        logger.debug(f"会话 {session_id} 在引擎 {engine_url} 的命中率预测: 时间因素={time_based_hit_rate:.2f}, " 
                    f"会话因素={session_based_hit_rate:.2f} (间隔{intervening_sessions}个会话), 最终={final_hit_rate:.2f}")
        
        return final_hit_rate
    
    def _calculate_engine_load_score(self, engine_url: str, 
                                    engine_stats: Dict[str, EngineStats], 
                                    request_stats: Dict[str, RequestStats]) -> float:
        """
        计算引擎的负载分数
        
        分数越低表示引擎负载越轻
        """
        if engine_url not in engine_stats:
            return 0.0  # 无统计数据，假设负载为0
            
        # 获取引擎统计数据
        stats = engine_stats[engine_url]
        
        # 基本负载因素：运行请求数和排队请求数
        running_load = stats.num_running_requests * 1.0  # 运行中请求权重
        queuing_load = stats.num_queuing_requests * 1.5  # 排队请求权重（略高）
        
        # 考虑QPS
        qps_factor = 0.0
        if engine_url in request_stats:
            qps = request_stats[engine_url].qps
            qps_factor = qps * 0.2  # QPS权重较低
        
        # 计算总负载分数
        total_load_score = running_load + queuing_load + qps_factor
        
        return total_load_score
    
    def _update_session_info(self, session_id: str, engine_url: str):
        """
        更新会话信息和引擎的访问历史
        """
        current_time = time.time()
        
        # 更新会话最后访问时间
        self.session_last_time[session_id] = current_time
        
        # 更新会话到引擎的映射
        self.session_to_engine[session_id] = engine_url
        
        # 更新引擎访问历史
        if engine_url not in self.engine_access_history:
            self.engine_access_history[engine_url] = []
            
        # 添加新的访问记录
        history = self.engine_access_history[engine_url]
        history.append((session_id, current_time))
        
        # 限制历史记录长度，只保留最近的记录
        if len(history) > self.max_history_per_engine:
            # 移除最旧的记录，保留最后max_history_per_engine条
            self.engine_access_history[engine_url] = history[-self.max_history_per_engine:]
            
        # 清理太旧的历史记录
        self._clean_engine_history(engine_url)
    
    def _clean_engine_history(self, engine_url: str):
        """
        清理指定引擎上太旧的访问历史记录
        """
        if engine_url not in self.engine_access_history:
            return
            
        current_time = time.time()
        cutoff_time = current_time - self.block_reuse_timeout * 2  # 保留2倍超时时间内的记录
        
        history = self.engine_access_history[engine_url]
        new_history = [(s_id, timestamp) for s_id, timestamp in history if timestamp >= cutoff_time]
        
        self.engine_access_history[engine_url] = new_history
    
    def _clean_stale_sessions(self):
        """
        清理过期会话信息和限制字典大小
        """
        current_time = time.time()
        stale_sessions = []
        
        # 清理过期会话（超过block_reuse_timeout的两倍）
        for session_id, last_time in self.session_last_time.items():
            if current_time - last_time > self.block_reuse_timeout * 2:  # 双倍超时时间
                stale_sessions.append(session_id)
        
        # 删除过期会话
        for session_id in stale_sessions:
            # 移除会话记录
            self.session_last_time.pop(session_id, None)
            self.session_to_engine.pop(session_id, None)
        
        # 限制字典大小，防止内存泄漏
        MAX_DICT_SIZE = 10000  # 最大允许的字典大小
        
        # 如果字典过大，删除最旧的条目
        if len(self.session_last_time) > MAX_DICT_SIZE:
            # 按访问时间排序，保留最近的MAX_DICT_SIZE个会话
            sorted_sessions = sorted(
                self.session_last_time.items(),
                key=lambda x: x[1],  # 按时间戳排序
                reverse=True  # 降序，最新的在前面
            )[:MAX_DICT_SIZE]
            
            # 重建字典，只保留需要的会话
            new_session_last_time = {sid: ts for sid, ts in sorted_sessions}
            new_session_to_engine = {}
            
            # 只保留still_active中的条目
            for sid in new_session_last_time:
                if sid in self.session_to_engine:
                    new_session_to_engine[sid] = self.session_to_engine[sid]
            
            # 更新字典
            self.session_last_time = new_session_last_time
            self.session_to_engine = new_session_to_engine
            
            # 清理所有引擎的历史记录
            for engine_url in list(self.engine_access_history.keys()):
                self._clean_engine_history(engine_url)
            
            logger.info(f"清理会话缓存，从 {len(sorted_sessions) + len(stale_sessions)} 减少到 {len(self.session_last_time)}")
    
    def _select_best_engine(self, session_id: str, endpoints: List[EndpointInfo],
                          engine_stats: Dict[str, EngineStats],
                          request_stats: Dict[str, RequestStats]) -> str:
        """
        选择最佳引擎，综合考虑负载均衡和缓存命中率
        
        对于每个引擎计算综合分数，选择分数最低的
        """
        best_engine_url = None
        best_score = float('inf')
        
        # 缓存命中权重因子
        cache_weight = 0.6  # 60%的权重给缓存命中
        load_weight = 0.4   # 40%的权重给负载均衡
        
        for info in endpoints:
            engine_url = info.url
            
            # 计算负载分数（分数越低越好）
            load_score = self._calculate_engine_load_score(engine_url, engine_stats, request_stats)
            
            # 预测缓存命中率（越高越好）
            cache_hit_rate = self._predict_cache_hit_rate(session_id, engine_url, engine_stats)
            
            # 将命中率转换为分数（分数越低越好）
            cache_score = 1.0 - cache_hit_rate
            
            # 计算综合得分，加权平均
            combined_score = (cache_score * cache_weight) + (load_score * load_weight)
            
            logger.debug(f"引擎 {engine_url} - 负载: {load_score:.2f}, 预估命中率: {cache_hit_rate:.2f}, 综合分数: {combined_score:.2f}")
            
            # 更新最佳引擎
            if combined_score < best_score:
                best_score = combined_score
                best_engine_url = engine_url
        
        # 如果找不到适合的引擎（极少发生），选择负载最低的
        if best_engine_url is None and endpoints:
            best_engine_url = min(
                [e.url for e in endpoints],
                key=lambda url: self._calculate_engine_load_score(url, engine_stats, request_stats)
            )
            
        return best_engine_url

    def route_request(
        self,
        endpoints: List[EndpointInfo],
        engine_stats: Dict[str, EngineStats],
        request_stats: Dict[str, RequestStats],
        request: Request,
    ) -> str:
        """
        智能路由请求，结合负载感知和缓存命中率预测
        
        对于有会话ID的请求，会基于KV缓存命中率预测和负载状况进行智能选择
        对于无会话ID的请求，会纯粹基于负载均衡选择引擎
        """
        # 定期清理过期会话
        self._clean_stale_sessions()
        
        # 提取会话ID
        session_id = request.headers.get(self.session_key, None)
        logger.debug(f"Got session id: {session_id}")
        
        if session_id is None:
            # 无会话ID，使用纯负载均衡策略
            engine_url = min(
                [e.url for e in endpoints],
                key=lambda url: self._calculate_engine_load_score(url, engine_stats, request_stats)
            )
        else:
            # 有会话ID，使用综合策略
            engine_url = self._select_best_engine(session_id, endpoints, engine_stats, request_stats)
            
            # 更新会话信息
            self._update_session_info(session_id, engine_url)
        
        self._update_and_print_stats(engine_url)
        return engine_url


# Instead of managing a global _global_router, we can define the initialization functions as:
def initialize_routing_logic(
    routing_logic: RoutingLogic, *args, **kwargs
) -> RoutingInterface:
    if routing_logic == RoutingLogic.ROUND_ROBIN:
        logger.info("Initializing round-robin routing logic")
        return RoundRobinRouter()
    elif routing_logic == RoutingLogic.SESSION_BASED:
        logger.info(f"Initializing session-based routing logic with kwargs: {kwargs}")
        return SessionRouter(kwargs.get("session_key"))
    elif routing_logic == RoutingLogic.CACHE_AWARE_LOAD_BALANCING:
        logger.info(f"Initializing cache-aware load balancing routing logic with kwargs: {kwargs}")
        return CacheAwareLoadBalancingRouter(
            kwargs.get("session_key"),
            kwargs.get("kv_cache_size_per_request_gb"),
            kwargs.get("gpu_memory_size_gb"),
            kwargs.get("block_reuse_timeout")
        )
    else:
        raise ValueError(f"Invalid routing logic {routing_logic}")


def reconfigure_routing_logic(
    routing_logic: RoutingLogic, *args, **kwargs
) -> RoutingInterface:
    # Remove the existing routers from the singleton registry
    for cls in (SessionRouter, RoundRobinRouter, CacheAwareLoadBalancingRouter):
        if cls in SingletonABCMeta._instances:
            del SingletonABCMeta._instances[cls]
    return initialize_routing_logic(routing_logic, *args, **kwargs)


def get_routing_logic() -> RoutingInterface:
    # Look up in our singleton registry which router (if any) has been created.
    for cls in (SessionRouter, RoundRobinRouter, CacheAwareLoadBalancingRouter):
        if cls in SingletonABCMeta._instances:
            return cls()
    raise ValueError("The global router has not been initialized")
